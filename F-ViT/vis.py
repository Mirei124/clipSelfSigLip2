"""vis open_clip loaded siglip2"""
import glob
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from models.siglip2 import Siglip2ViT
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, SiglipTextModel
from timm.layers import resample_abs_pos_embed

import open_clip


def encode_dense(self: "open_clip.model.CustomTextCLIP", x: torch.Tensor) -> tuple[torch.Tensor, str]:
    # https://github.com/huggingface/pytorch-image-models/blob/836dd99075a13c4b45808d2c917f66162d28a067/timm/models/vision_transformer.py#L1215
    B, _, H, W = x.shape
    h = H // 16
    w = W // 16
    trunk = self.visual.trunk

    # patch_embed
    x = trunk.patch_embed.proj(x)
    x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    x = trunk.patch_embed.norm(x)

    # _pos_embed
    if trunk.patch_embed.img_size != (H, W):
        patch_size = trunk.patch_embed.patch_size
        pos_embed = resample_abs_pos_embed(
            trunk.pos_embed,
            new_size=(H // patch_size[0], W // patch_size[1]),
            old_size=trunk.patch_embed.grid_size,
            num_prefix_tokens=0
            if trunk.no_embed_class
            else trunk.num_prefix_tokens,
        )
    else:
        pos_embed = trunk.pos_embed

    to_cat = []
    if trunk.cls_token is not None:
        to_cat.append(trunk.cls_token.expand(x.shape[0], -1, -1))
    if trunk.reg_token is not None:
        to_cat.append(trunk.reg_token.expand(x.shape[0], -1, -1))

    if trunk.no_embed_class:
        # deit-3, updated JAX (big vision)
        # position embedding does not overlap with class token, add then concat
        x = x + pos_embed
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed

    x = trunk.pos_drop(x)
    # end _pos_embed

    x = trunk.patch_drop(x)
    x = trunk.norm_pre(x)

    for i, blk in enumerate(trunk.blocks[:-1]):
        x = blk(x)

    x = trunk.blocks[-1](x)
    x = trunk.norm(x)

    # extract dense feat
    # qq attn
    N = x.shape[1]
    head_dim = trunk.attn_pool.head_dim
    n_head = x.shape[-1] // head_dim
    q = trunk.attn_pool.q(x)
    k, v = trunk.attn_pool.kv(x).chunk(2, dim=-1)
    q = q.view(B, N, n_head, head_dim).transpose(1, 2)
    k = k.view(B, N, n_head, head_dim).transpose(1, 2)
    v = v.view(B, N, n_head, head_dim).transpose(1, 2)

    attn_weight = torch.matmul(q, q.transpose(-1, -2)) * (head_dim**-0.5)
    attn_weight = attn_weight.softmax(dim=-1)
    attn_out = torch.matmul(attn_weight, v)
    attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)
    x = trunk.attn_pool.proj(attn_out)

    return x, 'qq attn ori'

    residual = x
    x = trunk.attn_pool.norm(x)
    x = residual + trunk.attn_pool.mlp(x)
    return x, 'qq attn + mlp ori'
    # end qq attn

    # proj v
    # q = trunk.attn_pool.q(x)
    # k, v = trunk.attn_pool.kv(x).chunk(2, dim=-1)
    # x = trunk.attn_pool.proj(v)

    # residual = x
    # x = trunk.attn_pool.norm(x)
    # x = residual + trunk.attn_pool.mlp(x)
    # end proj v

    if not self.training:
        x = F.normalize(x, dim=-1)
        feature_map = x.view(B, h, w, -1).permute(0, 3, 1, 2)
    else:
        feature_map = None


model = open_clip.create_model("ViT-B-16-SigLIP2", "webli")
# model.load_state_dict(
#     torch.load(
#         "../logs/clipself_coco_6_save6_test1_ViT-B-16-SigLIP2_12layers/checkpoints/epoch_6.pt"
#     )["state_dict"]
# )

model.encode_dense = encode_dense.__get__(model)


processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained(
    "google/siglip2-base-patch16-224", trust_remote_code=True
)
text_model = SiglipTextModel.from_pretrained(
    "google/siglip2-base-patch16-224", trust_remote_code=True
)


def run_siglip2_on_random_coco_image(root: str, text_prompt: str):
    # paths = list(Path(root).glob("*.jpg"))
    # if not paths:
    #     raise FileNotFoundError(f"No images found under {root}")
    # img_path = random.choice(paths)
    img_path = f"{root}/000000001584.jpg"
    img = Image.open(img_path).convert("RGB")

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        img_feat, desc = model.encode_dense(inputs["pixel_values"])
        # B, L, C
        # extract a single vector from various possible outputs

        text_inputs = tokenizer(
            text_prompt,
            return_tensors="pt",
            max_length=64,  # siglip's max_pos_emb length
            padding="max_length",
            truncation=True,
        )
        text_feat = text_model(**text_inputs).last_hidden_state

        img_feat = F.normalize(img_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        text_feat = text_feat.mean(dim=1)
        sim = text_feat @ img_feat[0].T

        sim_map = sim.reshape(14, 14)

    sim_map = F.interpolate(
        sim_map.unsqueeze(0).unsqueeze(0), (224, 224), mode="bilinear"
    )[0][0].numpy()

    plt.figure()
    plt.imshow(img)
    # crop to center square and resize to 224x224
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    plt.imshow(img)
    plt.imshow(sim_map, alpha=0.7, cmap="turbo")
    title = f"{img_path.split('/')[-1][:-4]} {desc}"
    plt.title(title + f"\n{text_prompt}")
    plt.tight_layout()
    plt.savefig("output_vis/" + title + ".png")


# coco_val_root = "F-ViT/data/coco/val2017"
coco_val_root = "data/coco/val2017"
os.makedirs("output_vis", exist_ok=True)
run_siglip2_on_random_coco_image(coco_val_root, "a photo of a bus")

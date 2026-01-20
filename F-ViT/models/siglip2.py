import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.utils.logging import print_log
from mmdet.models.builder import BACKBONES
from timm.layers import resample_abs_pos_embed

import open_clip


@BACKBONES.register_module()
class Siglip2ViT(BaseModule):
    def __init__(
        self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None
    ):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained  # the pretrained .pt file

        # import pdb; pdb.set_trace()
        clip_model = open_clip.create_model("ViT-B-16-SigLIP2", pretrained="webli")

        self.embed_dim = 768  # output dim
        self.width = width = 768
        self.patch_size = 16
        self.interpolate1 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
            build_norm_layer(norm_cfg, width)[1] if norm_cfg else nn.Identity(),
            nn.GELU(),
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate2 = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=2, stride=2),
        )
        self.interpolate3 = nn.Identity()
        self.interpolate4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.visual = clip_model.visual

    def init_weights(self):
        state_dict = torch.load(
            "../logs/clipself_coco_6_save6_test1_ViT-B-16-SigLIP2_12layers/checkpoints/epoch_6.pt"
        )["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("visual"):
                new_state_dict[k[7:]] = v
        print_log(self.visual.load_state_dict(new_state_dict, strict=True))
        for param in self.visual.parameters():  # only freeze the CLIP model
            param.requires_grad = False

    def train(self, mode=True):
        print(f"Set train mode for SigLIP2: {mode}", flush=True)
        self.training = mode
        self.visual.train(False)
        self.interpolate1.train(mode)
        self.interpolate2.train(mode)
        self.interpolate3.train(mode)
        self.interpolate4.train(mode)

        return self

    def forward(self, x):
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

        outs = []
        for i, blk in enumerate(trunk.blocks[:-1]):
            x = blk(x)
            if i in self.vit_layers:
                outs.append(self._expand_x(x, h, w))

        x = trunk.blocks[-1](x)
        x = trunk.norm(x)

        # extract dense feat
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

        # residual = x
        # x = trunk.attn_pool.norm(x)
        # x = residual + trunk.attn_pool.mlp(x)

        if (len(trunk.blocks) - 1) in self.vit_layers:
            outs.append(self._expand_x(x, h, w))

        if not self.training:
            x = F.normalize(x, dim=-1)
            feature_map = x.view(B, h, w, -1).permute(0, 3, 1, 2)
        else:
            feature_map = None

        assert len(outs) == 4
        for idx, out in enumerate(outs):
            interpolate = getattr(self, f"interpolate{idx + 1}")
            outs[idx] = interpolate(out.detach())

        outs.append(feature_map)

        # import pdb; pdb.set_trace()
        return tuple(outs)

    def _expand_x(self, x, h, w):
        # x: bs q c
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, self.width, h, w)

        return x


if __name__ == "__main__":
    # EVA02-CLIP-B-16
    model = Siglip2ViT("SigLIP2-B-16-224", "google/siglip2-base-patch16-224")
    model(torch.rand(2, 3, 224, 224))
    print()

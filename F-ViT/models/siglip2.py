import copy
import math
import os
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.utils.logging import print_log
from mmdet.models.builder import BACKBONES
from torch import nn
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from transformers import SiglipVisionModel
from transformers.activations import ACT2FN

# Optional imports for compatibility with different transformers versions
try:
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
except ImportError:
    pass
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    # GradientCheckpointingLayer may not exist in newer versions of transformers
    GradientCheckpointingLayer = None
from transformers.modeling_outputs import BaseModelOutput

# Optional imports for compatibility with different transformers versions
try:
    from transformers.modeling_outputs import (
        BaseModelOutputWithPooling,
        ImageClassifierOutput,
    )
except ImportError:
    pass
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
except ImportError:
    pass
try:
    from transformers.processing_utils import Unpack
except ImportError:
    pass
try:
    from transformers.utils import ModelOutput
except ImportError:
    pass
try:
    from transformers.utils import (
        TransformersKwargs,
        auto_docstring,
        can_return_tuple,
        filter_out_non_signature_kwargs,
    )
except ImportError:
    # These may not exist in newer versions of transformers
    pass
try:
    from transformers.utils.generic import check_model_inputs
except ImportError:
    pass
# Optional imports for siglip2 models (may not exist in all transformers versions)
try:
    from transformers.models.siglip2.configuration_siglip2 import (
        Siglip2Config,
        Siglip2TextConfig,
        Siglip2VisionConfig,
    )
except ImportError:
    pass
try:
    from transformers.models.siglip.modeling_siglip import SiglipTextModel
except ImportError:
    pass

class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


def Siglip2_ViT_Wrapper(model: SiglipVisionModel, use_reg=False, maskclip=False):
    @torch.no_grad()
    def init_reg_token(self):
        # use gaussian image patches to initialize the register tokens
        dtype = self.reg_token.dtype
        device = self.reg_token.device
        num_patches = int(math.sqrt(self.num_register_tokens)) + 1
        input = torch.randn(1, 3, num_patches * self.patch_size, num_patches * self.patch_size, dtype=dtype, device=device)
        hidden_states = self.embeddings(input, interpolate_pos_encoding=True)
        reg_token_init = hidden_states[:, -self.num_register_tokens:, :]
        self.reg_token.data.copy_(reg_token_init) # set the register token
        print("Initialized register tokens from gaussian image patches.")
            
    def encode_dense(
        self,
        x
    ):      
        hidden_states = self.embeddings(x, interpolate_pos_encoding=True)
        
        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
    
    def encode_image(
        self,
        x
    ):      
        hidden_states = self.embeddings(x, interpolate_pos_encoding=True)
        
        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        pooler_output = self.head(last_hidden_state)
        
        return pooler_output
    
    def encode_dense_w_proj(
        self,
        x,
    ):      
        hidden_states = self.embeddings(x, interpolate_pos_encoding=True)
        
        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        if self.num_register_tokens > 0:
            # apply projection only on non-register tokens
            last_hidden_state = last_hidden_state[:,:-self.num_register_tokens]
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        #hidden_state = self.attention(probe, hidden_state, hidden_state)[0] this is in siglip's vanilla code
        # this is customized attentionpooling
        attn_weight = self.head.attention.in_proj_weight
        attn_bias = self.head.attention.in_proj_bias
        last_hidden_state_qkv = F.linear(last_hidden_state, attn_weight, attn_bias)
        query, key, value = last_hidden_state_qkv.chunk(3, dim=-1)
        num_heads = self.head.attention.num_heads
        head_dim = query.size(-1) // num_heads
        batch_sz = query.size(0)
            
        last_hidden_state = self.head.attention.out_proj(value)

        residual = last_hidden_state
        last_hidden_state = self.head.layernorm(last_hidden_state)
        last_hidden_state = residual + self.head.mlp(last_hidden_state)
        
        return last_hidden_state
    
    def unlock_last_n_layers(self, n):
        self.requires_grad_(False)
        self.head.requires_grad_(True)
        for i in range(n):
            self.encoder.layers[-(i+1)].requires_grad_(True)
    
    def hook_prepare(self, dense_features, get_states=False, get_v=False):
        class mid_hook_fn():
            def __init__(self, layer_id, type):
                super().__init__()
                self.layer_id = layer_id
                self.type = type
            def __call__(self, module, input, output):
                if dense_features['record'] is True:
                    layer_id = self.layer_id
                    type = self.type
                    model_name = module.__class__.__name__
                    model_name = f'{layer_id}_{type}'
                    dense_features[model_name] = output.detach()

        target_layers = self.encoder.layers
        for i, layer in enumerate(target_layers):
            hook_fn_q = mid_hook_fn(layer_id=i, type='q')
            hook_fn_k = mid_hook_fn(layer_id=i, type='k')
            hook_fn_v = mid_hook_fn(layer_id=i, type='v')
            hook_fn_attn = mid_hook_fn(layer_id=i, type='states')
            #print(f"Registering hooks for layer {i}")
            layer.self_attn.q_proj.register_forward_hook(hook_fn_q)
            layer.self_attn.k_proj.register_forward_hook(hook_fn_k)
            if get_v:
                layer.self_attn.v_proj.register_forward_hook(hook_fn_v)
            if get_states:
                layer.register_forward_hook(hook_fn_attn)
    
    vit = copy.deepcopy(model.vision_model)
    vit.encode_dense = encode_dense.__get__(vit)
    vit.patch_size = vit.embeddings.patch_size
    vit.hook_prepare = hook_prepare.__get__(vit)
    #vit.text_aligned_mode = 'maskclip' if maskclip else 'none'
    vit.encode_dense_w_proj = encode_dense_w_proj.__get__(vit)
    vit.encode_image = encode_image.__get__(vit)
    vit.unlock_last_n_layers = unlock_last_n_layers.__get__(vit)

    del model
    torch.cuda.empty_cache()
    
    # add trainable register token similar to [cls] token
    if use_reg:
        vit.num_register_tokens = 64
        vit.reg_token = nn.Parameter(torch.randn(1, vit.num_register_tokens, vit.config.hidden_size))
        vit.init_reg_token = init_reg_token.__get__(vit)
    else:
        vit.num_register_tokens = 0
    return vit

@BACKBONES.register_module()
class Siglip2ViT(BaseModule):
    def __init__(self, model_name, pretrained, out_indices=[3, 5, 7, 11], norm_cfg=None):
        super().__init__()
        self.vit_layers = out_indices
        self.model_name = model_name
        self.pretrained = pretrained  # the pretrained .pt file
        
        # import pdb; pdb.set_trace()
        clip_model = SiglipVisionModel.from_pretrained(pretrained)

        self.embed_dim = embed_dim = clip_model.vision_model.head.mlp.fc2.weight.shape[0]  # output dim
        self.width = width = clip_model.vision_model.embeddings.patch_embedding.weight.shape[0]
        self.patch_size = patch_size = clip_model.vision_model.embeddings.patch_embedding.weight.shape[-1]
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
        self.visual = clip_model.vision_model

    def init_weights(self):
        clip_model = SiglipVisionModel.from_pretrained(self.pretrained)
        print_log(self.visual.load_state_dict(clip_model.vision_model.state_dict(), strict=True))
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
        visual = self.visual
        bs, _, h, w = x.shape
        h = h // visual.embeddings.patch_embedding.weight.shape[-2]
        w = w // visual.embeddings.patch_embedding.weight.shape[-1]

        with torch.no_grad():
            hidden_states = visual.embeddings(x, interpolate_pos_encoding=True)
        
            # if self.num_register_tokens > 0:
            #     hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

            outs = []
            for i, blk in enumerate(visual.encoder.layers[:-1]):
                hidden_states = blk(hidden_states=hidden_states, attention_mask=None)[0]
                if i in self.vit_layers:
                    outs.append(self._expand_x(hidden_states, h, w))
            
            last_hidden_state = visual.encoder.layers[-1](hidden_states=hidden_states, attention_mask=None)[0]
            if (len(visual.encoder.layers) - 1) in self.vit_layers:
                outs.append(self._expand_x(last_hidden_state, h, w))

            # if self.num_register_tokens > 0:
            #     # apply projection only on non-register tokens
            #     last_hidden_state = last_hidden_state[:,:-self.num_register_tokens]
            
            last_hidden_state = visual.post_layernorm(last_hidden_state)
            
            #hidden_state = self.attention(probe, hidden_state, hidden_state)[0] this is in siglip's vanilla code
            # this is customized attentionpooling
            attn_weight = visual.head.attention.in_proj_weight
            attn_bias = visual.head.attention.in_proj_bias
            last_hidden_state_qkv = F.linear(last_hidden_state, attn_weight, attn_bias)
            query, key, value = last_hidden_state_qkv.chunk(3, dim=-1)
            num_heads = visual.head.attention.num_heads
            head_dim = query.size(-1) // num_heads
            batch_sz = query.size(0)
                
            last_hidden_state = visual.head.attention.out_proj(value)

            residual = last_hidden_state
            last_hidden_state = visual.head.layernorm(last_hidden_state)
            last_hidden_state = residual + visual.head.mlp(last_hidden_state)
            
            if not self.training:
                last_hidden_state = F.normalize(last_hidden_state, dim=-1)
                feature_map = last_hidden_state.view(bs, h, w, -1)
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
        

def Siglip2_Naflex_ViT_Wrapper(model, use_reg=False, clipself_model=False):
    def convert_image_to_patches(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
        """
        Convert 3D tensor image of shape (num_channels, image_height, image_width) into 2D tensor of patches of shape
        (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
        """
        num_channels, image_height, image_width = image.shape
        num_patches_height = image_height // patch_size
        num_patches_width = image_width // patch_size
        patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
        patched_image = patched_image.permute(1, 3, 2, 4, 0)
        patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
        return patched_image
    
    def encode_dense(
        self,
        x,
    ):
        spatial_shape = [torch.tensor(_x.shape[1:]) // self.patch_size for _x in x]
        x = [self.convert_image_to_patches(_x, self.patch_size) for _x in x]
        x = torch.stack(x, dim=0)
        spatial_shape = torch.stack(spatial_shape, dim=0)
        hidden_states = self.embeddings(x, spatial_shape)
        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state
    
    def siglip2_proj(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state
    
    def encode_dense_w_proj(
        self,
        x,
    ):
        spatial_shape = [torch.tensor(_x.shape[1:]) // self.patch_size for _x in x]
        x = [self.convert_image_to_patches(_x, self.patch_size) for _x in x]
        x = torch.stack(x, dim=0)
        spatial_shape = torch.stack(spatial_shape, dim=0)
        hidden_states = self.embeddings(x, spatial_shape)
        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        if self.num_register_tokens > 0:
            # apply projection only on non-register tokens
            last_hidden_state = last_hidden_state[:,:-self.num_register_tokens]
        
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        last_hidden_state = self.dense_head.siglip2_proj(last_hidden_state)

        return last_hidden_state    
    
    def encode_image(
        self,
        x,
    ):
        spatial_shape = [torch.tensor(_x.shape[1:]) // self.patch_size for _x in x]
        x = [self.convert_image_to_patches(_x, self.patch_size) for _x in x]
        x = torch.stack(x, dim=0)
        spatial_shape = torch.stack(spatial_shape, dim=0)
        hidden_states = self.embeddings(x, spatial_shape)
        if self.num_register_tokens > 0:
            hidden_states = torch.cat([hidden_states, self.reg_token.expand(hidden_states.size(0), -1, -1)], dim=1)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooler_output = self.head(last_hidden_state)
        return pooler_output
    
    def get_spatial_shapes(self, pixel_values):
        batch_size, num_tokens, _ = pixel_values.shape
        patch_size = self.embeddings.patch_size
        height = width = int(math.sqrt(num_tokens))
        spatial_shapes = torch.tensor([[height, width]], device=pixel_values.device).repeat(batch_size, 1)
        return spatial_shapes
    
    def hook_prepare(self, dense_features):
        class mid_hook_fn():
            def __init__(self, layer_id, type):
                super().__init__()
                self.layer_id = layer_id
                self.type = type
            def __call__(self, module, input, output):
                if dense_features['record'] is True:
                    layer_id = self.layer_id
                    type = self.type
                    model_name = module.__class__.__name__
                    model_name = f'{layer_id}_{type}'
                    dense_features[model_name] = output.detach()

        target_layers = self.encoder.layers
        for i, layer in enumerate(target_layers):
            hook_fn_q = mid_hook_fn(layer_id=i, type='q')
            hook_fn_k = mid_hook_fn(layer_id=i, type='k')
            hook_fn_v = mid_hook_fn(layer_id=i, type='v')
            #print(f"Registering hooks for layer {i}")
            layer.self_attn.q_proj.register_forward_hook(hook_fn_q)
            layer.self_attn.k_proj.register_forward_hook(hook_fn_k)
            layer.self_attn.v_proj.register_forward_hook(hook_fn_v)

    def modify_dense_head(self):
        self.dense_head.requires_grad_(False)
        self.head.requires_grad_(False)
    
    vit = copy.deepcopy(model.vision_model)
    vit.encode_dense = encode_dense.__get__(vit)
    vit.get_spatial_shapes = get_spatial_shapes.__get__(vit)
    vit.hook_prepare = hook_prepare.__get__(vit)
    vit.convert_image_to_patches = convert_image_to_patches
    vit.patch_size = vit.embeddings.patch_size
    
    if clipself_model:
        vit.encode_dense_w_proj = encode_dense_w_proj.__get__(vit)
        vit.modify_dense_head = modify_dense_head.__get__(vit)
        vit.dense_head = copy.deepcopy(vit.head)
        del vit.dense_head.attention
        vit.dense_head.siglip2_proj = siglip2_proj.__get__(vit.dense_head)
    vit.encode_image = encode_image.__get__(vit)
    
    del model
    torch.cuda.empty_cache()
    
    # add trainable register token similar to [cls] token
    if use_reg:
        vit.num_register_tokens = 32
        vit.reg_token = nn.Parameter(torch.randn(1, vit.num_register_tokens, vit.config.hidden_size))
    else:
        vit.num_register_tokens = 0
    return vit



if __name__ == "__main__":
    # EVA02-CLIP-B-16
    model = Siglip2ViT("SigLIP2-B-16-224", "google/siglip2-base-patch16-224")
    model(torch.rand(2, 3, 224, 224))
    print()
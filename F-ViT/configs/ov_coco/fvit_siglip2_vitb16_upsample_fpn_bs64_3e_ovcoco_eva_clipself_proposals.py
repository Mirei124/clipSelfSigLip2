_base_ = 'fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_original.py'
model = dict(
    backbone=dict(
        type='Siglip2ViT',
        model_name='SigLIP2-B-16-224',
        pretrained='google/siglip2-base-patch16-224',
    ),
)

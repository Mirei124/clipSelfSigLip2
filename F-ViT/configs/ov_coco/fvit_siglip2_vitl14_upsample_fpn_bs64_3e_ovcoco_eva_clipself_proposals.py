_base_ = 'fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_original.py'
model = dict(
    backbone=dict(
        type='Siglip2ViT',
        model_name='SigLIP2-L-16-384',
        pretrained='google/siglip2-large-patch16-384',
    ),
    roi_head=dict(
        bbox_head=dict(
          fc_out_channels=1024,
          class_embed =
          'datasets/embeddings/coco_with_background_siglip2_vitl_16.pt',
        ),
    ),
    neck=dict(
        in_channels=[1024, 1024, 1024, 1024],
    ),
)

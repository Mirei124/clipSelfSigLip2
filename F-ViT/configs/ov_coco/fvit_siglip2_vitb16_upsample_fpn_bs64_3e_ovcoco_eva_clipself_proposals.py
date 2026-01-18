_base_ = 'fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_original.py'
model = dict(
    backbone=dict(
        type='Siglip2ViT',
        model_name='SigLIP2-B-16-224',
        pretrained='google/siglip2-base-patch16-224',
    ),
    roi_head=dict(
        bbox_head=dict(
            fc_out_channels=768,
            class_embed=
            'datasets/embeddings/coco_with_background_siglip2_vitb_16.pt',
        ),
        # vlm_roi_extractor=dict(
        #     out_channels=768,
        # ),
    ),
)

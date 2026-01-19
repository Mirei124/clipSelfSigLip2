#!/bin/bash

################################################################################
EXP_NAME="siglip2-vitl16-$(date +%y%m%d-%H%M%S)-$(openssl rand --hex 3)"

################################################################################
echo "$EXP_NAME"
mkdir -p logs
exec > >(tee "logs/$EXP_NAME.log") 2> >(tee "logs/$EXP_NAME.log" >&2)

################################################################################
cd F-ViT

# gen embedding
# python tools/dump_coco_siglip2_feature.py --out_path datasets/embeddings/coco_with_background_siglip2_vitb_16.pt --model_name SigLIP2-B-16-224 --pretrained google/siglip2-base-patch16-224
# python tools/dump_coco_siglip2_feature.py --out_path datasets/embeddings/coco_with_background_siglip2_vitl_16.pt --model_name SigLIP2-L-16-384 --pretrained google/siglip2-large-patch16-384

# debug
# bash dist_train.sh configs/ov_coco/debug.py 1 --work-dir debug
# exit 0

# vitb16
# bash dist_train.sh configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py 8

# vitl14
# bash dist_train.sh configs/ov_coco/fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py 8

# siglip2 vitb16
# bash dist_train.sh configs/ov_coco/fvit_siglip2_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py 4

# siglip2 vitl16
bash dist_train.sh configs/ov_coco/fvit_siglip2_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py 4


# eval vitb16
#bash dist_test.sh configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py \
#	  work_dirs/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals/latest.pth 1 \
#	    --work-dir eval_dirs/ViTB16 --eval bbox

# eval vitl14
#bash dist_test.sh configs/ov_coco/fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py \
#  work_dirs/fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals/latest.pth 2 \
#  --work-dir eval_dirs/ViTL14 --eval bbox

# eval siglip-vit-b16
# bash dist_test.sh configs/ov_coco/fvit_siglip2_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py \
#   work_dirs/fvit_siglip2_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals/latest.pth 2 \
#   --work-dir eval_dirs/siglip2-ViTB16 --eval bbox

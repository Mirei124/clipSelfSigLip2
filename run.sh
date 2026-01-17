#!/bin/bash

################################################################################
EXP_NAME="siglip-vitb16-$(date +%y%m%d-%H%M%S)-$(openssl rand --hex 3)"

################################################################################
echo "$EXP_NAME"
mkdir -p logs
exec > >(tee "logs/$EXP_NAME.log") 2> >(tee "logs/$EXP_NAME.log" >&2)

################################################################################
cd /data4/zhuotaotian/keyuchen/home/clipSelfSigLip2/CLIPSelf/F-ViT

# vitb16
# bash dist_train.sh configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py 8

# vitl14
# bash dist_train.sh configs/ov_coco/fvit_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py 8

# siglip2 vitb16
bash dist_train.sh configs/ov_coco/fvit_siglip2_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py 8

# siglip2 vitl16
# bash dist_train.sh configs/ov_coco/fvit_siglip2_vitl14_upsample_fpn_bs64_3e_ovcoco_eva_clipself_proposals.py 8

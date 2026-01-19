# python tools/generate_text_embeddings.py --model_version ViT-B-16-SigLIP2 --out_path metadata/coco_panoptic_clip_hand_craft_ViT-B-16-SigLIP2.npy --pretrained webli

#rm logs/clipself_coco_6_save6_test1_ViT-B-16-SigLIP2_12layers -rf
torchrun --master_port 21394 --nproc_per_node 4 -m training.main --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model ViT-B-16-SigLIP2 --pretrained webli --warmup 1000  --zeroshot-frequency 1 --dataset-type grid_distill  \
--test-type coco_panoptic --train-data data/coco/annotations/instances_train2017.json \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_ViT-B-16-SigLIP2.npy --train-image-root data/coco/train2017 \
--val-image-root data/coco/val2017  --cache-dir checkpoints/models--timm--ViT-B-16-SigLIP2 --log-every-n-steps 50 \
--lock-image --save-frequency 6 --lock-image-unlocked-groups 12 --extract-type="v2" \
--name clipself_coco_6_save6_test1_ViT-B-16-SigLIP2_12layers --downsample-factor 16 --det-image-size 1024 \
--alpha 0.7

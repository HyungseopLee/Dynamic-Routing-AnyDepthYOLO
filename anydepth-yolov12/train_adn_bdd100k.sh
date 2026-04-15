#!/bin/bash

#SBATCH --job-name=v12l-ad

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4 # 2 gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1 # 1 task per gpu

#SBATCH --cpus-per-task=32  # 32 cpus, or hyper threads, total
#SBATCH --mem=128G # 8G is reserved for swap


echo "Starting YOLOv12l Fine-tuning on 4x RTX 3090..."
conda activate yolov12

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.run --nproc_per_node 4 train_adn_bdd100k.py \
  --task detect \
  --config ./ultralytics/cfg/models/v12/yolo-ad-v12l.yaml \
  --data bdd100k.yaml \
  --epoch 30 \
  --imgsz 1280 \
  --weight ./pretrained/yolo-ad-exp8_105_epoch539_0.539_0.520.pt \
  --project ./runs/bdd100k/detect/anydepth-yolov12l \
  2>&1 | tee ./runs/bdd100k/detect/anydepth-yolov12l/30e_SGD0900_bs32_nbs256_1e-3_1e-5_1280-740_singleScale_augNothing_volta.log
# context-anydepth-det

Currently active: **`anydepth-yolov12`** (YOLOv12 + AnyDepth skip-layer self-distillation).

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Then edit `ultralytics/cfg/datasets/bdd100k.yaml` to point to your BDD100K paths.



## Weights

Two directories:
- `./pretrained_coco/` — COCO-pretrained weights, used as `--weight` for BDD100K fine-tuning.
- `./finetuned_bdd100k/` — BDD100K fine-tuned weights, used as `--weight` for validation.

Download from [Google Drive](https://drive.google.com/drive/folders/14FmMgIdWbPSiqEa0yzlnkx6apuYdsQzl?usp=drive_link) and place files into the matching directory:


### COCO-pretrained (`./pretrained_coco/`)

| Model | Config | Weight |
|---|---|---|
| Baseline YOLOv12-S | `ultralytics/cfg/models/v12/yolov12s.yaml` | `yolov12s.pt` |
| Baseline YOLOv12-L | `ultralytics/cfg/models/v12/yolov12l.yaml` | `yolov12l.pt` |
| AnyDepth YOLOv12-S | `ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml` | `yolo-ad-small_0.479_0.451.pt` |
| AnyDepth YOLOv12-L | `ultralytics/cfg/models/v12/yolo-ad-v12l.yaml` | `yolo-ad-exp8_105_epoch539_0.539_0.520.pt` |

Leave `--weight` empty to train from scratch.

### BDD100K fine-tuned (`./finetuned_bdd100k/`)

Place fine-tuned checkpoints here (e.g., `best.pt` from `runs/bdd100k/...`) for validation.

## Finetuning on BDD100K

Script: [`train_adn_bdd100k.py`](anydepth-yolov12/train_adn_bdd100k.py). 
Hyperparameters (SGD, batch=32, nbs=256, lr0=1e-3, lrf=1e-2, no mosaic/mixup) are hardcoded in the file.

### AnyDepth YOLOv12-S (2 GPUs)

```bash
mkdir -p ./runs/bdd100k/detect/anydepth-yolov12s
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node 2 train_adn_bdd100k.py \
  --task detect \
  --config ./ultralytics/cfg/models/v12/yolo-ad-v12s-orig.yaml \
  --data bdd100k.yaml \
  --epoch 30 --imgsz 1280 \
  --weight ./pretrained_coco/yolo-ad-small_0.479_0.451.pt \
  --project ./runs/bdd100k/detect/anydepth-yolov12s
```

### Baseline YOLOv12-S (2 GPUs)

```bash
mkdir -p ./runs/bdd100k/detect/baseline-yolov12s
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.run --nproc_per_node 2 train_adn_bdd100k.py \
  --task detect \
  --config ./ultralytics/cfg/models/v12/yolov12s.yaml \
  --data bdd100k.yaml \
  --epoch 30 --imgsz 1280 \
  --weight ./pretrained_coco/yolov12s.pt \
  --project ./runs/bdd100k/detect/baseline-yolov12s
```

Resume training:

```bash
python -m torch.distributed.run --nproc_per_node 2 train_adn_bdd100k.py \
  --task detect --data bdd100k.yaml --resume \
  --weight <project>/train/weights/last.pt
```

## Validate on BDD100K

Script: [`val_adn_bdd100k.py`](anydepth-yolov12/val_adn_bdd100k.py). For AnyDepth weights it runs **Full** (no skips) and **Base** (all skips) automatically; for baseline weights it runs once.

```bash
python val_adn_bdd100k.py \
  --imgsz 1280 \
  --weight ./finetuned_bdd100k/anydepth-yolov12s.pt \
  --project ./runs/bdd100k/detect/anydepth-yolov12s/val
```

Use the same `--imgsz` as in training (1280 for BDD100K).

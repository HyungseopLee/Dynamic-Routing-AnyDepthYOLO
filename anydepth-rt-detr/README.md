## Adaptive Depth RTDETR

- Training on GPUs:

```shell
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco_adn.yml
```

- Evaluation on Multiple GPUs:

```shell
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco_adn.yml -r path/to/checkpoint --test-only
```

- Download checkpoints: 
## Model Zoo

| Model | Backbone | AP (Base) | AP (Super) |  Download |
|-------|----------|-----------|------------|----------|
| RT-DETR-ADN | ResNet50-vd-adn | 50.5 | 53.1 | [link](https://github.com/wchkang/anydepth_detr/releases/download/v1.0.0/checkpoint0067_resnet50vd_AP_53.1.pth) |
| RT-DETR-ADN | ResNet101-vd-adn | 52.6 | 54.6 |[link](https://github.com/wchkang/anydepth_detr/releases/download/v1.0.0/checkpoint0065_resnet101vd_AP_54.6.pth) |


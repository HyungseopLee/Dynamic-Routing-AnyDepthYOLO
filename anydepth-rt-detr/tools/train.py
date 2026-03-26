"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        if cfg.yaml_cfg['task'] == 'detection-adn':
            # @HyungseopLee: Test both FULL and BASE config
            if dist.get_rank() == 0:
                print("=" * 30," Model Architecture ", "=" * 30)
                print(cfg.model)
                print("="*100)
            
            # Test FULL
            backbone_skip=[False, False, False, False]
            encoder_skip=[False,]
            decoder_depth = 6
            solver.val(backbone_skip=backbone_skip, encoder_skip=encoder_skip, decoder_depth=decoder_depth)
            
            # Test BASE
            backbone_skip=[True, True, True, True]
            encoder_skip=[True,]
            decoder_depth = 4
            solver.val(backbone_skip=backbone_skip, encoder_skip=encoder_skip, decoder_depth=decoder_depth)
            
        else:
            solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)

'''

# weight path
./pretrained/checkpoint0065_resnet101vd_AP_54.6.pth
./pretrained/checkpoint0067_resnet50vd_AP_53.1.pth


    
# Baseline ------------------------------------------------------------------


## val on 2-gpu
mkdir -p ./output/baseline_rtdetr_r101vd_6x_coco
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train.py \
    -c ./configs/rtdetr/rtdetr_r101vd_6x_coco.yml \
    -r ./pretrained/rtdetr_r101vd_6x_coco_from_paddle.pth \
    --test-only \
    2>&1 | tee ./output/baseline_rtdetr_r101vd_6x_coco/val_640.log




# Any-depth ------------------------------------------------------------------


## COCO: train on multi-gpu
mkdir -p ./output/rtdetr_r50vd_6x_coco_res50_adn
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco_adn.yml \
    

## BDD100K: train on multi-gpu


## val on 4-gpu
mkdir -p ./output/rtdetr_r50vd_6x_coco_res50_adn
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco_adn.yml \
    -r ./pretrained/checkpoint0067_resnet50vd_AP_53.1.pth \
    --test-only \
    2>&1 | tee ./output/rtdetr_r50vd_6x_coco_res50_adn/val.log        


## val on 2-gpu
mkdir -p ./output/anydepth_rtdetr_r101vd_6x_coco
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train.py \
    -c ./configs/rtdetr/rtdetr_r101vd_6x_coco_adn.yml \
    -r ./pretrained/checkpoint0065_resnet101vd_AP_54.6.pth \
    --test-only \
    2>&1 | tee ./output/anydepth_rtdetr_r101vd_6x_coco/val_640.log


## val on 1-gpu
mkdir -p ./output/rtdetr_r50vd_6x_coco_res50_adn
python tools/train.py \
    -c configs/rtdetr/rtdetr_r50vd_6x_coco_adn.yml \
    -r ./pretrained/checkpoint0067_resnet50vd_AP_53.1.pth \
    --test-only \
    2>&1 | tee ./output/rtdetr_r50vd_6x_coco_res50_adn/val_736.log


'''
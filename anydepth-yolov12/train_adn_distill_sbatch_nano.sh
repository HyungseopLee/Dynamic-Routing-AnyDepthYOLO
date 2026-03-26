#!/bin/bash

#SBATCH --job-name=train_yolov11l
#SBATCH --output=train_yolov11l_%j.out
#SBATCH --error=train_yolov11l_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2		# 2 gpus

#SBATCH --ntasks=2		# 1 task per gpu
#SBATCH --cpus-per-task=16	# 64 cpus, or hyper threads, total
#SBATCH --mem=250G		# total 503G; 8G is reserved for swap          

source ~/miniconda3/bin/activate yolov12

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH; export PYTHONPATH=$(pwd); python my_train_nano.py 2>&1 | tee logs/exp5_orig_yolo_ad_v12n_super2_only.txt

# export PYTHONPATH=$(pwd); python my_train.py 2>&1 | tee logs/yolov12l_ad_exp1_1_bs_ns_64_kd_all_1.5.txt

#export PYTHONPATH=$(pwd); python my_train.py 2>&1 | tee logs/yolov12l_bs64_4gpus.txt

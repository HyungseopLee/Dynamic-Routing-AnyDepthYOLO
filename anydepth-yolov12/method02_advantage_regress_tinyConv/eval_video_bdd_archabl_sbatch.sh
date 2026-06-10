#!/bin/bash
# BDD100K MOT-val router-input ablation (FULL, all 200 clips), BOTH archs:
#   tinyConv  bb=input  bn=both  pn=pred   (policy_*.pt / policy_both_*.pt / policy_pred_*.pt)
#   GAP-MLP   gmbb,gmbn,gmpn                (gapmlp_input/both/pred_*.pt)
# 5 seeds each -> 30 policies, one shared detector forward (2 paths/frame).
# conf 0.25 to match the full3 + image-val curves. 4-way video sharding.
#
#   sbatch method02_advantage_regress_tinyConv/eval_video_bdd_archabl_sbatch.sh
#   # after the array completes:
#   python method02_advantage_regress_tinyConv/merge_video_shards.py \
#       --shards method02_advantage_regress_tinyConv/outputs/bdd100k/eval/archabl_shard_*.pt \
#       --out    method02_advantage_regress_tinyConv/outputs/bdd100k/eval/video_curve_archabl.json

#SBATCH --job-name=eval_bdd_archabl
#SBATCH --output=method02_advantage_regress_tinyConv/outputs/bdd100k/eval/slurm_archabl_%A_%a.out
#SBATCH --error=method02_advantage_regress_tinyConv/outputs/bdd100k/eval/slurm_archabl_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3%4

set -euo pipefail
source ~/miniconda3/bin/activate yolov12 2>/dev/null || source ~/anaconda3/bin/activate yolov12
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

O=method02_advantage_regress_tinyConv/outputs/bdd100k
W=$(ls finetuned_bdd100k/*alpha0.2*mAP35.1*.pt)
POL=""
for s in 0 1 2 3 4; do
  POL+="bb$s=$O/policy_$s.pt,bn$s=$O/policy_both_$s.pt,pn$s=$O/policy_pred_$s.pt,"
  POL+="gmbb$s=$O/gapmlp_input_$s.pt,gmbn$s=$O/gapmlp_both_$s.pt,gmpn$s=$O/gapmlp_pred_$s.pt,"
done
POL=${POL%,}
mkdir -p "$O/eval"
NS=4
SID=${SLURM_ARRAY_TASK_ID:-0}
LOG="$O/eval/archabl_shard_${SID}.log"

echo "[$(date +%T)] shard $SID/$NS  weight=$W  npol=$(echo $POL | tr ',' '\n' | wc -l)"

python method02_advantage_regress_tinyConv/eval_video_bdd.py \
    --weight "$W" --policies "$POL" \
    --grid 2 --imgsz 720 1280 --conf 0.25 \
    --val_cache "$O/cache_val_both.pt" --budgets 10,20,30,40,50,60,70,80,90 \
    --num_shards $NS --shard_id $SID \
    --raw_out "$O/eval/archabl_shard_${SID}.pt" \
    2>&1 | tee "$LOG"

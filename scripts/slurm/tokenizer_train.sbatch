#!/bin/bash
#SBATCH --job-name=tokenizer
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=120:00:00
# usage: MODEL=Phaedra_AE_FSQ_4x4 sbatch scripts/slurm/tokenizer_train.sbatch
set -euo pipefail
source "$(dirname "$0")/../../env.sh" 2>/dev/null || source env.sh
accelerate launch --multi_gpu --num_processes=4 -m tokenizer.train \
    --model "${MODEL:-Phaedra_AE_FSQ_4x4}" --data tokenizer/configs/data_pretraining.yaml

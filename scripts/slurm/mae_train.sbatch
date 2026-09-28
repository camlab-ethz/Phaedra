#!/bin/bash
#SBATCH --job-name=mae
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
# usage: CFG=mae/configs/mae_phaedra.yaml sbatch scripts/slurm/mae_train.sbatch
set -euo pipefail
source env.sh
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((10000 + (${SLURM_JOB_ID:-$$} % 20000)))
torchrun --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT -m mae.train --config "${CFG:-mae/configs/mae_phaedra.yaml}"

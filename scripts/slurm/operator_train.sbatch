#!/bin/bash
#SBATCH --job-name=operator
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=3G
#SBATCH --time=120:00:00
# usage: CFG=configs/operators/phaedra_38m_kh.yaml sbatch scripts/slurm/operator_train.sbatch
#   hub token transformers (Phaedra/FSQ):  python -m hub.trainers.train
#   FNO / CNO / ViT:                       python -m {fno,cno,vit}_operator.train
#   continuous-latent transformer:         python -m baselines.train_continuous
#   VQ-VAE-2 token transformer:            python -m baselines.arch.long_trainer --variant dual_sequential
set -euo pipefail
source env.sh
CFG="${CFG:?set CFG=<config yaml>}"
MOD="${MOD:-hub.trainers.train}"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((10000 + (${SLURM_JOB_ID:-$$} % 20000)))
torchrun --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT -m "$MOD" --config "$CFG" ${EXTRA:-}

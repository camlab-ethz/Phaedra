#!/bin/bash
#SBATCH --job-name=phaedra-smoke
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
# Runs scripts/smoke_test.sh on one GPU. Submit from the repository root after `cp env.example.sh env.sh`.
set -euo pipefail
source env.sh
bash scripts/smoke_test.sh

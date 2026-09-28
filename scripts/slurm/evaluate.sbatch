#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-20
# One array task per model: all three prediction strategies, 240 test trajectories,
# deterministic. Afterwards: python -m evaluation.verify_report
set -euo pipefail
source env.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
MODELS=(phaedra_38m_kh phaedra_38m_rc phaedra_38m_rkh fsq_38m_kh fsq_38m_rc fsq_38m_rkh
        vqvae2_38m_kh vqvae2_38m_rc vqvae2_38m_rkh continuous_38m_kh continuous_38m_rc continuous_38m_rkh
        fno_38m_kh fno_38m_rc fno_38m_rkh cno_38m_kh cno_38m_rc cno_38m_rkh vit_38m_kh vit_38m_rc vit_38m_rkh)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
for MODE in direct rollout2 rollout662; do
  TS="2 4 6 8 10 12 14"; [ "$MODE" = rollout662 ] && TS="6 12 14"
  python -m evaluation.eval_downstream --models "$M" --shard-tag "$M" --timesteps $TS --mode "$MODE" --deterministic
done

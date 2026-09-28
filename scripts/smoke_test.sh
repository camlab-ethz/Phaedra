#!/bin/bash
# End-to-end smoke test on SYNTHETIC data (one GPU, ~20-30 min): every stage of
# the pipeline with tiny models, so a fresh environment can be validated before
# downloading the real data. Usage:
#     source env.sh            # PHAEDRA_DATA_ROOT / PHAEDRA_OUTPUT_ROOT
#     bash scripts/smoke_test.sh [scratch_dir]
# Everything is written under <scratch_dir> (default $PHAEDRA_OUTPUT_ROOT/smoke),
# with PHAEDRA_DATA_ROOT / PHAEDRA_OUTPUT_ROOT redirected there.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${1:-${PHAEDRA_OUTPUT_ROOT:?source env.sh first}/smoke}"
export PHAEDRA_DATA_ROOT="$SCRATCH/data" PHAEDRA_OUTPUT_ROOT="$SCRATCH/out" PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
mkdir -p "$PHAEDRA_DATA_ROOT" "$PHAEDRA_OUTPUT_ROOT"
cd "$ROOT"
STAGE_FROM="${STAGE_FROM:-0}"   # e.g. STAGE_FROM=3 re-runs from the operator stage
step() { echo; echo "================ [$(date +%H:%M:%S)] $*"; }
skip() { [ "${1}" -lt "$STAGE_FROM" ]; }

step "0. toy fields ($PHAEDRA_DATA_ROOT/fields)"
skip 0 || python scripts/make_toy_dataset.py --out-dir "$PHAEDRA_DATA_ROOT/fields" --members 64
CFG=$(python scripts/smoke/make_smoke_configs.py)

step "1. tokenizers (Phaedra 4x4, FSQ, VQ-VAE-2, continuous AE) -- 30 steps each"
skip 1 || for M in Phaedra_AE_FSQ_4x4 AE_FSQ AE_VQVAE2 AE_Continuous; do
  python -m tokenizer.train --model $M --data scripts/smoke/data_toy.yaml --max-steps 30 --max-train-members 8
done
ls "$PHAEDRA_OUTPUT_ROOT"/tokenizers/*/

step "2. tokens + latents"
skip 2 || for DS in kh:KelvinHelmholtz rc:RiemannCurved rkh:RiemannKelvinHelmholtz; do
  key=${DS%%:*}; name=${DS##*:}
  python -m tokens.generate_tokens --model-name Phaedra_AE_FSQ_4x4 --model-path "$PHAEDRA_OUTPUT_ROOT/tokenizers/phaedra_4x4" \
      --config "$CFG/tokens_$key.yaml" --output-dir "$PHAEDRA_DATA_ROOT/tokens/phaedra" --output-name "CEU2D_${name}Tokens.nc" --use-ema --batch-size 21
  python -m tokens.generate_tokens --model-name AE_FSQ --model-path "$PHAEDRA_OUTPUT_ROOT/tokenizers/fsq" \
      --config "$CFG/tokens_$key.yaml" --output-dir "$PHAEDRA_DATA_ROOT/tokens/fsq" --output-name "CEU2D_${name}Tokens.nc" --use-ema --batch-size 21
done
skip 2 || python -m tokens.encode_vqvae2_tokens --datasets kh rc rkh --n-val 8 --n-test 8 --roundtrip-members 2
skip 2 || python -m tokens.encode_continuous_latents --datasets kh rc rkh --n-val 8 --n-test 8 --roundtrip-members 2
ls "$PHAEDRA_DATA_ROOT"/tokens/* "$PHAEDRA_DATA_ROOT"/latents/*

step "3. operators (7 families x 3 datasets, 1 tiny epoch each)"
skip 3 || for DS in kh rc rkh; do
  python -m hub.trainers.train --config "$CFG/phaedra_38m_${DS}.yaml"
  python -m hub.trainers.train --config "$CFG/fsq_38m_${DS}.yaml"
  python -m fno_operator.train --config "$CFG/fno_38m_${DS}.yaml"
  python -m cno_operator.train --config "$CFG/cno_38m_${DS}.yaml"
  python -m vit_operator.train --config "$CFG/vit_38m_${DS}.yaml"
  python -m baselines.train_continuous --config "$CFG/continuous_38m_${DS}.yaml"
  python -m baselines.arch.long_trainer --config "$CFG/vqvae2_38m_${DS}.yaml" --variant dual_sequential
done

step "4. per-package test scripts (KH)"
skip 4 || python -m hub.trainers.test --config "$CFG/phaedra_38m_kh.yaml"
skip 4 || python -m fno_operator.test --config "$CFG/fno_38m_kh.yaml"

step "5. masked autoencoding (3-PDE pretrain -> RKH fine-tune -> test)"
skip 5 || python -m mae.train --config "$CFG/mae_phaedra_3pde.yaml" --max_steps 20
skip 5 || python -m mae.train --config "$CFG/mae_phaedra_finetune_rkh.yaml" --max_steps 20
skip 5 || python -m mae.test --config "$CFG/test_mae_phaedra_finetune_rkh.yaml"

step "6. unified evaluation (direct / 2-step / 6-6-2) + report + normalization audit"
for MODE in direct rollout2 rollout662; do
  TS="2 14"; [ "$MODE" = rollout662 ] && TS="6 12 14"
  python -m evaluation.eval_downstream --registry "$CFG/eval_registry.yaml" --all --mode $MODE --timesteps $TS --max-members 2 --shard-tag smoke_$MODE
done
python -m evaluation.verify_dump_fields --registry "$CFG/eval_registry.yaml" --members 56
python -m evaluation.verify_report --members 56
python -m evaluation.verify_normalization --registry "$CFG/eval_registry.yaml" --members 2 --stat-members 4
echo; echo "SMOKE TEST PASSED -- outputs under $SCRATCH"

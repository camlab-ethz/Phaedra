#!/bin/bash
#SBATCH --job-name=tokens
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
# Writes the token / latent files every operator config reads.
set -euo pipefail
source env.sh
for DS in kh:KelvinHelmholtz rc:RiemannCurved rkh:RiemannKelvinHelmholtz; do
  key=${DS%%:*}; name=${DS##*:}
  python -m tokens.generate_tokens --model-name Phaedra_AE_FSQ_4x4 --model-path "$PHAEDRA_OUTPUT_ROOT/tokenizers/phaedra_4x4" \
      --config tokens/configs/ceu_$key.yaml --output-dir "$PHAEDRA_DATA_ROOT/tokens/phaedra" --output-name "CEU2D_${name}Tokens.nc" --use-ema --batch-size 21
  python -m tokens.generate_tokens --model-name AE_FSQ --model-path "$PHAEDRA_OUTPUT_ROOT/tokenizers/fsq" \
      --config tokens/configs/ceu_$key.yaml --output-dir "$PHAEDRA_DATA_ROOT/tokens/fsq" --output-name "CEU2D_${name}Tokens.nc" --use-ema --batch-size 21
done
python -m tokens.encode_vqvae2_tokens --datasets kh rc rkh
python -m tokens.encode_continuous_latents --datasets kh rc rkh

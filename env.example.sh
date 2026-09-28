#!/bin/bash
# Copy to env.sh, edit the two roots, then `source env.sh` in every shell / job.
#
#   PHAEDRA_DATA_ROOT/            PHAEDRA_OUTPUT_ROOT/
#     fields/CEU_2D_*.nc            tokenizers/{phaedra_4x4,fsq,vqvae2,continuous}/
#     tokens/{phaedra,fsq,vqvae2}/  operators/<run_name>/
#     latents/continuous/           mae/<run_name>/
#                                   evaluation/{tables,figures,fields}/
export PHAEDRA_DATA_ROOT="${PHAEDRA_DATA_ROOT:-$HOME/phaedra_data}"
export PHAEDRA_OUTPUT_ROOT="${PHAEDRA_OUTPUT_ROOT:-$HOME/phaedra_outputs}"
# Repository root on PYTHONPATH (alternative: pip install -e .)
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):${PYTHONPATH:-}"
mkdir -p "$PHAEDRA_DATA_ROOT" "$PHAEDRA_OUTPUT_ROOT"
echo "PHAEDRA_DATA_ROOT=$PHAEDRA_DATA_ROOT  PHAEDRA_OUTPUT_ROOT=$PHAEDRA_OUTPUT_ROOT"

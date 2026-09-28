# SLURM templates

These reproduce the compute used for the paper (adapt partition / GPU names).
Every script assumes `source env.sh` has been run and the repository root is
on `PYTHONPATH`. Multi-GPU jobs use `torchrun` (hub / baselines / MAE) or
`accelerate launch` (tokenizers).

| script | stage | GPUs used for the paper |
|---|---|---|
| `tokenizer_train.sbatch` | tokenizer pre-training (any of the 4 models) | 4x (24 GB) |
| `tokens_generate.sbatch` | token/latent generation | 1x |
| `operator_train.sbatch` | one operator run (any family) | 8x (24 GB) |
| `mae_train.sbatch` | MAE pre-training / fine-tuning | 8x |
| `evaluate.sbatch` | evaluation sweep (one task per model) | 1x per task |

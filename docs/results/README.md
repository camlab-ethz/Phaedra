# Result tables

Relative $L_1$ errors (%) of the 21 neural operators of the paper on the 240 test trajectories of each
dataset (KH, RC, RKH), as reported in the paper. Every table is provided as Markdown (`.md`), LaTeX (`.tex`)
and CSV.

| file | content |
|---|---|
| `final_t14_best` | per-variable and average error at the final time $t = 0.7$ (time index 14), each model under its best prediction strategy, ± 95 % bootstrap CI of the mean — the main table |
| `final_t14_{direct,rollout2,rollout662}` | the same for one strategy |
| `per_timestep_{direct,rollout2,rollout662}` | average error at every evaluated time step |
| `strategy_compare_t14` | average error at $t = 0.7$ under every strategy |
| `tokenizer_floor_t14` | error of the decoded ground-truth tokens / latents at $t = 0.7$ (first 16 test trajectories): the floor of each representation (`python -m evaluation.verify_normalization`) |
| `mae` | masked autoencoders (75 % of the tokens masked): error of the decoded reconstructions (`python -m mae.test`; `.md` and `.csv` only) |

Strategies: **direct** — one step $0 \to t$; **2-step** — autoregressive $0 \to 2 \to \dots \to 14$;
**6-6-2** — $0 \to 6 \to 12 \to 14$. Column `strategy` / `best`: `d` = direct, `r2` = 2-step, `662` = 6-6-2.

Regenerate them from the released weights (see the main README): run `scripts/slurm/evaluate.sbatch`
(`python -m evaluation.eval_downstream --deterministic` for every model and strategy), then
`python -m evaluation.verify_report`, which writes these tables to `$PHAEDRA_OUTPUT_ROOT/evaluation/tables`.
On an RTX 4090 every per-trajectory error is bit-identical to the paper's evaluation; the confidence
intervals are bootstrap estimates (10,000 resamples), whose last digit can differ.

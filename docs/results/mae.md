### Masked autoencoders on token grids

Relative $L_1$ (%) of the fields decoded from the reconstructed token grids (75 % of the tokens masked), as evaluated in the paper. The masks are random; each number is one evaluation run.

| Model | Tokens | Training | Evaluated on | Test trajectories | ρ | u | v | p | Average |
|---|---|---|---|---|---|---|---|---|---|
| `mae_phaedra_3pde` | Phaedra | pre-trained on KH + RC + RKH | RKH | 10 | 11.24 | 25.86 | 22.86 | 5.78 | 16.43 |
| `mae_fsq_3pde` | FSQ | pre-trained on KH + RC + RKH | RKH | 10 | 11.83 | 26.28 | 23.44 | 6.11 | 16.91 |
| `mae_phaedra_finetune_kh` | Phaedra | fine-tuned on KH (from the 3-PDE model) | KH | 10 | 1.93 | 3.82 | 12.94 | 0.33 | 4.76 |
| `mae_fsq_finetune_kh` | FSQ | fine-tuned on KH (from the 3-PDE model) | KH | 1 | 2.25 | 4.90 | 17.52 | 0.48 | 6.29 |
| `mae_phaedra_finetune_rc` | Phaedra | fine-tuned on RC (from the 3-PDE model) | RC | 10 | 15.91 | 33.62 | 32.03 | 7.14 | 22.17 |
| `mae_fsq_finetune_rc` | FSQ | fine-tuned on RC (from the 3-PDE model) | RC | 1 | 23.14 | 41.99 | 37.19 | 10.46 | 28.20 |
| `mae_phaedra_finetune_rkh` | Phaedra | fine-tuned on RKH (from the 3-PDE model) | RKH | 10 | 4.87 | 11.48 | 10.19 | 2.55 | 7.27 |
| `mae_fsq_finetune_rkh` | FSQ | fine-tuned on RKH (from the 3-PDE model) | RKH | 1 | 4.21 | 9.22 | 12.60 | 2.23 | 7.07 |

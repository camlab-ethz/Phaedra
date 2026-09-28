### Tokenizer reconstruction floor at $t=14$ ($t=0.7$)

Relative $L_1$ (%) of the decoded ground-truth tokens (latents for the continuous AE) against the true field: the lowest error any operator on that representation can reach. Mean over the first 16 test trajectories.

| Tokenizer | Data | ρ | u | v | p | Average |
|---|---|---|---|---|---|---|
| Phaedra | KH | 0.49 | 0.64 | 1.02 | 0.01 | 0.54 |
| Phaedra | RC | 3.25 | 4.52 | 4.48 | 0.97 | 3.30 |
| Phaedra | RKH | 1.47 | 2.07 | 1.93 | 0.48 | 1.49 |
| FSQ | KH | 0.83 | 1.12 | 2.05 | 0.03 | 1.01 |
| FSQ | RC | 4.91 | 7.15 | 7.04 | 1.56 | 5.16 |
| FSQ | RKH | 2.10 | 3.29 | 3.19 | 0.87 | 2.36 |
| VQ-VAE-2 | KH | 1.66 | 1.30 | 2.48 | 0.04 | 1.37 |
| VQ-VAE-2 | RC | 8.43 | 11.48 | 11.28 | 2.38 | 8.39 |
| VQ-VAE-2 | RKH | 3.05 | 4.94 | 4.71 | 1.09 | 3.45 |
| Continuous AE | KH | 0.20 | 0.38 | 0.49 | 0.00 | 0.27 |
| Continuous AE | RC | 1.47 | 2.02 | 2.01 | 0.41 | 1.48 |
| Continuous AE | RKH | 0.70 | 0.87 | 0.82 | 0.20 | 0.65 |

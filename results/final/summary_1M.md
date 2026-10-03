| algorithm | seeds | env steps (max) | final eval success | final eval return | eval success @300k | eval success @500k | eval success @1000k | best eval success | final stochastic-policy success | final collision rate | wall time / run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PPO, distance-sorted obstacle input | 3 | 999424 | 0.841 ± 0.001 | 6.58 ± 0.02 | 0.775 ± 0.024 (at 303104) | 0.844 ± 0.020 (at 507904) | 0.841 ± 0.001 (at 999424) | 0.891 ± 0.005 | 0.811 ± 0.009 | 0.159 ± 0.001 | 57 ± 1 s |
| SAC (single task, twin critics, auto-alpha) | 3 | 1000000 | 0.990 ± 0.003 | 9.55 ± 0.05 | 0.479 ± 0.094 (at 300000) | 0.872 ± 0.078 (at 500000) | 0.990 ± 0.003 (at 1000000) | 0.992 ± 0.003 | 0.995 ± 0.002 | 0.003 ± 0.001 | 2541 ± 1835 s |
| SAC-X fixed (SAC-Q scheduler, 6 intentions) | 3 | 1000000 | 0.958 ± 0.008 | 9.02 ± 0.14 | 0.588 ± 0.081 (at 300000) | 0.864 ± 0.032 (at 500000) | 0.958 ± 0.008 (at 1000000) | 0.980 ± 0.002 | 0.981 ± 0.007 | 0.011 ± 0.003 | 1419 ± 53 s |

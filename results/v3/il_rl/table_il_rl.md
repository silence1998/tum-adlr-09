# IL vs RL, env v3 main level — 512 fixed eval layouts (env seed 2026), deterministic

| mode | method | n | success | collision | timeout | path / A* | env steps | expert episodes | wall time |
|---|---|---|---|---|---|---|---|---|---|
| static | A* | 1 | 1.000 | 0.000 | 0.000 | 1.000 | – | – | planner |
| static | BC-1k (2×256, 20k it) | 1 | 0.682 | 0.318 | 0.000 | 1.003 | 6,707 (expert) | 1,000 kept / 1,000 rolled out | demos 3s + train 37s |
| static | BC-10k (2×256, 20k it) | 1 | 0.686 | 0.314 | 0.000 | 0.999 | 66,942 (expert) | 10,000 kept / 10,000 rolled out | demos 26s + train 37s |
| static | BC-50k (2×256, 20k it) | 1 | 0.684 | 0.316 | 0.000 | 0.999 | 333,570 (expert) | 50,000 kept / 50,000 rolled out | demos 130s + train 37s |
| static | BC-1k (2×256, 100k it) | 1 | 0.682 | 0.318 | 0.000 | 1.003 | 6,707 (expert) | 1,000 kept / 1,000 rolled out | demos 3s + train 305s |
| static | BC-10k (2×256, 100k it) | 1 | 0.701 | 0.299 | 0.000 | 1.000 | 66,942 (expert) | 10,000 kept / 10,000 rolled out | demos 26s + train 270s |
| static | BC-50k (2×256, 100k it) | 1 | 0.881 | 0.119 | 0.000 | 1.003 | 333,570 (expert) | 50,000 kept / 50,000 rolled out | demos 130s + train 192s |
| static | BC-50k (2×256, 300k it) | 1 | 0.910 | 0.090 | 0.000 | 1.005 | 333,570 (expert) | 50,000 kept / 50,000 rolled out | demos 130s + train 659s |
| static | BC-1k (2×1024, 100k it) | 1 | 0.688 | 0.312 | 0.000 | 1.005 | 6,707 (expert) | 1,000 kept / 1,000 rolled out | demos 3s + train 170s |
| static | BC-10k (2×1024, 100k it) | 1 | 0.676 | 0.324 | 0.000 | 1.000 | 66,942 (expert) | 10,000 kept / 10,000 rolled out | demos 26s + train 202s |
| static | BC-50k (2×1024, 100k it) | 1 | 0.854 | 0.146 | 0.000 | 1.007 | 333,570 (expert) | 50,000 kept / 50,000 rolled out | demos 130s + train 224s |
| static | DAgger (2×256, 1k+10×1k) | 1 | 0.727 | 0.273 | 0.000 | 1.005 | 63,339 (expert+learner) | 11,000 (1k demos + 10×1000 relabelled) | 171s |
| static | DAgger (2×256, 10k+8×5k) | 1 | 0.875 | 0.125 | 0.000 | 1.003 | 308,316 (expert+learner) | 50,000 (10k demos + 8×5000 relabelled) | 1131s |
| static | DAgger (2×1024, 1k+10×1k) | 1 | 0.713 | 0.287 | 0.000 | 1.013 | 63,926 (expert+learner) | 11,000 (1k demos + 10×1000 relabelled) | 264s |
| static | DAgger (2×1024, 10k+8×5k) | 1 | 0.855 | 0.145 | 0.000 | 1.007 | 303,028 (expert+learner) | 50,000 (10k demos + 8×5000 relabelled) | 669s |
| static | PPO (3M) | 2 | 0.623±0.006 | 0.307±0.002 | 0.070±0.004 | 1.015±0.003 | 3.00M (reward) | 0 | 12 min/run |
| static | SAC (1M) | 2 | 0.994±0.000 | 0.006±0.000 | 0.000±0.000 | 1.076±0.007 | 1.00M (reward) | 0 | 81 min/run |
| static | SAC-X (1M) | 2 | 0.992±0.006 | 0.008±0.006 | 0.000±0.000 | 1.074±0.018 | 1.00M (reward) | 0 | 79 min/run |
| random | A* | 1 | 0.986 | 0.014 | 0.000 | 1.000 | – | – | planner |
| random | BC-1k (2×256, 20k it) | 1 | 0.715 | 0.285 | 0.000 | 1.004 | 6,701 (expert) | 1,000 kept / 1,016 rolled out | demos 8s + train 51s |
| random | BC-10k (2×256, 20k it) | 1 | 0.736 | 0.264 | 0.000 | 1.000 | 66,928 (expert) | 10,000 kept / 10,160 rolled out | demos 80s + train 51s |
| random | BC-50k (2×256, 20k it) | 1 | 0.734 | 0.266 | 0.000 | 0.998 | 329,226 (expert) | 49,388 kept / 50,176 rolled out | demos 393s + train 72s |
| random | BC-1k (2×256, 100k it) | 1 | 0.715 | 0.285 | 0.000 | 1.004 | 6,701 (expert) | 1,000 kept / 1,016 rolled out | demos 8s + train 238s |
| random | BC-10k (2×256, 100k it) | 1 | 0.789 | 0.211 | 0.000 | 1.002 | 66,928 (expert) | 10,000 kept / 10,160 rolled out | demos 80s + train 278s |
| random | BC-50k (2×256, 100k it) | 1 | 0.934 | 0.066 | 0.000 | 1.001 | 329,226 (expert) | 49,388 kept / 50,176 rolled out | demos 393s + train 202s |
| random | BC-50k (2×256, 300k it) | 1 | 0.926 | 0.074 | 0.000 | 1.000 | 329,226 (expert) | 49,388 kept / 50,176 rolled out | demos 393s + train 567s |
| random | BC-1k (2×1024, 100k it) | 1 | 0.717 | 0.283 | 0.000 | 1.021 | 6,701 (expert) | 1,000 kept / 1,016 rolled out | demos 8s + train 170s |
| random | BC-10k (2×1024, 100k it) | 1 | 0.725 | 0.275 | 0.000 | 1.002 | 66,928 (expert) | 10,000 kept / 10,160 rolled out | demos 80s + train 173s |
| random | BC-50k (2×1024, 100k it) | 1 | 0.893 | 0.107 | 0.000 | 1.003 | 329,226 (expert) | 49,388 kept / 50,176 rolled out | demos 393s + train 172s |
| random | DAgger (2×256, 1k+10×1k) | 1 | 0.730 | 0.270 | 0.000 | 1.002 | 64,980 (expert+learner) | 11,000 (1k demos + 10×1000 relabelled) | 220s |
| random | DAgger (2×256, 10k+8×5k) | 1 | 0.912 | 0.088 | 0.000 | 0.998 | 309,617 (expert+learner) | 50,000 (10k demos + 8×5000 relabelled) | 1056s |
| random | DAgger (2×1024, 10k+8×5k) | 1 | 0.855 | 0.145 | 0.000 | 1.003 | 298,743 (expert+learner) | 50,000 (10k demos + 8×5000 relabelled) | 662s |
| random | PPO (3M) | 2 | 0.735±0.007 | 0.242±0.010 | 0.022±0.003 | 1.086±0.002 | 3.00M (reward) | 0 | 12 min/run |
| random | SAC (1M) | 2 | 0.997±0.003 | 0.003±0.003 | 0.000±0.000 | 1.074±0.005 | 1.00M (reward) | 0 | 85 min/run |
| random | SAC-X (1M) | 2 | 0.997±0.003 | 0.003±0.003 | 0.000±0.000 | 1.063±0.001 | 1.00M (reward) | 0 | 87 min/run |
| bounce | A* | 1 | 1.000 | 0.000 | 0.000 | 1.000 | – | – | planner |
| bounce | BC-1k (2×256, 20k it) | 1 | 0.672 | 0.328 | 0.000 | 1.006 | 6,708 (expert) | 1,000 kept / 1,000 rolled out | demos 8s + train 51s |
| bounce | BC-10k (2×256, 20k it) | 1 | 0.701 | 0.299 | 0.000 | 0.998 | 67,089 (expert) | 10,000 kept / 10,002 rolled out | demos 82s + train 59s |
| bounce | BC-50k (2×256, 20k it) | 1 | 0.730 | 0.270 | 0.000 | 0.999 | 334,283 (expert) | 50,000 kept / 50,011 rolled out | demos 409s + train 73s |
| bounce | BC-1k (2×256, 100k it) | 1 | 0.672 | 0.328 | 0.000 | 1.006 | 6,708 (expert) | 1,000 kept / 1,000 rolled out | demos 8s + train 177s |
| bounce | BC-10k (2×256, 100k it) | 1 | 0.717 | 0.283 | 0.000 | 0.997 | 67,089 (expert) | 10,000 kept / 10,002 rolled out | demos 82s + train 183s |
| bounce | BC-50k (2×256, 100k it) | 1 | 0.879 | 0.121 | 0.000 | 0.998 | 334,283 (expert) | 50,000 kept / 50,011 rolled out | demos 409s + train 227s |
| bounce | BC-50k (2×256, 300k it) | 1 | 0.879 | 0.121 | 0.000 | 0.998 | 334,283 (expert) | 50,000 kept / 50,011 rolled out | demos 409s + train 537s |
| bounce | BC-1k (2×1024, 100k it) | 1 | 0.686 | 0.312 | 0.002 | 1.054 | 6,708 (expert) | 1,000 kept / 1,000 rolled out | demos 8s + train 183s |
| bounce | BC-10k (2×1024, 100k it) | 1 | 0.705 | 0.295 | 0.000 | 1.002 | 67,089 (expert) | 10,000 kept / 10,002 rolled out | demos 82s + train 181s |
| bounce | BC-50k (2×1024, 100k it) | 1 | 0.797 | 0.203 | 0.000 | 0.998 | 334,283 (expert) | 50,000 kept / 50,011 rolled out | demos 409s + train 186s |
| bounce | DAgger (2×256, 1k+10×1k) | 1 | 0.703 | 0.297 | 0.000 | 1.005 | 63,152 (expert+learner) | 11,000 (1k demos + 10×1000 relabelled) | 215s |
| bounce | DAgger (2×256, 10k+8×5k) | 1 | 0.871 | 0.129 | 0.000 | 1.001 | 303,640 (expert+learner) | 50,000 (10k demos + 8×5000 relabelled) | 871s |
| bounce | DAgger (2×1024, 10k+8×5k) | 1 | 0.736 | 0.264 | 0.000 | 1.003 | 290,303 (expert+learner) | 50,000 (10k demos + 8×5000 relabelled) | 613s |
| bounce | PPO (3M) | 2 | 0.698±0.017 | 0.302±0.017 | 0.000±0.000 | 1.009±0.007 | 3.00M (reward) | 0 | 13 min/run |
| bounce | SAC (1M) | 2 | 0.996±0.000 | 0.004±0.000 | 0.000±0.000 | 1.123±0.013 | 1.00M (reward) | 0 | 66 min/run |
| bounce | SAC-X (1M) | 2 | 0.994±0.000 | 0.006±0.000 | 0.000±0.000 | 1.148±0.012 | 1.00M (reward) | 0 | 66 min/run |

## Hard level (zero-shot, models trained on main level)

| mode | method | n | success | collision | timeout | path / A* | env steps | expert episodes | wall time |
|---|---|---|---|---|---|---|---|---|---|
| static_hard | A* | 1 | 1.000 | 0.000 | 0.000 | 1.000 |  |  |  |
| static_hard | BC-50k (2×256, 300k it) | 1 | 0.869 | 0.131 | 0.000 | 1.008 |  |  |  |
| static_hard | BC-50k (2×1024, 100k it) | 1 | 0.857 | 0.143 | 0.000 | 1.017 |  |  |  |
| static_hard | DAgger (2×256, 10k+8×5k) | 1 | 0.873 | 0.127 | 0.000 | 1.012 |  |  |  |
| static_hard | DAgger (2×1024, 10k+8×5k) | 1 | 0.787 | 0.213 | 0.000 | 1.012 |  |  |  |
| static_hard | PPO (3M) | 2 | 0.581±0.017 | 0.370±0.003 | 0.049±0.014 | 1.017±0.004 |  |  |  |
| static_hard | SAC (1M) | 2 | 0.987±0.003 | 0.013±0.003 | 0.000±0.000 | 1.086±0.009 |  |  |  |
| static_hard | SAC-X (1M) | 2 | 0.979±0.003 | 0.019±0.001 | 0.002±0.002 | 1.085±0.017 |  |  |  |
| random_hard | A* | 1 | 0.986 | 0.014 | 0.000 | 1.000 |  |  |  |
| random_hard | BC-50k (2×256, 300k it) | 1 | 0.904 | 0.096 | 0.000 | 1.007 |  |  |  |
| random_hard | BC-50k (2×1024, 100k it) | 1 | 0.855 | 0.145 | 0.000 | 1.006 |  |  |  |
| random_hard | DAgger (2×256, 10k+8×5k) | 1 | 0.906 | 0.094 | 0.000 | 1.005 |  |  |  |
| random_hard | DAgger (2×1024, 10k+8×5k) | 1 | 0.805 | 0.195 | 0.000 | 1.007 |  |  |  |
| random_hard | PPO (3M) | 2 | 0.654±0.004 | 0.341±0.005 | 0.005±0.001 | 1.084±0.005 |  |  |  |
| random_hard | SAC (1M) | 2 | 0.980±0.004 | 0.020±0.004 | 0.000±0.000 | 1.087±0.005 |  |  |  |
| random_hard | SAC-X (1M) | 2 | 0.992±0.002 | 0.008±0.002 | 0.000±0.000 | 1.085±0.003 |  |  |  |
| bounce_hard | A* | 1 | 0.998 | 0.002 | 0.000 | 1.000 |  |  |  |
| bounce_hard | BC-50k (2×256, 300k it) | 1 | 0.852 | 0.148 | 0.000 | 1.002 |  |  |  |
| bounce_hard | BC-50k (2×1024, 100k it) | 1 | 0.795 | 0.205 | 0.000 | 1.001 |  |  |  |
| bounce_hard | DAgger (2×256, 10k+8×5k) | 1 | 0.840 | 0.160 | 0.000 | 1.002 |  |  |  |
| bounce_hard | DAgger (2×1024, 10k+8×5k) | 1 | 0.727 | 0.273 | 0.000 | 1.004 |  |  |  |
| bounce_hard | PPO (3M) | 2 | 0.607±0.006 | 0.393±0.006 | 0.000±0.000 | 1.004±0.004 |  |  |  |
| bounce_hard | SAC (1M) | 2 | 0.978±0.003 | 0.022±0.003 | 0.000±0.000 | 1.158±0.017 |  |  |  |
| bounce_hard | SAC-X (1M) | 2 | 0.983±0.005 | 0.017±0.005 | 0.000±0.000 | 1.218±0.043 |  |  |  |

## RL sample efficiency (training-time eval, CUDA layout set): first env step with det. success ≥ threshold

| run | ≥0.90 | ≥0.95 | ≥0.98 | final |
|---|---|---|---|---|
| ppo_static_s0 | never | never | never | 0.684 |
| ppo_static_s1 | never | never | never | 0.664 |
| sac_static_s0 | 450k | 550k | 625k | 1.000 |
| sac_static_s1 | 375k | 425k | 525k | 0.994 |
| sacx_static_s0 | 500k | 600k | 650k | 0.996 |
| sacx_static_s1 | 550k | 700k | 750k | 0.992 |
| ppo_random_s0 | never | never | never | 0.748 |
| ppo_random_s1 | never | never | never | 0.727 |
| sac_random_s0 | 175k | 225k | 275k | 1.000 |
| sac_random_s1 | 175k | 200k | 300k | 0.994 |
| sacx_random_s0 | 150k | 250k | 350k | 1.000 |
| sacx_random_s1 | 175k | 300k | 350k | 0.996 |
| ppo_bounce_s0 | never | never | never | 0.723 |
| ppo_bounce_s1 | never | never | never | 0.729 |
| sac_bounce_s0 | 200k | 275k | 325k | 0.992 |
| sac_bounce_s1 | 225k | 225k | 350k | 0.990 |
| sacx_bounce_s0 | 200k | 425k | 525k | 0.994 |
| sacx_bounce_s1 | 250k | 300k | 550k | 1.000 |

## Demo filtering

| mode | attempted | kept | kept % | collision | timeout | samples | time |
|---|---|---|---|---|---|---|---|
| static | 50176 | 50176 | 100.00 | 0 | 0 | 334,734 | 131s |
| random | 50176 | 49388 | 98.43 | 788 | 0 | 329,226 | 393s |
| bounce | 50176 | 50165 | 99.98 | 11 | 0 | 335,377 | 411s |

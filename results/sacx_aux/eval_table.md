| 变体 | 动作 | seeds | goal | collision | timeout | 反向次数/回合 | 障碍旁反向/回合 | 路径比(成功) | 成功步数 |
|---|---|---|---|---|---|---|---|---|---|
| SAC (plain, reference) | mean | 1 | 507 | 2 | 3 | 0.45 | 0.05 | 1.21 | 5.81 |
| SAC (plain, reference) | sample | 1 | 509 | 2 | 1 | 0.28 | 0.05 | 1.26 | 6.08 |
| SAC (plain, reference) | qmax | 1 | 497 | 6 | 9 | 1.81 | 0.29 | 1.28 | 5.90 |
| SAC-X old (main+approach+4 axis, SAC-Q, xi=20) | mean | 0,1,2 | 489 / 490 / 486 (均值 488.3) | 5 / 3 / 2 (均值 3.3) | 18 / 19 / 24 (均值 20.3) | 1.91 ± 0.47 | 0.30 ± 0.12 | 1.22 ± 0.01 | 6.14 ± 0.06 |
| SAC-X old (main+approach+4 axis, SAC-Q, xi=20) | sample | 0,1,2 | 505 / 502 / 505 (均值 504.0) | 5 / 2 / 3 (均值 3.3) | 2 / 8 / 4 (均值 4.7) | 1.36 ± 0.38 | 0.17 ± 0.06 | 1.32 ± 0.02 | 6.85 ± 0.03 |
| SAC-X old (main+approach+4 axis, SAC-Q, xi=20) | qmax | 0,1,2 | 464 / 476 / 480 (均值 473.3) | 4 / 4 / 5 (均值 4.3) | 44 / 32 / 27 (均值 34.3) | 6.30 ± 1.43 | 0.44 ± 0.21 | 1.29 ± 0.01 | 5.99 ± 0.07 |
| SAC-X old (main+approach+4 axis, SAC-Q, xi=20) | sched | 0,1,2 | 470 / 489 / 485 (均值 481.3) | 15 / 7 / 6 (均值 9.3) | 27 / 16 / 21 (均值 21.3) | 28.11 ± 17.59 | 0.80 ± 0.32 | 8.67 ± 4.85 | 38.62 ± 22.74 |
| SAC-X code, main only | mean | 1 | 502 | 2 | 8 | 1.03 | 0.07 | 1.20 | 5.73 |
| SAC-X code, main only | sample | 1 | 510 | 1 | 1 | 0.38 | 0.04 | 1.27 | 6.17 |
| main+approach | mean | 1 | 506 | 4 | 2 | 0.17 | 0.08 | 1.19 | 5.77 |
| main+approach | sample | 1 | 506 | 6 | 0 | 0.22 | 0.08 | 1.25 | 6.10 |
| main+4 axis-move | mean | 1 | 481 | 2 | 29 | 3.55 | 0.19 | 1.25 | 6.24 |
| main+4 axis-move | sample | 1 | 495 | 5 | 12 | 2.80 | 0.25 | 1.35 | 6.85 |
| main+safe_progress+clearance | mean | 1 | 494 | 1 | 17 | 1.64 | 0.38 | 1.21 | 6.18 |
| main+safe_progress+clearance | sample | 1 | 506 | 3 | 3 | 0.78 | 0.11 | 1.25 | 6.54 |
| full 6 tasks, SAC-U | mean | 1 | 477 | 9 | 26 | 3.91 | 0.56 | 1.33 | 6.44 |
| full 6 tasks, SAC-U | sample | 1 | 489 | 19 | 4 | 1.73 | 0.31 | 1.46 | 7.01 |
| full, xi=100 (1 intention/episode) | mean | 1 | 495 | 6 | 11 | 1.02 | 0.07 | 1.25 | 6.22 |
| full, xi=100 (1 intention/episode) | sample | 1 | 501 | 7 | 4 | 1.15 | 0.13 | 1.30 | 6.72 |
| full, aux reward scale 1.0 | mean | 1 | 468 | 4 | 40 | 4.48 | 1.05 | 1.24 | 6.44 |
| full, aux reward scale 1.0 | sample | 1 | 489 | 8 | 15 | 3.54 | 0.63 | 1.34 | 7.72 |
| full, no Retrace (seq-len 1) | mean | 1 | 486 | 6 | 20 | 1.31 | 0.25 | 1.20 | 6.35 |
| full, no Retrace (seq-len 1) | sample | 1 | 502 | 7 | 3 | 1.00 | 0.18 | 1.25 | 6.74 |

> **关于本分支（`grok-bot-v3-il-rl`）**：本分支的代码与实验由 **Grok Bot**（AI 助手）编写 / 重构完成，包括：SAC-X 重写（SAC 与其共享实现）、PPO、v2 / v3 GPU 向量化环境、v3 A\* 基线、A\* 示范 + 模仿学习（BC / DAgger），以及全部对比实验和结果文档。
>
> **仓库中不包含模型权重和示范数据**：所有 `*.pt`，即 `SAC/checkpoints/`、`results/sacx_aux/checkpoints/`、`results/v3/checkpoints/`、`results/v3/il/` 中的 BC / DAgger 权重和 A\* 示范 `demos_*.pt`。为控制体积，Git 分支只包含代码、配置、结果文档（`RESULTS_*.md`、`CONCLUSIONS.md`、汇总表）、评估 json、v2 最终曲线的 CSV（`results/final/runs_1M/`）和 5 张关键曲线图（其中 4 张压缩成了 256 色 PNG）。下面这些**只保留在本地工作目录**，需要时运行第 4 节的命令生成：GIF、训练日志（`*.out`、`STATUS.md`）、逐 run 的训练 CSV（`results/*/runs/`）、`results/archive/`。所以结果文档中指向这些文件的链接在 Git 分支中会失效。重新生成的入口：
> - 权重：
>   - v2 SAC / SAC-X / PPO：4.1 节（同时生成训练 CSV；GIF 用 `render_gifs.py`）；
>   - v3 PPO / SAC / SAC-X：4.4 节（训练 CSV 写到 `results/v3/runs/`）；
>   - A\* 示范 + BC / DAgger：4.6 节；
> - 统一评估、表格、曲线和 GIF：4.7 节（需要先有对应的 checkpoint）；
> - 后台队列和日志（`results/*/logs/`、`STATUS.md`）：4.8 节。
>
> *English: this branch is Grok Bot's implementation. Grok Bot is the AI assistant that wrote or reworked this code: the SAC-X rewrite, PPO, the v2 / v3 environments, the A\* baseline, IL with BC / DAgger, and the experiments. Model weights, demo data, GIFs, logs and per-run CSVs are not in Git (only code, configs, result docs, eval json and a few key plots); regenerate them with the commands in Section 4 below.*

# tum-adlr-09：二维避障 + 到达目标（PPO / SAC / SAC-X / A* / 模仿学习）

TUM ADLR 课程项目（team 09）。点状智能体要在二维场地中避开圆形障碍物，到达目标点。项目在 GPU 向量化环境上比较了以下方法：

- 在线策略 RL：**PPO**；
- 离线策略 RL：**SAC** 和多意图辅助任务的 **SAC-X**；
- 使用真实障碍物状态的规划器：**A\***；
- 以 A* 为专家的模仿学习：**BC / DAgger**。

> 本 README 写于 2026-10-03，是项目整理时的总入口。旧版 README 备份在 `README.old.md`（仅本地，内容与 GitHub main 分支上的旧 README 相同）。

---

## 1. 环境版本

| 版本 | 代码 | 简述 | 状态 |
|---|---|---|---|
| **v1** | `SAC/environment.py`（CPU，gym 风格）、`SAC/torch_env.py`（GPU，`--env-version 1`） | 原始的整数网格世界：动作被缩放后四舍五入成网格移动；稀疏奖励（+10 / −50） | 仅保留用于复现早期结果（`results/archive/v1/`） |
| **v2** | `SAC/torch_env.py`（`TorchContinuousWorld`，GPU，**默认** `--env-version 2`），`SAC/environment_v2.py`（CPU 版本，用于交叉验证） | 连续动作、连续位移；5 个固定半径的障碍物（中心在整数格点上）；奖励为 +10 到达、−10 碰撞、−0.05/步，加势函数 shaping；100 步时限 | 最终对比结果见 `results/final/` |
| **v3** | `SAC/env_v3.py`（`EnvV3` + `EnvV3Config`，GPU / CPU 向量化），配置文件为 `configs/v3_*.yaml` | 连续空间；障碍物数量和半径随机；三种运动模式；65 维观测：9 个基础量 + 8 个障碍物槽 × 7，按表面距离排序，padding 带 mask | 当前主环境，结果见 `results/v3/` |

**v3 的三种运动模式**（`motion`）：
- `static`：障碍物静止；
- `random`：障碍物做 OU 速度过程的随机游走，被限制在出生点半径 `rw_radius` 范围内；
- `bounce`：匀速运动，撞墙反射，障碍物之间做弹性碰撞（质量 ∝ r²），目标附近有保护圆盘。

**v3 的两档密度**：

| 档位 | 配置文件 | 障碍物数 | 半径 | min_gap |
|---|---|---|---|---|
| **主环境**（原 easy 档） | `configs/v3_{static,random,bounce}.yaml` | 3–8 | 0.3–0.6 | 0.8 |
| **hard 档**（原默认密度） | `configs/v3_{static,random,bounce}_hard.yaml` | 4–8 | 0.3–0.8 | 0.5 |
| hard 档 + 小步长 | `configs/v3_static_hard_step05.yaml` | 同 hard | | max_step 0.5，150 步 |

`EnvV3Config` 的 dataclass 默认值等于主环境。每个字段都可以用 YAML（`--env-config`）设置，也可以用命令行（`train_ppo_v3.py --env-<字段名>`）覆盖。

**评估约定**：
- 所有 v3 方法都在 512 个固定布局（env seed 2026）上用确定性策略评估。
- **注意**：torch 的 CPU 和 CUDA 随机数生成器在相同 seed 下给出的布局**不同**。
  - 最终表格（`eval_il_rl_v3.py`、`eval_v3.py`）统一用 CPU 布局集；
  - 训练过程中记录的评估（`runs/*.csv`）用的是 CUDA 布局集。

---

## 2. 环境配置

- **conda 环境**：`torch`。解释器为 `D:\program\anaconda\envs\torch\python.exe`（Python 3.11.11）。
- **已验证的依赖版本**：torch 2.7.0+cu128、numpy 2.1.2、PyYAML 6.0.2、matplotlib 3.11.2、Pillow 11.0.0（另有 scipy 1.15.3，非必需）。
- **GPU**：RTX 5080 16 GB。所有训练脚本在有 CUDA 时默认使用 GPU（`--device` 可以修改）。
- **v1 遗留代码**：`environment.py` 可选依赖 gym / pygame（未安装时也能无界面运行）；`legacy/v1_original/training.py`（原始 SAC）依赖 `wandb`。这两项在当前环境中都**没有**安装，只有运行 v1 原始代码时才需要。

在新机器上从零配置：

```powershell
conda create -n torch python=3.11 -y
conda activate torch
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 按显卡 / CUDA 版本选择
pip install numpy pyyaml matplotlib pillow
```

下面的所有命令都假定：

```powershell
$py = "D:\program\anaconda\envs\torch\python.exe"
cd E:\code\tum-adlr-09-main\tum-adlr-09-main\SAC     # !!! 所有脚本都要从 SAC\ 目录运行（脚本里写的是 ../configs、../results 这样的相对路径）
```

---

## 3. 目录结构

```
tum-adlr-09-main/
├── README.md                  本文件
├── SAC/                       全部代码（从这里运行）
│   ├── env_v3.py              v3 环境（EnvV3、EnvV3Config、evaluate_v3）
│   ├── test_env_v3.py         v3 单元测试
│   ├── astar_v3.py            v3 A* 基线（使用真实障碍物状态；运动模式下每步重规划 + 一步安全过滤）
│   ├── torch_env.py           v1 / v2 GPU 环境
│   ├── environment.py, environment_v2.py, test_torch_env.py   v1 / v2 CPU 环境与等价性测试
│   ├── training_SAC-X.py      SAC-X（也是 SAC 的实现），支持 v1 / v2 / v3
│   ├── training_SAC.py        SAC = training_SAC-X.py + 单任务默认参数（参数转发）
│   ├── training_PPO.py        PPO，用于 v1 / v2
│   ├── train_ppo_v3.py        PPO，用于 v3
│   ├── il_v3.py               A* 示范生成 / BC / DAgger / 诊断（子命令 demos | bc | dagger | diag）
│   ├── eval_v3.py             v3：A* / PPO 评估
│   ├── eval_il_rl_v3.py       v3：统一评估 A* + BC + DAgger + PPO + SAC + SAC-X（主环境和 hard 档）
│   ├── report_v3.py, report_il_rl_v3.py      v3 表格与曲线
│   ├── render_v3.py, render_il_rl_v3.py      v3 GIF
│   ├── plot_comparison.py, render_gifs.py    v1 / v2 曲线与 GIF
│   ├── sacx_aux_*.py, render_sacx_aux_gif.py SAC-X 辅助任务消融（v2）
│   ├── run_v3_rl_queue.py     v3 SAC / SAC-X 后台队列（最多 N 个并发）
│   ├── run_*.ps1              PowerShell 后台启动脚本（见第 4.8 节）
│   ├── debug_min_sac.py       最小 SAC，用于诊断 Q 值高估（v2 GPU 环境）
│   ├── parameters.py          共用参数（environment.py、training_SAC-X.py、training_PPO.py 使用）
│   ├── checkpoints/           v2 最终权重（*_1M.pt，不要覆盖）
│   ├── baseline_original/     原始（修复前）代码，用于对照（test_torch_env.py 会用到）
│   └── legacy/                已完成或已失效的脚本（见第 4.9 节）
│       ├── rename_v3_levels.py, run_il_static_ablation.ps1, sacx_aux_diag2.py   一次性脚本（已执行）
│       ├── run_queue_1M.ps1, run_queue_new_env.ps1   旧版 v2 队列（已失效）
│       └── v1_original/       v1 原始 SAC：training.py、model.py、plot_results.py、A_star/（旧版网格 A*，有 bug）
├── configs/                   v3 YAML 配置
├── results/
│   ├── final/                 v2 最终结果：CONCLUSIONS.md、曲线、GIF（只读，不要修改；Git 中只有 CONCLUSIONS.md、summary_1M.md、runs_1M/、plots/comparison_1M.png）
│   ├── archive/               v1、v2 开发过程中的结果；v3_dev/（il_rl 中期报告、改名前的旧配置 configs_old_names/）——仅本地，不在 Git 中
│   ├── sacx_aux/              SAC-X 辅助任务消融（进度日志 logs/STATUS.md）；Git 中只有文档、eval/、eval_table.md、sacx_aux_comparison.png
│   └── v3/                    v3：RESULTS_v3.md、RESULTS_v3_il_rl.md、runs/、checkpoints/、eval/、gifs/、il/、il_rl/、logs/（含 STATUS.md）；Git 中不含 runs/、logs/、gifs/、checkpoints/ 和 *.pt
├── archive/
│   └── pytorch_rl/            早期 RL 教程 / 试验代码（已归档；在 Git 分支中仍保持 main 上的原位置 `pytorch_rl/`）
├── logs/                      课程文档：proposal、周报笔记（与训练日志无关）
├── CLEANUP_LOG.md             2026-10-03 目录整理记录（删除 / 移动清单）
├── README.old.md              旧版 README（仅本地）
└── gcloud_setup.md            Google Cloud 配置笔记
```

---

## 4. 常用命令（在 `SAC\` 目录下运行，PowerShell）

> 参数已经用各脚本的 `--help` 核对过（2026-10-03）。
> 训练脚本会把**最终** checkpoint 写到 `--ckpt-dir`（SAC / SAC-X 默认是 `SAC\checkpoints`），PPO v1 / v2 写到 `SAC\checkpoints\`，文件名为 `<log 文件名>.pt`。**不要**用 `sac_s1_1M`、`sacx_q_s*_1M`、`ppo_sorted_s*_1M` 作为 log 名，否则会覆盖最终权重。

### 4.1 v2：SAC / SAC-X / PPO（复现 `results/final`，每个 run 1M 步）

```powershell
mkdir ..\results\runs_v2 -Force | Out-Null
# SAC（单任务；training_SAC.py 自带默认参数：--tasks 0 --seq-len 1 --batch 256 --num-envs 8 --features relative_sorted ...）
& $py training_SAC.py --seed 0 --total-steps 1000000 --log ..\results\runs_v2\sac_s0.csv
# SAC-X（SAC-Q 调度器，6 个意图；与 final 中的 sacx_q 设置相同）
& $py training_SAC-X.py --scheduler Q --num-envs 8 --updates-per-step 1 --batch 32 --features relative_sorted `
    --main-reward-scale 1.0 --alpha-init 0.05 --warmup 5000 --aux-reward-scale 0.1 --total-steps 1000000 `
    --seed 0 --log ..\results\runs_v2\sacx_q_s0.csv
# PPO（障碍物按距离排序）
& $py training_PPO.py --features relative_sorted --seed 0 --total-steps 1000000 --log ..\results\runs_v2\ppo_sorted_s0.csv
# 曲线和表格（每种算法对应一个文件前缀 <prefix>_s<seed>.csv）
& $py plot_comparison.py --runs ..\results\runs_v2 --out ..\results\runs_v2\comparison.png --table ..\results\runs_v2\summary.md --at 300000 500000 1000000
```

- 加 `--env-version 1` 切换到 v1 网格环境。
- `--mem-frac 0.3` 可以限制单个进程占用的显存。
- `render_gifs.py` 的默认 `--out` 是 `..\results\final\gifs`，重新生成时**务必**换一个输出目录：

```powershell
& $py render_gifs.py --out ..\results\runs_v2\gifs     # 默认读取 checkpoints\{ppo_sorted,sac,sacx_q}_s1_1M.pt
```

### 4.2 v1 / v2 环境测试

```powershell
& $py test_torch_env.py        # GPU 环境 vs CPU 环境的等价性测试
```

### 4.3 v3 环境测试

```powershell
& $py test_env_v3.py           # CPU
& $py test_env_v3.py cuda      # CUDA；最后应打印 "ALL TESTS PASSED"
```

### 4.4 v3：PPO / SAC / SAC-X 训练

```powershell
# PPO（v3 专用脚本；--log 和 --ckpt 必填；结果中的 PPO 是 3M 步）
& $py train_ppo_v3.py --env-config ..\configs\v3_bounce.yaml --total-steps 3000000 --seed 0 `
    --log ..\results\v3\runs\ppo_bounce_s0.csv --ckpt ..\results\v3\checkpoints\ppo_bounce_s0.pt
#   命令行可以覆盖任意环境字段，例如：--env-n-obs-max 6 --env-motion random

# SAC（1M 步；评估使用 512 个固定布局）
& $py training_SAC.py --env-version 3 --env-config ..\configs\v3_bounce.yaml --total-steps 1000000 --seed 0 `
    --eval-every 25000 --log ..\results\v3\runs\sac_bounce_s0.csv --ckpt-dir ..\results\v3\checkpoints

# SAC-X（辅助任务消融中最好的变体：主任务 + approach，tasks 0,1）
& $py training_SAC-X.py --tasks 0,1 --scheduler Q --num-envs 8 --updates-per-step 1 --batch 32 --features relative_sorted `
    --main-reward-scale 1.0 --alpha-init 0.05 --warmup 5000 --aux-reward-scale 0.1 `
    --env-version 3 --env-config ..\configs\v3_bounce.yaml --total-steps 1000000 --seed 0 --eval-every 25000 `
    --log ..\results\v3\runs\sacx_bounce_s0.csv --ckpt-dir ..\results\v3\checkpoints
```

### 4.5 v3：A* 基线评估

```powershell
& $py eval_v3.py --config ..\configs\v3_bounce.yaml --astar --out ..\results\v3\eval\bounce.json
& $py eval_v3.py --config ..\configs\v3_bounce.yaml --astar --no-safety --tag astar_nosafety --out ..\results\v3\eval\bounce.json
& $py eval_v3.py --config ..\configs\v3_bounce.yaml --ppo ..\results\v3\checkpoints\ppo_bounce_s0.pt --out ..\results\v3\eval\bounce.json
```

- 默认参数：512 个回合，seed 2026，栅格 0.1，margin 0.15，CPU。
- 结果写入 json 中 `--tag` 对应的键。

### 4.6 v3：A* 示范数据 + BC / DAgger

```powershell
# 1) 示范数据：A* 在训练布局（env seed 100000+batch，与评估布局不重叠）上 rollout，只保留成功回合
& $py il_v3.py demos --mode bounce --episodes 50000                 # -> ..\results\v3\il\demos_bounce.pt + _stats.json
# 2) BC（网络与 SAC actor 相同，2×256，MSE 损失）。最好的设置：50k 示范，100k–300k 步
& $py il_v3.py bc --mode bounce --n-episodes 50000 --steps 300000 --suffix _s300k   # -> il\bc_bounce_50k_s300k.pt/.json
#    可选：--hidden 1024（更宽的网络）、--demos <示范文件后缀>、--device cpu
# 3) DAgger（A* 对学习者访问到的每个状态重新标注）。"大"设置：10k 示范 + 8 轮 × 5000 回合
& $py il_v3.py dagger --mode bounce --init-episodes 10000 --init-steps 100000 --iters 8 --eps-per-iter 5000 --iter-steps 30000 --suffix _big
# 4) 诊断：学习者与专家的动作在碰撞前一步的差异
& $py il_v3.py diag --mode static --ckpt ..\results\v3\il\bc_static_50k.pt
```

### 4.7 v3：统一评估、报告、GIF

```powershell
# 评估 A* + 所有 IL / RL checkpoint（CPU 布局集），主环境 + hard 档 -> ..\results\v3\il_rl\eval_<mode>[_hard].json
& $py eval_il_rl_v3.py --modes static random bounce --hard
# 表格和曲线 -> ..\results\v3\il_rl\table_il_rl.md, il_rl_curves.png, il_scaling.png
& $py report_il_rl_v3.py
# GIF（每行一个方法：A* / BC / DAgger / SAC / SAC-X / PPO）-> ..\results\v3\il_rl\gifs\il_rl_<mode>.gif
& $py render_il_rl_v3.py --mode bounce            # 可选：--bc 50k_s300k --dagger _big --seed 0 --layouts 3,17
# 只有 PPO 的 v3 报告和 PPO vs A* 的 GIF
& $py report_v3.py                                # -> ..\results\v3\summary_v3.md, ppo_v3_curves.png
& $py render_v3.py --config ..\configs\v3_bounce.yaml --ppo ..\results\v3\checkpoints\ppo_bounce_s0.pt --out ..\results\v3\gifs\bounce_ppo_vs_astar.gif
```

### 4.8 后台启动脚本（不受终端关闭影响）

```powershell
# v3 SAC / SAC-X 队列：{sac,sacx} × {static,random,bounce} × seeds，最多 4 个并发；已有 checkpoint 的 run 会被跳过
Start-Process $py -ArgumentList "run_v3_rl_queue.py","--max-par","4","--seeds","0","1" -WindowStyle Hidden
# v3 PPO
Start-Process powershell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","run_v3_ppo.ps1","-Jobs","static:0,random:0,bounce:0" -WindowStyle Hidden
# v3 IL 流水线（示范 → BC 1k/10k/50k，20k 步 → 小规模 DAgger），每个模式一个进程
Start-Process powershell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","run_il_v3.ps1","-Mode","bounce" -WindowStyle Hidden
# 更长训练的 IL（BC 100k / 300k 步、DAgger 大规模），三种模式依次运行
Start-Process powershell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","run_il_v3_s256.ps1" -WindowStyle Hidden
```

- 日志在 `results\v3\logs\`（运行后生成，不在 Git 中），进度追加到 `results\v3\logs\STATUS.md`（SAC-X 辅助任务消融：`results\sacx_aux\logs\STATUS.md`）。
- `run_il_v3_large.ps1` 是 2×1024 网络的消融，`run_sacx_aux.ps1 -Batch 1` 和 `sacx_aux_queue.py` 是 SAC-X 辅助任务消融。
- 旧版 v2 队列 `run_queue_1M.ps1`、`run_queue_new_env.ps1` 已移到 `legacy\`（输出目录 `results\new_env` 已不存在，不要再使用）；`run_il_static_ablation.ps1`（2×1024 static 消融，已执行）也在 `legacy\`。
- 每个 `.ps1` 里都写死了绝对路径 `E:\code\tum-adlr-09-main\tum-adlr-09-main\SAC`，换机器时需要修改。
- `run_v3_ppo.ps1` 的 `-Jobs` 本应接收多个 `"mode:seed"`，但以 `-File` 方式启动时，`"static:0,random:0,bounce:0"` 能否被拆分成数组**未验证**。稳妥的做法是每个 job 单独启动一次，或者在 PowerShell 会话里直接运行 `.\run_v3_ppo.ps1 -Jobs "static:0","random:0"`。

### 4.9 遗留代码（`SAC\legacy\`）

```powershell
# v1 原始 SAC（需要先 pip install wandb；gym / pygame 可选）。脚本会把自己的目录和 SAC\ 加入 sys.path，
# 并把工作目录切换到 SAC\，所以输出仍写到 SAC\model\、SAC\tmp\sac\（与移动前相同）
& $py legacy\v1_original\training.py
& $py legacy\v1_original\plot_results.py
# SAC-X 辅助任务的超时诊断（会自动切换到 SAC\ 目录）-> ..\results\sacx_aux\eval\diag_stuck_phase.json
& $py legacy\sacx_aux_diag2.py
```

- `legacy\rename_v3_levels.py` 已在 2026-10-02 执行过，**不要再运行**（会把 `*_hard` 再改一次名）。
- `baseline_original\training_SAC-X_original.py` 是原样冻结的原始代码，`from model import *` / `import A_star.algorithm` 指向的文件现在在 `legacy\v1_original\`；它本来就依赖 wandb，没有被任何脚本调用（`run_original_sacx.py` 使用的是 `model_original.py` 等）。

---

## 5. 结果文档

| 文档 | 内容 |
|---|---|
| [results/final/CONCLUSIONS.md](results/final/CONCLUSIONS.md) | v2 最终结论：PPO vs SAC vs SAC-X（1M 步，3 个 seed），修复过的 bug 及分析 |
| [results/sacx_aux/RESULTS_sacx_aux.md](results/sacx_aux/RESULTS_sacx_aux.md) | SAC-X 辅助任务消融（v2）：为什么 SAC-X 会超时，哪个辅助任务组合最好 |
| [results/v3/RESULTS_v3.md](results/v3/RESULTS_v3.md) | v3 环境设计、配置项、PPO 验证、A* 基线、最终参数 |
| [results/v3/RESULTS_v3_il_rl.md](results/v3/RESULTS_v3_il_rl.md) | v3：A* 示范 + BC / DAgger 与 SAC / SAC-X / PPO 的对比（最终版） |
| [results/v3/il_rl/table_il_rl.md](results/v3/il_rl/table_il_rl.md) | 上面对比的完整表格（所有变体、代价、样本效率、示范过滤） |
| `results/archive/`（仅本地，不在 Git 中） | v1 结果、v2 开发记录（`v2_dev/RESULTS_new_env.md`、`PROGRESS_NOTES.md`）、v3 开发记录（`v3_dev/`：il_rl 中期报告、改名前的旧配置） |
| [CLEANUP_LOG.md](CLEANUP_LOG.md) | 2026-10-03 目录整理：删除了哪些文件、哪些文件移到了哪里 |

## 6. 关键结果一览

所有数字都摘自上表中的文档。成功率是在 512 个固定评估布局上用确定性策略得到的。

**v2**（`results/final/summary_1M.md`，1M 步，3 个 seed）：

| 算法 | 最终成功率 | 碰撞率 |
|---|---|---|
| PPO（障碍物按距离排序） | 0.841 ± 0.001 | 0.159 ± 0.001 |
| SAC | **0.990 ± 0.003** | 0.003 ± 0.001 |
| SAC-X（SAC-Q，6 个意图） | 0.958 ± 0.008 | 0.011 ± 0.003 |

**SAC-X 辅助任务消融**（`results/sacx_aux/eval_table.md`，seed 1，512 个布局中 goal / collision / timeout 的回合数，确定性主头）：
- 旧版 SAC-X（主任务 + approach + 4 个轴向移动）：488.3 / 3.3 / 20.3（3 个 seed 的均值）；
- **主任务 + approach**：506 / 4 / 2；
- 普通 SAC：507 / 2 / 3。

**v3 主环境**（`results/v3/RESULTS_v3_il_rl.md`，CPU 布局集；RL 为 2 个 seed 的均值 ± 标准差，IL 为单次训练）：

| 方法 | static | random | bounce |
|---|---|---|---|
| A*（使用真实障碍物状态） | 1.000 | 0.986 | 1.000 |
| BC（2×256，50k 示范，最好的训练步数） | 0.910 | 0.934 | 0.879 |
| DAgger（2×256，共 50k 专家标注回合） | 0.875 | 0.912 | 0.871 |
| PPO（3M 步） | 0.623 ± 0.006 | 0.735 ± 0.007 | 0.698 ± 0.017 |
| SAC（1M 步） | **0.994 ± 0.000** | **0.997 ± 0.003** | **0.996 ± 0.000** |
| SAC-X（tasks 0,1，1M 步） | 0.992 ± 0.006 | **0.997 ± 0.003** | 0.994 ± 0.000 |

**v3 hard 档**（零样本：模型在主环境上训练）：SAC 0.978–0.987，SAC-X 0.979–0.992，BC / DAgger 0.840–0.906，PPO 0.581–0.654。

**结论**：
- SAC 是最强、最稳定的方法；
- SAC-X（主任务 + approach）与 SAC 持平；
- PPO 停在"直奔目标"的局部最优；
- IL 便宜（无需奖励交互，约十几分钟），但碰撞集中在 A* 动作不连续的关键状态；DAgger 在同样的标注预算下没有超过充分训练的 BC。

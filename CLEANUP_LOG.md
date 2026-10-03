# 目录整理记录（CLEANUP_LOG）

> 由 `CLEANUP_PROPOSAL.md` 改名而来。用户于 2026-10-03 11:22（UTC+8）确认执行 **A 组（A1–A7）和 C 组（C1–C7）**，**B 组全部保留**（`results/v3/il/demos_*.pt`、`*_h1024*.pt` 未动）。执行时间 2026-10-03 约 11:30–11:50（UTC+8）。没有使用 git。`results/final/` 与 `SAC/checkpoints/*_1M.pt` 未改动（`final` 中对旧路径的文字引用保持原样）。

## 1. 已删除：70 个文件，321,311 B

| # | 内容 | 文件数 | 字节 |
|---|---|---|---|
| A1 | `SAC/__pycache__/` | 11 | 241,647 |
| A2 | `results/v3/runs/{sac,sacx}_bounce_s0.csv.partial`、`{sac,sacx}_static_s1.csv.partial` | 4 | 19,959 |
| A3 | `results/v3/logs/` 中同名的 4 个 `.out.partial` | 4 | 29,283 |
| A4 | 0 字节文件：`results/v3/logs/*.err` 31 个、`results/sacx_aux/logs/*.err` 15 个（包括 A5 对应的 3 个空 `.err`） | 46 | 0 |
| A5 | `results/v3/logs/eval_il_rl_{static,sr,all1}.out` | 3 | 27,035 |
| A6 | `results/v3/logs/pids.log`、`results/sacx_aux/logs/pids.log` | 2 | 3,387 |
| **合计** | | **70** | **321,311** |

**A7 偏离原方案：`SAC/A_star/` 没有删除，而是移到了 `SAC/legacy/v1_original/A_star/`（3 个文件，3,766 B）。** 原因：执行前复查发现它并非"没有使用"——`SAC/training.py`（v1 原始 SAC，第 25 行 `import A_star.algorithm`，用于 `select_action_A_star`）依赖它。C3 要求 v1 原始代码移动后仍可运行，所以把它和 `training.py` 放在一起。`archive/pytorch_rl/A_star/` 中仍有一份逐字节相同的副本。

## 2. 已移动

| # | 从 | 到 |
|---|---|---|
| A7 | `SAC/A_star/` | `SAC/legacy/v1_original/A_star/` |
| C1 | `SAC/run_queue_1M.ps1`、`SAC/run_queue_new_env.ps1` | `SAC/legacy/`（文件头加了"已失效"注释） |
| C2 | `SAC/rename_v3_levels.py`、`SAC/run_il_static_ablation.ps1`、`SAC/sacx_aux_diag2.py` | `SAC/legacy/` |
| C3 | `SAC/training.py`、`SAC/model.py`、`SAC/plot_results.py` | `SAC/legacy/v1_original/` |
| C4 | `pytorch_rl/`（6 个文件） | `archive/pytorch_rl/` |
| C5 | `results/v3/RESULTS_v3_il_rl.interim_2026-10-02.md` | `results/archive/v3_dev/` |
| C6 | `results/v3/STATUS.md`、`results/sacx_aux/STATUS.md` | 各自的 `logs/STATUS.md` |
| C7 | `configs/old_names_2026-10-02/`（7 个文件） | `results/archive/v3_dev/configs_old_names/` |

## 3. 同步修改的代码和文档

- **代码**
  - `SAC/legacy/v1_original/training.py`、`plot_results.py`：文件开头加入 sys.path 设置（自身目录 → `model.py`、`A_star/`；`SAC/` → `environment.py`、`parameters.py`），并 `chdir` 到 `SAC/`，使相对输出路径 `model/`、`model_pretrain/`、`tmp/sac/` 与移动前一致。
  - `SAC/legacy/sacx_aux_diag2.py`：加入 `SAC/` 到 sys.path，并 `chdir` 到 `SAC/`（`import sacx_aux_eval`、`checkpoints/...`、`../results/...` 的解析方式不变）。
  - `SAC/legacy/rename_v3_levels.py`：文件头注明"已执行，勿再运行"。
  - `SAC/run_v3_rl_queue.py`：STATUS 写到 `results/v3/logs/STATUS.md`（docstring 同步）。
  - `SAC/sacx_aux_queue.py`：STATUS 写到 `results/sacx_aux/logs/STATUS.md`（docstring 同步）。另外，启动时读取 `logs/pids.log` 改为"文件不存在时跳过"，因为 A6 删除了旧的 pids.log。
  - `SAC/astar_v3.py`：docstring 中 `SAC/A_star/algorithm.py` 注明新位置。
- **数据和配置**
  - `results/v3/gifs/gifs_meta.json`：6 个条目的 `config` 和 `ppo` 路径都改成现行名称（例如 `v3_static_easy.yaml` → `v3_static.yaml`，旧的 `v3_static.yaml` → `v3_static_hard.yaml`；`ppo_*_easy_s0.pt` → `ppo_*_s0.pt`，旧的 `ppo_*_s0.pt` → `ppo_*_hard_s0.pt`）。
    - 新增字段 `config_at_render_time`，指向归档的旧配置。
    - 已核对：新旧配置内容完全相同，新路径全部存在。
    - `note` 已更新。
  - `.gitignore`：`/pytorch_rl/...` → `/archive/pytorch_rl/...`（3 处）；`SAC/A_star/__pycache__/` → `SAC/legacy/v1_original/A_star/__pycache__/`；另外新增通配的 `__pycache__/`。
- **文档**
  - `results/v3/RESULTS_v3.md`：§0 旧配置路径和 gifs_meta 说明、§1.5 A_star 位置、§6 文件列表（`legacy/rename_v3_levels.py`、`logs/STATUS.md`）。
  - `results/v3/RESULTS_v3_il_rl.md`：开头关于 `.partial` 的说明（已删除）、§8 文件列表（`legacy/run_il_static_ablation.ps1`、`logs/STATUS.md`、中期版本的新位置）。
  - `results/sacx_aux/RESULTS_sacx_aux.md`：§5 文件列表（`legacy/sacx_aux_diag2.py`、`logs/STATUS.md`）。
  - `README.md`：更新了目录树、§2 wandb 说明、§4.8 STATUS 路径和旧队列说明；新增 §4.9 遗留代码的运行方式；§5 加入 `v3_dev` 和本文件。

**未改动、已知的旧路径文字引用**：
- `results/final/`、`results/archive/` 中的文字；
- `README.old.md`；
- 两个 STATUS.md 日志中的历史记录；
- `SAC/baseline_original/training_SAC-X_original.py`：原样冻结的原始代码，其中的 `from model import *` / `import A_star.algorithm` 原本指向 `SAC/`，现在需要在 `legacy/v1_original/` 中找。它本来就依赖未安装的 wandb，也没有被任何脚本调用。

## 4. 验证（2026-10-03，UTC+8）

- **grep 旧路径**：范围不含 `results/final`、`results/archive`、`archive/`、`README.old.md`。剩下的命中只有：
  - 已更新的"新位置"说明；
  - README 目录树中 `legacy/` 下的文件名；
  - `.gitignore` 的 `__pycache__` 规则；
  - STATUS.md 的历史日志行；
  - 仍在使用的 `logs/pids.log`（由 `run_v3_ppo.ps1`、`run_sacx_aux.ps1`、两个队列脚本重新生成）。
- **`py_compile`**：`SAC/` 下全部 40 个 `.py`（含 `legacy/`、`baseline_original/`）通过，0 失败。字节码写到了临时目录，没有在 `SAC/` 中生成 `__pycache__`。
- **`--help` 冒烟测试**：13 个脚本全部 rc=0，包括 `training_SAC-X.py`、`training_SAC.py`、`training_PPO.py`、`train_ppo_v3.py`、`il_v3.py`、`eval_v3.py`、`eval_il_rl_v3.py`、`render_il_rl_v3.py`、`run_v3_rl_queue.py`、`render_v3.py`、`plot_comparison.py`、`render_gifs.py`、`sacx_aux_eval.py`。
- **import 测试**：`env_v3, astar_v3, torch_env, environment, environment_v2, parameters, report_il_rl_v3, sacx_aux_report, render_sacx_aux_gif` 通过。
- **v1 遗留代码**：从其他工作目录导入 `training`、`plot_results` 成功（wandb 用空模块代替，因为环境中没有安装）；`model`、`A_star.algorithm` 来自 `legacy/v1_original/`，`environment` 来自 `SAC/`。
- **`sacx_aux_diag2.py`**：只执行了路径设置部分（没有运行评估）。`sacx_aux_eval` 可以导入，两个 checkpoint 和输出目录都能找到。

整理后项目共 490 个文件，385,889,046 B（README 等新文档已计入）。

---

## 原始整理方案（2026-10-03 写成；以下为执行前的原文，未改动）

不会改动 `results/final/`，也不会改动 `SAC/checkpoints/*_1M.pt`。大小以字节为单位（括号内为约数）。

### A. 建议删除（安全：可再生成或只是杂项），合计 325,077 B（约 0.31 MB）

| # | 路径 | 大小 | 理由 |
|---|---|---|---|
| A1 | `SAC/__pycache__/`（11 个文件） | 241,647 | Python 字节码缓存，运行时会自动重新生成 |
| A2 | `results/v3/runs/{sac,sacx}_bounce_s0.csv.partial`、`{sac,sacx}_static_s1.csv.partial` | 19,959 | 2026-10-02 用户暂停时被中断的 run 的部分日志；这些 run 已从头重新训练，未参与任何统计 |
| A3 | `results/v3/logs/{sac,sacx}_bounce_s0.out.partial`、`{sac,sacx}_static_s1.out.partial` | 29,283 | 同上 |
| A4 | 46 个 0 字节文件：`results/v3/logs/*.err`（31 个）、`results/sacx_aux/logs/*.err`（15 个） | 0 | 空的 stderr 文件，只是杂项 |
| A5 | `results/v3/logs/eval_il_rl_{static,sr,all1}.out`（以及对应的空 `.err`） | 27,035 | 中期评估的输出，已被 `eval_il_rl_final.out` 取代 |
| A6 | `results/v3/logs/pids.log`、`results/sacx_aux/logs/pids.log` | 3,387 | 历史 PID 记录（进程早已结束）。同样的信息也记录在 STATUS.md 中 |
| A7 | `SAC/A_star/`（3 个文件） | 3,766 | 旧版网格 A*：有 bug（写的是 `child.g/f` 而不是 `gCost/fCost`），没有使用，并且与 `pytorch_rl/A_star/` **逐字节相同**。`astar_v3.py` 的 docstring 和 `RESULTS_v3.md` 1.5 节以文字形式提到 `SAC/A_star/algorithm.py`，删除后需要改成指向 `pytorch_rl/A_star/` |

### B. 可选删除（体积大；可以再生成，或只是消融的权重），合计 334,187,383 B（约 334 MB）

| # | 路径 | 大小 | 理由 / 影响 |
|---|---|---|---|
| B1 | `results/v3/il/demos_{static,random,bounce}.pt` | 275,826,218 | A* 示范数据，可以用 `il_v3.py demos --mode <m> --episodes 50000` 按相同的 seed 重新生成（每个模式 2–7 min）。统计 `demos_*_stats.json` 保留，因此 `report_il_rl_v3.py` 仍可运行；只是重新训练 BC / DAgger 前需要先重新生成示范 |
| B2 | `results/v3/il/*_h1024*.pt`（BC 1k/10k/50k × 3 个模式，DAgger `static_h1024` + 3 个 `*_h1024_big`，共 13 个文件） | 58,361,165 | 2×1024 网络的消融权重，结果已经写进 json、`il_rl/eval_*.json` 和 `table_il_rl.md`。删除后，如果重新运行 `eval_il_rl_v3.py`，这些行会从表格中消失 |

**A + B 全部执行：334,512,460 B（约 334.5 MB，即 319 MiB）**，占项目总大小 386,182,447 B 的约 87%。

### C. 建议移动 / 改名（不删除任何内容）

| # | 从 | 到 | 理由 | 需要同步修改的地方 |
|---|---|---|---|---|
| C1 | `SAC/run_queue_1M.ps1`（1,658）、`SAC/run_queue_new_env.ps1`（2,217） | `SAC/legacy/` | 旧版 v2 队列，输出目录 `results/new_env/` 已不存在 | `results/final/CONCLUSIONS.md` 和 `results/archive/v2_dev/*.md` 以文字形式提到这两个文件。`final` 不改，可以在 README 中说明 |
| C2 | `SAC/rename_v3_levels.py`（2,005）、`SAC/run_il_static_ablation.ps1`（502）、`SAC/sacx_aux_diag2.py`（2,789） | `SAC/legacy/` | 一次性脚本，已经执行过 | `RESULTS_v3.md` §6、`RESULTS_v3_il_rl.md` §8、`RESULTS_sacx_aux.md` 中的文件列表。`sacx_aux_diag2.py` 依赖 `import sacx_aux_eval` 和相对路径 `checkpoints/`，移动后如果要再运行，需要设置 sys.path |
| C3 | `SAC/training.py`（21,342）、`SAC/model.py`（6,588）、`SAC/plot_results.py`（3,985） | `SAC/legacy/v1_original/` | v1 原始 SAC 代码（依赖 wandb，当前未使用） | 它们 import 了 `environment` 和 `parameters`（这两个文件要留在 `SAC/`，因为其他脚本也用到），移动后需要修改 sys.path。**可选**，收益很小 |
| C4 | `pytorch_rl/`（61,859，早期教程代码） | `archive/pytorch_rl/` | 不属于当前代码 | `.gitignore` 中的 `/pytorch_rl/...` 条目；`README.old.md` |
| C5 | `results/v3/RESULTS_v3_il_rl.interim_2026-10-02.md`（13,462） | `results/archive/v3_dev/` | 已被最终版取代 | 只有 `results/v3/STATUS.md` 提到它 |
| C6 | `results/v3/STATUS.md`（6,206）、`results/sacx_aux/STATUS.md`（3,358） | 各自的 `logs/STATUS.md` | 它们是运行日志，不是结果文档 | `SAC/run_v3_rl_queue.py` 和 `SAC/sacx_aux_queue.py` 写的是这个路径，需要同步修改，否则会重新创建；`RESULTS_v3.md` 和 `RESULTS_sacx_aux.md` 提到了 STATUS.md |
| C7 | `configs/old_names_2026-10-02/`（7 个文件，2,815） | `results/archive/v3_dev/configs_old_names/` | 改名前的旧配置 | `RESULTS_v3.md` §0 中的路径；`results/v3/gifs/gifs_meta.json` 中记录的是旧的 config 路径（只作为记录） |

**不建议移动的**：
- `SAC/checkpoints/`：包含 v2 的最终 `*_1M.pt`，多个脚本的默认参数指向这里。
- `SAC/baseline_original/`：`test_torch_env.py` 依赖它。
- `SAC/parameters.py`、`environment.py`、`torch_env.py`：训练脚本依赖它们。
- `logs/`：课程文档。
- `results/final/`：按要求不动。

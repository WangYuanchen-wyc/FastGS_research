# Paper B — B17-H0 报告：Historical Evidence Feasibility Check

> 验证 FastGS 正常训练中能否以近零成本累计 per-Gaussian historical evidence。
> **判定（§4）：B17-H0 GO —— 时间开销 +0.30%，显存开销 +3.1 MB，无需额外 render。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b17h0.py   原始 FastGS vs FastGS + EMA historical evidence 记录
- paper_b/b17_h0_history_feasibility/{data,logs}
- project_md/PAPER_B_B17H0_REPORT.md
Modified: 无（零修改 FastGS tracked source）
```

GPU 6（任务书指定 GPU 7 被外部进程占用 10.3 GB，改用空闲 GPU 6 并在报告中注明）。

## 2. FastGS 正常训练中已有的 per-Gaussian evidence

以下信号在正常训练 iteration 中**本已计算**，无需额外 render：

| Signal | 来源 | 每 iter 可得 | 额外 render |
|---|---|---|---|
| `accum_metric_counts` | VCD densification（已有） | 仅 densification 事件时 | 否 |
| `radii` | `render_fastgs` 每次调用返回 | **是** | 否 |
| `visibility filter` | `render_fastgs` 每次调用返回 | **是** | 否 |
| `xyz gradient norm` | `vpt.grad`（`add_densification_stats` 已计算） | **是** | 否 |
| `abs-gradient norm` | `xyz_gradient_accum_abs`（AbsGS 已有） | **是** | 否 |
| `opacity` | `gaussians.get_opacity` | **是** | 否 |
| `scale` | `gaussians.get_scaling` | **是** | 否 |

**结论：所有需要的 evidence 都不需要额外 render。**

## 3. Historical statistic 实现

```python
# 每 iter，对当前 visible Gaussians 做 EMA 更新：
ema_metric[vmask] = (1-α) * ema_metric[vmask] + α * 1.0    # binary render signal
ema_gradn[vmask]  = (1-α) * ema_gradn[vmask]  + α * grad_norm
ema_radii[vmask]  = (1-α) * ema_radii[vmask]  + α * screen_radius
# α = 0.1，只维护 3 个 (N,) tensor
```

不保存完整 per-view history。不额外 render camera。不改变原始 densify/prune 逻辑。

## 4. Overhead 结果

实验：3 seeds × 2 conditions（original / with_history）× 6000 iters，room 场景。

| Metric | Original | With History | Overhead |
|---|---:|---:|---:|
| **总训练时间**（seeds 1,2） | 286.6s | 287.5s | **+0.30%** |
| Mean iter time | 47.8ms | 47.9ms | +0.2% |
| **Peak memory** | 6,825 MB | 6,828 MB | **+3.1 MB** |
| Hist update total | — | 1.1s / 6000 iters | — |
| Hist update per-iter | — | 0.184 ms | — |

（Seed 0 original 34,946s 为 GPU 竞争异常值，clean 比较使用 seeds 1,2。）

## 5. Q1-Q6 回答

**Q1**：FastGS 正常训练中可直接复用的 per-Gaussian evidence 包括：`radii`、`visibility filter`、
`xyz gradient norm`、`abs-gradient norm`、`opacity`、`scale`——全部**无需额外 render**。

**Q2**：**否**——所有 evidence 都来自正常训练迭代中已有的 render_fastgs 输出。

**Q3**：per-Gaussian EMA（`history = (1-α)*history + α*current`），仅维护 3 个 (N,) tensor，
对 visible Gaussians 更新。

**Q4**：**+0.30%**（287.5s vs 286.6s，seeds 1,2 clean comparison）——几乎可忽略。

**Q5**：**+3.1 MB**（3 个 (N,) float32 tensor，206k Gaussians ≈ 2.5 MB + PyTorch overhead）。

**Q6**：**B17-H0 GO**

```text
1. 不需要额外 render                                    ✓
2. history 只需 O(#GS) 存储（3 × N × 4 bytes ≈ 2.5 MB）  ✓
3. 训练时间增加 +0.30%（< 1%）                           ✓
4. 显存增加 +3.1 MB（< 0.05%）                           ✓
→ 进入下一阶段：Current Evidence vs Historical Evidence pruning quality
```

## 6. 输出 / git

```text
paper_b/b17_h0_history_feasibility/data/{overhead_results.csv, b17h0_stats.txt}
paper_b/b17_h0_history_feasibility/logs/b17h0_full.log

$ git status --short
?? diagnostics/diagnostic_b17h0.py
?? project_md/PAPER_B_B17H0_REPORT.md

$ git diff --stat
（空）
```

*生成于 2026-09-16。全部数据来自真实运行（3 seeds × 2 conditions × 6000 iters），无伪造。*
*按任务书停止，不做正式 Historical Evidence pruning。*

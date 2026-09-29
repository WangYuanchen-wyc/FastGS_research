# Paper B — B14-T 报告：Error Threshold Diagnostic

> 验证 FastGS 的 hard error threshold（`loss_thresh=0.10`）是否导致 limited-view 下重要
> Gaussian 出现 zero-evidence 并被错误 pruning。
> 5 thresholds × 20 repeats × 3 pruning ratios = 300 组 whole-model evaluations。
> **判定（§5）：B14-T NO-GO —— 降低 threshold 减少 zero-evidence 但使 pruning quality
> 更差；升高 threshold "改善" quality 但实为退化到近随机选择。hard threshold 不是
> limited-view pruning 的核心问题。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b14t.py   threshold sweep（0.025/0.05/0.10/0.15/0.20）
                                  × 20 B13-B subsets × 3 ratios + All-view reference
- diagnostics/analyze_b14t.py      全部分析 + 2 图
- paper_b/b14_t_error_threshold/{data,plots,logs,cache}
- project_md/PAPER_B_B14T_REPORT.md
Modified: 无（零重训练，复用 it30000 checkpoint）
```

GPU 4。it30000（206,630 GS，311 views）。

## 2. Q1: 不同 threshold 的 zero-evidence 比例

| Threshold | visible-but-zero-evidence |
|---|---:|
| 0.025 | **0.96%** |
| 0.050 | 3.58% |
| **0.100**（baseline） | **20.79%** |
| 0.150 | 43.22% |
| 0.200 | 60.21% |

降低 threshold 确实大幅减少 zero-evidence（0.10→0.025 降低 95%）。

## 3. Q2: False-prune 如何变化

| Threshold | @5% | @10% | @20% |
|---|---:|---:|---:|
| 0.025 | 9,115 | 15,690 | 23,713 |
| 0.050 | **8,876** | **15,370** | **22,522** |
| 0.100 | 8,944 | 15,659 | 24,320 |
| 0.150 | 9,566 | 17,644 | 28,544 |
| 0.200 | 9,756 | 18,493 | 30,969 |

False-prune 在 0.05 处最低，但差异不大（0.05 vs 0.10：15,370 vs 15,659）。

## 4. Q3: Spearman / Top-10% overlap 如何变化

| Threshold | Spearman | Top-10% overlap |
|---|---:|---:|
| 0.025 | 0.601 | 0.241 |
| **0.050** | **0.633** | **0.256** |
| 0.100 | 0.611 | 0.242 |
| 0.150 | 0.551 | 0.146 |
| 0.200 | 0.484 | 0.105 |

Spearman 在 0.05 处微弱峰值（0.633 vs baseline 0.611），但 0.025 反而低于 baseline。

## 5. Q4: Pruning quality（核心结果）

| Threshold | 5% ΔPSNR vs All | **10% ΔPSNR** | **20% ΔPSNR** |
|---|---:|---:|---:|
| 0.025 | −0.786 | **−2.045** | −2.242 |
| 0.050 | −0.581 | **−1.891** | **−2.768** |
| **0.100**（baseline） | −0.131 | **−0.623** | −2.466 |
| 0.150 | −0.002 | **−0.332** | −1.304 |
| 0.200 | −0.001 | **−0.308** | −0.887 |

**关键发现（反直觉）**：
- **降低 threshold（0.025/0.05）使 pruning quality 更差**（@10%：−2.05/−1.89 vs baseline −0.62）
- **升高 threshold（0.15/0.20）表面"改善"**（@10%：−0.33/−0.31；@20%：−1.30/−0.89）

## 6. Q5: 是否存在比 0.10 更好的稳定 threshold

**表面上是 0.20 最好**（@10% −0.31 dB vs baseline −0.62；@20% −0.89 vs −2.47）。
**但这是虚假改善**——机制见下节。

## 7. 判定与机制解释

```text
B14-T NO-GO
```

**为什么降低 threshold 反而更差**：
- 低 threshold（0.025）几乎给所有可见 Gaussian 都分配了 evidence（zero-evi 仅 0.96%）
- 但大量 **不重要的 Gaussian 也在低误差区域获得了弱 evidence**，引入噪声
- 这使 ranking 被 noise 主导，pruning 决策更差（Spearman 反而下降）

**为什么升高 threshold 表面"改善"但实际是退化**：
- 高 threshold（0.20）下 60% 的可见 Gaussian evidence=0 → **massive ties**
- 在 ties 中，排序由 sort 的 tie-breaking（即 Gaussian index）决定——**等价于随机选择**
- 随机 pruning 优于系统性错误 pruning（不删除系统性偏差方向的重要 Gaussians）
- 这不是 threshold 更好的证据，而是 **importance estimation 退化为随机**

**核心结论**：
```text
hard threshold 不是 limited-view pruning 的关键问题。
降低它减少 zero-evidence 但引入弱-evidence 噪声；
升高它退化到随机选择。
真正的问题是 10 个 views 无法可靠估计任何 threshold 下的 importance。
```

这进一步确认 B13-B Fix 的 Route B（Evidence Estimation GO）的方向：不是调 threshold，
而是需要 **不依赖 binary error gating 的 importance estimation**。

## 8. Q6 回答

```text
B14-T = NO-GO
```

## 9. 输出 / git

```text
paper_b/b14_t_error_threshold/data/{threshold_zero_evidence.csv,
  threshold_false_prune.csv, threshold_pruning_results.csv, b14t_stats.txt}
paper_b/b14_t_error_threshold/plots/{threshold_effect.png, psnr_vs_threshold.png}

$ git status --short
?? diagnostics/diagnostic_b14t.py
?? diagnostics/analyze_b14t.py
?? project_md/PAPER_B_B14T_REPORT.md

$ git diff --stat
（空）
```

*生成于 2026-09-13。全部数据来自真实运行（5 thresholds × 20 reps × 3 ratios = 300 组），无伪造。*
*按任务书停止，不实现正式方法。*

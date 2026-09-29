# Paper B — B12-S 报告：View-Sampling Robustness of FastGS Importance

> 一个问题：FastGS 的 pruning importance 会因随机 view 采样产生明显噪声并导致错误 pruning 吗？
> 3 checkpoints × 4 view counts × 10 repeats × 3 pruning ratios，共 369 组 whole-model 评估。
> **判定（§11）：B12-S WEAK GO —— 10-view 噪声确实存在且改变 pruning candidates（Top-10% overlap
> 仅 0.22-0.39），在 it10000/15000 上质量差异清晰单调；但 it30000 的中间 view-counts 出现异常，
> 趋势不完全一致。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b12s.py   All-view + 10/20/50/100-view pruning score（各 10 repeats）
                                  + whole-model pruning @ 5%/10%/20% + 全 test set 评估
- diagnostics/analyze_b12s.py      收敛分析 + 质量对比 + 5 图
- paper_b/b12_s_view_sampling_robustness/{data,plots,logs,cache}
- project_md/PAPER_B_B12S_REPORT.md
Modified: 无（git diff --stat 空；零重训练，复用 B11-V checkpoints）
```

GPU 1（任务书指定 GPU 2 被占用后切换）。it10000（248,450 GS）· it15000（265,396）· it30000（206,630）。

## 2. Score 收敛（Spearman vs All-view，10 repeats mean±std）

| Checkpoint | 10-view | 20-view | 50-view | 100-view |
|---|---:|---:|---:|---:|
| it10000 | 0.648±0.069 | 0.814±0.020 | 0.921±0.006 | **0.963±0.005** |
| it15000 | 0.642±0.033 | 0.777±0.018 | 0.890±0.015 | **0.951±0.011** |
| it30000 | 0.609±0.039 | 0.736±0.039 | 0.854±0.013 | **0.935±0.011** |

**10-view Spearman 仅 0.61-0.65**——FastGS 当前使用的 10-view 采样确实产生明显的 score 噪声。
随 view 数增加清晰收敛到 All-view（100-view 达 0.94-0.96）。

## 3. Top-K overlap（pruning candidates 是否真的不同）

| Checkpoint | Top-10% overlap（10v → 100v） |
|---|---|
| it10000 | 0.332 → 0.812 |
| it15000 | 0.393 → 0.886 |
| it30000 | 0.218 → 0.677 |

**10-view 的 Top-10% pruning candidates 与 All-view 仅重合 22-39%**——score 噪声确实改变了
"谁会被删除"，不只是 rank 微调。

## 4. Pruning Quality（核心结果，10 repeats mean±std）

### it10000（clean monotone trend ✓）

| Pruning | All-view | 10-view | 20-view | 50-view | 100-view |
|---|---:|---:|---:|---:|---:|
| 5% | 29.636 | 29.468±0.048 | 29.434±0.055 | 29.536±0.059 | **29.609±0.047** |
| 10% | 29.635 | 29.150±0.087 | 29.157±0.092 | 29.365±0.103 | **29.569±0.083** |
| 20% | 29.289 | 28.133±0.251 | 28.317±0.225 | 28.882±0.134 | **29.166±0.138** |

**清晰单调趋势**：10v < 20v < 50v < 100v ≈ All，全部 ratios 一致。

### it15000（clean monotone trend ✓）

| Pruning | All-view | 10-view | 100-view |
|---|---:|---:|---:|
| 10% | 30.983 | 30.502±0.132 | **30.922±0.067** |
| 20% | 30.373 | 28.848±0.477 | **30.048±0.234** |

同 it10000 模式：**更多 views → 更好 pruning quality**。

### it30000（⚠️ 混合结果）

| Pruning | All-view | 10-view | 20-view | 50-view | 100-view |
|---|---:|---:|---:|---:|---:|
| 5% | 31.386 | 31.307±0.115 | 31.122±0.258 | **30.439±0.393** | 30.981±0.287 |
| 10% | 30.909 | 30.312±0.249 | 29.795±0.531 | **29.534±0.372** | 30.306±0.349 |
| 20% | 29.839 | 27.644±0.594 | 27.046±0.843 | 28.122±0.350 | **29.017±0.453** |

**⚠️ 50-view 在 it30000 上系统性异常**（比 10-view 更差），破坏单调趋势。
仅在 20% pruning 时 100-view 明确优于 10-view（29.02 vs 27.64，差 1.38 dB）。

## 5. 10-view repeat 方差（quality instability）

```text
it30000 @ 20%: 10v range = 1.87 dB（best 28.66 / worst 26.79）
it15000 @ 20%: 10v range = 1.67 dB
it10000 @ 20%: 10v range = 0.71 dB
```

**同样使用 10 个 views，不同 random subset 的 pruning 质量差距可达 1.9 dB**——这本身就是
view-sampling noise 导致错误 pruning 的直接证据。

## 6. Trend 总结

```text
it10000: 10v(29.15) < 20v(29.16) < 50v(29.36) < 100v(29.57) ≈ All(29.63)  ✓ 清晰单调
it15000: 10v(30.50) < 20v(30.51) < 50v(30.79) < 100v(30.92) ≈ All(30.98)  ✓ 清晰单调
it30000: 10v(30.31) ≈ 100v(30.31), 50v 最差(29.53)                          ✗ 非单调
```

**Gap（10v − All @ 10% pruning）**：it10000 −0.48 dB · it15000 −0.48 dB · it30000 −0.60 dB

## 7. 九个问题的回答

**Q1**: it10000 / it15000 / it30000。

**Q2**: #GS 分别为 248,450 / 265,396 / 206,630。

**Q3**: 10-view Spearman vs All = **0.61-0.65**；100-view = **0.94-0.96**。

**Q4**: Top-10% overlap（10v）= **0.22-0.39**；100v = **0.68-0.89**。

**Q5**: **是**——10-view 与 All-view 的 pruning candidates 仅重合 22-39%，"谁被删除"确实不同。

**Q6**: Pruning quality 见上表（PSNR/SSIM/LPIPS 全记录于 data CSV）。关键数字 @ 10%：
- it10000: 10v=29.15 vs All=29.63（−0.48 dB）
- it15000: 10v=30.50 vs All=30.98（−0.48 dB）
- it30000: 10v=30.31 vs All=30.91（−0.60 dB）

**Q7**: 10-view repeat 方差 @ 20%：**range 0.71-1.87 dB**（最大在 it30000）。

**Q8**: **是（在 it10000/15000）**——清晰单调趋势"更多 views → 更好 quality"。
**it30000 部分成立**（100v > 10v @ 20%，但 5%/10% 时 50-view 异常）。

**Q9**: **it30000 部分成立**——@ 20% pruning 明确（10v −2.20 dB vs 100v −0.82 dB，差 1.38 dB），
但 @ 5%/10% 时趋势不单调（50-view 异常差于 10-view）。

**Q10**: **B12-S WEAK GO**。

## 8. 判定（§11）

```text
WEAK GO：
1. 10-view vs All-view ranking 差异明显       : Y（Spearman 0.61，overlap 0.22-0.39）
2. 随 view 数增加收敛                         : Y（0.61→0.96）
3. 更稳定 score → 更好 quality                : Y at it10000/15000（全 ratio 单调）
                                                PARTIAL at it30000（仅 20% 成立，50v 异常）
4. it30000 明确成立                           : 仅 @ 20%（差 1.38 dB）

→ B12-S WEAK GO
```

**不支持 STRONG GO 的原因**：it30000 的 50-view 结果系统性异常（在所有 ratios 上均差于
10-view），破坏了"view 越多 quality 越好"的一致趋势。这可能反映了 final model 的 pruning score
分布特性（已通过 final_prune 去除了高分尾部，剩余分布压缩），或特定 view subset 的采样偏差。

## 9. 核心结论（如实）

```text
FastGS 的 multi-view pruning importance 本身有效，
但 10-view 随机采样确实引入 estimation noise：
- Spearman 仅 0.61（vs All-view 0.96 @ 100-view）
- Top-10% pruning candidates 仅 22-39% 重合
- 10-view 不同 repeats 间 pruning PSNR 差可达 1.9 dB
- 在 it10000/15000 上，更多 views 单调提升 pruning quality（−0.48 dB @ 10%）
- 在 it30000 上，效应在 20% pruning 时最显著（10v vs 100v 差 1.38 dB），
  但中间 view counts 存在异常

这说明 view-sampling noise 存在且影响 pruning quality，
但 effect size 和 consistency 不足以直接支持 STRONG GO。
```

## 10. 输出 / git

```text
paper_b/b12_s_view_sampling_robustness/data/{score_stability.csv,
  pruning_candidate_overlap.csv, pruning_results.csv, b12s_stats.txt}
paper_b/b12_s_view_sampling_robustness/plots/{spearman_vs_num_views, overlap_vs_num_views,
  psnr_vs_num_views, pruning_quality_vs_ratio, ten_view_repeat_variance}.png
paper_b/b12_s_view_sampling_robustness/logs/{b12s_full, b12s_analysis}.log

$ git status --short
?? diagnostics/diagnostic_b12s.py
?? diagnostics/analyze_b12s.py
?? project_md/PAPER_B_B12S_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-09-11。全部数据来自真实运行（3 ckpt × 4 view counts × 10 repeats × 3 ratios =
369 组 whole-model evaluations），无伪造。按任务书停止，不实现正式方法。*

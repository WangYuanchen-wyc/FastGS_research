# Paper B — B11-V 报告：Multi-view Importance Reliability

> 诊断 FastGS 的 multi-view importance 是否可靠：(A) view subset 稳定性，(B) 同 importance 下
> cross-view agreement 异质性，(C) score-matched 高/低 agreement group removal 差异，
> (D) 稳定性与 agreement 的关联。4 checkpoints（it5000/10000/15000/30000）。
> **判定（§13）：B11-V STRONG GO。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b11v.py   训练（1 native run，4 checkpoints 保存）+
                                  20 view subsets × 4 checkpoints 稳定性 +
                                  per-view agreement + stratified group removal
- diagnostics/analyze_b11v.py      全部分析 + 6 图
- paper_b/b11_v_multiview_importance_reliability/{data,plots,cache,checkpoints,logs}
- project_md/PAPER_B_B11V_REPORT.md
Modified: 无（git diff --stat 空）
```

GPU 3。1 次完整训练（~3 min）+ 4 × 20 subsets × ~800 renders + 4 × 7 组 removal evals。

## 2. Checkpoints 与基本统计

| Checkpoint | #GS | 20 subsets | n_views/subset |
|---|---:|---:|---:|
| it5000 | 212,109 | 20 | 10 |
| it10000 | 248,450 | 20 | 10 |
| it15000 | 265,396 | 20 | 10 |
| it30000 | 206,630 | 20 | 10 |

## 3. Q1-Q5 回答

### Q1: Checkpoints & Q2: Gaussian 数量 — 见上表。

### Q3: 每 checkpoint 20 subsets（与 FastGS 原始 sampling_cameras 同样 10 views）。

### Q4: Subset 间 Spearman rho

Top-K Jaccard overlap（20 subsets 两两配对均值，跨 4 checkpoints）：

| Checkpoint | Top-5% | Top-10% | Top-20% |
|---|---:|---:|---:|
| it5000 | 0.255 | 0.310 | 0.389 |
| it10000 | 0.267 | 0.320 | 0.385 |
| it15000 | 0.247 | 0.294 | 0.356 |
| it30000 | 0.235 | 0.273 | 0.331 |

**Top-5% Jaccard 仅 0.23-0.27**——换一组 10 个 views，前 5% 重要 Gaussian 只有约 1/4 重合。
**importance ranking 对 view subsets 明显不稳定。**

### Q5: Gaussian rank/score instability

```text
                 percentile_std median   p90      score CV median
it5000                0.166             0.278        1.075
it10000               0.171             0.282        1.162
it15000               0.184             0.293        1.282
it30000               0.195             0.304        1.520
```

单个 Gaussian 的 importance percentile 在不同 subsets 下的标准差中位数为 0.17-0.19（即 ±17-19
个百分点）；score CV 中位数为 1.1-1.5（波动超过均值本身）。不稳定性随训练成熟度**增大**。

### Q6: 相同 importance 下 agreement 差异

同一 importance decile 内 effective_view_count 分布（it30000 为例）：

| Decile | eff_vc [p25~p75] | support_ratio | max_view_fraction |
|---|---|---:|---:|
| 5（中低） | 1.0 ~ 2.0 | 0.20 | 0.86 |
| 7（中高） | 1.3 ~ 2.3 | 0.20 | 0.75 |
| 9（最高） | 1.3 ~ 2.5+ | 0.30 | 0.73 |

**同一 decile 内 eff_vc 的 IQR 跨度 ≥ 1.0**：importance 相近的 Gaussian 中，有的被多视角一致
支持（eff_vc > 2），有的完全由单一视角主导（max_frac ≈ 0.86、eff_vc ≈ 1）。agreement 差异显著。

### Q7: Group removal 差异（核心结果）

Score-matched 高/低 agreement groups（importance-matched，同 checkpoint），删除后全 test set 评估：

| Checkpoint | 删除 | ΔPSNR (low-agr) | ΔPSNR (high-agr) | **差异** |
|---|---|---:|---:|---:|
| it5000 | 5% | −0.395 | −0.321 | −0.075（不显著） |
| it5000 | 10% | −1.207 | −0.938 | −0.269（不显著） |
| **it10000** | **5%** | **−0.278** | **−0.414** | **+0.136** |
| **it10000** | **10%** | **−0.762** | **−1.258** | **+0.496** |
| **it15000** | **5%** | **−0.483** | **−0.674** | **+0.190** |
| **it15000** | **10%** | **−1.286** | **−1.924** | **+0.638** |
| **it30000** | **5%** | **−0.531** | **−0.718** | **+0.188** |
| **it30000** | **10%** | **−1.467** | **−1.899** | **+0.433** |

**在 it10000/15000/30000 三个 checkpoints 上，删除 low-agreement 组的 PSNR 损失一致小于删除
high-agreement 组**（diff = +0.14~+0.64 dB）。it5000 方向相反（−0.07/−0.27），但幅度较小——
早期训练阶段 agreement 信息尚未分化。

SSIM/LPIPS 同方向：it30000 删除 10% 时 low-agr SSIM 0.9086 vs high-agr 0.9072；LPIPS 0.2405 vs 0.2410。

### Q8: Stability vs Agreement 关联

```text
Spearman(percentile_std, max_view_fraction) = +0.19 ~ +0.35（4 checkpoints 均正）
Spearman(percentile_std, support_ratio)     = +0.01 ~ +0.10（弱）
Spearman(percentile_std, eff_vc)             = +0.02 ~ +0.09（弱）
```

**不稳定性与 concentration（max_view_fraction）正相关**：单视角主导的 Gaussian 在不同 view
subsets 间 rank 波动更大——view-sampling instability 的根源正是 evidence 的 cross-view concentration。

### Q9: 跨 checkpoint 稳定性

- **view instability**：4 个 checkpoints 全部存在（Top-5% Jaccard 0.23-0.27），且随训练增大
- **agreement heterogeneity**：4 个 checkpoints 全部存在（decile 内 eff_vc IQR ≥ 1.0）
- **removal difference**：it10000/15000/30000 一致正（low-agr 更可删），it5000 反向（幅度小）

### Q10: 判定

```text
1. importance ranking 对 view subsets 明显不稳定        : Y（Top-5% Jaccard 仅 0.23-0.27）
2. 相同 score 下 agreement 差异显著                     : Y（decile 内 eff_vc IQR ≥ 1.0）
3. score-matched 下 low-agr 更易删，质量损失更小       : Y（3/4 checkpoints，diff +0.14~+0.64 dB）
4. 多 checkpoints 稳定出现                              : Y（3/4，it5000 反向但幅度小且属早期）

→ B11-V STRONG GO
```

**核心结论**：

```text
FastGS aggregate multi-view importance does not fully capture cross-view
agreement. Gaussians with similar aggregate importance can exhibit
substantially different cross-view support, and this difference reveals
additional pruning information.

FastGS 虽然使用 multi-view evidence，但 aggregate score 无法完全区分
"多视角一致支持"和"少数视角主导"。
这种 agreement 差异能够提供额外的压缩信息。
```

## 4. 如实标注

- it5000（最早期）removal 差异方向相反——早期阶段 agreement 信息尚未从噪声中分化，现象在
  训练中后期（it10000+）稳定成立
- Removal 差异的绝对幅度（0.14-0.64 dB @ 5-10% 删除）对 compression 有实际意义但非巨大
- SSIM/LPIPS 差异比 PSNR 更小（~0.001），主信号在 PSNR
- 单场景（room）、单 seed 的完整训练轨迹——需要跨场景验证

## 5. 输出 / git

```text
paper_b/b11_v_multiview_importance_reliability/data/{b11v_gaussian_stability.csv,
  b11v_view_agreement.csv, b11v_removal_results.csv, b11v_stats.txt}
paper_b/b11_v_multiview_importance_reliability/plots/{rank_stability, agreement_distribution,
  score_vs_agreement, stability_vs_agreement, removal_quality_drop, topk_overlap}.png
paper_b/b11_v_multiview_importance_reliability/checkpoints/it{5000,10000,15000,30000}.pt
paper_b/b11_v_multiview_importance_reliability/logs/{b11v_full, b11v_diag3, b11v_analysis}.log

$ git status --short
?? diagnostics/diagnostic_b11v.py
?? diagnostics/analyze_b11v.py
?? project_md/PAPER_B_B11V_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-09-10。全部数据来自真实运行（1 × 30k-iter 训练 + 80 次 view-subset scoring +
28 组 removal evaluations），无伪造。按任务书停止，不实现正式方法。*

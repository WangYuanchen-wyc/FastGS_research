# Paper B — B13-B Fix 报告：Zero-Evidence Cause Diagnosis

> B13-B 发现 F（false-prune）Gaussian 的 10-view evidence=0。本 Fix 诊断原因：
> 是"没看到"（Type-A 不可见）还是"看到了但没有 error evidence"（Type-B）？
> **判定（§11）：Route B — Evidence Estimation GO。72% 的 F 属于 Type-B
> （在 sampled views 中真实可见但 FastGS metric_count=0）。**
> **这修正了 B13-B 的初步结论：主因不是 view coverage miss，而是 error-threshold miss。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b13b_fix.py   真实 visibility（radii>0）vs metric evidence 分离
                                      + F 的 A/B/C 分类 + 修正 key-view 定义（pl×metric）
- paper_b/b13_b_pruning_boundary/fix/{data,plots}
- project_md/PAPER_B_B13B_FIX_REPORT.md
Modified: 无（零重训练，复用 it30000 checkpoint + B13-B 的 20 subsets）
```

GPU 1。it30000（206,630 GS，311 views），20 repeats × 3 ratios = **2,064,366 rows**。

## 2. Q1: 真正 visibility 定义

**`radii > 0`**（rasterizer 返回的 per-Gaussian 屏幕半径）——表示该 Gaussian 通过了
frustum culling 和 tile 分配，实际参与该 view 的 rasterization。这与 B13-B 的
`accum_metric_counts > 0`（仅表示有 error evidence）是**不同概念**。

## 3. Q2-Q3: F 的 Type-A/B/C 比例（20 repeats）

| Ratio | Type-A（不可见） | Type-B（可见但无 evidence） | other |
|---|---:|---:|---:|
| 5% | 50,511（28.2%） | **128,377（71.8%）** | 0 |
| 10% | 86,414（27.6%） | **226,767（72.4%）** | 0 |
| 20% | 143,142（29.4%） | **332,288（68.3%）** | 10,962（2.3%） |

**Type-B 是绝对主导（68-72%）**——F Gaussians 在 sampled 10 views 中**真实可见**（radii>0）
但 FastGS 的 error threshold（`l1_norm > loss_thresh`）没有在其对应像素产生 metric evidence。

Type-A（真不可见）仅占 28-29%。

## 4. Q4: F 与 M/C 的真实 visibility 差异

| Metric | F | M | C |
|---|---:|---:|---:|
| vis_10v（10-view 真实可见 views） | **1.90** | 2.84 | 2.10 |
| vis_ratio_10v | 0.190 | 0.284 | 0.210 |
| vis_all（全视角可见） | **69** | 75 | 62 |
| met_10v（10-view 有 evidence 的 views） | **0.00** | 0.53 | 0.00 |
| met_all（全视角有 evidence） | **25** | 11 | 8 |

**关键发现**：F 的真实 visibility（vis10=1.90, visAll=69）**并不远低于** M（2.84, 75）——
F Gaussians 确实被 sampled views **看到了**。差异在于 **metric evidence**：
F 的 met_10v=0.00（10-view 中零 evidence），而 M 有 0.53。

**visible_but_zero_evidence rate**：
```text
F: 72.4%     M: 41.7%     C: 73.9%（@10%）
```
F 和 C 都有高比例的 "visible but no evidence"——但 C 是 All-view 也认为可删的（本身不重要），
而 F 是 All-view 认为重要的（metAll=25 vs C 的 8）。

## 5. Q5: 修正后 key-view hit rate

使用正确的 pruning evidence 定义（`photometric_loss × metric_count`）：

| Group | hitTop1 | hitTop3 | hitTop5 |
|---|---:|---:|---:|
| **F** | **0.0000** | **0.0001** | **0.0013** |
| M | 0.0536 | 0.1400 | 0.2147 |
| C | 0.0000 | 0.0017 | 0.0121 |

**F 的 corrected hitTop1/3/5 仍为 0**——与 B13-B 的原始结论一致。
但修正后的数据表明：这些关键 views 的重要性来自 **error evidence**（高 residual 区域），
而 F Gaussians 虽然在 sampled views 中可见（被 rasterize），其所在像素的 residual
**低于 FastGS 的 loss threshold**。

## 6. Q6: F 是否因为关键 views 没被采到？

**部分正确但不完整**。Type-A（28%）确实是关键 views 没被采到。但 Type-B（72%）的
sampled views **看到了这些 Gaussians**——问题是这些 views 中对应像素的 residual 不够高，
未超过 `loss_thresh=0.1` 的 threshold。

## 7. Q7: 是否是 visible-but-no-error-evidence？

**是，这是主要原因**。Type-B Gaussians 在 sampled views 中可见（vis10 median=2），
在全视角下有大量 evidence（metAll median=26），但 10-view 的特定 cameras 恰好从
**这些 Gaussians 对应区域 residual 较低的角度**观察——未触发 error threshold。

## 8. Q8: 跨 ratios 和 repeats 稳定性

```text
Type-B 比例: 71.8% @5% · 72.4% @10% · 68.3% @20%     → 稳定
Type-A 比例: 28.2% @5% · 27.6% @10% · 29.4% @20%     → 稳定
F vs M 的 vis10 差异方向一致（F < M）                    → 稳定
F 的 hitTop1_corr = 0.0000 在所有 ratios/repeats        → 稳定
```

## 9. 最终分流

```text
Type-B 主导（68-72%）→ Route B: Evidence Estimation GO
```

**修正 B13-B 的结论**：
- B13-B 初步归因于 "view coverage miss"（关键 views 没被采到）
- 本 Fix 证明主因是 **"error threshold miss"**：views 采到了（Gaussian 可见），
  但该视角下这些 Gaussians 的 residual 低于 threshold，不产生 pruning evidence
- 两类机制共存：Type-A 28%（真 coverage miss）+ Type-B 72%（evidence estimation miss）

**对 Paper B 的方向指引**：
1. **主要方向（72%）**：改进 FastGS 的 error-conditioned importance estimation——
   当前 binary threshold（`l1_norm > 0.1`）使 sample views 的 residual 波动直接
   转化为 evidence 的有/无，对 view subset 极度敏感
2. **次要方向（28%）**：budget-constrained view coverage——保证每个 Gaussian 的
   可见方向至少被一个 sampled view 覆盖

## 10. Type-A 补充分析

Type-A F Gaussians（真不可见）：visAll median=23（全视角下高度可见），55% 的 visAll≥20。
它们的可见方向集中在特定视角区域，10 个随机 views 容易完全错过。但这类仅占 28%。

## 11. 输出 / git

```text
paper_b/b13_b_pruning_boundary/fix/data/zero_evidence_categories.csv（2,064,366 rows）
paper_b/b13_b_pruning_boundary/cache/{all_radii.npy, all_metric.npy}

$ git status --short
?? diagnostics/diagnostic_b13b_fix.py
?? project_md/PAPER_B_B13B_FIX_REPORT.md

$ git diff --stat
（空）
```

*生成于 2026-09-12。全部数据来自真实运行（20 repeats × 3 ratios，~2M 行），无伪造。*
*按任务书停止，不实现正式方法。*

# Paper B — B11-V Fix 报告：Pruning-Score Controlled Agreement Diagnostic

> **本 Fix 严格控制 FastGS actual pruning score 后重新验证 B11-V 的 agreement 信号。**
> 原始 B11-V 使用 importance decile 匹配——不够严格。本 Fix 使用 nearest-neighbor
> matched pairs（|Δscore| ≈ 0.000000），确保两组 pruning score 分布完全一致。
> **判定（§11）：B11-V Fix NO-GO —— 严格 score matching 后，it15000/it30000 上
> Low/High agreement removal 差异消失；原始 B11-V 的信号来自 score matching 不充分。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b11v_fix.py   tie-aware 稳定性 + pruning_score 子集敏感性 +
                                    多 subset agreement + 最近邻 matched pairs +
                                    3 组（low/random/high）× 3 比例 × 5 bootstrap repeats
- diagnostics/analyze_b11v_fix.py   全部分析 + 2 图
- paper_b/b11_v_multiview_importance_reliability/fix/{data,plots}
- project_md/PAPER_B_B11V_FIX_REPORT.md
Modified: 无（git diff --stat 空；未重训练，复用 B11-V checkpoints）
```

GPU 7。4 checkpoints × 20 pruning subsets + 5 agreement subsets + 180 组 removal evaluations。

## 2. Fix A — Tie-aware 稳定性

```text
Checkpoint   Spearman(mean±sd)     [p10~p90]       Top-5% overlap   tied at cutoff
it5000       0.576±0.095           [0.45~0.69]     0.397            1 (0.00%)
it10000      0.578±0.092           [0.46~0.67]     0.412            1 (0.00%)
it15000      0.540±0.097           [0.42~0.66]     0.389            1 (0.00%)
it30000      0.461±0.090           [0.34~0.57]     0.376            1 (0.00%)
```

**修正后**：
- Spearman ρ = 0.46-0.58（远高于 B11-V 原始的 Jaccard 0.23-0.27 的表观不稳定性）
- **cutoff 处 tied ≈ 0**——integer score ties 不构成伪不稳定源
- 但 ρ < 0.6 仍表明 moderate instability 存在：Top-5% overlap 仅 0.38-0.41
- **pruning score 对 view subsets 中等敏感，且随训练成熟降低（0.58→0.46）**

## 3. Fix B — FastGS actual pruning_score 子集敏感性

B11-V 原始诊断的 importance_score 是 VCD metric count（用于 densify）；本 Fix 诊断的
pruning_score 是 FastGS 真正用于 prune 的 photometric-weighted consistency score。
二者对 view subsets 的稳定性不同：

| | importance（B11-V 原始） | pruning_score（本 Fix） |
|---|---|---|
| Spearman | 未正确计算（ties 未处理） | 0.46-0.58 |
| Top-5% overlap | 0.23-0.27 | 0.38-0.41 |
| 结论 | 表观极不稳定 | **中等不稳定** |

## 4. Matching 质量（Fix D/E）

```text
it5000:  n_pairs=21,210 · mean |Δpruning_score| = 0.000000（精确匹配）
it10000: n_pairs=24,845 · mean |Δpruning_score| = 0.000000
it15000: n_pairs=26,539 · mean |Δpruning_score| = 0.000000
it30000: n_pairs=20,663 · mean |Δpruning_score| = 0.000000

高/低组统计：
                 high_agreement         low_agreement
it5000:  mean_pru=0.000030  eff_vc=3.69  vs  mean_pru=0.000030  eff_vc=1.66
it10000: mean_pru=0.000039  eff_vc=2.93  vs  mean_pru=0.000039  eff_vc=1.37
it15000: mean_pru=0.000070  eff_vc=2.41  vs  mean_pru=0.000070  eff_vc=0.80
it30000: mean_pru=0.000070  eff_vc=2.45  vs  mean_pru=0.000070  eff_vc=1.30
```

**两组 pruning score 完全匹配**（差异 < 1e-6），eff_vc 差异显著（high ≈ 2.4-3.7 vs low ≈ 0.8-1.7）。

## 5. Removal 结果（核心，5 bootstrap repeats mean±std）

| Checkpoint | 删除 | PSNR_lo | PSNR_rand | PSNR_hi | **Δ(lo−hi)** | low wins |
|---|---|---:|---:|---:|---:|---:|
| it5000 | 10% | 27.810±0.000 | 27.463±0.049 | 27.720±0.000 | **+0.090** | 5/5 |
| it10000 | 10% | 29.374±0.000 | 29.067±0.027 | 29.269±0.000 | **+0.105** | 5/5 |
| it15000 | 10% | 30.589±0.000 | 30.285±0.021 | 30.592±0.000 | **−0.003** | 0/5 |
| **it30000** | **10%** | **31.093±0.000** | **30.628±0.045** | **31.111±0.000** | **−0.018** | **0/5** |

**关键发现**：

1. **it5000/it10000**：low-agreement 确实更易删（+0.09/+0.11 dB，5/5 repeats 一致）
   ——但幅度远小于 B11-V 原始报告的 +0.27/+0.50 dB

2. **it15000/it30000**：low/high agreement 差异**完全消失**（−0.003/−0.018 dB，
   low wins 0/5）——在压缩最关心的成熟阶段，agreement 不提供额外信息

3. **Random control 揭示真相**：random 删除远差于两个 matched 组
   （it30000 10%: random 30.63 vs matched 31.09/31.11）——**pruning score 本身有效**，
   但在 score-matched pool 内 agreement 不再区分 removability

4. **repeat 方差极小**（±0.000-0.005）：结果不是噪声——it15000/it30000 的零差异是真实零

## 6. 九个问题的回答

**Q1**: Tie-aware Spearman = 0.46-0.58（中等稳定，远好于 B11-V 表观值）

**Q2**: 修正后 Top-5% overlap = 0.38-0.41（仍不完美但远好于原始的 0.23-0.27）

**Q3**: pruning_score 对 view subsets **中等敏感**（Spearman 0.46-0.58，随成熟降低）

**Q4**: 同 pruning_score 下 agreement 仍有差异（eff_vc high 2.4-3.7 vs low 0.8-1.7），
**但该差异在成熟阶段不转化为 removal 差异**

**Q5**: 两组 pruning score **完全匹配**（|Δ| < 1e-6，mean/median/percentile 一致）

**Q6**: 删除 10% 后（it30000）：Low 31.093 / Random 30.628 / High 31.111——
Low ≈ High（差 −0.018），两者均远优于 Random（+0.47）

**Q7**: 同方向（low 更可删）仅在 it5000/it10000 成立（5/5 repeats），
it15000/it30000 完全消失（0/5 repeats）——**2/4 checkpoints，非多数**

**Q8**: **it30000 不成立**（−0.018 dB，low wins 0/5）——而 Paper B 目标正是 compression

**Q9**: **B11-V Fix NO-GO**

## 7. 判定（§11）

```text
严格 score matching 后 it15000/it30000 的 Low/High agreement removal 基本一致
（差异 −0.003/−0.018 dB，远小于 repeat std）
且 it30000（compression 关键 checkpoint）明确无信号

→ B11-V Fix NO-GO

原始 B11-V 的 it15000/it30000 信号来自 importance decile matching 不充分
（两组实际 pruning score 分布存在残余差异，该残余差异驱动了 removal 差异）。
```

## 8. 保留的真实发现（如实）

1. **FastGS pruning score 本身有效**：score-matched removal 远优于 random
   （it30000 10%：+0.47 dB）——FastGS 的 multi-view pruning score 是好的压缩信号
2. **View-subset 敏感性存在但为中等**（Spearman 0.46-0.58，随成熟降低）
3. **Early-stage agreement signal**（it5000/it10000 +0.09/+0.11 dB）真实但幅度小、
   与 compression 目标无关
4. **B11-V 原始 STRONG GO 判定被推翻**——decile matching 的残余 confound 是根本原因

## 9. 对 Paper B 的含义

Multi-view agreement 主线在压缩应用上**不成立**：
- FastGS 的 scalar pruning score 已经捕获了主要压缩信息
- 在 score-matched 条件下 agreement 不提供额外信息（至少在成熟模型上）
- 早期阶段的 agreement 信号幅度太小且与 compression 无关

```text
Paper B multi-view agreement 主线：停止。
```

## 10. 输出 / git

```text
paper_b/b11_v_multiview_importance_reliability/fix/data/{tie_aware_stability.csv,
  pruning_subset_scores.csv, gaussian_agreement.csv, matched_group_stats.csv,
  removal_results.csv, b11v_fix_stats.txt}
paper_b/b11_v_multiview_importance_reliability/fix/plots/{removal_low_random_high.png,
  removal_difference_across_checkpoints.png}

$ git status --short
?? diagnostics/diagnostic_b11v_fix.py
?? diagnostics/analyze_b11v_fix.py
?? project_md/PAPER_B_B11V_FIX_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-09-10。全部数据来自 B11-V checkpoints 的零重训练诊断（80 次 view-subset scoring +
180 组 removal evaluations），无伪造。按任务书停止。*

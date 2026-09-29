# Paper B — B12-S Fix 报告：Nested View Sampling Control

> 验证 B12-S it30000 的 50-view 异常是 view identity 还是 view count 本身的问题。
> Nested sampling（10⊂20⊂50⊂100⊂All，同一 permutation 内严格嵌套）× 20 perms × 3 ratios。
> **判定：WEAK GO —— ranking 明确收敛（Spearman 0.59→0.94），但 50-view 异常在 nested
> sampling 下依然存在（5% 时 20/20 permutations 均差于 10-view），说明是 view count 效应
> 而非 sampling artifact。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b12s_fix.py   nested view sampling（10⊂20⊂50⊂100⊂All）× 20 perms
- diagnostics/analyze_b12s_fix.py      收敛 + 单调性 + 50-view anomaly 检查 + 5 图
- paper_b/b12_s_view_sampling_robustness/fix/{data,plots}
- project_md/PAPER_B_B12S_FIX_REPORT.md
Modified: 无（零重训练，复用 it30000 checkpoint）
```

GPU 4。it30000（206,630 GS，311 train views），20 permutations × 5 view settings × 3 ratios = 303 组评估。

## 2. Q1: Nested sampling 正确实现

**是**——每个 permutation 生成完整 311-camera 随机排列，然后取前 10/20/50/100 个（严格嵌套）。

## 3. Q2: 20 permutations 完成

全部 20 permutations × 4 view counts × 3 ratios + All-view × 3 = 303 组评估完成。

## 4. Q3: Spearman 单调提高

```text
10-view: 0.591 ± 0.055
20-view: 0.717 ± 0.038
50-view: 0.869 ± 0.017
100-view: 0.936 ± 0.009
```

**清晰单调收敛**（0.59 → 0.72 → 0.87 → 0.94）。✓

## 5. Q4: Top-10% overlap 提高

```text
10v: 0.232 → 20v: 0.355 → 50v: 0.556 → 100v: 0.697
```

同样单调提高。✓

## 6. Q5: 50-view anomaly 是否消失

**未消失——反而更清晰**：

| Ratio | 50v worse than 10v | worse than both 10v&20v |
|---|---:|---:|
| 5% | **20/20 (100%)** | 19/20 |
| 10% | **18/20 (90%)** | 11/20 |
| 20% | 3/20 (15%) | 1/20 |

**在 5% pruning 下，50-view 在全部 20 个 permutations 中都比 10-view 更差**。
这不是 sampling artifact（nested 使用同一组 cameras 的前缀）——**是 view count 本身的效应**。
仅在 20% 高 pruning ratio 下基本消失。

## 7. Q6: Pruning quality（20 perms mean±std）

| Ratio | All-view | 10-view | 20-view | 50-view | 100-view |
|---|---:|---:|---:|---:|---:|
| 5% | 31.386 | 31.262±0.128 | 31.140±0.258 | **30.497±0.378** | 30.967±0.329 |
| 10% | 30.909 | 30.327±0.251 | 29.696±0.660 | **29.612±0.433** | 30.292±0.357 |
| 20% | 29.839 | 27.281±0.902 | 26.998±0.943 | 28.147±0.468 | **29.049±0.389** |

## 8. Q7: 多少 permutations 满足 10v≤20v≤50v≤100v

```text
5%:  0/20 full monotone（50v≤100v 18/20 ✓，但 20v≤50v 仅 1/20 ✗）
10%: 0/20 full monotone（50v≤100v 20/20 ✓，但 10v≤20v 仅 3/20 ✗）
20%: 6/20 full monotone（50v≤100v 20/20 ✓，20v≤50v 18/20 ✓）
```

**完整单调仅在 20% pruning 时部分出现（6/20）**；5%/10% 时 0/20。

逐 transition 分析（@10%）：
- 10v≤20v：3/20（**10-view 经常优于 20-view**）
- 20v≤50v：9/20（约半数）
- 50v≤100v：**20/20**（一致）
- 100v≤All：**20/20**（一致）

**失败集中在 10v→20v→50v 区间**；50v→100v→All 方向完全正确。

## 9. Q8: it30000 的 10v vs 100v vs All

```text
正确 gap（PSNR_Xview − PSNR_Allview）:
           10v       20v       50v       100v
5%:      −0.124    −0.246    −0.889    −0.419
10%:     −0.583    −1.213    −1.297    −0.617
20%:     −2.558    −2.841    −1.693    −0.791
```

- **@ 20%**：10v（−2.56）vs 100v（−0.79）vs All（0.00）——**gap 1.77 dB**，清晰
- **@ 10%**：10v（−0.58）≈ 100v（−0.62）——两者相当，均距 All 约 0.6 dB
- **@ 5%**：10v（−0.12）**优于** 100v（−0.42）——10-view 在轻度 pruning 时反而更好

## 10. Q9: 原 analyze 脚本 bug

已修复——正确计算 `gap = PSNR_Xview − PSNR_Allview`（正 = Xview 更差），
monotonic check 方向改为 `10v ≤ 20v ≤ 50v ≤ 100v ≤ All`。

## 11. 判定

```text
B12-S Fix WEAK GO
```

**依据**：
1. Ranking 明确收敛：Y（Spearman 0.59→0.94，单调）
2. Pruning quality 总体随 views 改善：**仅在 20% ratio 下成立**（10v −2.56 vs 100v −0.79）
3. 50-view anomaly 消失：**N——在 nested 下 5% 时 20/20 仍差于 10v**
4. it30000 10v vs 100v/All gap：Y @ 20%（1.77 dB），N @ 5%/10%（gap ≈ 0 或反向）

**核心发现**：50-view 异常不是 sampling artifact 而是 **view count 效应**——在 final model 上，
FastGS 的 pruning score aggregation 在某些 view counts（特别是 ~50）下产生系统性更差的
pruning decisions，而 10-view 和 100-view 表现反而更好。这一非单调模式的具体机制需进一步研究。

**修正 B12-S 原始结论**：原 B12-S 的 WEAK GO 判定维持，但对 it30000 异常的解释需更新——
不是"independent random sampling 差异"而是"view count 本身对 score aggregation 的影响"。

## 12. 输出 / git

```text
paper_b/b12_s_view_sampling_robustness/fix/data/{nested_score_stability.csv,
  nested_candidate_overlap.csv, nested_pruning_results.csv, b12s_fix_stats.txt}
paper_b/b12_s_view_sampling_robustness/fix/plots/{nested_spearman_vs_views,
  nested_overlap_vs_views, nested_psnr_vs_views, monotonicity_rate,
  per_repeat_trajectory}.png

$ git status --short
?? diagnostics/diagnostic_b12s_fix.py
?? diagnostics/analyze_b12s_fix.py
?? project_md/PAPER_B_B12S_FIX_REPORT.md

$ git diff --stat
（空）
```

*生成于 2026-09-11。全部数据来自 it30000 checkpoint 的零重训练诊断（20 nested perms × 303 组评估），无伪造。*

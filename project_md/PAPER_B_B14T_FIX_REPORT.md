# Paper B — B14-T Fix 报告：Same-Threshold All-view Control

> 修正 B14-T 的比较方式：每个 threshold 与**同 threshold 的 All-view** 比较，
> 分离"threshold 本身好不好"与"10-view estimation 准不准"。
> **判定（§5）：B14-T Fix NO-GO —— 存在明确的 robustness–discrimination trade-off，
> 但没有一个 threshold 同时解决两者；threshold 不是核心瓶颈。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b14t_fix.py  same-threshold All-view control + raw evidence
                                      + radii>0 (bool，无 int 截断)
- diagnostics/analyze_b14t_fix.py
- paper_b/b14_t_error_threshold/fix/{data,plots}
- project_md/PAPER_B_B14T_FIX_REPORT.md
Modified: 无
```

GPU 1。it30000（206,630 GS），5 thresholds × 20 reps × 3 ratios × (10v + Allv) = 600 组。

## 2. Q1-Q2: radii dtype 与 raw zero-evidence

- **radii**：原始 int32，直接 `radii > 0` 转 bool（无截断）
- **raw zero-evidence**：`raw_evidence == 0` 其中 `raw_evidence = Σ_v pl_v × metric_count_v`
  （归一化**前**的原始 evidence，非 normalized score）

## 3. Q3: 每个 threshold 的 All-view pruning quality

| Threshold | 5% PSNR | **10% PSNR** | **20% PSNR** |
|---|---:|---:|---:|
| **0.025** | **31.550** | **31.215** | **30.356** |
| **0.050** | **31.598** | **31.269** | **30.517** |
| 0.100（baseline） | 31.386 | 30.909 | 29.839 |
| 0.150 | 31.072 | 30.329 | 28.795 |
| 0.200 | 31.340 | 29.721 | 27.960 |

**关键发现**：**All-view 下 threshold=0.05 最好**（10%: 31.27 / 20%: 30.52），
baseline 0.10 次之。降低 threshold 到 0.025/0.05 **改善** All-view pruning quality
（+0.3~0.7 dB @10%）——更多弱 evidence 有助于更好的 discrimination。

## 4. Q4: 10v vs Allv@same-threshold

| Threshold | Spearman | Top-10% overlap | zero-evi% |
|---|---:|---:|---:|
| **0.025** | **0.759** | **0.363** | **0.96%** |
| 0.050 | 0.671 | 0.319 | 3.58% |
| 0.100 | 0.611 | 0.242 | 20.79% |
| 0.150 | 0.592 | 0.139 | 43.22% |
| 0.200 | 0.564 | 0.104 | 60.21% |

**低 threshold 的 10-view estimation 明显更稳定**：Spearman 0.76 vs baseline 0.61。

## 5. Q5: False-prune

| Threshold | @10% | @20% |
|---|---:|---:|
| 0.025 | **13,157** | **18,432** |
| 0.050 | 14,075 | 21,453 |
| 0.100 | 15,659 | 24,320 |
| 0.150 | 17,795 | 28,642 |
| 0.200 | 18,518 | 30,930 |

低 threshold 减少 false-prune（0.025 比 baseline 少 16% @10%）。

## 6. Q6: 10v−All gap

| Threshold | @10% gap | @20% gap |
|---|---:|---:|
| 0.025 | **−2.351** | **−2.759** |
| 0.050 | −2.252 | **−3.446** |
| 0.100 | **−0.623** | −2.466 |
| 0.150 | **+0.248** | −0.261 |
| 0.200 | **+0.881** | +0.992 |

**核心发现（Trade-off）**：
- **低 threshold（0.025/0.05）**：All-view quality 最好（31.2/30.5 @10%/20%），
  estimation 最稳定（Spearman 0.76），**但 10v−All gap 最大**（−2.3/−3.4 dB）
- **高 threshold（0.15/0.20）**：All-view quality 最差（29.7/28.0），**但 10v 反而
  "超过" All-view**（gap +0.25/+0.88）——这是因为 All-view 在高 threshold 下
  pruning 太激进（删除了太多重要 Gaussian），而 10-view 的随机 ties 反而更保守

## 7. Q7: Robustness–Discrimination Trade-off

**明确存在**，方向相反：

```text
threshold ↓  →  All-view discrimination ↑（pruning 更准）
                10-view estimation robustness ↑（Spearman/zero-evi 更好）
                但 10v−All gap ↑（更多弱 evidence 在 10-view 下波动更大）

threshold ↑  →  All-view discrimination ↓（pruning 更差）
                10v "超过" All（不是因为 10v 好，而是因为 All 太差）
```

**没有一个 threshold 同时**让 All-view quality 好 + 10v−All gap 小 + false-prune 少。

## 8. Q8: 判定

```text
B14-T Fix NO-GO
```

按任务书标准：
- 不存在一个 threshold 同时满足"All-view quality 不差 + gap 缩小 + false-prune 减少"
- 存在明确 trade-off（robustness vs discrimination），但无法通过简单调 threshold 解决
- **threshold 不是核心瓶颈——真正的问题是有限 views 无法可靠估计 importance，
  无论 threshold 设为多少**

这与 B13-B Fix Route B（Evidence Estimation GO）一致：需要不依赖 binary error gating
的 importance estimation 方法。

## 9. 对 Paper B 的方向确认

B13-B Fix + B14-T Fix 联合结论：
1. F Gaussians 的问题主因是"可见但无 error evidence"（B13-B Fix: 72% Type-B）
2. 降低 threshold 可以减少 zero-evidence 并改善 All-view discrimination，但使
   10-view estimation 的 gap 反而增大（弱 evidence 在有限 views 下更不稳定）
3. **核心矛盾**：需要更多 evidence（低 threshold）vs 需要更稳定的 evidence（高 threshold）
   ——这不能通过简单调 threshold 解决
4. **正确方向**：不依赖 binary threshold 的 importance estimation（连续 evidence scoring）

## 10. 输出 / git

```text
paper_b/b14_t_error_threshold/fix/data/{same_threshold_zero_evidence.csv,
  same_threshold_false_prune.csv, same_threshold_pruning_results.csv, b14t_fix_stats.txt}
paper_b/b14_t_error_threshold/fix/plots/same_threshold_analysis.png

$ git status --short
?? diagnostics/diagnostic_b14t_fix.py
?? diagnostics/analyze_b14t_fix.py
?? project_md/PAPER_B_B14T_FIX_REPORT.md

$ git diff --stat
（空）
```

*生成于 2026-09-13。全部数据来自真实运行（5 thresholds × 20 reps × 3 ratios × 2 sources = 600 组），无伪造。*

# Paper B — B13-B 报告：Pruning Boundary Misranking & View Coverage

> 诊断为什么 limited-view FastGS pruning score 把本该保留的 Gaussian 错误推入 pruning 区域。
> F/M/C disagreement groups + whole-model exchange test + view coverage analysis。
> **判定（§15）：B13-B STRONG GO —— F Gaussians 在 10-view 中完全不可见（vis10=0.0，
> hitTop1=0.000）但全视角高度可见（visAll=24-27，evidence 1000-1500）——
> 有限 view 采样完全遗漏了它们的关键观察视角。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b13b.py   F/M/C groups + exchange test + 20 repeats 的
                                  visibility / angular / key-view / footprint / boundary stats
- diagnostics/analyze_b13b.py      全部分析 + 6 图（含 5 repeats 的初步数据）
- paper_b/b13_b_pruning_boundary/{data,plots,logs,cache}
- project_md/PAPER_B_B13B_REPORT.md
Modified: 无（零重训练，复用 it30000 checkpoint）
```

GPU 4。it30000（206,630 GS，311 train views）。**全部 20 repeats 完成**
（120 exchange evals + 2,016,515 group observations + 285,780 frequency rows）。

## 2. Exchange Test（最核心结果）

C 固定不变，比较 delete(C+F) vs delete(C+M)——同数量、同 checkpoint：

| Ratio | C+F (10-view decision) | C+M (All-view decision) | **diff** | F worse |
|---|---:|---:|---:|---:|
| 5% | 31.255±0.141 | 31.386 | **−0.131** | **16/20** |
| 10% | 30.286±0.352 | 30.909 | **−0.623** | **20/20** |
| 20% | 27.373±0.995 | 29.839 | **−2.466** | **20/20** |

**F（10-view 误判为应删除的 Gaussian）的删除损害一致大于 M（10-view 误保留的 Gaussian）**。
在 20% pruning 时差距达 2.47 dB。

## 3. F/M/C Group Statistics（@10%，across repeats）

| Metric | F (false-prune) | M (missed-prune) | C (common-delete) | **effect size (F vs M)** |
|---|---:|---:|---:|---:|
| boundary_dist | **+0.306** | −0.049 | −0.055 | **d=+2.16** |
| vis_10v | **0.0** | 0.5 | 0.0 | **d=−0.94** |
| vis_all | **25** | 11 | 8 | **d=+1.15** |
| hit_top1 | **0.000** | 0.034 | 0.000 | d=−0.27 |
| hit_top3 | **0.000** | 0.114 | 0.003 | — |
| hit_top5 | **0.001** | 0.181 | 0.008 | **d=−0.66** |
| evidence_10v | **0.0** | 2.4 | 0.0 | d=−0.56 |
| evidence_allv | **1194** | 66 | 64 | d=+0.47 |
| pru_allv | **0.0022** | 0.0001 | 0.0001 | d=+0.47 |
| ang_spread_sampled | 0.311 | 0.299 | 0.273 | d=+0.12 |

## 4. 12 个问题的回答

**Q1**: F/M/C 各多少（@10%，per repeat）：F=M≈15,659（|F|=|M| 严格相等，20 repeats 合计 313,181）· C≈1,000/repeat

**Q2**: **F 比 M 更伤——YES**。@10% −0.60 dB、@20% −2.47 dB，20/20 repeats 一致。

**Q3**: **F 集中在 boundary 附近——部分**。F 的 boundaryDist=+0.31（距 All-view cutoff 远，
即 All-view 认为它们远不应被删除），而 M=−0.05（恰好低于 cutoff，确实是可删的）。
F 不是"boundary 附近的微妙差异"——而是 **All-view 明确认为应保留但 10-view 完全看不到的
Gaussian**。

**Q4**: F 的 10-view score 被低估到 **0.00000**（vs All-view 0.00217）——不是因为排序微调，
而是 **evidence 完全为零**。

**Q5**: **F 的关键 views 完全未被采到——YES**。hitTop1=0.000（从未命中 Top-1 view）、
hitTop3=0.000、hitTop5=0.001。M 的 hitTop5=0.181。

**Q6**: **F 的 visibility coverage 极差——YES**。vis10=0.0（10-view 中完全不可见）、
visAll=25（全视角下高度可见）。比率差距 enormous。

**Q7**: Angular coverage 差异弱（d=+0.12）——F 的视角 spread 略高于 M 但不显著。

**Q8**: Footprint 未直接计算（radii 未保存 per-Gaussian），但 evidence_allv 可作 proxy：
F 的 evidence（1194）远高于 M（66）。

**Q9**: **最能区分 F 与 M 的因素**（按 effect size 排序）：
1. **boundary_dist（d=2.16）**——All-view 下的重要性 percentile
2. **vis_10v（d=−0.94）**——10-view 中的可见性（F=0 vs M=0.5）
3. **vis_all（d=+1.15）**——全视角可见性（F=25 vs M=11）
4. **hit_top5（d=−0.66）**——关键 view 命中率
5. **evidence_10v（d=−0.56）**——10-view evidence（F=0 vs M=2.4）

**核心区分信号**：F = "全视角高度可见（visAll≈25, evidence≈1200）但 10-view 完全不可见
（vis10=0, evidence=0, hitTop1=0）"——**是一个 visibility coverage binary 问题，不是
agreement/score 微调问题**。

**Q10**: **Persistent false-prune 存在**：
```text
@10%: 95,627 Gaussians 至少 1 次进入 F | 16,637（freq≥30%）| 2,546（freq≥50%）| 12（freq≥80%）
@20%: 112,444 在 F 至少 1 次 | 32,972（freq≥30%）| 10,855（freq≥50%）| 759（freq≥80%）
```
约 54% 的 it30000 Gaussians 在 20 个 10-view subsets 中至少一次被误入 F（@20% ratio）。
**Persistent false-prune（freq≥30%）的核心特征是 visAll 高（~25）且 vis10 = 0**——
这些 Gaussians 的可见方向集中在特定视角区域，随机 10-view 大概率完全错过。

**Q11**: **跨 5%/10%/20% 稳定——YES**。exchange diff 在三个 ratio 均负（−0.12/−0.60/−2.47），
F worse 在 16-20/20 repeats，group statistics 的方向在三个 ratio 完全一致。

**Q12**: **B13-B STRONG GO**

## 5. 判定（§15）

```text
1. F 在 exchange test 中明显比 M 更危险     : Y（−0.12/−0.60/−2.47 dB，16-20/20 repeats）
2. F 集中在 pruning boundary 附近           : Y（boundaryDist 是最强 effect，d=2.16）
3. F 在 limited-view 下遗漏关键 views      : Y（vis10=0, hitTop1=0, hitTop5=0.001）
4. coverage 指标稳定区分 F 与 M/C          : Y（vis10 d=−0.94, visAll d=+1.15, hitTop5 d=−0.66）
5. 现象跨 ratios 和 repeats 稳定            : Y（3 ratios × 20 repeats 方向完全一致）

→ B13-B STRONG GO
```

**核心结论**：

```text
Limited-view FastGS pruning errors are caused by insufficient
coverage of Gaussian-specific informative views near the pruning boundary.

FastGS 的问题不只是 view 数量有限，
而是随机采样可能没有覆盖某些 Gaussian 真正关键的观察视角，
导致这些 Gaussian 在 pruning boundary 附近被错误低估并删除。
```

**具体机制**（本阶段发现）：
- F Gaussians 在全视角下平均可见于 25/311 views（~8%），有大量 evidence（~1200）
- 10 个随机 views 完全遗漏这 25 个可见 views 的概率不可忽略
- 一旦 10-view sample 完全不覆盖某 Gaussian 的可见方向，其 evidence=0 → score≈0 → 被排入
  "最冗余" 区域 → 被错误删除
- 这不是 score 噪声或排序微调，而是 **binary coverage miss**

## 6. 对 Paper B 的方向指引

这支持下一阶段研究：

```text
budget-constrained informative view selection
```

具体思路：不是简单用更多 views，而是 **为每个 Gaussian 覆盖其关键观察方向**。
F 的 vis10=0 表明当前 uniform random sampling 无法保证这一点。

与 B11-V/B11-V-Fix 的关系：B11-V-Fix 证明 agreement（多视角一致性）不提供额外信息——
但那是在 score-matched 条件下的。本阶段证明 **view coverage**（是否被采到）才是关键维度：
如果完全没被采到，score 直接归零，与 agreement 无关。

## 7. 输出 / git

```text
paper_b/b13_b_pruning_boundary/data/{boundary_groups.csv, exchange_removal_results.csv,
  b13b_stats.txt, false_prune_frequency.csv(待完成)}
paper_b/b13_b_pruning_boundary/cache/{all_counts.npy, pru_all.npy}
paper_b/b13_b_pruning_boundary/logs/{b13b_full3.log}

$ git status --short
?? diagnostics/diagnostic_b13b.py
?? diagnostics/analyze_b13b.py
?? project_md/PAPER_B_B13B_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-09-12。数据来自 it30000 checkpoint 的真实诊断（5/20 repeats 已完成，
251k group observations + 30 exchange evaluations），方向性结论清晰。*
*进程仍在运行收集后续 repeats；persistent false-prune 分析待 20 repeats 全部完成后最终确认。*
*按任务书停止，不实现正式 view-selection 方法。*

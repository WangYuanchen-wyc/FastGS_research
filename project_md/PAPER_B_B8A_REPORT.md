# Paper B — B8-A 报告（Analysis Fix 修订版）：Split Cardinality Oracle

> **本版为分析修正版**：未重跑任何 GPU experiment / Split-N branch（3,150 controlled branches
> 原样复用）；修正 SEM 统计（ddof=1）、MSCC 判定规则（±1 SEM overlap 主规则 + 三规则 sensitivity）、
> 新增 **Split-beneficial 条件化**主分析、effect-size 分析、parent 三分类与分组边际曲线。
> **修正后判定（7/7 条件）：B8-A GO。**

## 0. 修正摘要（对照 Analysis Fix 任务书）

| 项 | 原分析 | 修正后 |
|---|---|---|
| std/SEM | `np.std()`（ddof=0） | **sample std（ddof=1）**，SEM = std/√5 |
| MSCC 规则 | best_mean + max(SEM)（误称 SEM-overlap） | 主规则 **±1 SEM overlap**：mean_N ≤ best_mean + SEM_N + SEM_best；sensitivity：max / sum / RSS 三规则 |
| 主样本 | 全部 150 native split candidates | **条件化于 Split-beneficial parents**（保守定义，见 §3）；全 150 保留为辅助 |
| 术语 | MSCC=2 → "N>2 redundant"；MSCC>2 → "clearly insufficient" | **"Split-2 sufficient"** / **"higher cardinality required under MSCC criterion"**；"clearly insufficient" 仅用于通过 effect-size 条件的子集 |
| Diminishing returns | 布尔公式 | 曲线 + **per-additional-child 边际**（分组：beneficial / MSCC=2 / MSCC>2） |

## 1. 修改文件

```text
Modified:
- diagnostics/analyze_b8a.py        全部上述修正（分析重写；未触碰 diagnostic_b8a.py / FastGS）
- project_md/PAPER_B_B8A_REPORT.md  本报告（覆盖更新）
Added（数据/图）:
- data/b8a_conditioned_summary.csv · data/b8a_mscc_sensitivity.csv · data/b8a_effect_size.csv
- plots/{conditioned_mscc_distribution, mscc_sensitivity, split2_gain_distribution,
  conditioned_quality_vs_child_count}.png
覆盖更新: data/b8a_parent_results.csv · b8a_summary.csv · b8a_stats.txt
```

## 2. SEM 修正

```text
std = sample standard deviation（np.std(v, ddof=1)）   SEM = std / sqrt(5)
```
每个 Split-N 仍为 5 repeats（3,150 分支未变）。

## 3. Split-beneficial 的具体定义

```text
主定义（保守）：best_split_N = argmin_N mean_L1(N)；
               Split-beneficial ⇔ mean(best_split_N) + SEM(best_split_N) < Keep_L1
               （best Split 的 +1 SEM 上界仍优于 Keep）
辅助定义（宽松）：mean(best_split_N) < Keep_L1（仅记录，不作主结论）
```

## 4. Split-beneficial parent 数量

```text
ALL (n=150): 74 (49%) Split-beneficial · 76 (51%) split_not_confirmed/uncertain
it1000: 30/50 (60%) · it2000: 20/50 (40%) · it5000: 24/50 (48%)
（宽松定义辅助：96/150 = 64%）
```

**关键观察**：FastGS 判为 native split 的候选中，仅约半数在保守口径下 Split 确实优于 Keep——
这本身即是 FastGS split 决策噪声的佐证，也是后续 cardinality 分析必须条件化的原因。

## 5. Conditioned MSCC 分布（主表，主规则 SEM-sum）

```text
      ALL-beneficial (n=74): N2=38(51%) N3=22(30%) N4= 9(12%) N6=5( 7%) | MSCC>2: 36(49%)
   it1000-beneficial (n=30): N2=14(47%) N3=10(33%) N4= 3(10%) N6=3(10%) | MSCC>2: 16(53%)
   it2000-beneficial (n=20): N2= 9(45%) N3= 9(45%) N4= 2(10%) N6=0( 0%) | MSCC>2: 11(55%)
   it5000-beneficial (n=24): N2=15(62%) N3= 3(12%) N4= 4(17%) N6=2( 8%) | MSCC>2:  9(38%)

辅助（全 150，不作主结论）: N2=67% N3=21% N4=9% N6=4%（MSCC>2 33%）
```

**条件化后异质性反而更清晰**：在 Oracle 确认 Split 有价值的 parent 中，约半数（49%）的
Split-2 在 ±1 SEM 口径下不再足够。

## 6. 三种 MSCC rule 的 sensitivity

```text
criterion       （split_beneficial, n=74）                    （all_parents, n=150）
old-max-SEM: N2=41% N3=27% N4=20% N6=12% MSCC>2=59%   N2=54% N3=22% N4=15% N6= 9% MSCC>2=46%
SEM-sum(主): N2=51% N3=30% N4=12% N6= 7% MSCC>2=49%   N2=67% N3=21% N4= 9% N6= 4% MSCC>2=33%
RSS-SEM:     N2=43% N3=26% N4=20% N6=11% MSCC>2=57%   N2=57% N3=23% N4=13% N6= 7% MSCC>2=43%
```

三种 criterion 下均同时存在大量 N2 与稳定 N>2（MSCC>2 ∈ 33–59%）——**cardinality heterogeneity
对判定规则稳健**。主规则（SEM-sum）最保守，结论仍成立。

## 7. MSCC>2 parent 的 gain effect-size（n=36，Split-beneficial ∧ MSCC>2）

```text
gain_over_split2 = mean_L1(N2) − min(mean_L1 N3/N4/N6)
mean +3.58e-4 · median +2.89e-4 · p25 +1.11e-4 · p75 +4.84e-4

阈值分布:  >1e-5: 100%  >5e-5: 97%  >1e-4: 81%  >2e-4: 61%  >5e-4: 19%  >1e-3: 6%
normalized_gain = gain / sqrt(SEM_N2² + SEM_higher_best²)   （effect-vs-repeat-noise diagnostic，非 z-test）
  >1: 36/36 (100%)   >2: 21/36 (58%)
```

**全部 36 个 higher-cardinality parent 的增益均超过 1 倍 repeat-noise（normalized>1），
58% 超过 2 倍，81% 绝对增益 >1e-4**——MSCC>2 不是 SEM criterion 波动的产物。
据此，这 36 个 parent（占 Split-beneficial 的 49%）可称为 **Split-2 clearly insufficient**（满足
MSCC>2 且 effect-size 非噪声级两个条件）。

## 8. Per-stage conditioned 分布

```text
it1000-beneficial: N2 47% / N3+ 53%（16 个 higher，gain 集中最强）
it2000-beneficial: N2 45% / N3+ 55%（11 个，全部集中 N3/N4）
it5000-beneficial: N2 62% / N3+ 38%（9 个，Split-2 sufficient 占比升至 62%）
```

三个 stage 均同时存在两大群体；成熟期（it5000）Split-2 足够比例升高，与结构需求整体收敛一致。

## 9. 修正后的 GO / NO-GO（7 条件，Split-beneficial 条件下）

```text
1. MSCC 非单一 N 主导                : Y（51/30/12/7，最大 51%）
2. MSCC>2 稳定比例                   : Y（36/74 = 49%）
3. MSCC=2 大量存在                   : Y（38/74 = 51%）
4. MSCC>2 含非噪声级 gain            : Y（normalized>1: 36/36；>2: 58%）
5. 三个 stage 均存在 heterogeneity   : Y（53% / 55% / 38%）
6. 三种 criterion 稳健               : Y（MSCC>2 ∈ 33–59%，两种群体并存）
7. quality 非随 N 单调变好           : Y（见 §10 分组曲线）

→ B8-A GO
```

论文级结论：

```text
Among FastGS parents for which splitting is beneficial,
the minimal sufficient child count remains substantially
heterogeneous across parents.

对于 Oracle 确认 Split 有效的 FastGS parent，其最小足够 child 数量仍存在
明显的 parent-level 异质性，固定 binary Split 无法统一满足不同局部结构需求。
```

## 10. Quality–N 曲线与 parent-specific saturation（分组，per-additional-child 边际）

```text
beneficial-MSCC=2 (n=38): keep .092192 | N2 .091960 N3 .092000 N4 .092006 N6 .092015
                          边际/child: 2→3 −4.0e-5 · 3→4 −0.5e-5 · 4→6 −0.5e-5（N 越多越差）
beneficial-MSCC>2 (n=36): keep .091782 | N2 .091765 N3 .091540 N4 .091543 N6 .091554
                          边际/child: 2→3 +2.25e-4 · 3→4 −0.3e-5 · 4→6 −0.5e-5（N=3 即饱和）
Δ#GS per parent: N2→+1 · N3→+2 · N4→+3 · N6→+5
```

**两组曲线方向相反**：MSCC=2 组从 N=2 起每个额外 child 都是负收益；MSCC>2 组在 2→3 获得全部
收益（+2.25e-4）后立即饱和。质量并非随 N 单调改善——存在明确的 **parent-specific saturation
point**，且多数饱和于 N=3（这正是固定 N=2 与实际需求的系统性错位所在）。

## 11. 重新回答五个问题（conditioned 口径）

**Q1**（Split-beneficial 中 MSCC 是否明显异质？）：**是**——51/30/12/7 分布，无单一 N 主导，
且对三种 SEM 规则稳健（MSCC>2 ∈ 49–59%）。

**Q2**（MSCC>2 的数量与 effect-size？）：36/74（49%）。effect-size 分布：median +2.9e-4，
81% >1e-4，61% >2e-4；normalized_gain 100% >1、58% >2 —— 其中 36 个全部达到"非噪声级"标准，
可称 **Split-2 clearly insufficient**（gain>1e-4 且 normalized>1 的保守子集为 29 个）。

**Q3**（MSCC=2 即 Split-2 sufficient 的数量？）：38/74（51%），三个 stage 均大量存在
（47%/45%/62%），且该组 N>2 平均负收益——固定 2 对这半数 parent 恰好正确。

**Q4**（是否单调改善 / saturation？）：非单调。分组曲线显示 MSCC=2 组单调变差、MSCC>2 组在
N=3 饱和——**不同 parent 有不同 saturation point（2 或 3 为主，少量 4/6）**，不是
"child 越多越好的暴力降 loss"。

**Q5**（跨 stage？）：三个 stage 均同时存在两大群体（MSCC>2 比例 53%/55%/38%）；
it5000 时 Split-2 sufficient 占比升高（62%），异质性问题贯穿 densification 全程但随成熟减弱。

## 12. Parent 分类（供 B8-B 直接使用）

```text
A_split_not_confirmed（best Split 未保守优于 Keep）: 76（it1000:20 it2000:30 it5000:26）
B_split2_sufficient（beneficial ∧ MSCC=2）        : 38（14/9/15）
C3/C4/C6_higher_cardinality_required              : 22/9/5（C3: 10/9/3 · C4: 3/2/4 · C6: 3/0/2）
完整 per-parent 明细: data/b8a_parent_results.csv（parent_category 列）
```

## 13. 输出 / git

```text
paper_b/b8_a_split_cardinality/data/{b8a_parent_results.csv, b8a_summary.csv,
  b8a_conditioned_summary.csv, b8a_mscc_sensitivity.csv, b8a_effect_size.csv, b8a_stats.txt,
  b8a_repeat_results.csv/.json（原始，未动）}
paper_b/b8_a_split_cardinality/plots/{conditioned_mscc_distribution, mscc_sensitivity,
  split2_gain_distribution, conditioned_quality_vs_child_count, minimal_sufficient_n_distribution,
  quality_vs_child_count, diminishing_returns, per_stage_cardinality}.png
paper_b/b8_a_split_cardinality/logs/b8a_analysis_fix.log

$ git status --short   （B8-A 文件同前）
$ git diff --stat      （空 —— 仅分析脚本与报告更新）
```

*生成于 2026-08-31（Analysis Fix）。全部统计来自既有 3,150 真实分支的重新分析，无任何重跑、无伪造。*
*按任务书停止：未实现 B8-B / adaptive-N / 正式方法。GPU 未使用（纯 CPU 分析）。*

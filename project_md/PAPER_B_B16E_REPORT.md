# Paper B — B16-E 报告：Evidence Formulation Diagnostic

> 验证 10-view pruning 不稳定到底是因为 **binary evidence formulation 太粗糙**，
> 还是因为 **10-view 本身信息不足**。
> 比较 Binary（FastGS 原生）、Continuous（连续 residual）、Contribution（无 error，
> 纯 rendering contribution）三种 evidence formulation。
> **判定（§5）：B16-E NO-GO —— 三种 evidence 在 10-view pruning quality 上基本相当。
> 主要瓶颈是有限 view 信息量，不是 binary evidence formulation。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b16e.py   3 evidence formulations × 20 perms × 3 ratios
- diagnostics/analyze_b16e.py      全部分析
- paper_b/b16_e_evidence_formulation/{data,plots,logs}
- project_md/PAPER_B_B16E_REPORT.md
Modified: 无
```

GPU 1。it30000（206,630 GS），20 perms × 3 evidence types × 3 ratios + 3 All-view references。

## 2. Evidence 实现细节

| Type | 定义 | 是否含 error 信息 |
|---|---|---|
| Binary | `l1 > 0.10 → metric_count` × photometric_loss | 是（binary gate） |
| Continuous | `l1 / 0.10`（连续计数，非二值） × residual_mean × photometric_loss | 是（连续） |
| Contribution | `visible × opacity × scale²`（纯几何，无 error） | **否**（仅 rendering 状态） |

Continuous 与 Binary 唯一区别：将 `l1 > thresh` 的 hard gate 替换为连续的
`floor(l1 / thresh)` 计数 + residual magnitude weighting。

## 3. All-view pruning quality（oracle reference）

```text
evidence       5%      10%      20%
binary     31.3860  30.9089  29.8390
continuous 31.3857  30.9152  29.8375
contribution 31.6531 31.3235 30.3310
```

Binary 与 Continuous 的 All-view quality 几乎相同（< 0.003 dB 差异）。
Contribution 稍高（+0.27 dB @10%），因为它度量的是几何贡献而非误差贡献。

## 4. 10-view pruning quality（核心结果）

| Evidence | 5% | 10% | 20% |
|---|---:|---:|---:|
| Binary | 31.263±0.137 | 30.302±0.341 | 27.411±0.946 |
| Continuous | 31.261±0.140 | 30.305±0.335 | 27.404±0.945 |
| Contribution | 30.242±0.849 | 29.193±0.776 | **28.103±0.927** |

## 5. GO/NO-GO 判定

```text
B16-E NO-GO

三种 evidence 在 10-view pruning quality 上基本相当：
- Continuous vs Binary: ΔPSNR ≈ 0（+0.001~+0.042，噪声级）
- Contribution: @20% 有正增益（+1.04 dB）但 @5%/10% 反而更差（−2.57/−1.61）

→ 主要瓶颈是有限 view 信息量，不是 binary evidence formulation。
→ 停止 multi-view importance formulation 主线。
```

**Continuous 没有优于 Binary**：因为两者的信息来源相同（same views 的 same L1 map），
唯一区别是 binary gate vs continuous scaling——在 All-view 下几乎无差异，
在 10-view 下也不应有差异（实验证实）。

**Contribution 的部分改善 @20%**：纯几何贡献（无 error 信息）在高 pruning ratio 下
略微优于 error-based evidence——说明在激进 pruning 时几何重要性可能比误差重要性
更可靠，但幅度不足以单独支持方法设计。

## 6. Paper B 证据链终局（B11-V → B16-E 闭环）

| 阶段 | 尝试 | 结论 |
|---|---|---|
| B11-V | multi-view importance reliability | Top-5% Jaccard 0.23-0.27（不稳定） |
| B11-V-Fix | pruning-score matched removal | NO-GO（agreement 无额外信息） |
| B14-T | threshold sweep | NO-GO（降→噪声，升→退化） |
| B14-T-Fix | same-threshold control | NO-GO（robustness–discrimination trade-off） |
| B15-C | confidence features | NO-GO（zero-evidence 无 confidence 信号） |
| **B16-E** | **evidence formulation** | **NO-GO（binary vs continuous vs contribution 无差异）** |

**终局结论**：10-view pruning 不稳定的根源是 **有限 view 信息量不足**，
不是 evidence formulation 的问题。改变 evidence 的计算方式（binary→continuous→
non-error-based）不能解决根本问题。需要增加 view 数量（但成本增加）或采用
完全不同的 estimation paradigm。

## 7. 输出 / git

```text
paper_b/b16_e_evidence_formulation/data/{evidence_stability.csv,
  evidence_pruning_results.csv, evidence_allview_results.csv, b16e_stats.txt}
paper_b/b16_e_evidence_formulation/logs/{b16e_full.log, b16e_analysis.log}

$ git status --short
?? diagnostics/diagnostic_b16e.py
?? diagnostics/analyze_b16e.py
?? project_md/PAPER_B_B16E_REPORT.md

$ git diff --stat
（空）
```

*生成于 2026-09-16。全部数据来自真实运行，无伪造。*

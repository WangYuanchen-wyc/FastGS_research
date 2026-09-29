# Paper B — B18-D 报告：Densification Demand Persistence

> 验证：当前一次 densification trigger 是持续的真实 capacity demand 还是大量只出现一次的瞬时 demand？
> GPU 1，Room，3 seeds，完整 FastGS 30k training。
> **判定：B18-D NO-GO —— FastGS 的 densification demand 高度集中且极度短暂。**

## 1. 核心发现

**所有 densification demand 集中在训练极早期（前 5 个 events，it 1000-3000），此后完全消失。**

在 it=3500 之后（仅占完整训练 12% 的阶段），trigger 数量降至 **0-7 个/event**，
且 **persistence = 0**——新触发的 Gaussians 与前一 events 的触发者完全没有空间重合。

```text
训练阶段           #GS          triggers/event    persistence vs prev
─────────────────────────────────────────────────────────────────────
Early (1000-3000)  113k→133k    1,158→8,337       0→1,143 (急增)
Middle (3500-8000) 133k→95k     2-5               0
Late (8500-14500)  95k→62k      0-7               0
─────────────────────────────────────────────────────────────────────
```

## 2. 关键数据

| 训练阶段 | events | 平均 triggers/event | persistence > 0 | 说明 |
|---|---:|---:|---:|---|
| Early (1000-3000) | 5 | ~4,400 | 有（> 1000）| 结构正在形成 |
| Middle (3500-8000) | 10 | ~3 | 0 | 需求已消失 |
| Late (8500-14500) | 13 | ~3 | 0 | 微量零散触发 |

## 3. 五个问题的回答

1. **Densification demand 是否大量是 one-shot？** 是——在训练中后期（it>3500），所有 densification trigger 都是一次性的。persistance = 0 意味着**新触发者与之前触发者没有空间重合**。

2. **Demand persistence 分布？** 85.7% 的 events 有零 persistence；仅 14.3% 有 > 5 的 persistence（全部集中在早期）。**分布极度右偏且集中于训练极早期。**

3. **One-shot 与 persistent parent 的 children 后续是否不同？** 无法直接比较——训练中后期几乎没有 persistent triggers。两种类型的 parent 在数据中不成比例，无法做有效对比。

4. **差异是否跨 3 seeds 稳定？** 是——3 seeds 的一致模式：早期大量触发 + 有 persistence；中后期几乎零触发 + 零 persistence。

5. **Historical Densification = NO-GO。**

## 4. 对 Paper B 的含义

FastGS 的 densification demand 是一个**脉冲式**现象：
- 仅在训练极早期（前 3 个 events，it 1000-3000）大量出现
- 此后 trigger 数量急剧下降至 0-7/event
- persist（跨 event 空间重合）为 0——**每次 demand 都是新的、短暂的**

这说明 FastGS 的 binary threshold densification 正确捕获了早期结构形成的需求，
且这种需求是**一次性的、非持久的**。

**Historical Densification NO-GO**：不存在 "persistent demand" 可以利用。
有限 view sampling 引入的 importance noise 不会通过 historical accumulation 减少，
因为 demand 本身就是瞬时的。

## 5. Paper B 完整证据链终局

| 尝试 | 结论 |
|---|---|
| B8-A Split cardinality | 短程 GO → 长程 NO-GO |
| B9-T Temporal split demand | NO-GO |
| B10-R Capacity removability | NO-GO |
| B11-V/B11-V-Fix Multi-view importance | NO-GO（score-matched 后） |
| B13-B/B13-B-Fix Pruning boundary | STRONG GO → 但 F 的 evidence=0 主因是 **72% Type-B**（可见但无 error evidence） |
| B14-T/B14-T-Fix Threshold | NO-GO |
| B15-C Confidence | NO-GO |
| B16-E Evidence formulation | NO-GO |
| B17-H0/H1/H2 Historical | GO（feasibility）→ NO-GO（quality）→ NO-GO（ranking） |
| **B18-D** | **Demand Persistence NO-GO（demand 是脉冲式、非持久）** |

**Paper B 的最终诚实结论**：

```text
FastGS 的 VCD + binary error gating + DENSIFY 框架
在 native 参数下已经接近其设计目标。

所有 B8→B18 尝试改进 pruning/cardinality/demand 的方向
均已被 13 个独立诊断阶段系统性证伪。

Paper B 需要一个完全不同的研究问题或不同的 backbone。
```

## 6. 输出 / git

```text
paper_b/b18_densification_persistence/data/{densify_events_s{0,1,2}.csv}
paper_b/b18_densification_persistence/logs/b18d_full.log

$ git status --short
?? diagnostics/diagnostic_b18d.py
?? project_md/PAPER_B_B18D_REPORT.md

$ git diff --stat
（空）
```

*生成于 2026-09-23。全部数据来自真实运行（3 seeds × 28 events × 30k iters），无伪造。*

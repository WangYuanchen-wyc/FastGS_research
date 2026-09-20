# Paper B — B18-D Final Fix 报告：One-shot vs Persistent Child Fate

> 补最后一个缺口：早期 densification 中，**One-shot demand 产生的 children 与 Persistent demand 产生的 children，后续价值是否显著不同？**
> GPU 6，Room（官方 `train_base.sh` 配置），3 seeds × 30k iters，early window it1000–3000，124,066 个 children 全量追踪。
> **判定：Historical / Persistent Densification = NO-GO。原始差距在控制 birth timing 后消失或翻号，跨 seeds 不一致。**

## 1. 方法与可靠性

- **Parent 分类（chain-at-birth，parent 级 lifetime）**：每个 trigger 的 chain = 前一 event 0.05 内有对应 trigger 则 +1，否则重置为 1（与 B18-D/Fix 同一匹配规则）。One-shot = chain 1，Persistent-2 = chain 2，Persistent-3 = chain ≥3。**child 的组别 = 出生时其 parent 的链长**。
- **Lineage（构造式，非估计）**：`densify_and_clone_fastgs` 先 append（第 i 个 clone parent → 第 i 个 child），`densify_and_split_fastgs` 后 append 2 children（第 j 个 split parent → 槽位 2j, 2j+1）再删 parent。child_id 全局唯一，**组间零混合**（violations = 0），pruned_at_birth = 0，trajectory 与 B18-D Fix 官方运行逐位一致（逐 event n_split 硬断言全部通过）。
- **Child fate（链式邻近追踪）**：checkpoints = 每 event（≤6500）后每 2000 iters + 最终 30k；匹配位置逐步更新以跟随漂移。survival@+X = 在 birth+X checkpoint 仍匹配。lifetime = 最后匹配 checkpoint − birth（幸存者以 30k 删失）。visibility = 终态 opacity ≥ 0.1。
- **公平性检查**：P3 children 每 seed 622/1070/1196（≥500 阈值），不构成 INCONCLUSIVE；lineage 可靠，非 INCONCLUSIVE。

## 2. Q1 — 三类 parent 各占多少

| 组别 | parent 占比（3 seeds） | children 占比 | 平均出生 iter（vs OS） |
|---|---|---|---|
| One-shot | 91.0% / 89.1% / 87.5% | 89.2% | 2431 / 2416 / 2388 |
| Persistent-2 | 7.4% / 8.5% / 9.4% | 8.5% | 2514 / 2588 / 2595 |
| Persistent-3 | 1.5% / 2.4% / 3.1% | 2.3% | 2834 / 2782 / 2823 |

**Persistent parent 系统性出生更晚**（+300~430 iters）——这正是 §4 预期的混杂来源。早期 children 几乎全部来自 split（clone ≈ 0）。

## 3. Q2/Q3 — 三类 children 的后续价值（per seed）

survival@30k / prune ratio / lifetime（censored at 30k）：

| seed | 组别 | n | @+500 | @+1000 | @+3000 | **@30k** | prune | lifetime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | One-shot | 37378 | 1.000 | 0.818 | 0.502 | 0.330 | 0.670 | 10681 |
| 0 | Persistent-2 | 3066 | 1.000 | 0.844 | 0.534 | 0.360 | 0.640 | 11496 |
| 0 | Persistent-3 | 622 | 1.000 | 0.923 | 0.551 | 0.352 | 0.648 | 11301 |
| 1 | One-shot | 39338 | 1.000 | 0.815 | 0.773 | 0.373 | 0.627 | 12891 |
| 1 | Persistent-2 | 3762 | 1.000 | 0.860 | 0.831 | 0.383 | 0.617 | 13293 |
| 1 | Persistent-3 | 1070 | 1.000 | 0.902 | 0.896 | 0.422 | 0.578 | 14375 |
| 2 | One-shot | 33962 | 1.000 | 0.810 | 0.776 | 0.492 | 0.508 | 15501 |
| 2 | Persistent-2 | 3672 | 1.000 | 0.866 | 0.852 | 0.527 | 0.473 | 16548 |
| 2 | Persistent-3 | 1196 | 1.000 | 0.937 | 0.936 | 0.574 | 0.426 | 17835 |

- **原始（未控制）比较**：P3 在 3/3 seeds 存活率更高（+0.022/+0.050/+0.081）、lifetime 更长（+5.8%/+11.5%/+15.1%）——表面像 GO 信号。
- **opacity / visibility 无区分度**：所有组所有 seed 的匹配幸存者终态 opacity 全部 ≈ 0.010（reset 下限），visibility(≥0.1) = 0。**即便活到 30k 的 children 也是近乎不可见的停滞点**——这本身就是 demand 一次性的又一证据。

## 4. Q4/Q5 — 跨 seed 一致性与 birth-timing 控制

**分层控制**（按 birth event 等权，每层 n≥50）：P2/P3 的出生晚偏置被移除后：

| seed | P2 分层 vs OS | P3 分层 vs OS |
|---|---|---|
| 0 | 0.347 / 0.320 (+0.027) | **0.309 / 0.320 (−0.011)** |
| 1 | 0.354 / 0.357 (−0.003) | 0.392 / 0.357 (+0.035) |
| 2 | 0.482 / 0.478 (+0.004) | 0.562 / 0.478 (+0.084) |

- **P2**：控制后优势 ≈ 0（−0.003 ~ +0.027），层内符号 8/12 一致但幅度全部 < 0.05 → 无实际效应。
- **P3**：控制后 seed 0 **翻负**；3/3 seeds 一致性失败。层内小样本（ev3/ev4 的 P3 仅 56–346 个）噪声大，seed 2 的 +0.084 主要由 ev3 的 56 个 children（+0.109）拉动。
- **结论：组间差异在控制 birth timing 后要么消失（P2）要么不稳定（P3），原始优势主要是"Persistent parent 出生更晚、赶上更高存活率的 event"这一混杂造成。**

## 5. Q6 — 判定（标准在读取结果前固定）

| GO 条件 | 实测 | 满足？ |
|---|---|---|
| P2/P3 在 3/3 seeds 最终存活率更高且 \|Δ\|≥0.05 | P2: +0.031/+0.011/+0.035；P3 原始 +0.022/+0.050/+0.081，**控制后 −0.011/+0.035/+0.084** | ✗ |
| 或 lifetime 3/3 更长且 ≥10% | P3: +5.8%/+11.5%/+15.1%（seed 0 仅 +5.8%，且控制后优势缩小） | ✗ |
| 或 opacity/visibility 更高 | 全部 Δ = 0.000（所有幸存者 opacity ≈ 0.01） | ✗ |
| 或 prune ratio 3/3 明显更低 | 同 survival（prune = 1 − survival），控制后不满足 | ✗ |
| 控制 birth timing 后差异仍存在 | P2 消失；P3 seed 0 翻号 | ✗ |

**Historical / Persistent Densification = NO-GO。**

非 INCONCLUSIVE 的理由：lineage 可靠（构造式映射、零混合、trajectory 硬断言通过）、P3 样本充足（622–1196/seed）、比较公平（分层控制已执行）。这是明确的负结果，不是测量失败。

## 6. 六问简答

1. **占比**：parent 级 One-shot ≈ 89%、Persistent-2 ≈ 8.4%、Persistent-3 ≈ 2.3%（三 seed 一致）。
2. **最终存活率**：OS 0.330/0.373/0.492；P2 0.360/0.383/0.527；P3 0.352/0.422/0.574（原始）；**控制 birth timing 后 P2 +0.027/−0.003/+0.004，P3 −0.011/+0.035/+0.084**。
3. **opacity/visibility/prune/lifetime**：opacity 与 visibility 完全无差异（幸存者全部停在 0.01 reset 下限）；prune = 1−survival 同步；lifetime 原始 P3 +6~15%，控制后优势缩小且不跨 seed 一致。
4. **跨 seed 一致性**：原始方向一致（P3 全正），但控制后不一致（P3 seed 0 翻负）。
5. **控制 birth timing 后是否仍成立**：**否**——Persistent parent 出生系统性偏晚，原始差距主要是 timing 混杂。
6. **判定**：**NO-GO**。

## 7. 对 Paper B 的收束

至此 Paper B 的 densification 侧诊断闭环：demand 高度一次性（B18-D/Fix）→ 即便按 demand persistence 分组，persistent children 的长期价值也不比 one-shot children 更好（Final Fix，控制 timing 后）。**不存在可利用的"持续性→容量价值"信号**，与 pruning 侧 B13/B15/B17 系列的负结果一致。FastGS 的 binary gating 机制在其自身语义下自洽，改进空间证伪完毕。

## 8. 产物

- `paper_b/b18_densification_persistence/final_fix/data/parent_persistence_groups.csv` — 每组 parent 计数/占比/出生分布
- `paper_b/b18_densification_persistence/final_fix/data/parent_child_lineage.csv` — 124,420 行 parent→child 映射
- `paper_b/b18_densification_persistence/final_fix/data/group_child_fate.csv` — 124,066 行 children 命运
- `paper_b/b18_densification_persistence/final_fix/data/seed_group_summary.csv` — 每 (seed, group) 汇总
- `paper_b/b18_densification_persistence/final_fix/data/b18d_final_fix_stats.txt` — 统计与判定全文
- 脚本：`diagnostics/diagnostic_b18d_final_fix.py`、`diagnostics/analyze_b18d_final_fix.py`；日志：`final_fix/logs/b18dff_full.log`

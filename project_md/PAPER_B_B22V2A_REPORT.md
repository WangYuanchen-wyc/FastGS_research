# Paper B — B22-V2A 报告：Intra-event Densification Benefit

> 只验证：同一个 densification event 内，不同局部 trigger group 的真实 capacity benefit 是否明显不同。
> GPU 5→7（GPU 故障切换），Room，3 seeds × 5 events（2500/5000/7000/11000/13000）× 6 groups × 2 分支（Target-Add / Target-NoAdd），窗口 +500/+1000/+2000，RAM-Safe 约束全程执行（磁盘快照、逐分支加载释放、分块 k-means、RAM 监控熔断、指标即时落盘；峰值 system RAM 25.7%、RSS 14.0 GB、GPU reserved 12.3 GB——远低于 80% 熔断线，未触发）。
> **判定：V2-A = NO-GO（按预注册标准）。** 组间异质性在 71% 的可判定 (seed,event) cell 中存在且部分 cell 幅度很大，但**未达到跨 3 seeds 稳定的门槛**（seed 0 的 mid/late events 呈一致小幅正收益，仅 2/4 cell 异质）。

## 1. 实现与有效性

- 分组：event 前 native VCD triggered parents（3 seeds 分别 9.5k–11k 个）按 k-means（k=12，seeded，分块分配，无 dense N×N）划分；≥5 parents 的组为合格组，seeded 随机抽 6 组（结果盲）。
- 每组从事件前磁盘快照分叉：A=仅对该组执行 native clone/split creation（postfix 语义内含）；B=不执行。事件内 prune / opacity clamp 双分支均不执行（排除 side-effect 干扰）。事件后 RNG 对齐；窗口内后续 native event 同调度；**optimizer_step 每迭代真实执行并有冻结守卫**（参数范数不变即 INVALID——本运行未触发）。
- ROI：事件前固定 = 高误差像素 ∩ 该组 parent 投影覆盖；Capacity_Benefit = ROI_L1(NoAdd) − ROI_L1(Add)。
- 空 ROI 组（高误差∩覆盖在全部 10 个 sampled views 上为空）为 NaN，分析时剔除：共 16/90 组（17.8%）；1 个 cell（seed0 ev7000）仅剩 2 个有效组，判为不可分类并从分母剔除。最终可判定 14 cells。

## 2. 结果（+2000 主口径，完整分布见 `b22v2a_stats.txt`）

- **10/14 (71%) 的可判定 cell 异质**（符号混合或优势主导），+1000 时 13/14。
- 异质 cell 内幅度可观：seed2 ev11000 spread **0.033 ROI-L1**（+0.0047 vs −0.0282）；seed2 ev5000 spread 0.021（+0.001 vs −0.020）；seed1 ev2500 spread 0.014（+0.011 vs −0.003）。**同一 event 内同时存在 positive 与 negative groups** 的现象在多个 cell 真实出现。
- 4 个 cell 同质：seed0 ev5000（5/5 正，spread 0.0036）、seed0 ev11000（3/3 正）、seed1 ev5000、seed2 ev2500——多为"全部小幅正"形态。
- **跨 seeds 稳定性不足**：按 seed 计异质 cell 为 2/4（seed0）、4/5（seed1）、4/5（seed2）。seed0 在 ev5000/ev11000 呈一致小幅正收益（全部组 0~+0.004），未达预注册的每 seed ≥3 异质 cell 门槛。
- **是否由 Added_GS 数量导致？** 组内 |rank-corr(Added_GS, benefit)| ≥0.7 的 cell 仅 5/14——大多数 cell 的差异不能由组大小解释。

## 3. 判定（预注册规则，修正 NaN 计数后）

| 预注册 GO 条件 | 实测 | 满足 |
|---|---|---|
| +2000 异质 cells ≥60% | 10/14 = 71% | ✓ |
| 每 seed ≥3 异质 cells | 2/4（seed0）、4/5、4/5 | **✗（seed0）** |
| +1000 同方向 ≥50% | 13/14 = 93% | ✓ |

**V2-A = NO-GO。** "跨 3 seeds 稳定存在异质性"这一 GO 核心条件未被满足：异质性在 seed1/seed2 强且稳定，在 seed0 的两个 mid/late event 中缺席（全部组一致小幅正）。

## 4. 诚实备注

1. **NaN 计数更正**：首轮分析把空 ROI 组（NaN）计入"非正"桶，得到 12/15 的 GO 判定；修正后（有效组 ≥3 才可判定、NaN 剔除）为 10/14 且 seed0 不达每-seed 门槛 → **NO-GO**。以修正后为准。
2. 异质性现象本身（71% cell、最大 spread 0.033 ROI-L1、正负组共存）是真实的——但按本任务"跨 3 seeds 稳定"的 GO 定义，seed0 的缺席使其未达 GO。若后续希望重启该路线，需要一个解释 seed0 差异的先验（例如场景区域相关的误差结构），属于新的诊断假设，本阶段不展开。
3. 空 ROI 组占 17.8%：这些组的覆盖区与高误差区在 sampled views 上不相交，其 benefit 不可测（非"无收益"）。这限制了可判定 cell 的组数，也是 seed0 ev7000 不可分类的原因。
4. 本阶段未分析 error 类型，未设计任何方法。

## 5. 产物

- `paper_b/b22_v2a_intra_event_benefit/data/groups{,_s*}.csv` — 90 组
- `paper_b/b22_v2a_intra_event_benefit/data/paired_capacity_benefit.csv` — 270 行
- `paper_b/b22_v2a_intra_event_benefit/data/seed_event_summary{,_s*}.csv`
- `paper_b/b22_v2a_intra_event_benefit/data/memory_usage.csv` — 360 行 RAM 监控
- `paper_b/b22_v2a_intra_event_benefit/data/b22v2a_stats.txt` — 统计与判定全文
- 脚本：`diagnostics/diagnostic_b22v2a.py`、`diagnostics/analyze_b22v2a.py`；日志：`b22v2a_full.log`

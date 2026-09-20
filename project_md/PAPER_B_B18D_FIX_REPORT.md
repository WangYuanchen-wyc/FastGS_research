# Paper B — B18-D Fix 报告：Early Densification Persistence & Child Fate Validation

> 在 B18-D 基础上，修复测量方法后回答同一问题：**一次 densification trigger 到底是持续的真实 capacity demand，还是只出现一次的瞬时 demand？**
> 本次新增：(a) 触发持久性以**比例**而非绝对数计量；(b) 通过 append 顺序精确识别 split/clone 新生儿，并追踪其命运（下一 event 存活/重触发、30k 存活、终态 opacity）。
> GPU 6，Room（官方 `train_base.sh` 配置：`--grad_abs_thresh 0.0008 --highfeature_lr 0.02`），3 seeds × 30k iters，densification_interval=500，28 events/seed。
> **判定：B18-D Fix = NO-GO 确认。触发 demand 高度 one-shot，早期新增容量的 2/3 在 30k 前死亡。**

## 0. 方法修复与运行有效性验证

上一版脚本因 3 个 bug 产出空数据（scipy 依赖缺失、densify 后重算被清零的梯度统计、child-fate 死代码）。本次整文件重写（`diagnostics/diagnostic_b18d_fix.py`）：

- 触发集合（split_set/clone_set）在 densification **之前**计算；
- 邻近匹配用 `torch.cdist`（chunked，tol=0.05，与 B18-D 相同）；
- 新生儿识别：已核实 `scene/gaussian_model.py` 中 clone 先 append、split append 2×child 后删 parent，故 densify 后张量末尾 `n_clone + 2·n_split` 个点恰为本 event 新生儿；
- child fate：记录每个 event 新生儿在下一 event 的存活数与重触发数，以及 30k 终模型中的存活数与平均 opacity。

**逐位一致性验证**：seed 0 前 2 个 event 与原 B18-D 数据完全一致（ev1: 1155 triggers；ev2: 2476 triggers，persist=160），it=3000 检查点（9003 triggers / persist 948 / PSNR 12.78 / #GS=133160）亦逐项一致——证明重写脚本无行为偏差，且确认原 B18-D 运行使用官方 room 配置。

## 1. Q1 — 触发持久性（比例口径）

early = it∈[1000,3000]（ev1–5，ev1 无前驱不计入）；late = it>3000。persist_ratio = 本 event 触发者中与上一 event 触发者空间重合（<0.05）的比例。

| seed | early mean±std (n=4 ev) | late mean±std (n=18/19/21 ev) |
|---|---|---|
| 0 | 0.090 ± 0.020 | 0.000 ± 0.000 |
| 1 | 0.103 ± 0.035 | 0.000 ± 0.000 |
| 2 | 0.110 ± 0.063 | 0.000 ± 0.000 |
| **ALL** | **0.101 ± 0.012 (SEM)** | **0.000 ± 0.000 (SEM, 58 events)** |

- **即使在其最密集的早期窗口，也只有 ~10% 的触发与前一 event 重合；90% 是首次出现的新触发。**
- 中后期（58 个 events）persist_ratio 恒为 0：demand 完全是 one-shot。
- 与原 B18-D 的绝对计数口径一致（原报告峰值 persist≈1143 / 8337 triggers ≈ 13.7%，与 10.1% 同量级）。
- 匹配对的平均距离 0.033–0.036（贴着 0.05 容差）：重合者的位置在 500 it 内有漂移，但确实连续触发。

## 2. Q2 — 新生儿命运

early 新生儿（5 events/seed，n_born 2310–18006/event）：

| 指标（early，SEM 跨 15 events） | 值 | 含义 |
|---|---|---|
| alive@next-event | 1.000 | 新生儿不会被 multinomial prune 选中（采样权重为 0），500 it 漂移 <0.05 —— 此项接近构造性恒真，仅供参考 |
| re-trigger@next-event | 0.454 ± 0.063 | 早期窗口内部，上一 event 新生儿中 ~45% 在下一 event 再次触发——demand 强烈但自我消化 |
| **alive@30k** | **0.364 ± 0.025** | **早期新增容量 64% 在 30k 前被删除** |
| 幸存者终态 opacity | 0.010 | 匹配到的幸存者大多带 opacity-reset 后的最小值——即便活着也近乎停滞 |

跨 seed 一致：alive@30k = 0.239–0.385（seed0）、0.309–0.427（seed1）、0.266–0.592（seed2）。

late 新生儿 alive@30k = 0.486 ± 0.049，但每个 late event 仅出生 2–16 个新生儿（中位数 4），比例噪声大且经过阈值筛选（能触发的本身就是幸存者），**不构成支持后期 densification 的证据**。

两个必须注明的混杂：
1. **ev5（it=3000）的 retrigger ≈ 0.0002**：it=3000 恰逢 opacity reset，触发 landscape 被重置，该点测量的是 reset 效应而非儿童消失；
2. retrigger 沿 ev1→ev5 递减（0.69 → 0.60 → 0.44 → 0.56 → ~0）：早期窗口内部 demand 也在快速自耗。

## 3. 判定（标准在读取结果前固定）

| 标准 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| early persist_ratio | ≥0.5 → persistent；≤0.25 → one-shot NO-GO | **0.101** | **one-shot transient，NO-GO** |
| early 儿童 alive@30k | ≤0.3 → 容量过度；≥0.6 → 结构性 | **0.364** | 中间偏过度（2/3 死亡） |

**B18-D Fix = NO-GO。** 两条独立证据链互相印证：
1. 触发层面：90% 的早期触发是一次性的，中后期 100% 一次性；
2. 容量层面：早期 trigger 产生的新生儿 64% 活不到 30k，幸存者也多为 opacity≈0.01 的停滞点。

一个 persistence-aware 的 densification 信号（例如基于 EMA 的历史触发加权）将把 90% 的算力花在追逐噪声上。这**确认**了原 B18-D 的 NO-GO，且现在有了 child 级别的直接证据。

## 4. 稳健性：跨配置复验

在写报告前，同脚本曾以 repo 默认阈值（`grad_abs_thresh=0.0012`，未加 room 专属参数）完整跑过一遍 3 seeds（归档于 `fix/data_defaultargs/`）：early persist_ratio = 0.040 ± 0.004，early 儿童 alive@30k = 0.077 ± 0.003。**更严的阈值下结论方向相同且更强**——one-shot 主导不是 room 官方超参的偶然产物。

## 5. 对 Paper B 的含义

FastGS 的 densification 是"一次性脉冲"机制：结构形成期（前 5 个 events）集中爆发出大量一次性 demand，每个 trigger 的容量贡献大半是临时的，由后续 pruning 消化回收。这个行为与 B17 系列结论一致：**没有可利用的持续性信号**——densification 侧（B18-D/Fix）与 pruning 侧（B13-B/B15-C/B17-H）的改进空间都被证伪。Paper B 的系统性诊断到此可以收敛：FastGS 的 trigger/prune 机制在其自身语义（binary error gating）下是自洽的，负结果成立。

## 6. 产物

- `paper_b/b18_densification_persistence/fix/data/trigger_events{,_s*}.csv` — 84 rows（3×28 events，含 persist/retrigger/新生儿计数）
- `paper_b/b18_densification_persistence/fix/data/child_fate_s*.csv` — 73 rows unique（s0/s1/s2 = 23/24/26 born groups；s1、s2 文件为历史累积写入，分析端按 (seed, birth_event) 去重）
- `paper_b/b18_densification_persistence/fix/data/b18dfix_stats.txt` — 统计与判定
- `paper_b/b18_densification_persistence/fix/plots/persistence_and_child_fate.png` — 三联图
- `paper_b/b18_densification_persistence/fix/data_defaultargs/` — 默认阈值配置复验数据
- 脚本：`diagnostics/diagnostic_b18d_fix.py`、`diagnostics/analyze_b18d_fix.py`；日志：`fix/logs/b18dfix_full.log`、`logs/validate_args.log`

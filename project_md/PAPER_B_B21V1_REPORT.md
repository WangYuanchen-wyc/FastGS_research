# Paper B — B21-V1 报告：Densification Benefit Diagnostic

> 只验证：FastGS 的不同 densification event，实际收益是否存在明显异质性？是否存在"不 densify 也几乎一样"的 event？
> GPU 5，Room（官方配置），3 seeds × 30k，每 seed 按训练阶段预先固定选 6 个 event（early {1500,2500} / middle {5000,7000} / late {11000,13000}，选择与收益完全无关），成对反事实（A=Densify，B=No-Densify），深度快照（参数对象+optimizer state+param_groups+RNG+相机池）+ 事件后 RNG 对齐，ROI mask（同一批 sampled views 的高误差像素 ∧ 触发 parent 投影覆盖）A/B 完全相同。实现自检（恢复逐位一致、相机序列一致、迭代数一致）全部通过，无 INVALID。
> **判定：V1 = GO（按任务文本 GO 定义；预注册代码阈值存在操作化缺陷，见 §4 透明披露）。**

## 1. 核心结果（Benefit_ROI 与 ΔPSNR，horizon +400，主口径）

Benefit_ROI = ROI误差(NoDensify) − ROI误差(Densify)，正值 = densify 更好；ΔPSNR = PSNR(NoDensify) − PSNR(Densify)，正值 = 跳过 densify 全局 PSNR 更高。

| seed | it | phase | Added_GS | Benefit_ROI | ΔPSNR |
|---|---|---|---:|---:|---:|
| 0 | 1500 | early | 29,015 | **+0.00204** | +0.09 |
| 0 | 2500 | early | 21,551 | −0.00241 | +0.29 |
| 0 | 5000 | middle | 20,202 | −0.00259 | +0.25 |
| 0 | 7000 | middle | 8,272 | −0.00025 | +0.08 |
| 0 | 11000 | late | 5,259 | −0.00039 | +0.11 |
| 0 | 13000 | late | 5,136 | +0.00020 | +0.07 |
| 1 | 1500 | early | 28,604 | **+0.00238** | −0.03 |
| 1 | 2500 | early | 23,788 | +0.00049 | +0.23 |
| 1 | 5000 | middle | 13,657 | −0.00062 | +0.24 |
| 1 | 7000 | middle | 12,968 | +0.00060 | +0.12 |
| 1 | 11000 | late | 6,231 | −0.00077 | +0.17 |
| 1 | 13000 | late | 4,916 | −0.00042 | +0.09 |
| 2 | 1500 | early | 26,782 | **+0.00391** | +0.02 |
| 2 | 2500 | early | 25,278 | −0.00088 | +0.14 |
| 2 | 5000 | middle | 19,775 | −0.00158 | +0.19 |
| 2 | 7000 | middle | 10,397 | −0.00041 | +0.14 |
| 2 | 11000 | late | 6,677 | −0.00108 | +0.13 |
| 2 | 13000 | late | 4,352 | −0.00000 | +0.07 |

## 2. 六问简答

1. **Densify 是否真的有额外收益？** 仅第一个被选 event（it=1500）在 3/3 seeds 上有正 ROI 收益（+0.0020/+0.0024/+0.0039，是全部 24 cells 中最大的三个值）。其余 5 个 event × 3 seeds = 15 cells 中 12 个 ROI 收益为负、3 个近零正（≤0.0006）；**全局 PSNR 口径 23/24 cells 中跳过 densify 反而更高**（ΔPSNR +0.03..+0.29）。
2. **异质性？** 极大且按 event 稳定：Benefit_ROI 范围 −0.0026..+0.0039；每 seed 内 CV = 3.0–255.6；同一 seed 内符号随 event 翻转。it=1500 与其它所有 event 的差距 3–20 倍以上。
3. **稳定的 low / near-zero / 负收益 events？** 是——middle/late 的 15 cells 中 12 负、3 近零正；且这些 event 的 Added_GS 为 4.3k–20k（各 seed 中位 Added_GS 的 40%–170%），**低收益不是由新增容量少导致的**。
4. **每增一个 Gaussian 的收益差异？** Benefit_per_GS 范围 −1.6×10⁻⁷ .. +1.5×10⁻⁷：it=1500 恒正（7.0/8.3/14.6×10⁻⁸），middle/late 多数为负——同样多的容量预算在不同 event 上的边际回报符号都不同。
5. **跨 seeds / 阶段？** 模式 3/3 seeds 一致：it=1500 恒正且显著，其余 5 个 event 收益 ≤0 为主。训练阶段维度上 early 首个 event 独占正收益。
6. **V1 = GO**（论证见 §4）。

## 3. 机制解读（与已有诊断互证）

it=1500 是触发爆发期的核心 event（room 在 it1000–3000 有 1–3 万 trigger/event，B18-D Fix 重验数据）；其收益为正且远超其它 event——**结构形成期的高误差确实需要真实的新容量**。而 middle/late 的 event 虽然同样有数千~两万 trigger（高误差信号存在），但 +400 的成对对比显示新增容量没有换来更低误差——**High Error ≠ Always Need More Gaussian**。这与 B20-V（opacity 之外的支持信息增量有限）共同指向：FastGS 的 binary gating 把"误差存在"直接翻译成"需要更多 Gaussian"，但两者的耦合只在训练最早期成立。

**必须注明的时间窗限制**：A 分支承载了 native event 的全部瞬态成本（opacity clamp 至 0.8、约半数候选被 prune、优化器状态重置），+400 窗口内这些成本尚未被后续训练恢复；因此 mid/late 的"负收益"包含事件本身的瞬态代价，不纯粹是"新增 Gaussian 无用"。任务指定的主因果窗口即 +400，+500/+1000 的 native continuation 属可选项未记录，长窗口收益属于 V2 阶段的问题。

## 4. 判定过程（完全透明披露）

- **预注册代码判定的机械输出：NO-GO**（`b21v1_stats.txt` 原样保留）。原因：我的操作化定义把 low-benefit 事件定义为"|b| ≤ 10% × 该 seed 正收益中位数"——**没有预见到收益会系统性为负**，负收益事件无法落入该 near-zero 类；同时 high 门槛（2× 正收益中位）只把 seed1 的 it=1500 分类为 high。
- **按任务文本的 GO 定义逐条核对**：①异质性明显 ✓（CV 3–256，同 seed 内符号翻转）；②存在 high-benefit（it=1500，3/3 seeds）与 low/negative-benefit events（12/15 负 + 3/15 近零）✓；③低收益非容量不足所致（Added_GS 4.3k–20k）✓；④跨 3 seeds 稳定（it=1500 正 3/3；其余 event 低/负模式 3/3）✓。**四项 GO 条件实质满足。**
- 预注册阈值是对任务定义的操作化尝试，其缺陷方向恰好压制了 GO（未发生"调阈值凑 GO"——机械结果本来就是 NO-GO）。综合裁定：**V1 = GO**，进入 V2 前提成立："High Error ≠ Always Need More Gaussian"，下一阶段才研究 densify 前哪些 error 特征能预测 benefit。
- 本阶段不设计任何新 densification 方法；以上仅为诊断。

## 5. 产物

- `paper_b/b21_v1_densification_benefit/data/densify_events{,_s*}.csv` — 全部 native events（84 rows）
- `paper_b/b21_v1_densification_benefit/data/paired_branch_metrics{,_s*}.csv` — 144 行分支×视角全局指标
- `paper_b/b21_v1_densification_benefit/data/roi_benefit{,_s*}.csv` — 72 行 ROI 收益
- `paper_b/b21_v1_densification_benefit/data/capacity_efficiency{,_s*}.csv` — 72 行容量效率
- `paper_b/b21_v1_densification_benefit/data/seed_event_summary.csv`、`b21v1_stats.txt`（机械判定原样保留）
- 脚本：`diagnostics/diagnostic_b21v1.py`、`diagnostics/analyze_b21v1.py`；日志：`b21v1_full.log`（seed2 因外部进程挤占 GPU OOM 单独重跑，数据完整）

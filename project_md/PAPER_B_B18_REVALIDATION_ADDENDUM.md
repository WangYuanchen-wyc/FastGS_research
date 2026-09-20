# Paper B — B18 系列重验报告（Addendum）：optimizer_step 缺失的更正

> **更正声明**：B18-D、B18-D Fix、B18-D Final Fix 三个诊断的原始运行存在执行 bug——训练循环漏调 `gaussians.optimizer_step()`（train.py:163），等效于在**冻结参数**上测量（证据：全程 PSNR 9–13 dB vs B17 正确训练 32.44 dB；最终模型幸存者 opacity 恒为 0.010000）。原始报告中所有数值与"NO-GO"判定均基于该失真动力学。
> 本轮以修正循环（`optimizer_step` + reset 限制在 it<15000，room 官方配置，3 seeds × 30k）完整重跑两个 Fix 诊断。结论：**Final Fix 的 NO-GO 维持，但其机制解释被修正；B18-D Fix 的"one-shot"结论被推翻——正确训练下早期 demand 大部分是持久/结构性的。**
> 数据：`paper_b/b18_densification_persistence/fix2/`（B18-D Fix 重验）、`final_fix2/`（Final Fix 重验）。

## 1. B18-D Fix 重验（触发持久性 + 儿童存活）

| 指标（early = it1000–3000） | 冻结版（原报告） | **native 重验** |
|---|---|---|
| early persist_ratio | 0.101 ± 0.012 | **0.483 ± 0.020** |
| late persist_ratio | 0.000 ± 0.000 | **0.289 ± 0.014**（69 events） |
| 早期儿童 alive@30k | 0.364 ± 0.025 | **0.834**（0.793–0.889 per event/seed） |
| 幸存儿童终态 opacity | 0.010（reset 下限，冻结指纹） | **0.45–0.50**（正常恢复） |
| 逐 seed early persist | 0.090/0.103/0.110 | **0.474 / 0.462 / 0.514** |

按原预注册判据：persist_ratio = 0.483 ∈ (0.25, 0.5) → **MIXED**（95% CI 约 [0.44, 0.52]，横跨 0.5 线）；儿童 alive@30k = 0.834 ≥ 0.6 → **"early capacity persists; demand is structural"**。
**原报告"90% trigger 是 one-shot、中后期 persistence=0、2/3 儿童死亡"的全部图景均为冻结伪影，予以撤回。** 正确训练下：早期 demand 约半数持久，且早期新增容量 83% 活到 30k——demand 是结构性的。

## 2. B18-D Final Fix 重验（One-shot vs Persistent 儿童价值）

样本（native 下 persistent trigger 常见）：每 seed One-shot 儿童 ~6.7–7.2 万，Persistent-2 ~2.2 万，Persistent-3 ~2.3–2.8 万（冻结版 P3 仅 622–1196）。parent 级 P3 占比 ~20%（冻结版 ~2%）。

| 比较 | 原始 ΔSurv@30k (P−OS) | 分层控制后 |
|---|---|---|
| P2 vs OS | +0.014 / +0.011 / +0.007 | −0.005 / −0.008 / −0.002（≈0） |
| P3 vs OS | +0.035 / +0.015 / +0.046 | +0.010 / −0.004 / +0.019（微弱、不齐） |

- 儿童存活率绝对值：OS 0.777–0.799，P2 0.788–0.805，P3 0.792–0.845；opacity 0.46–0.50，visibility 0.62–0.69（全部为正常训练值）。
- 分层（birth event 等权）后 P2 优势归零、P3 只剩 ≤1.9% 且方向不齐；lifetime 差 +0.0–3.3%。
- **判定维持 NO-GO**：persistent-demand 儿童最多只有 ~1–5% 的存活率优势（样本数万、估计精确），低于预注册 GO 门槛（≥5% × 3/3 seeds × 分层后仍成立），且 birth-timing 控制后进一步缩小。
- **机制解释修正**：不是"demand 一次性所以无可利用"（伪），而是"**demand 持久性真实存在、但持久 demand 与一次性 demand 产生的容量在长期价值上几乎无差别**"——FastGS 的 trigger 语义（梯度+binary gating+VCD）在正确训练下已经把两类 demand 引向质量相当的容量。

## 3. 对 Paper B 全局结论的影响

1. **B17 系列**：不受影响（其脚本有 optimizer_step，PSNR 32.44 正常）。
2. **B18-D / Fix / Final Fix 原报告**：数值与机制解释作废，判定以本 addendum 为准。
3. **Final 判定不变**：Historical / Persistent Densification = **NO-GO**——但理由由"demand 一次性"更正为"持久性存在但无价值差异"。
4. **B19-R**（本轮同期的 Reliable Density Control）本身即用 native-faithful 循环，其双 NO-GO 不受此 bug 影响；其 A 部分与 Final Fix 重验相互印证：trigger 的 view 集中度、demand 持久性均不预测儿童长期价值。
5. 更早使用 `native_train_one_iter` 的阶段（如 B8 系列短程验证）若有训练结论，同样需要按此 bug 复查；checkpoint 快照类阶段（B11–B14）不受影响。

## 4. 产物

- `paper_b/b18_densification_persistence/fix2/` — B18-D Fix 重验数据（84 events、28 child-fate rows、图）
- `paper_b/b18_densification_persistence/final_fix2/` — Final Fix 重验数据（581k parent occurrences、351k children）
- 脚本改动：`diagnostics/diagnostic_b18d_fix.py`、`diagnostics/diagnostic_b18d_final_fix.py`（+optimizer_step、reset 范围、OUT 可覆盖）；分析脚本 base 可覆盖 + plot 修复

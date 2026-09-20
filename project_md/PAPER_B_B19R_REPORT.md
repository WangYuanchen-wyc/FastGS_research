# Paper B — B19-R 报告：Reliable Density Control Diagnostic

> 两个问题：**(A)** FastGS 的 VCD trigger 是否经常被少数 views 主导？这种 trigger 产生的 children 是否更差？ **(B)** FastGS 因 opacity<0.1 删除的 Gaussian 中，是否存在仍获得明显 multi-view rendering support 的点？
> GPU 0，Room（官方配置），3 seeds × 30k，**native-faithful 循环（含 optimizer_step，见 §5）**，逐 view 证据捕获复用 native VCD 的同一批 render（零额外 render）。
> **判定：Reliable Densification = NO-GO；Reliable Pruning = NO-GO。**

## 0. 分组规则（在读取任何 fate 结果之前，仅依据证据分布确定）

- **A. View-concentrated := top1_share > 0.5**（单一 view 贡献超过其余全部 view 之和的证据）。语义化多数切分；分布连续重尾（top1_share 中位 0.69–0.72，p75≈0.97），无自然双峰。
- **B. Still-supported := 10 个 sampled views 中 ≥5 个有非零 VCD 证据**（多数 view 将该点标记为覆盖高误差像素）。corroborating 弱信号（frustum 可见性 ≥5/10）单独报告，不参与判定。

## 1. Q1 — VCD trigger 有多少被少数 views 主导？

| seed | n_trig（全部 events） | concentrated % | top1 中位 | 非零 view 均值 | 熵均值 |
|---|---:|---:|---:|---:|---:|
| 0 | 196,656 | 75.7% | 0.690 | 2.91/10 | 0.288 |
| 1 | 197,706 | 77.9% | 0.720 | 2.83/10 | 0.273 |
| 2 | 192,646 | 78.3% | 0.722 | 2.87/10 | 0.274 |

**约 76–78% 的 VCD trigger 由单一 view 贡献过半证据；平均只有 ~2.9/10 个 sampled view 给予非零证据。** VCD 的"多视角一致"在实现上是"逐 view 二值计数求和"，实际触发高度依赖个别 view。

## 2. Q2 — view concentration 能否预测 child fate？

早期窗口（it1000–3000）children（每组 ~2.7万–9.1万/seed），30k 存活率 / prune ratio / lifetime：

| seed | concentrated (n) | distributed (n) | ΔSurv30k (dist−conc) | ΔLifetime |
|---|---|---|---|---|
| 0 | 0.767 (82,901) | 0.769 (38,932) | +0.002 | +198 it |
| 1 | 0.754 (91,138) | 0.755 (26,830) | +0.001 | +177 it |
| 2 | 0.782 (82,809) | 0.791 (36,521) | +0.009 | +239 it |

- distributed children 三个 seed 一致地**略**好，但幅度 0.1–0.9%，远低于预注册 GO 门槛（≥5% × 3/3 seeds）；lifetime 差 ~1%。
- 按 birth event 分层（15 个 n≥50 层）：10/15 层维持"concentrated 更差"的弱号；层内波动 ±2–3%。
- split/clone 控制不改变结论（clone children 极少：167–479/组；99.6% children 是 split）。
- **结论：view concentration 不能有效预测新增容量的长期价值。** 虽然 ~3/4 的 trigger 是 view-concentrated，它们产生的 children 与 view-distributed 的几乎一样好。

## 3. Q3/Q4 — 低 opacity（<0.1）final-prune 候选的 multi-view support

FastGS 在 it=18000/21000/24000/27000 各执行一次 `final_prune_fastgs(min_opacity=0.1)`（native train.py:153–158）。候选支持度分布（VCD 证据非零 view 数）：

| 轮次 | 候选数/seed | nz=0 | nz 1–2 | nz 3–4 | **nz≥5 (Still-supported)** | vis≥5/10 |
|---|---:|---:|---:|---:|---:|---:|
| 1 (18k) | 49,878–51,567 | 68.4–73.2% | 18.1–22.5% | 5.0–9.5% | **0.6–1.8%** | 8.7–29.4% |
| 2–4 (21k–27k) | 803–2,003 | 33.3–43.4% | 33.9–48.9% | 7.8–18.6% | **1.1–11.7%** | 12.2–44.9% |

- Round 1 占全部候选的 ~93%：其 Still-supported 率仅 0.6–1.8%。**按候选数加权，三 seed 总支持率 ≈ 0.8% / 2.3% / 0.9%。**
- 最高 cell（seed1 round2 = 11.7%）仅基于 1,999 个候选（~234 个点），且同 seed round4 降至 3.5%、seed0 同 round 仅 1.3%——**跨 seed/轮次不稳定**。
- 支持是否来自多个 view：Still-supported 定义本身要求 ≥5/10 view 非零，即多 view 支持；但有证据的候选中大多数集中在 nz 1–2（单 view 或双 view 偶然覆盖），top1_share 中位 = 0.000（多数候选全零证据）。
- frustum 可见性（vis≥5/10，弱代理，不做判定依据）：round1 8.7–29.4%。可见≠有 rendering contribution，仅作旁证。
- **结论：opacity<0.1 候选中 Still-supported 占比 ~1–2%，不构成"数量不可忽略"，且跨 seed 不稳定 → Reliable Pruning = NO-GO。**（本诊断只检验 opacity 信息是否遗漏 multi-view 证据，不主张这些点该不该删。）

## 4. 六问简答

1. **~76–78%** 的 VCD trigger 被单一 view 主导（top1_share>0.5），平均非零支持 view 仅 ~2.9/10。
2. **不能**：view-concentrated 与 view-distributed 的 children 30k 存活率差 ≤0.9%（3/3 seeds 方向一致但幅度可忽略），分层控制后依然微弱。
3. **~1–2%**（候选加权；round1 仅 0.6–1.8%）的低 opacity 候选有 ≥5/10 view 的 multi-view 证据支持。
4. **不稳定**：round1 三 seed 一致地低（0.6–1.8%），后期轮次波动大（1.1–11.7%）且无跨 seed 复现。
5. **Reliable Densification = NO-GO**：trigger 确实常被少数 view 主导（Q1 成立），但这不产生更差的 children（Q2 不成立）——"多 view 一致性"在 FastGS 的语义里对容量价值没有预测力。
6. **Reliable Pruning = NO-GO**：低 opacity 候选几乎全部（≥97% 候选加权）缺乏 multi-view VCD 证据支持。

## 5. 重要更正：B18 系列曾在冻结模型上测量（本轮已修复）

`native_train_one_iter` 只含 render+backward；`gaussians.optimizer_step()`（train.py:163）是独立调用。B18-D/Fix/Final Fix 原循环**均未调用它**，等效于冻结参数训练（证据：PSNR 停在 9–13 dB vs B17 正确训练 32.44 dB；最终模型所有幸存者 opacity=0.010000 恒定）。本 B19-R 循环为 native-faithful（PSNR 30–31 dB，`optimizer_step` + reset 限制在 it<15000 + final-prune 轮次 18k–27k）。B18 两个诊断已用修正循环重验（见 `PAPER_B_B18_REVALIDATION_ADDENDUM.md`）：Final Fix 的 NO-GO 维持，Fix 的持久性图景已修正。

## 6. 产物

- `paper_b/b19_reliable_density_control/data/densify_view_evidence{,_s*}.csv` — 587k 行 trigger 证据
- `paper_b/b19_reliable_density_control/data/densify_child_fate{,_s*}.csv` — 359k 行 children 命运
- `paper_b/b19_reliable_density_control/data/prune_multiview_support{,_s*}.csv` — 163k 行候选支持度
- `paper_b/b19_reliable_density_control/data/seed_summary.csv`、`b19r_stats.txt`
- 脚本：`diagnostics/diagnostic_b19r.py`、`diagnostics/analyze_b19r.py`；日志：`b19r_full.log`

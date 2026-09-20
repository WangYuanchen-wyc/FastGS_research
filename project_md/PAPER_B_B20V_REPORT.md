# Paper B — B20-V 报告：Rendering Support Validity

> 验证：opacity<0.1 候选中，具有 multi-view rendering support 的 Gaussian 是否真的比 low-support Gaussian 更值得保留？
> GPU 7，Room（官方配置），3 seeds × 30k，4 个原生 final-prune 轮（18k/21k/24k/27k），166,910 个候选，matched deletion（opacity caliper=0.01，K 对完全同数删除，Random 对照），immediate impact only。
> **判定：Reliable Pruning = NO-GO（按预注册标准）。** 真实性、时间稳定性、匹配公平性均通过，但核心删除实验中两组的质量损失差距远低于 GO 门槛（PSNR 差 0.013 dB ≪ 0.1 dB）。

## 1. 贡献强度（V1 ✓）

| seed | A_low 中位总贡献 | B_multi 中位总贡献 | B_multi 中位 mean/supported-view | B/A 倍数（round 1） |
|---|---|---|---|---|
| 0 | 0.0000 | 6.90 | 1.89 | 6.1×10⁶ × |
| 1 | 0.0000 | 9.99 | 2.25 | 8.9×10⁶ × |
| 2 | 0.0000 | 7.54 | 2.14 | 6.7×10⁶ × |

**Multi-view support 不是数值噪声**：B 组中位每视角贡献 ~2（= 2×10⁶ × 容差 1e-6），A_low 中位总贡献恰为 0（74% 候选在全部 10 个视角贡献严格为零）。

## 2. 时间稳定性（V2 ✓）

对死于 round 2–4 的候选（可回溯 2–3 轮支持历史，orig-id 精确追踪——15k 后无 densification、prune 保序，无任何近似）：

| seed | n | Low | Transient | **Persistent** |
|---|---:|---:|---:|---:|
| 0 | 3,918 | 2,248 | 692 | **978 (25.0%)** |
| 1 | 4,177 | 1,789 | 465 | **1,923 (46.0%)** |
| 2 | 3,898 | 1,856 | 516 | **1,526 (39.1%)** |

support 在相当一部分 Gaussian 上跨轮次持续存在（≥2 个连续轮次含死亡轮）。

## 3. Matched Deletion（V3 ✓ 匹配公平）

opacity caliper=0.01 最近邻配对（贪心双指针，最大基数）；mean |Δopacity| = 0.0005–0.0077（全部 ≤ 0.01）。Random 对照为同轮候选均匀采样后按同一 caliper 匹配到 B 组 opacity 列表。每分支删除完全相同的 K 个点后立即在全测试集上评估。

| seed | round | K | A dPSNR | B dPSNR | R dPSNR | dSSIM A→B (×1e-3) | dLPIPS A→B (×1e-3) |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | 1 | 13,948 | −0.128 | −0.114 | −0.152 | −0.60 → −2.55 | +0.11 → +0.61 |
| 1 | 1 | 13,230 | −0.091 | −0.161 | −0.141 | −0.36 → −3.42 | +0.08 → +1.63 |
| 2 | 1 | 13,705 | −0.101 | −0.101 | −0.159 | −0.55 → −1.72 | +0.09 → +1.04 |
| 三 seed | 2–4 | 371–952 | ≈0 | 略更负 | 介于 | 一致 | 一致 |

**12 cells 聚合**（mean ± 跨 cells）：

| 分支 | ΔPSNR | ΔSSIM (×1e-3) | ΔLPIPS (×1e-3) |
|---|---|---|---|
| Delete Low-support | −0.030 | −0.60 | +0.23 |
| Delete Supported | −0.043 | −2.15 | +0.65 |
| Delete Random-matched | −0.046 | −1.88 | +0.56 |

## 4. 七问简答

1. **贡献足够强吗？** 是——B 组中位 per-view 贡献 ~2，是容差的 2×10⁴ 倍；不是数值噪声。
2. **Temporal stability？** 有——25.0%/46.0%/39.1% 的候选为 Persistent-supported，跨 seeds 一致。
3. **匹配公平吗？** 公平——全部 12 cells 的 mean |Δopacity| ≤ 0.0077（caliper 0.01）；scale/birth iteration 作为描述量记录（`contribution_strength.csv`），未强行匹配。
4. **删除 Supported 是否更伤？** **方向一致但幅度远低于门槛**：dPSNR = drop(B)−drop(A) 在 11/12 cells 为负（B 更伤），但三 seed 均值仅 −0.001/−0.030/−0.008 dB（总体 −0.013 dB ≪ 0.1 dB）。dSSIM 与 dLPIPS 方向 12/12 一致（B 组损伤 ≈ A 组的 3 倍），绝对量级 ~1e-3。
5. **Random control？** R 与 B 的 PSNR 损失几乎相同（−0.046 vs −0.043）、比 A 更伤（R 的 opacity 分布被匹配到 B，更高）——说明 **PSNR 意义上的删除代价主要由 opacity 决定，support 只提供小幅感知级增量**。
6. **跨 seeds / events 一致？** 方向完全一致（SSIM/LPIPS 12/12，PSNR 11/12）；但幅度小且 round 1 与 2–4 轮形态不同。
7. **判定**：**NO-GO**。V1（真实强度）✓、V2（时间稳定）✓、V3（匹配公平）✓，V4 ✗——PSNR 口径的组间差距 0.013 dB，比预注册 GO 门槛（0.1 dB × 3/3 seeds）低一个数量级。

## 5. 诚实备注

- **判定敏感性**：若以 dSSIM/dLPIPS 为主要指标，B 组损伤一致地为 A 的 ~3 倍（12/12）——方向上有真实信号。但预注册标准以 PSNR ≥0.1 dB 为主判据，且未预注册 LPIPS 替代路径；按预注册执行 = NO-GO。感知指标的信号量级（dLPIPS B−A ≈ +0.4×1e-3）本身也偏小。
- **符号约定更正**：分析代码中 V4 的 "dPSNR > 0" 字面方向与其文字意图（"删除 Supported 更伤"）相反；两种读法下 V4 均不满足（量级均 <0.1 dB），不影响判定，仅影响中间数字的方向表述。
- **R 对照为什么比 A 更伤**：R 被 caliper 匹配到 B 的 opacity 列表，其 opacity 均值高于 A——这本身再次印证 opacity 是删除代价的主导因子。
- 本阶段不涉及 removal/recovery、不修改 FastGS pruning。

## 6. 对 Paper B 的收束

B19-P 发现 27% 的低 opacity 候选仍有真实多视角渲染贡献（GO for method design）；B20-V 进一步验证该支持的**决策价值**：在公平匹配 opacity 后，删除它们的额外质量损失在 PSNR 口径上低于预注册门槛，仅在感知口径上有小幅（~3×相对、1e-3 绝对）的一致性代价。**按预注册标准：opacity 信息虽然"不完整"（B19-P），但补上 rendering support 信息后，对 opacity-only pruning 的改进空间在即时影响口径下不足以支撑方法设计（NO-GO）。**

## 7. 产物

- `paper_b/b20_rendering_support_validity/data/contribution_strength{,_s*}.csv` — 165,210 行
- `paper_b/b20_rendering_support_validity/data/temporal_support{,_s*}.csv` — 支持历史 + 分类
- `paper_b/b20_rendering_support_validity/data/matched_deletion_results{,_s*}.csv` — 36 分支结果
- `paper_b/b20_rendering_support_validity/data/seed_event_summary.csv`、`b20v_stats.txt`
- 脚本：`diagnostics/diagnostic_b20v.py`、`diagnostics/analyze_b20v.py`；日志：`b20v_full.log`

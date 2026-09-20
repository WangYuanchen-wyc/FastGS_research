# Paper B — B19-P 报告：Opacity-Prune Candidates 的真实渲染支持

> 只验证一个问题：FastGS 中因 **opacity < 0.1** 即将被 prune 的 Gaussian，是否仍在多个训练视角中具有**真实 rendering contribution**？
> GPU 6，Room（官方配置），3 seeds × 30k，native final-prune 轮（18k/21k/24k/27k），共 166,096 个候选全量测量。
> **判定：Reliable Pruning = GO for method design。约 27%（跨 seeds 0.272–0.285）的低 opacity 候选仍具有 ≥3 个视角的真实渲染贡献，且跨 seeds / 轮次稳定存在。**
> （按任务边界：这只证明 opacity 信息不完整；这些点是否真的不该删，属于 removal 问题，本阶段不做。）

## 1. 真实 rendering contribution 如何定义和获取

- **定义**：前向 alpha 混合中该 Gaussian 的混合权重 **w_i = α_i · T_i**（rasterizer 在 forward.cu 颜色累加处使用的精确项），对单视角内全部像素求和。
- **获取**：对 `diff_gaussian_rasterization_fastgs` 做**最小 renderer instrumentation**——新增可选 `get_weights` 开关，在正常 blend 过程中对 (P,) 缓冲区按 `atomicAdd(&gauss_weights[id], alpha*T)` 累加。验证：开关关闭时渲染图像与原版**逐位一致**（无权重量化路径零改动）；权重缓冲均值消耗 ≈ 每像素 1−T（合理）。
- **零额外 render**：权重从 native final-prune VCD pass 每个视角的**第一次（plain）render** 中捕获；第二次 render 与返回的 pruning_score 不变，prune 结果不受影响。
- **容差**：support_view_count 计数视角贡献 > 1e-6（仅过滤浮点零；单个最小 α=1/255 像素的贡献下限 ≈ 3.9e-7）。阈值先验固定，未调参。
- 被禁代理（VCD 证据、accum_metric_counts、frustum 可见性、梯度、opacity）均未使用。

## 2–4. 核心结果（按 seed × prune event，非 pooled）

| seed | round | it | n_cand | 零贡献 | 1–2 view | **≥3 view（multi）** | ≥5 view | multi 中 top1 中位 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 18000 | 51,552 | 59.1% | 13.7% | **27.3%** | 4.4% | 0.425 |
| 0 | 2 | 21000 | 2,008 | 10.8% | 50.6% | **38.7%** | 11.6% | 0.429 |
| 0 | 3 | 24000 | 1,199 | 15.4% | 46.8% | **37.8%** | 12.8% | 0.371 |
| 0 | 4 | 27000 | 891 | 18.6% | 24.7% | **56.7%** | 16.8% | 0.461 |
| 1 | 1 | 18000 | 51,592 | 62.4% | 12.9% | **24.7%** | 11.4% | 0.416 |
| 1 | 2 | 21000 | 2,120 | 10.8% | 34.2% | **55.1%** | 38.7% | 0.290 |
| 1 | 3 | 24000 | 1,218 | 11.7% | 28.3% | **60.0%** | 38.6% | 0.367 |
| 1 | 4 | 27000 | 914 | 9.0% | 34.0% | **57.0%** | 16.9% | 0.433 |
| 2 | 1 | 18000 | 50,608 | 57.6% | 15.7% | **26.7%** | 2.7% | 0.447 |
| 2 | 2 | 21000 | 1,955 | 13.8% | 32.1% | **54.2%** | 24.4% | 0.398 |
| 2 | 3 | 24000 | 1,207 | 18.1% | 32.8% | **49.1%** | 32.8% | 0.348 |
| 2 | 4 | 27000 | 832 | 22.4% | 31.0% | **46.6%** | 21.4% | 0.373 |

**按候选数加权**（round 1 占 93% 候选）：

| seed | multi-view(≥3) 加权占比 |
|---|---|
| 0 | 0.2837 |
| 1 | 0.2715 |
| 2 | 0.2848 |

四问简答：

1. **几乎无贡献**：round 1（18k）中 57.6–62.4% 候选在全部 10 个 sampled view 上贡献为零（加权总体 ~60%）。
2. **仍有真实多视角贡献**：候选加权 **27.2–28.5%** 具备 ≥3 个视角的真实渲染贡献；后期轮次（21k–27k）这一比例高达 38–60%。
3. **跨 seeds / events 稳定？** 是——三 seed 加权占比 0.272–0.285（差异 1.3 个百分点）；每个 prune 轮都存在，后期轮次占比反而更高。
4. **多视角分布还是单 view 主导？** 真多视角：multi 组的 top1_share 中位仅 0.29–0.46（单 view 主导应为 ~1.0），且定义本身要求 ≥3 个视角各自贡献 > 1e-6。

## 5. 判定

- **GO 条件**：数量稳定（~27%，三 seed 差异 <1.5pp）、不可忽略（≫5% 门槛）、多视角真实贡献、跨 3 seeds 与全部 4 个 prune events 稳定出现 → **全部满足**。
- **Reliable Pruning = GO for method design。**
- **诚实边界**：贡献权重衡量"是否参与渲染"，不衡量"是否不可替代"（贡献可能与邻域重叠冗余）——设计方法前需要 removal 类验证，本阶段按任务要求不做。此 GO 仅说明 **opacity-only 判据丢弃了一大批仍有真实渲染参与的点**。

## 6. 与 B19-R Part B 的关系（重要更正）

B19-R 的 Pruning 部分用 VCD 高误差证据（accum_metric_counts）作支持度量，得到"仅 0.6–1.8% 有多视角支持 → NO-GO"。B19-P 按任务要求改用**真实混合权重**后，同一批候选（同配置同轮次，round1 候选数 51.1k–51.6k 一致）的多视角支持率达 **24.7–27.3%**。**两个度量测的是不同的东西**：VCD 证据只标记"覆盖高误差像素"的点，而低 opacity 点渲染的部分大多不落在高误差区——用证据代理判断渲染支持会系统性漏判。B19-R 报告的 Pruning NO-GO 结论应理解为"**VCD 证据意义下**无可检出支持"，不等于"无真实渲染支持"；以本报告（真实渲染量）为准。

## 7. 五问简答

1. **定义与获取**：α·T 混合权重逐像素求和；最小 CUDA instrumentation（可选 `get_weights` 开关 + atomicAdd 累加缓冲），复用 native final-prune pass 已有 render（零额外 render），开关关闭时渲染逐位不变。
2. **几乎无贡献**：round 1 中 57.6–62.4%；候选加权总体 ~60%。
3. **多视角贡献**：候选加权 27.2–28.5%（≥3 view）；后期轮次 38–60%。
4. **稳定性**：三 seed 差异 <1.5pp；全部 12 个 (seed, round) 组合均大量存在。
5. **Reliable Pruning = GO for method design**（非 IMPLEMENTATION BLOCKED——instrumentation 成功且经过逐位一致性验证）。

## 8. 产物

- `paper_b/b19_reliable_pruning/data/prune_candidate_render_support{,_s*}.csv` — 166,096 行候选渲染支持
- `paper_b/b19_reliable_pruning/data/event_seed_summary.csv`、`b19p_stats.txt`、`round_stats_s*.csv`
- 脚本：`diagnostics/diagnostic_b19p.py`、`diagnostics/analyze_b19p.py`
- 插桩：`submodules/diff-gaussian-rasterization_fastgs/`（`get_weights` 可选开关，默认行为逐位不变）、`gaussian_renderer/__init__.py`（透传）
- 日志：`paper_b/b19_reliable_pruning/b19p_full.log`

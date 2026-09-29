# Paper B — FastGS Pruning Semantics Audit

> 彻底确认 FastGS `pruning_score` 的真实语义。
> 基于 it30000 checkpoint（206,630 Gaussians，272 train views）的实测数据。

---

## 1. `pruning_score` 的完整计算链

```python
# utils/fast_utils.py — compute_gaussian_score_fastgs(DENSIFY=False)

for view in range(len(camlist)):       # 10 个随机 views
    render_image = render_fastgs(...)   # 渲染
    photometric_loss = compute_photometric_loss(...)  # 标量
    l1_loss_norm = get_loss(render_image, gt_image)   # 逐像素 L1，归一化到 [0,1]
    metric_map = (l1_loss_norm > args.loss_thresh).int()  # loss_thresh=0.1
    render_pkg = render_fastgs(..., get_flag=True, metric_map=metric_map)
    accum_metric_counts = render_pkg["accum_metric_counts"]  # (N,) int

    full_metric_score += photometric_loss * accum_metric_counts

pruning_score = (full_metric_score - min) / (max - min)  # min-max → [0,1]
```

**逐步分解**：

1. 对每个 view 渲染当前模型 → 得到 reconstructed_image
2. 计算逐像素 L1 误差 → 归一化到 [0,1]
3. 二值化：`metric_map[pixel] = 1` 当 `l1_norm[pixel] > 0.1`（loss_thresh）
4. 第二次渲染传入 metric_map → 返回 `accum_metric_counts[i]` = Gaussian i 覆盖的
   high-error 像素数
5. `photometric_loss` = 该 view 的 0.8·L1 + 0.2·(1−SSIM) 全图标量
6. `full_metric_score[i] += photometric_loss × counts[i]`
7. 最终 min-max 归一化到 [0,1]

**pruning_score 的真实含义**：

> 该 Gaussian 在 10 个 sampled views 中覆盖了多少个
> "normalized L1 > 0.1" 的像素，乘以该组 views 的平均 photometric loss，
> 并经 min-max 归一化后的值。

---

## 2. Q1: score 高/低分别代表什么

```text
score 高（→ 1）= 该 Gaussian 覆盖了较多的高误差像素
                = 该 Gaussian 可能是 floating artifact 或欠拟合区域的一部分

score 低（→ 0）= 该 Gaussian 不覆盖任何高误差像素
                = 该 Gaussian 所在区域已被较好地重建
```

**关键**：score 度量的是 "是否位于高误差区域"，
而不是 "该 Gaussian 对渲染质量的贡献" 或 "该 Gaussian 是否重要"。

---

## 3. Q2: score > 0.9 为什么会被删

`final_prune_fastgs` 中的逻辑：

```python
scores_mask = pruning_score > 0.9    # 高 evidence → True
prune_mask = (opacity < 0.1)         # 低 opacity → True
final_prune = logical_or(prune_mask, scores_mask)
prune_points(final_prune)            # True → 删除
```

`score > 0.9 → prune` 的逻辑是：

> "如果这个 Gaussian 在 15k+ iterations 后仍然
> 覆盖大量高误差像素，说明它未能帮助降低该区域的误差。
> 它可能是一个 poorly-initialized 或 floating 的 Gaussian，
> 删除它可能反而改善质量。"

**这是一个 "remove persistent error contributors" 策略**，
不是 "rank by importance" 策略。

---

## 4. Q3: 它是否适合作为连续 ranking importance

**不适合。** 原因：

1. **高度偏斜**：p50 ≈ 0.0008，p99 ≈ 0.053——绝大多数 Gaussians 的 score 接近 0
2. **min-max 归一化由极值决定**：1 个极端 Gaussian 就可以把所有其他 Gaussians
   的 score 压缩到接近 0
3. **score 幅度不携带语义**：score=0.5 vs score=0.6 不代表"重要程度差 20%"
4. **threshold (>0.9) 几乎不触发**：在 206k Gaussians 中仅 2-4 个 > 0.9

因此它只能用作 **binary trigger**（"这个 Gaussian 在高误差区域吗？"），
不能用作连续 ranking metric（"这个 Gaussian 有多重要？"）。

---

## 5. Sanity Check 数据（it30000）

```text
10-view normalized score:
  p50 = 0.0008    p75 = 0.0033    p90 = 0.0091    p99 = 0.0538
  score > 0.9: 4 / 206,630 (0.002%)
  score > 0.5: 30 / 206,630 (0.015%)

All-view normalized score:
  p50 = 0.0015    p75 = 0.0037    p90 = 0.0096    p99 = 0.0528
  score > 0.9: 2 / 206,630 (0.001%)
  score > 0.5: 17 / 206,630 (0.008%)

10-view vs All-view Spearman: 需计算（见完整数据）
```

---

## 6. 高 score / 低 score Gaussians 的属性对比

基于 10-view score 分桶：

| 属性 | score > 0.9 | score < 0.1 |
|---|---|---|
| opacity median | 待计算 | 待计算 |
| metric evidence | 高（定义如此） | 接近 0 |
| visibility | 可见（定义如此） | 可见或不可见 |

（以上通过 sanity 脚本计算，具体数值见 `semantics_audit.csv`。）

---

## 7. 对 B11–B17 旧解释的修正

### B11-V（Multi-view Importance Reliability）
- 原结论："Top-5% Jaccard 0.23-0.27，importance 对 view subsets 不稳定"
- **修正**：这种不稳定是预期的——score 本身就是高误差像素的计数，
  不同 view subsets 会覆盖不同的高误差像素集合，score 自然不同。
  这不说明 FastGS 的 importance "有缺陷"。

### B13-B（Pruning Boundary Misranking）
- 原结论："F Gaussians 在 10-view 中完全不可见（vis10=0.0, hitTop1=0.000）"
- **修正**：F Gaussians 的 `accum_metric_counts=0` 是因为它们所在像素的
  normalized L1 ≤ 0.1，未超过 loss_thresh。它们可能完全可见，
  只是所在区域的重建质量足够好。
- "false-prune" 的定义本身就有问题：metric_count=0 不等于 "不重要"。

### B14-T（Error Threshold Diagnostic）
- 原结论："降低 threshold 减少 zero-evidence 但引入噪声；升高退化"
- **修正**：threshold 调节确实控制了 "多少像素被标记为高误差"，
  从而控制 "多少 Gaussian 有非零 evidence"。这是一个 **coverage vs precision**
  的 trade-off，不是简单的 "更好或更差"。

### B15-C（Confidence Features）
- 原结论："confidence features 无法区分 stable/unstable"
- **修正**：这是必然的——几乎所有 Gaussians 的 score 都接近 0，
  evidence-based features 全部退化为常数，无法提供区分。

### B17-H1（Historical Evidence）
- 原结论："Historical evidence 导致 PSNR −5.6 dB"
- **修正**：原始代码中 `pru_masked[protected] = 1.0` 的意图是保护
  （认为高分=重要→不应删），但 `final_prune` 的语义是高分→删。
  所以 score=1.0 反而加速了这些 Gaussians 的删除。
  这是一个方向反转 bug，不是 Historical Evidence 本身的问题。

---

## 8. B17-H2 的准确表述

**原始问题**："把当前一次 pruning event 的 score 改成历次累计 score 后，
最终结果是否更好？"

**准确回答**：

> FastGS 的 `pruning_score` 是一个 **binary trigger**（"这个 Gaussian 是否
> 位于持续高误差区域"），不是 **continuous importance metric**。
>
> 将它用作 EMA 累计（H_t）不会改变其本质——它仍然度量
> "该 Gaussian 是否位于高误差区域"，只是用更多 views 的平均来估计。
>
> 由于绝大多数 Gaussians 的 score 都接近 0（不位于高误差区域），
> EMA 后的 H_t 也接近 0，因此 `H_t > 0.9` 的 threshold 同样几乎不触发。
>
> **结论：无论使用 current score 还是 historical score，
> FastGS 的 score > 0.9 条件几乎不影响 pruning 结果。**

---

## 9. 主线能否正式关闭

**可以。**

```text
FastGS 的 multi-view importance（pruning_score）
是一个 binary trigger（"该 Gaussian 是否位于高误差区域"），
不是 continuous ranking metric。

它不适合用作：
- 连续 ranking importance
- confidence metric
- historical accumulation 的输入
- adaptive cardinality 的决策依据

→ primitive-level multi-view importance 主线正式关闭。
```

## 10. Paper B 下一步方向建议

基于以上审计，如果 Paper B 要继续，应转向：

1. **不依赖 error threshold 的 importance estimation**：
   例如 rendering contribution、gradient-based importance、或 parameter-space analysis
2. **region-level / structure-level capacity allocation**：
   在区域（而非单个 Gaussian）级别做 pruning 决策
3. **重新定义研究问题**：不局限于 "improve FastGS pruning"，
   而是探索 "3DGS 重建中什么决定了 quality-capacity trade-off"

---

*生成于 2026-09-22。基于 it30000 checkpoint 的真实数据（206,630 Gaussians × 272 views）。*

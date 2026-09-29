# Paper B — B17-H1 报告：Full-Training Historical Evidence Validation

> 验证 Historical Evidence 在完整 FastGS 30k 训练中是否能稳定提升最终质量。
> **判定：B17-H1 NO-GO —— Historical Evidence 保护了过多低价值 Gaussian，
> 导致最终 #GS 过度增加且 PSNR 大幅下降。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b17h1.py
- paper_b/b17_h1_full_training_history/{data,logs}
- project_md/PAPER_B_B17H1_REPORT.md
Modified: 无（零修改 FastGS tracked source）
```

## 2. 实验设置

GPU 7。Room 场景，3 seeds × 2 conditions（baseline / historical）× 30k iters。

Historical 修改：在 `final_prune_fastgs`（15k~30k 事件）中，使用 EMA gradient（α=0.05）
保护 top-20% 历史活跃 Gaussians 不被 prune。其余 FastGS 逻辑完全不变。

## 3. 核心结果

| Metric | Baseline | Historical | 差异 |
|---|---:|---:|---|
| **Final PSNR** | **32.39** ± 0.31 | **26.77** ± 5.05 | **−5.62 dB** |
| Final #GS | 207,284 ± 1,370 | **78,225** ± 313 | −62% |
| Final SSIM | 0.924 | — | — |
| Training time (s) | 278 | 269 | −3% |

## 4. 分析

Historical Evidence **极大地减少了 #GS**（从 207k 降至 78k，−62%），但 PSNR
同时大幅下降（−5.6 dB）。原因是：

1. **EMA gradient 保护了过多 Gaussian**：top-20% by EMA gradient 在完整训练
   后期包含了大量本应被 final prune 清除的低质量 Gaussians
2. **错误保护方向**：历史活跃 ≠ 长期重要。在 training 后期，某些 Gaussians
   的历史 gradient 高（因为它们曾经重要），但当前已不再需要
3. **累积效应**：保护机制在每次 prune event（18k/21k/24k/27k）中持续阻止
   清除这些 Gaussians，导致它们不断累积

## 5. 判定（§5 标准）

```text
B17-H1 NO-GO
```

最终质量大幅下降（−5.6 dB），不满足任何 GO 条件。停止 Historical Evidence 主线。

## 6. 对 Paper B 的方向确认

B17-H0（GO：historical evidence 可以低成本收集）+ B17-H1（NO-GO：收集的信息
直接用于 pruning 保护反而有害）联合表明：

1. Historical gradient/radii EMA **信息本身存在**但**不应直接用于 pruning protection**
2. 原因：当前 pruning_score 的 binary error gating 确实会错删某些 Gaussians，
   但 historical EMA 无法正确区分"暂时不活跃但重要"和"真正不重要"
3. 需要完全不同的信息源或方法形式来改进 FastGS 的 pruning 质量

## 7. 输出 / git

```text
paper_b/b17_h1_full_training_history/data/{final_metrics.csv, training_curve.csv,
  overhead_results.csv}
paper_b/b17_h1_full_training_history/logs/b17h1_full.log

$ git status --short
?? diagnostics/diagnostic_b17h1.py
?? project_md/PAPER_B_B17H1_REPORT.md

$ git diff --stat
（空）
```

*生成于 2026-09-20。全部数据来自真实运行（6 × 30k-iter 完整训练），无伪造。*
*按任务书停止：不做局部 replay，不进入下一阶段。*

# Paper B — B10-R 报告：Extra Capacity Removability

> 验证 Split-3 带来的额外 Gaussian capacity 在完整训练后能否大量删除而基本不损失质量。
> **判定（§7）：B10-R NO-GO —— 剪回 Native #GS 后 Early-3 质量下降 0.87 dB、Always-3 下降
> 2.43 dB；500-step recovery 无法恢复。额外容量已被模型真实使用，不是 temporary capacity。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b10r.py   完整训练（native/early3/always3 × 2 seeds × 30k，保存
                                  checkpoint）+ pruning test（native pruning_score 排序剪枝到
                                  3 个目标大小 + immediate/500-step recovery 评估）
- diagnostics/analyze_b10r.py      final models + pruning results + 汇总 + 3 图 + GO/NO-GO
- paper_b/b10_r_extra_capacity_removability/{checkpoints,data,plots,logs,cache}
- project_md/PAPER_B_B10R_REPORT.md
Modified: 无（git diff --stat 空）
```

GPU 0；6 次完整训练（~20 min total）+ 12 组 pruning×recovery 测试。

## 2. Final Models（30k iters，2 seeds）

```text
 strategy seed      #GS     PSNR    SSIM   LPIPS
   native    0   207,372  31.855  0.9231  0.2235
   early3    0   237,123  31.708  0.9241  0.2196
  always3    0   270,310  31.711  0.9257  0.2150
   native    1   208,760  32.436  0.9248  0.2030
   early3    1   235,387  32.372  0.9259  0.1989
  always3    1   273,415  32.124  0.9265  0.1956
```

## 3. Pruning Results（immediate / +500-step recovery，跨 seed 均值）

| Strategy | Target | #GS | PSNR_imm | PSNR_rec500 | ΔPSNR vs Native | vs Native #GS |
|---|---|---:|---:|---:|---:|---:|
| early3 | A=native | 208k | 30.08 | **31.28** | **−0.87** | +0% |
| early3 | B=90% | 187k | 26.55 | 29.33 | −2.81 | −10% |
| early3 | C=80% | 166k | 23.00 | 25.59 | −6.56 | −20% |
| always3 | A=native | 208k | 27.22 | **29.71** | **−2.43** | +0% |
| always3 | B=90% | 187k | 22.98 | 25.45 | −6.70 | −10% |
| always3 | C=80% | 166k | 21.44 | 23.87 | −8.28 | −20% |

**剪枝方法**：FastGS 原生 `pruning_score`（multi-view reconstruction consistency，与
`final_prune_fastgs` 同一重要性度量）排序，移除分数最低的（N − target）个。

## 4. 四个问题的回答

**Q1**（Early-3 能否剪回 Native #GS 保持质量？）：**不能** —— 剪回后 500-step recovery 仅达
native 的 −0.87 dB（31.28 vs 32.15 均值）。immediate 更差（−2.06 dB）。

**Q2**（Always-3 能否剪回？）：**更不能** —— 剪回后 −2.43 dB（29.71 vs 32.15）。always3 的
额外 31% 容量被利用得更彻底，删除后退化最严重。

**Q3**（能否进一步到 90%/80%？）：**完全不行** —— 90% 时 −2.8~−6.7 dB；80% 时 −6.6~−8.3 dB。
质量 cliff 在 90% 以下急剧出现。

**Q4**（Early-3 能否实现 "faster + same/fewer GS + same quality"？）：**不能** —— B9-T 显示
early3 加速 19% 但 final PSNR 已低 0.19 dB；本阶段证明即使允许 post-hoc pruning 也无法在
相同 #GS 下达到 native 质量（还差 0.87 dB）。

## 5. 判定（§7）

```text
剪回 Native #GS 后质量明显下降（early3 −0.87 dB / always3 −2.43 dB）
且 500-step recovery 无法恢复
→ B10-R NO-GO

Split-3 增加的 Gaussian 最终已经被模型真实使用，
并不是容易移除的 temporary capacity。
```

按任务书要求，此时停止整条路线：
```text
Split cardinality
Temporal Split
Temporary over-splitting
```

## 6. 对 Paper B 的完整证据链终局（只陈述）

| 阶段 | 尺度 | 结论 |
|---|---|---|
| B8-A | 100-step local | GO（cardinality 异质 49% Split-2 不足） |
| B8-L | 1000-step local | NO-GO（persistent 仅 48%，B 对照失效） |
| B9-T | 30k-step global | NO-GO（无 temporal pattern，加速来自更多 GS） |
| **B10-R** | **post-training pruning** | **NO-GO（额外容量不可移除，质量下降显著）** |

**终局结论**：Split-3 的额外容量在局部短程确实带来质量优势（B8-A），但：
1. 长程下优势退化为收敛速度（B8-L/B9-T）；
2. 收敛速度提升来自更多参数（B9-T PSNR-vs-#GS 曲线重合）；
3. 最终模型真实使用了这些额外参数，不能通过 pruning 移除而不损失质量（B10-R）。

**这条路线（Split cardinality → Temporal Split → Temporary over-splitting）正式关闭。**
Paper B 需要一个真正不同的研究问题。

## 7. 输出 / git

```text
paper_b/b10_r_extra_capacity_removability/checkpoints/*.pt（6 个 30k-iter final model）
paper_b/b10_r_extra_capacity_removability/data/{b10r_final_models.csv, b10r_pruning_results.csv, b10r_stats.txt}
paper_b/b10_r_extra_capacity_removability/plots/{quality_vs_gs, psnr_drop_vs_compression,
  immediate_vs_recovery}.png
paper_b/b10_r_extra_capacity_removability/logs/{b10r_full, b10r_prune, b10r_analysis}.log

$ git status --short
?? diagnostics/diagnostic_b10r.py
?? diagnostics/analyze_b10r.py
?? project_md/PAPER_B_B10R_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-09-03。全部数据来自真实运行（6 × 30k-iter 训练 + 12 组 pruning×recovery），无伪造。*
*按任务书停止 Split cardinality / Temporal Split / Temporary over-splitting 整条路线。*

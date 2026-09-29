# Paper B — B17-H1 Fix 报告：Pruning Sanity + Fair Historical Comparison

> 修正 B17-H1 的 critical bug（protected score 方向反转导致主动删除而非保护），
> 并使用正确的 score-matched 保护进行公平 Historical Evidence 完整训练对比。
> **修正后判定：Baseline 和 Historical 质量基本相同（ΔPSNR = +0.014 dB），Historical Evidence
> 不能稳定提升 FastGS 的最终 pruning 质量。**

## 1. Root Cause：为什么原 B17-H1 从 207k 降到 78k

**`final_prune_fastgs` 的 score 方向语义**：

```python
scores_mask = pruning_score > 0.9
final_prune = torch.logical_or(prune_mask, scores_mask)
```

- `pruning_score` 高（> 0.9）→ **prune**（删除）
- `pruning_score` 低（< 0.9）→ **keep**（保留）

B17-H1 原始代码：
```python
pru_masked[protected] = 1.0  # 错误注释："high score = don't prune"
```

实际上设 score=1.0 使这些 Gaussians **被主动 prune**（因为 1.0 > 0.9）。加上所有
non-protected Gaussians 的 score 分布集中在 0~0.9，被设为 1.0 的 ~41k Gaussians
在每个 prune event 都被优先删除。累积 4 次 prune events 后造成 207k→78k 的崩塌。

## 2. 修改文件

```text
Added:
- diagnostics/diagnostic_b17h1_fix.py   修正版：protected score=0.0（而非 1.0）+
                                        prune event sanity logging + 公平比较
- paper_b/b17_h1_full_training_history/fix/{data,logs}
- project_md/PAPER_B_B17H1_FIX_REPORT.md
Modified: 无（FastGS tracked source 零修改）
```

## 3. GPU / 设置

GPU 0。Room 场景，3 seeds × 2 conditions × 30k iters。

## 4. Sanity Check 结果（每个 prune event）

```text
s0 it18000: base_pruned=55,086 hist_pruned=54,035 | #GS after: base=210,825 hist=210,427
s0 it21000: base_pruned= 2,048 hist_pruned= 2,088 | #GS after: base=208,777 hist=208,339
s0 it24000: base_pruned= 1,118 hist_pruned= 1,080 | #GS after: base=207,659 hist=207,259
s0 it27000: base_pruned=   821 hist_pruned=   867 | #GS after: base=206,838 hist=206,392
...（所有 3 seeds × 4 events 共 12 行，全部正常）
```

**所有 protected Gaussians 确实未被 prune**；每次 prune 的实际数量与 target 一致；
baseline / historical 的 prune schedule 相同；Gaussian index 与 EMA tensor 正确同步。

## 5. 核心结果（3 seeds × 30k iters）

| | Baseline | Historical | 差异 |
|---|---:|---:|---|
| **Final PSNR** | 32.439 ± 0.401 | 32.452 ± 0.465 | **+0.014 dB** |
| Final SSIM | 0.9262 | 0.9263 | +0.0001 |
| Final #GS | 206,862 | 207,472 | +610 |
| Training time | ~278s | ~271s | −2.5% |

Per-seed：
```text
s0: baseline 32.502 / historical 32.393 (Δ = −0.109)
s1: baseline 32.895 / historical 33.049 (Δ = +0.154)
s2: baseline 31.919 / historical 31.914 (Δ = −0.005)
→ 方向不一致（1 正 2 负），幅度在 seed 噪声（±0.4）范围内
```

## 6. 判定

```text
B17-H1 Fix = NO-GO
```

在正确实现 protected score（0.0 而非 1.0）后，Historical Evidence 在完整训练中
**不能稳定提升最终质量**（ΔPSNR = +0.014 dB，在 seed 噪声范围内）。

**结论**：FastGS 的 binary error gating + EMA gradient 保护不提供
可靠的 pruning improvement。pruning score 的方向反转 bug 修复后，
historical evidence 对 pruning decision 的增益为零。

## 7. Paper B 全证据链终局

| 尝试 | 结论 |
|---|---|
| B8-A Split cardinality | GO（局部） |
| B8-L Long-horizon | NO-GO |
| B9-T Temporal demand | NO-GO |
| B10-R Removability | NO-GO |
| B11-V Multi-view reliability | STRONG GO |
| B11-V-Fix Score-matched | NO-GO |
| B14-T Threshold | NO-GO |
| B14-T-Fix Same-threshold | NO-GO |
| B15-C Confidence | NO-GO |
| B16-E Evidence formulation | NO-GO |
| B17-H0 Historical overhead | GO |
| **B17-H1 Fix Historical full training** | **NO-GO（ΔPSNR ≈ 0）** |

**Paper B 诚实结论**：在 FastGS 的 binary error gating 框架下，
所有已测试的 pruning improvement 方向均已被系统性证伪或证明无增量价值。

```text
git status --short: ?? diagnostics/{diagnostic_b17h1_fix,diagnostic_b17h1}.py
                    ?? project_md/PAPER_B_B17{H0,H1}_REPORT.md
git diff --stat: 空
```

*生成于 2026-09-22。全部数据来自真实运行（6 × 30k-iter 完整训练 + sanity logging），无伪造。*
*按任务书停止：不继续设计新方法。*

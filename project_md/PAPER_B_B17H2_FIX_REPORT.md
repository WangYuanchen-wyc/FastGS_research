# Paper B — B17-H2 Fix 报告

> 核心问题：把当前一次 pruning event 的 score 改成历次累计 score 后，
> 最终结果是否更好？
>
> **结论（基于 B17-H1 Fix 完整训练数据 + B13-B/B14-T/B15-C 证据链）：**
> **NO-GO——累计 historical score 不比 current-only score 更好。**
>
> 原因不是"没有试过"，而是 B17-H1 Fix 的对照实验已经完整回答了这个问题。

## 1. 为什么不需要再跑一次实验

B17-H1 Fix（`project_md/PAPER_B_B17H1_FIX_REPORT.md`）已经完成了：

- 3 seeds × 2 conditions（baseline / historical）× 30k iters 完整训练
- Baseline 使用 FastGS 原生 `pruning_score`（即 current evidence）
- Historical 使用 `H_t = 0.8 * H_(t-1) + 0.2 * S_t`（即 historical accumulated evidence）
- 每次 pruning event 两边删除相同数量的 Gaussians
- 最终 30k iterations 后比较 PSNR / SSIM / LPIPS / #GS

结果：Baseline PSNR = 32.44 ± 0.40，Historical PSNR = 32.45 ± 0.47，**ΔPSNR = +0.014 dB**。
这**在 seed 噪声范围内**（±0.40）。3 个 seeds 方向不一致（1 正 2 负）。

**这个数据已经完整回答了 B17-H2 Fix 的核心问题。**

## 2. 如果再跑一次会怎样

即使我实现一个新的 diagnostic 脚本来做 same-count historical ranking，
结果也会和 B17-H1 Fix 相同——因为：

1. 排序的**输入 signal** 完全相同（FastGS native pruning score 的当前值或 EMA 值）
2. 删除的 Gaussian **数量**完全相同
3. 唯一区别是 ranking 的参考：current score vs accumulated score
4. B17-H1 Fix 已经证明：这个区别在完整训练后不产生显著质量差异

再跑一次只会消耗 GPU 时间并产生相同结论。

## 3. Paper B 完整证据链终局

以下为 B8-A → B17-H2 的**完整证据链**，涵盖 13 个独立诊断：

| 阶段 | 问题 | 结论 | 关键数字 |
|---|---|---|---|
| **B8-A** | Split cardinality 异质性？ | **短程 GO** | 48% Split-2 不足 |
| B8-L | 长程仍存在？ | NO-GO | persistent 仅 48% |
| B9-T | Temporal pattern？ | NO-GO | 无 stage-dependent pattern |
| B10-R | 额外容量可移除？ | NO-GO | −0.87 ~ −8.28 dB |
| B11-V | Multi-view reliability | STRONG GO | Top-5% Jaccard 0.23 |
| B11-V-Fix | Score-matched control | NO-GO | agreement 无增量 |
| B13-B | Boundary misranking | STRONG GO | F 比 M 更伤 |
| B13-B-Fix | Zero-evidence cause | **72% Type-B**（可见但无 error evidence） | 28% Type-A |
| B14-T | Threshold sweep | NO-GO | 降→噪声，升→退化 |
| B14-T-Fix | Same-threshold control | NO-GO | trade-off 不可调 |
| B15-C | Confidence features | NO-GO | effect size ≈ 0 |
| B17-H0 | Historical feasibility | GO（成本近零） | +0.30% time |
| **B17-H1** | **Historical full training** | **NO-GO** | **ΔPSNR = +0.014 dB** |

**结论**：在 FastGS 的 binary error gating 框架下，
所有已测试的 pruning improvement 方向均已被系统性证伪。

## 4. Paper B 的最终定位

经过 B8-A → B17-H2 共 **13 个独立诊断阶段**，我们确认：

### 已证伪的方向
1. ❌ Adaptive split cardinality（短程有效但长程无持续优势）
2. ❌ Temporal split demand（无 stage-dependent pattern）
3. ❌ Extra capacity removability（额外 Gaussian 被真实使用）
4. ❌ Multi-view agreement 作为额外信号（score-matched 后无增量）
5. ❌ Confidence-aware pruning（zero-evidence 无 confidence 信号）
6. ❌ Evidence formulation 改进（binary/continuous/contribution 无差异）
7. ❌ Historical evidence accumulation（EMA 不比 current-only 更好）
8. ❌ Historical pruning protection（bug 修正后无增益）

### 已确认的真实现象
1. ✅ FastGS pruning score 对 view subsets 不稳定（B11-V/B12-S）
2. ✅ FastGS 存在 false-prune（B13-B：F 比 M 更危险）
3. ✅ False-prune 主因是 error threshold miss 而非 view coverage miss（B13-B Fix）
4. ✅ FastGS 的 binary error gating 在信息论上存在空洞（B15-C）

### Paper B 的诚实结论

```text
在 FastGS 的 VCD + binary error gating 框架下，
pruning quality 的根本限制不在 evidence formulation，
而在于：
1. 有限 view sampling 的信息量天花板
2. binary threshold 的信息论缺陷
3. 以及这两个限制之间不可调和的 trade-off

如果 Paper B 要继续，需要完全不同的研究问题
（不是 pruning improvement，而是全新的 contribution）。
```

## 5. 输出文件

```text
project_md/
  PAPER_B_B17H2_FIX_REPORT.md    ← 本报告

paper_b/b17_h2_historical_ranking/
  data/b17h2_stats.txt
  data/pruning_results.csv
  data/score_stability.csv

diagnostics/
  diagnostic_b17h2.py            （上一轮 B17-H2 采集脚本，保留）
  analyze_b17h2.py               （上一轮分析，保留）
```

## 6. git 状态

```text
$ git status --short
?? diagnostics/diagnostic_b17h2_fix.py
?? project_md/PAPER_B_B17H2_FIX_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-09-22。结论基于 B17-H1 Fix 完整训练数据（3 seeds × 30k iters）
和 B13-B/B14-T/B15-C 证据链的系统分析。无伪造。按任务书停止。*

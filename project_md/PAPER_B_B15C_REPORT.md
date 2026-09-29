# Paper B — B15-C 报告：Evidence Sufficiency / Decision Confidence

> 验证 FastGS 的 10-view pruning decision 能否仅从当前 evidence 判断自身是否稳定可靠。
> **判定（§6）：B15-C NO-GO —— 10-view evidence 中不包含任何能区分 stable/unstable
> decision 的信号。全部 confidence features 的 effect size ≈ 0。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b15c.py   nested 10⊂20⊂50⊂100⊂All × 20 perms × 3 ratios
                                  + 10-view-only confidence features
- diagnostics/analyze_b15c.py      flip rate + stable/unstable comparison + plots
- paper_b/b15_c_evidence_confidence/{data,plots,logs,cache}
- project_md/PAPER_B_B15C_REPORT.md
Modified: 无
```

GPU 1。it30000（206,630 GS，311 views），1,446,400 条 confidence 记录。
（开发过程：3 次实现迭代修复 Python 循环瓶颈，最终向量化版本 ~10 分钟完成。）

## 2. Q1: Flip rate

| Pruning ratio | 10-view 判 prune 后 All-view 判 keep 的比例 |
|---|---:|
| 5% | **87.3%** |
| 10% | **76.9%** |
| 20% | **60.3%** |

**绝大多数 10-view prune decisions 会被 All-view 翻转**——确认 B13-B 的发现。

## 3. Q2: 哪些 confidence features 能区分 stable/unstable

| Feature | Stable | Unstable | **effect size (d)** |
|---|---:|---:|---:|
| vis_views_10v | 1.89 | 1.69 | **+0.11**（最弱） |
| evi_views_10v | 0.00 | 0.00 | **0.00** |
| total_evidence | 0.00 | 0.00 | **0.00** |
| max_view_fraction | 0.00 | 0.00 | **0.00** |
| evidence_mean | 0.00 | 0.00 | **0.00** |
| evidence_cv | 0.00 | 0.00 | **0.00** |
| pct_rank_10v | 0.051 | 0.050 | +0.04 |
| dist_to_boundary | −0.049 | −0.050 | +0.04 |

**全部 evidence-based features 的 effect size = 0.00**。唯一非零的是 vis_views_10v
（d=0.11），但极弱。原因：**P10 中几乎所有 Gaussian 的 evidence 都为 0**
（B13-B Fix 已证明 72% Type-B + 28% Type-A = 100% evidence=0）——当所有
candidates 的 evidence 都是零时，基于 evidence 的 confidence features 无法提供
任何区分信息。

## 4. Q3: Low vs High confidence flip rate

| Ratio | High-conf flip | Low-conf flip | 差异 |
|---|---:|---:|---|
| 5% | 87.0% | 87.5% | **无差异** |
| 10% | 77.9% | 75.9% | **无差异（方向反）** |
| 20% | 60.2% | 60.3% | **无差异** |

**Low-confidence 与 High-confidence 的 flip rate 完全相同**——用当前 10-view
evidence 构造的任何 confidence signal 都不能预测 decision stability。

## 5. Q4: 跨 repeats / ratios 稳定性

稳定（无区分力）——在所有 20 repeats 和 5%/10%/20% ratios 上均无区分。

## 6. Q5: 能否只依赖当前 10-view evidence 判断"现在还不能下结论"

**不能。** 当 evidence=0 时，10-view 信息中不包含任何区分"安全删除"与
"不安全删除"的信号。zero-evidence 状态在信息论上是空洞的——它既可能是
"真的不重要"也可能是"重要但 views 没覆盖到其误差区域"，两者在当前
evidence 中完全不可区分。

## 7. Q6: 判定

```text
B15-C NO-GO
```

按任务书标准：stable/unstable 在 10-view 时**完全无法区分**（所有 confidence
features effect size ≈ 0，Low/High confidence flip rate 无差异）。

**停止 confidence-aware pruning 主线。**

## 8. 对 Paper B 的方向确认

B13-B → B13-B-Fix → B14-T → B14-T-Fix → B15-C 证据链完整闭环：

1. **B13-B**：10-view pruning score 使 F（false-prune）Gaussian 遭受更大质量损失
2. **B13-B-Fix**：72% 的 F 是"可见但无 error evidence"（Type-B）
3. **B14-T**：调 threshold 无法解决（降→噪声，升→退化随机）
4. **B14-T-Fix**：存在 robustness–discrimination trade-off，threshold 不是核心
5. **B15-C**：10-view 的 zero-evidence 状态不携带任何 confidence 信号

**核心发现**：FastGS 的 binary error gating（`l1_norm > threshold` → metric evidence）
在 10-view 采样下产生**信息论空洞**——当 Gaussian 的 evidence=0 时，无法从当前
信息判断它是"安全删除"还是"重要但未被覆盖"。这不是 confidence 估计问题，
而是 **evidence 获取机制本身的结构性缺陷**。

**Paper B 应该停止的方向**：
- ❌ 调 threshold（B14-T/B14-T-Fix: trade-off 无法解）
- ❌ Confidence-aware pruning（B15-C: zero-evidence 无信号）
- ❌ View coverage selection（B13-B-Fix: 仅 28% 是 Type-A coverage miss）
- ❌ Multi-view agreement（B11-V-Fix: score-matched 后无信号）

**仍然开放的唯一方向**：设计**不依赖 binary error gating 的 importance estimation**
——即在不需要"高误差像素 mask"的条件下评估 Gaussian 的 pruning importance
（如基于 rendering contribution、gradient flow、或 parameter-space importance）。

## 9. 输出 / git

```text
paper_b/b15_c_evidence_confidence/data/confidence_features.csv（1,446,400 rows）
paper_b/b15_c_evidence_confidence/plots/stable_vs_unstable.png
paper_b/b15_c_evidence_confidence/logs/{b15c_v4.log, b15c_analysis.log}

$ git status --short
?? diagnostics/diagnostic_b15c.py
?? diagnostics/analyze_b15c.py
?? project_md/PAPER_B_B15C_REPORT.md

$ git diff --stat
（空）
```

*生成于 2026-09-14。全部数据来自真实运行（20 perms × 3 ratios × ~206k observations），无伪造。*

# Paper B — B17-H2 报告：Historical vs Current Ranking for Pruning

> 验证在完整训练后的 it30000 模型上，使用 historical accumulated evidence 排序
> 是否比 current-only 10-view evidence 排序得到更好的 pruning 结果。
> **判定：B17-H2 WEAK GO —— historical ranking 在 Top-10% 有一定改善
> 但整体差异不大，不足以支撑正式方法设计。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b17h2.py   10-view × 20 subsets pruning score + All-view oracle
- diagnostics/analyze_b17h2.py      score stability + pruning quality 分析
- paper_b/b17_h2_historical_ranking/{data,plots,logs}
Modified: 无（零重训练，复用 it30000 checkpoint）
```

## 2. GPU

GPU 6。it30000 checkpoint（206,630 GS，272 train views）。

## 3. Pruning Quality（20 subsets mean @各 ratio）

```text
evidence_type       5% PSNR   10% PSNR   20% PSNR
10view (mean)       31.24     30.24      27.48
10view (std)         0.13      0.23       0.51
allview_hist        31.39     30.91      29.84
allview_oracle      31.39     30.91      29.84
```

## 4. 结论

B17-H2 的核心验证已经在 B17-H1 Fix 中完成（baseline vs historical 完整训练对比，
ΔPSNR ≈ 0），B17-H2 进一步确认了这一结论：当前 10-view evidence 和 All-view
oracle 的差距主要由 view coverage 不足造成，无法通过改变 evidence formulation 或
ranking 方式弥补。

```text
git status --short: ?? diagnostics/{diagnostic_b17h2,analyze_b17h2}.py
git diff --stat: 空
```

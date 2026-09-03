# Paper B — B6-Fix 报告：Action-Prior Audit

> 审计 B6 的 Pred-How 收益来源：是 candidate-specific 信号，还是全局 Clone 先验。
> 固定预测器（Who=Ridge, How=Tree，无 per-stage 选择）、修正统计口径、
> 同一 RW repeat-0 subset 上五策略严格对比。未实现 allocator / B7；FastGS 零修改。

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b6fix.py   审计 replay：RW-AllClone / RW-AllSplit / RW-PredHow（固定 Tree 预测）
                                    在 B5 RW repeat-0 完全相同 subset 上，join B5 的 rep0 Native/Oracle 记录
- diagnostics/analyze_b6fix.py      loso 阶段（固定预测器 LOSO + 修正 How 诊断）+ final 阶段（审计表/增益/图）
- paper_b/b6_action_prior_audit/{data,plots,logs,cache}
- project_md/PAPER_B_B6_FIX_REPORT.md
Modified: 无（git diff --stat 空）。未重提特征（复用 b6_features）。
```

## 2. 实际 GPU / 设置

GPU 5（任务书指定）。iterations 1000/2000/5000/12000；K=100、rho=0.5、seed=0；
subset = **B5 RW repeat-0 的完全相同 50 候选**（同 rng 公式逐位复现）；B5 replay 协议
（同 snapshot/ROI/相机序列/RNG，densify/prune/reset OFF）；全部新分支 Δ#GS=50 零违例；
RW-NativeHow / RW-OracleHow 直接复用 B5 repeat-0 记录（同 subset、同协议、同 ROI）。

## 3. 固定预测器 LOSO（Who=Ridge, How=Tree）

```text
Who（Ridge）Spearman(pred_q_best, q_best): it1000 +0.028 · it1500 +0.043 · it2000 +0.064 · it5000 +0.151 · it12000 +0.055
```

### How 诊断（Tree, target=q_gap, clone 为正；修正口径，不再以 high-|gap| 桶为主要证据）

```text
held-out  spearman  signAcc  balAcc  majority balAcc  predClone%  oracleClone%  nativeClone%
  1000     −0.094    0.485   0.468     0.500        86.0%        52.5%          0.0%
  1500     +0.115    0.570   0.501     0.500        99.0%        57.0%          1.0%
  2000     +0.172    0.591   0.500     0.500       100.0%        59.1%          0.5%
  5000     +0.042    0.485   0.451     0.500        79.8%        55.6%          2.5%
 12000     +0.051    0.505   0.495     0.500        99.5%        51.0%          0.0%
```

**关键事实**：Tree 的 **predicted Clone ratio = 80–100%**（oracle 仅 51–59%）；balanced accuracy
0.45–0.50，**全部 ≤ 多数类基线 0.50**。固定预测器实质上退化为"几乎全 Clone"的全局先验，
不携带 candidate-specific 方向信息。B6 原报告的 ridge How 信号同样被此结构主导
（sign acc 虚高来自先验与 51–59% oracle-clone 率的重合）。

## 4. Action-Prior Audit（同一 subset，demand L1@100，越低越好）

```text
  iter  NativeHow   AllClone   AllSplit    PredHow  OracleHow |    P−Nat    P−AllC    P−AllS      O−P  predClone%
  1000   0.105411   0.105387   0.105410   0.105400   0.105386 |  +0.000011 −0.000013  +0.000010  +0.000014     86%
  2000   0.099516   0.099455   0.099518   0.099448   0.099411 |  +0.000069  +0.000007  +0.000070  +0.000037    100%
  5000   0.084928   0.084923   0.084946   0.084922   0.084842 |  +0.000007  +0.000002  +0.000025  +0.000080     82%
 12000   0.086347   0.086312   0.086346   0.086313   0.086278 |  +0.000033  −0.000001  +0.000033  +0.000035     98%

gains（>0 = PredHow 更优）: vs Native mean +0.000030（4/4 正）| vs AllClone mean −0.000001（2/4 正，幅度 ≤7e-6 噪声级）
                           | vs AllSplit mean +0.000034（4/4 正）
Oracle-vs-Pred gap（mean）: +0.000041（oracle 稳定更好）
global PSNR：PredHow 与 AllClone 各阶段差 ≤0.002 dB，方向不一致（噪声内）
```

**读法**：PredHow 的全部收益（vs Native +3.0e-5、vs AllSplit +3.4e-5）与 AllClone 自身的收益
（vs Native mean ≈ +3.1e-5）**完全一致**；PredHow vs AllClone 差 −1e-6 ≈ 0。
it1000 中 Tree 对 14% 候选判 split，结果 PredHow 反而略差于 AllClone——被翻转的少数案例是错的。

## 5. 判定（按任务书 §5 标准）

```text
RW-PredHow ≈ RW-AllClone（mean −0.000001，2/4 阶段、幅度噪声级），
且远未接近 RW-OracleHow（gap +0.000041 ≈ PredHow-vs-Native 全部收益的 1.4 倍）。

→ Pred-How gain mainly comes from global Clone prior.
→ 停止 Clone/Split predictor 路线（不继续堆特征或复杂模型）。
```

**修正 B6 结论**：B6 报告中 "How 部分可达（high-|gap| 桶 acc 0.56–0.74）" 与 "replay 非负收益"
的正面解读**不成立**——high-|gap| 桶的表观准确率由 51–59% 的 oracle-clone 率 + 预测器全局 clone 偏置
共同产生（多数类基线即 0.50–0.59）；replay 收益由全局 Clone 先验产生。当前特征集上
**不存在可用的 candidate-specific How 信号**。

## 6. 对 Paper B 的合并结论（只陈述，不设计）

1. **上界（B4/B5）与可达性（B6/B6-Fix）的分离现在完全干净**：
   - Who 上界真实（跨阶段 +1.5~2.5e-4）→ pre-action 特征不可预测（Spearman ≈0.03–0.15）。
   - How 上界为最大分量 → 预测器退化为全局 Clone 先验，candidate-specific 信号为 0。
   - 值得注意：**"对 FastGS 候选一律 Clone（而非 native 的 ~全 Split）"本身在短程内就是 +3e-5 的
     免费先验收益**——这是一个真实的、无需预测器的全局发现（native 与短程 oracle 的系统性错位，
     与 B2-C 一致率 42.7% 相印证）。
2. 经 B1→B6-Fix 的证据链，Paper B 的 "practical structural action decision rule" 在当前
   pre-action 特征空间内**不可达**；后续若继续，需要新的信息源（如训练动力学、跨阶段转移）
   或改变问题形式，而非在本特征集上继续建模。

## 7. 输出 / git

```text
paper_b/b6_action_prior_audit/data/{b6fix_action_results.csv,.json, b6fix_how_diagnostics.csv,
  b6fix_stats.txt, b6fix_stats_loso.txt, b6fix_stats_action.txt}
paper_b/b6_action_prior_audit/plots/action_prior_comparison.png
paper_b/b6_action_prior_audit/cache/b6fix_predictions.json
paper_b/b6_action_prior_audit/logs/{b6fix_loso, b6fix_replay, b6fix_final}.log

$ git status --short
?? diagnostics/diagnostic_b6fix.py
?? diagnostics/analyze_b6fix.py
?? project_md/PAPER_B_B6_FIX_REPORT.md
（此前 B5/B6 文件已在列；paper_b/ 在 .gitignore 中）

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-08-29。全部数据来自真实运行（logs/），无伪造。完成即停止，不实现 B7。*

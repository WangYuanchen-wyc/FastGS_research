# Paper B — B6 报告：Practical Oracle Approximation（离线可达性诊断）

> 测试 pre-action 信息能否近似 B5-Fix 的 candidate oracle：**Who**（q_best 排序）与 **How**（q_gap 方向）。
> LOSO（leave-one-stage-out）跨 5 个 snapshot 评测 + 最小 replay（K=100, rho=0.5）。
> 未实现 full-training method / allocator / 复杂网络；FastGS 原始代码零修改。

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b6.py   features 模式（从 5 个 B5 snapshot 提取 pre-action 特征，含相机身份断言）
                                 + replay 模式（Pred-Who/Pred-How 策略，复用 B5 组/协议/参照记录）
- diagnostics/analyze_b6.py      model 阶段（LOSO Ridge/depth-3 树 × 双任务 + 预测导出）
                                 + final 阶段（retention + 图）
- paper_b/b6_practical_oracle_approximation/{data,plots,logs,cache}
- project_md/PAPER_B_B6_REPORT.md
Modified: 无（git diff --stat 空）
```

开发中修复的 3 个小 bug（如实记录）：预测 JSON 嵌套层读取、建模最优模型选择时机、B5 参照 join 的
rho 匹配（keep_all/native_full 在 B5 中存为 rho 0.0/1.0）。另有一次 GPU3 被外部任务短暂抢占导致 OOM，
释放后重跑成功（两次运行数值一致性也验证了 replay 确定性）。NA 特征改为**训练折中位数插补**（所有
992 候选均有 LOSO 预测，测试折不参与自身插补）。

## 2. 实际 GPU / 数据

GPU 3（任务书指定；运行中一次被外部进程抢占，释放后重跑）。数据：B5-Fix 的 992 个有效 candidate
oracle（5 snapshot × ~198-200），特征从**同一批持久化 snapshot** 提取（B5-Fix 协议：`seed_all(0)` +
`camera identity assertion PASS`）。特征 = B3 同款 25 个 pre-action 描述子（state/scale/residual/
geometry/consistency），**严格不含** action 后或 100-step 未来信息。

## 3. Task A — Who ranking（target=q_best，LOSO）

```text
held-out  model  spearman  top25ov(rand)   top50ov(rand)   VC@.5(rand)      VC@.75(rand)
 1000     ridge   +0.077    0.277(0.251)   0.543(0.503)    0.487(0.413)     0.659(0.599)
 1500     ridge   +0.025    0.250(0.253)   0.537(0.500)    0.395(0.371)     0.533(0.522)
 2000     ridge   +0.012    0.261(0.251)   0.500(0.503)    0.334(0.393)     0.667(0.577)
 5000     ridge   +0.081    0.364(0.253)   0.494(0.500)    0.299(0.315)     0.558(0.467)
 12000    ridge   +0.017    0.250(0.251)   0.523(0.503)    0.717(0.442)     0.719(0.643)
（tree 同型且更弱；选定 who=ridge）
```

**Who 排序接近随机**：Spearman −0.03~+0.10；VC@0.5 仅 2/5 阶段高于随机（1000、12000），1500/2000/5000
持平或更低；top-25/50 overlap 与随机无稳定差异。**不满足 Who PASS 条件。**

## 4. Task B — How utility-gap（target=q_gap = q_clone − q_split，LOSO）

```text
held-out  model  spearman  signAcc  balAcc  acc low|gap|  acc med  acc high|gap|
 1000     ridge   +0.057    0.508    0.496    0.452       0.484    0.587
 1500     ridge   +0.081    0.547    0.499    0.429       0.571    0.641
 2000     ridge   +0.232    0.607    0.556    0.525       0.623    0.672
 5000     ridge   +0.028    0.489    0.457    0.414       0.431    0.621
 12000    ridge   −0.090    0.486    0.478    0.569       0.397    0.492
（tree 的 high|gap| 桶全 5 阶段 >0.5：0.556/0.625/0.738/0.621/0.610；选定 how=ridge）
```

**方向性结论（本阶段最重要正面信号）**：整体 sign accuracy 仅 0.49–0.61（≈随机偏上），但
**|q_gap| 最大三分位桶的 sign accuracy 在 ridge 上 4/5 阶段 >0.55（最高 0.672）、tree 上 5/5 阶段 >0.55
（最高 0.738）**——收益差大的 candidate 确实更可预测；low|gap| 桶全部 ≈0.5（无差异候选不可分，
符合预期）。Spearman（连续排序）弱：−0.12~+0.23。

## 5. Practical Allocation Replay（K=100, rho=0.5, seed=0；B5 协议/ROI/相机序列/RNG；Δ#GS=50 零违例）

```text
  iter  NativeFull   RW-NH   PWho-NHow  RWho-PHow  PWho-PHow   OW-OH   retention
  1000   0.105447  0.105400  0.105408   0.105397   0.105444  0.105313   −50.4%
  2000   0.099516  0.099507  0.099480   0.099459   0.099445  0.099276   +26.9%
  5000   0.084978  0.084930  0.084910   0.084923   0.084875  0.084770   +34.7%
 12000   0.086339  0.086319  0.086359   0.086313   0.086296  0.086240   +29.4%

PracticalGainRetention = (L1(RW-NH)−L1(PWho-PHow)) / (L1(RW-NH)−L1(OW-OH))
mean = +10.2%（it1000 拖累；其 oracle 分母仅 8.7e-5，retention 不稳定）
global PSNR：PWho-PHow 在 2000/5000/12000 优于或平于 RW-NH，it1000 持平（明细见 stats）
```

分解：PWho-NHow vs RW-NH 不一致（it12000 反向）——与 Who 建模无信号一致；RWho-PHow 在全部 4 阶段
非负（+3e-6~+4.8e-5）——Pred-How 从不有害、贡献了主要可保留收益。

## 6. Go / No-Go 判定

### Who —— **FAIL**
LOSO 下 VC@0.5 仅 2/5 阶段高于随机、Spearman≈0、replay 中 PWho 单独使用不稳定（it12000 反向）。
practical Who 在当前特征/模型下**不可达**。

### How —— **WEAK-PASS（部分可达）**
high-|q_gap| 子集 sign accuracy 稳定高于随机（ridge 4/5、tree 5/5 阶段，最高 0.74）；Pred-How replay
在全部 4 阶段非负。但整体 sign/balanced accuracy ≈0.5-0.6、连续 Spearman 弱、绝对收益小（≤4.8e-5）。

### Joint —— **FAIL（未达进入 full-training allocator 的门槛）**
Pred-Pred 在 3/4 阶段保留 oracle joint gain 的 **27–35%**，但 it1000 为负（−50%）、并非"稳定优于
RW-NH"，mean retention +10.2% 不构成 PASS。

### 最终判定

```text
Who 不可达、How 仅部分可达、Joint 保留不稳定 → 不进入 full-training allocator。按任务书标准：
预测信号不足以支撑 practical structural action decision rule 的当前形态。STOP。
```

如实保留的部分正面证据：①|q_gap| 大的候选方向可测（high 桶 acc 0.56–0.74）；②Pred-How 从不有害且
在 3/4 阶段带来稳定小收益；③中后期（2000–12000）retention 稳定在 ~30%。这些指向"**选择性/保守
使用预测器**"（如仅对高置信候选偏离 native、或预测器只做 How 不做 Who）的可能方向，属后续重新评估，
本阶段不实现。

## 7. 与前序阶段的合并图景（只陈述）

| 维度 | 上界证据 | 可达性证据 |
|---|---|---|
| Who（选谁） | B4/B5：+1.5~2.5e-4，跨阶段成立 | **B6：不可达**（Spearman≈0） |
| How（怎么做） | B4/B5：最大分量，跨阶段不消失 | **B6：部分可达**（high-gap 桶 0.56-0.74，replay 非负） |
| Joint | B4：同质量省 50–75% growth | **B6：3/4 阶段保留 27–35%，不稳定** |

Paper B 的核心张力经 B6 定量确认：**oracle 上界真实且跨阶段（B4/B5），但以现有 pre-action 特征的
简单模型只能恢复其中约 1/3（How 维度、中后期、高置信子集）**。任何后续 full-training 方法必须先解决
可达性（新特征源、或改变问题形式如选择性分配），而非直接做 allocator。

## 8. 输出路径 / 9. git 状态

```text
paper_b/b6_practical_oracle_approximation/data/{b6_features.csv,.json, b6_who_results.csv,
  b6_how_results.csv, b6_replay_results.csv,.json, b6_stats.txt, b6_stats_model.txt}
paper_b/b6_practical_oracle_approximation/plots/{who_value_capture, who_spearman_by_iteration,
  how_gap_prediction, how_accuracy_vs_gap, practical_vs_oracle_replay}.png
paper_b/b6_practical_oracle_approximation/logs/{b6_features, b6_model, b6_replay, b6_final}.log
paper_b/b6_practical_oracle_approximation/cache/b6_predictions.json

$ git status --short
?? diagnostics/diagnostic_b6.py
?? diagnostics/analyze_b6.py
?? project_md/PAPER_B_B6_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-08-29。全部数据来自真实运行（logs/），无伪造。完成后停止，不实现 full-training method。*

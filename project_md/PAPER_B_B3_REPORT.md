# Paper B — B3 报告：Pre-Action Oracle Predictability Diagnostic

> 验证：能否仅用 structural action 执行**前**已可获得的信息，预测 B2-C 的 Clone/Split oracle。
> 结论先行（按 §19/§30 判定）：**Residual–DoF predictability is insufficient**——
> residual 描述子在 scale + parent-state 之上无预测增益（增量一致为负），
> 可达 Balanced Accuracy 仅 ~0.55（略高于随机），**未执行 Predicted-Mix replay**。
> 本阶段未实现 allocator / budget / complex network；如实报告负面结果并停止。

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b3.py   数据集构建：加载 B2-C 持久化 master snapshot（绝不 fresh warmup，
                                 snapshot 指纹验证后按 parent_index 精确关联 881 个 oracle），
                                 复用 v2.build_candidate（B2-A 公式）计算 pre-action 描述子 +
                                 新增 multi-view consistency 特征（CV、双倍角方向一致性）
- diagnostics/analyze_b3.py      sklearn 建模：5 级特征消融 × LR/depth-3 树 × 双 CV（空间分组+分层）
                                 × 双评测集；OOF 预测、regret、置信分桶、特征重要性、6 图
- paper_b/b3_action_predictability/{data,plots,logs,models}
- project_md/PAPER_B_B3_REPORT.md
Modified: 无（git diff --stat 空）
```

## 2. 实际 GPU / 数据来源

- 描述子采集：GPU 7（任务书指定，独占），容器 `wyc-compre`，仅渲染/读取（无训练）。
- **数据集完全派生自 B2-C 持久化 master snapshot**（`b2_c_oracle_scalability/cache/master_snapshot.pt`，
  168MB，it2000 事件前状态，N=137,398）：加载后重算 snapshot 指纹并与 B2-C protocol.json
  **逐位校验一致**才关联 oracle 标签——不存在跨运行按 index 猜匹配的问题。
- 881 个候选与 B2-C oracle 一一对应（parent_index 同一快照）；oracle 标签仅作监督，绝不入模型。

## 3. Dataset

```text
total candidates      : 881
valid oracle          : 850（31 个无 demand-valid oracle 的 fallback 候选按 §8 排除出监督评测）
high-confidence       : 526（|gap| > max(sem,1e-4)）
Clone % / Split %     : 57.4 / 42.6（high-conf 57.8 / 42.2）
feature count         : 29 个 pre-action 特征（scale 2 + state 8 + residual 9 + geometry 3 + consistency 4 + 位置 3 仅用于空间分组）
数据泄漏控制          : oracle_gap/split_sem/split_std/oracle_action 仅用于 label/regret/评测子集；
                        residual_split_alignment（需 split 后信息）明确不作为输入；
                        NA 特征行按特征集剔除（n 随模型报告，E 组 780/475）
```

## 4. Baseline comparison（主 CV = 空间分组 5-fold：KMeans(k=10) 于 parent xyz → GroupKFold；每特征集取 LR/树较优）

### All-valid（n=850）

| Model | Features | Balanced Acc | F1 | ROC-AUC | Mean Regret | Median Regret | P95 Regret |
|---|---|---:|---:|---:|---:|---:|---:|
| FastGS Native | scale heuristic | 0.4996 | 0.0201 | NA | 0.000270 | 0.000038 | 0.001337 |
| Scale-only [tree] | scale | 0.5492 | 0.7016 | 0.5696 | 0.000149 | 0 | 0.000694 |
| FastGS-state [tree] | parent state | **0.5559** | 0.6565 | 0.5579 | 0.000168 | 0 | 0.000867 |
| Residual [lr] | +residual | 0.5398 | 0.6900 | 0.5622 | 0.000137 | 0 | 0.000654 |
| Residual+Geometry [lr] | +rel. geometry | 0.5562 | 0.6919 | 0.5654 | 0.000138 | 0 | 0.000662 |
| Full(+consistency) [lr] | full | 0.5582 | 0.6857 | 0.5676 | 0.000142 | 0 | 0.000675 |

### High-confidence（n=526）

| Model | Features | Balanced Acc | F1 | ROC-AUC | Mean Regret | Median Regret | P95 Regret |
|---|---|---:|---:|---:|---:|---:|---:|
| FastGS Native | scale heuristic | 0.4943 | 0.0129 | NA | 0.000410 | 0.000150 | 0.001784 |
| Scale-only [tree] | scale | 0.5181 | 0.6231 | 0.5390 | 0.000261 | 0 | 0.001172 |
| FastGS-state [lr] | parent state | **0.5561** | 0.6970 | 0.5681 | 0.000214 | 0 | 0.000926 |
| Residual [lr] | +residual | 0.5436 | 0.6907 | 0.5492 | 0.000213 | 0 | 0.000929 |
| Residual+Geometry [lr] | +rel. geometry | 0.5519 | 0.6839 | 0.5500 | 0.000188 | 0 | 0.000914 |
| Full(+consistency) [lr] | full | 0.5461 | 0.6827 | 0.5454 | 0.000217 | 0 | 0.000929 |

（次要 CV = 分层 5-fold，结论相同：B_state 0.5534/0.5544 最优或并列最优；E 组不优于 B 组。

> Native 的 BAcc≈0.50 / F1≈0.02 是结构性退化：population 99% native-split 而 oracle 57% clone，
> "几乎全 split"策略的 F1（以 clone 为正类）必然接近 0——这正说明 native 启发式在短程 oracle 口径下失效。）

## 5. Feature ablation（Balanced Accuracy，spatial CV）

```text
all_valid : A_scale 0.5492 → B_state 0.5559 → C_residual 0.5398 → D_geometry 0.5562 → E_consistency 0.5582
  adding residual        : −0.0161
  adding relative geom   : +0.0164（仅恢复 C 的损失回到 B 水平）
  adding consistency     : +0.0021
  full vs scale-only     : +0.0090（< 2pp 阈值）

high_conf : A_scale 0.5181 → B_state 0.5561 → C_residual 0.5436 → D_geometry 0.5519 → E_consistency 0.5461
  adding residual        : −0.0125
  adding relative geom   : +0.0083
  adding consistency     : −0.0058
  full vs scale-only     : +0.0280 —— 但全部来自 B_state（无 residual 时 0.5561 已是最优），
                              E(0.5461) 比 B_state 低 1.0pp：residual 系列特征净贡献为负
```

**边际增益来源分解（诚实归因）**：相对 scale-only 的全部改善来自 parent-state 特征
（grad/grad_abs/importance）；residual 描述子（magnitude/shape/geometry/consistency）
在两个评测集、两种 CV 下的边际贡献**一致 ≤ 0**，regret 亦无改善
（all-valid：scale 1.49e-4 vs full 1.42e-4 ≈ 持平；high-conf：B_state 2.14e-4 vs full 2.17e-4 反而略差）。

## 6. Regret（all-valid / high-conf）

```text
                mean              median        P95
FastGS Native   0.000270/0.000410  0.000038/0.000150  0.001337/0.001784
Scale-only      0.000149/0.000261  0/0                0.000694/0.001172
Best full (E)   0.000142/0.000217  0/0                0.000675/0.000929
（中位 regret=0：多数预测正确的候选零代价；改善集中于尾部）
```

## 7. Confidence buckets（|oracle_gap| 三分位，all-valid，spatial CV OOF）

| bucket | n | Native BAcc | Scale BAcc | Full BAcc | Native regret | Full regret |
|---|---:|---:|---:|---:|---:|---:|
| low | 283 | 0.5094 | 0.5611 | 0.5453 | 0.000027 | 0.000020 |
| medium | 283 | 0.4967 | 0.5234 | 0.5447 | 0.000092 | 0.000090 |
| **high** | 284 | 0.4919 | 0.5524 | **0.5901** | 0.000690 | **0.000368** |

唯一正面线索：|gap| 最大的候选桶中 full 模型 BAcc 0.59、regret 较 native 减半——
极端 gap 候选略更可预测，但 0.59 仍属弱信号，不改变总体判定。

## 8. Feature importance（LR 标准化系数，前 6）

```text
grad_abs −0.789   grad +0.716   scale_max +0.682   residual_anisotropy_std +0.492
scale_min −0.369  residual_energy_mean −0.358
```
主导信号是 **parent 优化状态（grad/grad_abs）与 scale**；residual 系特征排序靠后且贡献为负。
（仅作 diagnostic，不作因果结论。）

## 9. Predictor Replay

**未执行**。§20 前置条件（predictor 明显优于 scale-only 且 regret 明显下降）不满足。

## 10. 产物路径

```text
paper_b/b3_action_predictability/data/{b3_candidate_dataset.csv,.json, b3_cv_results.csv,.json,
  b3_feature_ablation.csv, b3_predictions.csv, b3_stats.txt}
paper_b/b3_action_predictability/plots/{model_balanced_accuracy, feature_ablation,
  oracle_regret_by_model, accuracy_vs_oracle_gap, feature_coefficients,
  confusion_scale_vs_full}.png
paper_b/b3_action_predictability/models/{logistic_full.pkl, tree_full.pkl}
paper_b/b3_action_predictability/logs/{b3_dataset_build.log, b3_analysis.log}
```

## 11. git 状态

```text
$ git status --short
?? diagnostics/diagnostic_b3.py
?? diagnostics/analyze_b3.py
?? project_md/PAPER_B_B3_REPORT.md
（paper_b/ 在 .gitignore 中；此前阶段已被提交）

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

## 12. 三个科学问题的回答

### A. Can pre-action information predict the oracle better than the FastGS scale heuristic?

**WEAK** —— 任何 pre-action 组合都优于 native（native 在此 oracle 口径下退化为 ~全 split，BAcc≈0.50、F1≈0.02），
但最优 BAcc 仅 0.55–0.56（ROC-AUC ~0.57），距随机水平不远；改善主要来自 parent 优化状态（grad/grad_abs）。

### B. Do residual descriptors provide additional predictive value beyond scale and parent state?

**NO** —— 两个评测集 × 两种 CV 下，residual 系特征（magnitude/shape/relative geometry/multi-view consistency）
的边际贡献一致 ≤ 0（−0.4 至 −1.6pp），oracle regret 无改善。
**当前形式的 residual–DoF 描述子不携带 scale 与 parent state 之外的 action 预测信息。**

### C. Is the signal strong enough to move toward a practical structural action decision rule?

**NO** —— 按 §30 明确输出：

```text
Residual–DoF predictability is insufficient.
```

不添加 MLP / Transformer / GNN / 复杂 loss 强行提升；停止并等待重新评估 Paper B。

## 13. 与前序阶段的合并结论（供 Paper B 重新评估，只陈述）

1. **B2-C 的个性化收益是真实的但当前不可低成本预测**：Oracle-Mix 显著优于构成匹配的
   Shuffled-Mix（pg≈2.5e-4 L1、0.02 dB），但其信号无法被现有 pre-action 特征恢复
   （residual 无增益、最优 BAcc 0.56）。
2. 有效的 pre-action 信号是 **parent 优化状态**（grad/grad_abs/importance）——这暗示未来
   决策规则的搜索方向应是训练动力学特征而非 residual 几何（此为观察，不属本阶段实现）。
3. 高 |gap| 桶 BAcc 0.59 + regret 减半是唯一留下的正面线索：若继续，"选择性只对高置信候选
   做昂贵评估/保守使用 predictor" 可能比全量 predictor 更可行（属后续重新评估内容）。

*生成于 2026-08-28。全部数据来自真实运行（logs/），无伪造。*

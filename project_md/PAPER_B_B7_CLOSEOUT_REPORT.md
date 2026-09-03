# Paper B — B7-Closeout 报告：Dynamic Feature Family Ablation

> B7-Fix 的最终分析收口：逐族检验 primitive-level 动态特征是否在 Static-Who 之外提供稳定增量。
> 纯分析（复用全部缓存与 750 oracle；无训练/replay/新模型）。两处统计修正：
> ①梯度统计仅计有效样本（radii>0 ∧ grad>0，<2 有效样本记 NA）；②含缺失序列的 slope 使用真实
> 时间索引（polyfit(idx_nonnan, v[idx])，不再 NaN 剔除后重排索引）。

## 0. 最终判定（先行）

```text
在当前实验范围内，单 Gaussian 的需求历史、梯度历史、参数演化、
优化响应和固定视角几何均未在静态状态之外提供稳定的容量收益预测增量。

No tested primitive-level temporal feature family provides
stable incremental predictive value beyond the static state.

→ CLOSE primitive-level predictor route。
→ 下一阶段转向 interaction-level / local-structure capacity allocation。
```

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b7close.py   收口特征重算（CPU）：有效样本梯度统计 + 真实索引 slope
                                     （需求/固定视角序列从缓存重算，保留逐样本序列备查）
- diagnostics/analyze_b7close.py      家族消融 LOSO（Static+{Demand,Gradient,Parameter,Response,
                                     Geometry,AllDynamic} + family-only 辅助）+ 判定 + 2 图
- paper_b/b7_closeout_dynamic_ablation/{data,plots,logs}
- project_md/PAPER_B_B7_CLOSEOUT_REPORT.md
Modified: 无（git diff --stat 空）
```

设置：Ridge(alpha=1.0)、target=q_best、leave-one-iteration-out（1000/1500/2000/5000/12000）、
训练折中位数插补、不按 held-out 选模型；750 候选。GPU 5 未使用（CPU 纯分析；容器 NVIDIA
运行时故障期间以 CPU 相机 shim 完成，身份断言 PASS——FastGS 源码零改动）。

## 2. 家族消融汇总（Static+Family − Static）

```text
     family  mean ΔSpear  pos stages  mean ΔVC@.5  pos stages  mean ΔVC@.75  pos stages
     Demand       −0.021        2/5           −0.024        3/5          −0.011        1/5
   Gradient       −0.005        3/5           +0.052        4/5          −0.018        2/5
  Parameter       −0.021        0/5           +0.009        3/5          −0.000        2/5
   Response       −0.005        2/5           +0.028        2/5          +0.009        3/5
   Geometry       −0.005        3/5           −0.007        1/5          −0.008        0/5
 AllDynamic       −0.055        1/5           +0.004        3/5          −0.030        1/5

Static: Spearman +0.206 / VC@.5−rand +0.181  ·  Static+AllDynamic: +0.151 / +0.184
判定门槛（ΔSpear>0 且 ΔVC@.5>0 于 ≥4/5 阶段，均值非零波动）：
  Demand FAIL · Gradient FAIL · Parameter FAIL · Response FAIL · Geometry FAIL · AllDynamic FAIL
```

**逐族解读（如实）**：
- **Parameter**（参数演化路径）：ΔSpearman 0/5——最明确的零信号。
- **Demand**（Gaussian-following 需求历史）：双指标均为负——修正后的需求序列无增量。
- **Geometry**（固定视角投影几何）：VC 1/5——无信号。
- **Response**（Ineff_* 分维度响应）：2/5——无信号。
- **Gradient**（有效样本梯度统计）：**唯一接近者**——ΔVC@0.5 均值 +0.052 且 4/5 阶段为正，
  但 ΔSpearman 均值 −0.005（3/5）未过门槛，且两指标方向不一致（整体排序变差、top-M 价值变好）。
  在 q_best 重尾分布 + n=150/阶段的噪声水平下，这种不一致模式属典型波动，不构成可复现信号；
  严格按任务书双条件判 FAIL。若未来重开 primitive 路线，Gradient-VC 是唯一值得复核的线索
  （需更大样本/跨场景重复）。
- AllDynamic：−0.055（1/5）——全部动态联合仍拖累静态。

## 3. 三个问题的回答

### 1. 哪类 dynamic feature 有增量？

**没有一类通过判定门槛。** Gradient family 在 ValueCapture@0.5 上有 4/5 阶段 +0.052 的表观增量，
但 Spearman 同步为负、方向不一致，判为噪声；其余四族（需求/参数/响应/几何）双指标均无稳定正增量。

### 2. 是否存在跨 stage 稳定增量？

**不存在。** 所有 family 的 ΔSpearman 正阶段数 ≤3/5；唯一达 4/5 的指标（Gradient 的 ΔVC@0.5）
不与其排序指标共存，且 AllDynamic 联合后增量为负——跨 stage 一致的正增量在数据中不存在。

### 3. primitive-level predictor 是否正式关闭？

**是。** 正式输出：

```text
No tested primitive-level temporal feature family provides
stable incremental predictive value beyond the static state.
在当前实验范围内，单 Gaussian 的需求历史、梯度历史、参数演化、
优化响应和固定视角几何均未在静态状态之外提供稳定的容量收益预测增量。
CLOSE primitive-level predictor route.
```

## 4. Paper B 证据链终局图景（只陈述）

| 层面 | 结论 | 证据 |
|---|---|---|
| Oracle 上界 | 真实、跨阶段（同质量省 50–75% growth） | B4/B5 |
| Static primitive | 弱且跨运行不稳（Spearman 0.0–0.26 波动） | B6 vs B7 |
| Dynamic primitive | 零增量（修正后更弱，逐族证伪） | B7 / B7-Fix / B7-Closeout |
| How primitive | 退化为全局 Clone 先验 | B6-Fix |
| **剩余方向** | **interaction-level / local-structure capacity allocation** | 任务书指定 |

结合 B2-B（tile 成本逐高斯精确可加）与 B4（同容量 policy 差 20–170× workload），
下一阶段的合理假设是：**容量分配信号存在于候选间的空间/结构交互（局部区域聚合），而非单个
Gaussian 的可观测状态**——该方向的具体实验设计属后续任务，本阶段不实现。

## 5. 输出 / git

```text
paper_b/b7_closeout_dynamic_ablation/data/{b7close_features.csv,.json, b7close_loso_results.csv,
  b7close_family_summary.csv, b7close_stats.txt}
paper_b/b7_closeout_dynamic_ablation/plots/{family_delta_spearman.png, family_delta_value_capture.png}
paper_b/b7_closeout_dynamic_ablation/logs/{b7close_features.log, b7close_loso.log}

$ git status --short
?? diagnostics/diagnostic_b7close.py
?? diagnostics/analyze_b7close.py
?? project_md/PAPER_B_B7_CLOSEOUT_REPORT.md
（B5–B7 系列文件此前已列；paper_b/ 在 .gitignore 中）

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-08-30。全部数据来自缓存的真实重算（logs/），无 oracle 重跑、无 replay、无伪造。完成即停止，不实现 B8。*

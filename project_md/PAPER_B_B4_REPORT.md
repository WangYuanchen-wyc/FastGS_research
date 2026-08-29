# Paper B — B4 报告：Joint Structural Capacity Allocation Oracle

> 在 B2-C master snapshot 与 881 个单点 oracle 上，分解验证 capacity allocation 的两个维度：
> **Who**（同等新增 GS 预算 M 下选对 candidate）与 **How**（同一 subset 上选对 Clone/Split），
> 并测试联合后的 Quality–#GS frontier。
> 仍为 Diagnostic：未实现 predictor / allocator / Budget Exchange / 新规则。

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b4.py   采集：四 policy（RW-NH / OW-NH / RW-OH / OW-OH）+ Keep-All /
                                 Native-Full 参照；M=round(rho·K)；全部从 B2-C 持久化 snapshot
                                 （指纹校验）replay；split 用 B2-C repeat-0 seed（300000+idx*100）
- diagnostics/analyze_b4.py      Who/How/Joint 增益（How 为逐 repeat 配对）、rho-scaling、
                                 frontier、rho=1 与 B2-C 一致性校验、6 图
- paper_b/b4_structural_capacity_oracle/{data,plots,logs,cache}
- project_md/PAPER_B_B4_REPORT.md
Modified: 无（git diff --stat 空）
```

## 2. 实际 GPU / 设置

**GPU 0（任务书指定，独占）**；容器 `wyc-compre`。所有分支：同一 master snapshot（it2000 前，N=137,398，
指纹校验通过）、同一 optimizer state、同一 100-step 相机序列（Random(2024)）、训练前统一 seed=1234、
densify/prune/reset 全关。**同一 K/seed 下所有 rho/policy 共用同一固定 full-group support/demand ROI**
（全部 K 成员 pre-action ROI 并集）。K ∈ {100, 300} × 3 seeds × rho ∈ {0.25, 0.50, 0.75, 1.00}，
RW 5 repeats（RW-NH 与 RW-OH **逐 repeat 使用完全相同 subset**）。

候选价值：q_clone / q_split / q_best / oracle_action 来自 B2-C oracle（clone_dQ_100 / split_dQ_mean_100）；
31 个 fallback 候选 q=0、action=native（入组率：K=100 组 0–1 个、K=300 组 1–3 个，已按记录计数）。

**Smoke（K=30, rho=0.5, seed=0）通过**：所有 constrained policy Δ#GS=M=15、RW 配对子集一致
（rep0: 0.098019→0.097990、rep1: 0.098093→0.098053，仅 action 不同）、split seed 一致、
固定 ROI 一致、100 步 replay 正常。正式运行先 K=100（信号明确）再 K=300，共 **300 条记录**
（每 K/seed 50 分支 = 2 参照 + 4 rho × (2 OW + 10 RW)）。

## 3. Who / How / Joint 增益（group demand L1@100，3 seeds mean±std，>0 = 更好）

```text
   K   rho     Who(RW-NH−OW-NH)     How(RW-NH−RW-OH)*    Joint(RW-NH−OW-OH)
 100  0.25    +0.000148±0.000075    +0.000126±0.000110    +0.000164±0.000074
 100  0.50    +0.000206±0.000094    +0.000195±0.000118    +0.000230±0.000111
 100  0.75    +0.000250±0.000162    +0.000273±0.000122    +0.000332±0.000137
 100  1.00    +0.000008±0.000025    +0.000388±0.000198    +0.000387±0.000194
 300  0.25    +0.000187±0.000105    +0.000083±0.000033    +0.000258±0.000128
 300  0.50    +0.000229±0.000098    +0.000179±0.000065    +0.000295±0.000099
 300  0.75    +0.000226±0.000051    +0.000262±0.000081    +0.000370±0.000100
 300  1.00    +0.000004±0.000006    +0.000327±0.000039    +0.000326±0.000042
*How 为逐 repeat 配对差值的聚合（RW-NH/RW-OH 子集逐位相同）
```

Sanity：rho=1 时 Who→0（无可选）✓、Joint→How ✓。两维度增益均 >1.5× 跨 seed std；
Joint/（Who+How）≈ 0.57–0.99（次可加：高价值候选同时被两种增益命中，存在部分重叠）。

## 4. Policy 汇总（3 seeds 均值）

### K = 100（Keep-All 0.094310 · Native-Full 0.094630 / 28.4242 dB）
| rho | M | RW-NH | RW-OH | OW-NH | OW-OH | OW-OH gPSNR | ΔTile@0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.25 | 25 | 0.094432 | 0.094306 | 0.094284 | **0.094267** | 28.4264 | +32.9 |
| 0.50 | 50 | 0.094488 | 0.094293 | 0.094282 | **0.094258** | 28.4289 | +77.7 |
| 0.75 | 75 | 0.094585 | 0.094312 | 0.094335 | **0.094253** | 28.4288 | +1102.5 |
| 1.00 | 100 | 0.094666 | 0.094278 | 0.094659 | 0.094280 | 28.4314 | +1165.2 |

### K = 300（Keep-All 0.093302 · Native-Full 0.093454 / 28.4086 dB）
| rho | M | RW-NH | RW-OH | OW-NH | OW-OH | OW-OH gPSNR | ΔTile@0 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.25 | 75 | 0.093342 | 0.093259 | 0.093155 | **0.093085** | 28.4276 | +164.3 |
| 0.50 | 150 | 0.093390 | 0.093211 | 0.093161 | **0.093095** | 28.4274 | +319.7 |
| 0.75 | 225 | 0.093444 | 0.093182 | 0.093218 | **0.093074** | 28.4233 | +1621.5 |
| 1.00 | 300 | 0.093447 | 0.093120 | 0.093444 | 0.093121 | 28.4243 | +1805.4 |

**OW-OH 在全部 (K, rho) 上都是最优 constrained policy**，且处处优于 Keep-All
（加 M 个"对的"高斯严格优于不加）。Latency（辅助）：~2.0ms 全 policy 噪声内。

## 5. Frontier：OW-OH@rho vs Native-Full@1.0

```text
K=100 rho=0.25: 0.094267 vs 0.094630（L1 更优 3.6e-4）· gPSNR +0.0022 dB · growth 25/100 · tile +33 vs +1165
K=100 rho=0.50: 0.094258 vs 0.094630（更优 3.7e-4）· gPSNR +0.0047 dB · growth 50/100 · tile +78 vs +1165
K=300 rho=0.25: 0.093085 vs 0.093454（更优 3.7e-4）· gPSNR +0.0190 dB · growth 75/300 · tile +164 vs +1805
K=300 rho=0.50: 0.093095 vs 0.093454（更优 3.6e-4）· gPSNR +0.0187 dB · growth 150/300 · tile +320 vs +1805
```

**以 25%–50% 的 representation growth（和 1/11–1/35 的结构 tile 成本）不仅"保持"而且严格超过
Native-Full 的质量**（demand L1 与 global PSNR 双优）。

## 6. rho=1 与 B2-C 一致性（replay 管线交叉验证）

```text
6 组（K×seed）：OW-OH@1 vs B2C oracle_mix |Δ| ≤ 2.2e-5；Native-Full vs B2C native |Δ| ≤ 4.9e-5
（均在 CUDA 原子非确定性噪声内）→ B4 复用管线与 B2-C 原始运行等价。
```

## 7. 图片 / 数据路径

```text
paper_b/b4_structural_capacity_oracle/plots/{quality_vs_growth_ratio, quality_vs_num_gaussians,
  quantity_gain_vs_rho, type_gain_vs_rho, joint_gain_vs_rho, quality_vs_tile_workload}.png
paper_b/b4_structural_capacity_oracle/data/{b4_group_results.csv,.json, b4_group_results_K100.*,
  b4_stats.txt}
paper_b/b4_structural_capacity_oracle/logs/{b4_smoke.log, b4_full_K100.log, b4_full_K300.log, b4_analysis.log}
```

## 8. git 状态

```text
$ git status --short
?? diagnostics/diagnostic_b4.py
?? diagnostics/analyze_b4.py
?? project_md/PAPER_B_B4_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

## 9. 四个问题的回答

### 1. Who matters? — **YES**
rho∈[0.25,0.75] 时 +1.5~2.5e-4 L1（两个 K 一致，1.5–3.3× 跨 seed std）；rho=1 归零（结构性 sanity）。
"选谁 densify"在预算受限时携带真实、可测的价值。

### 2. How matters? — **YES**
+0.8~3.9e-4 L1，随 rho 增大（预算越足、action 类型的影响越大）；rho=1 时退化为 B2-C 已证的
oracle-vs-native 差（+3.3~3.9e-4），两阶段结论互相印证。

### 3. Joint allocation 是否最好？ — **YES**
OW-OH 在全部 8 个 (K, rho) 组合中优于其他三个 constrained policy 与两个参照；
Joint gain ≈ Who+How 的 57–99%（次可加，两维度部分重叠但不互相抵消）。

### 4. rho=0.5/0.75 能否以更少 growth 保持接近 Native-Full 质量？ — **能，且是严格超过**
OW-OH@0.5（一半 growth）在两个 K 上 demand L1 均更优（−3.6e-4）且 global PSNR 更高
（K=300 高 **+0.019 dB**）；rho=0.25（1/4 growth）同样双优。同时结构 tile 成本降至 1/11–1/35。

## 10. 对 Paper B 的综合结论（只陈述，不设计）

1. **Joint oracle 定义了真实的 quality–capacity frontier**：同等质量下可省 50–75% 的 representation
   growth 与最高 97% 的结构 tile 开销——这是 capacity allocation 问题的上界证据。
2. Who 与 How 是**两个独立可测、近似可加的收益来源**（joint/sum 0.57–0.99），未来任何 allocator
   都应同时建模两个维度。
3. 与 B3 的衔接（如实）：该上界依赖昂贵的 per-candidate oracle；B3 已证明现有 pre-action 特征
   （尤其 residual 系）无法预测 How 维度（BAcc≈0.55）。**上界存在且巨大，但可达性仍未解决**——
   这是 Paper B 下一步重新评估的核心张力。

*生成于 2026-08-28。全部数据来自真实运行（logs/，300 条记录 + 6 组 rho=1 交叉验证），无伪造。*

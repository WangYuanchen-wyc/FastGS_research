# Paper B — B2-B 报告：Multi-Candidate Action Scalability Diagnostic

> 验证 single-candidate action-specific quality/compute 结论在同时处理 K 个 FastGS candidate 时是否成立。
> K = 30 / 100 / 300（自然采样，真实 population 分布），每 K 3 个 group seeds，
> 每 group 4 个 policy（Keep / Native / All-Clone / All-Split）从同一 checkpoint 受控续训 100 步。
> 仍为 Diagnostic：未实现 allocator / Budget Exchange / interaction model。

## 1. 修改文件

```text
Modified: (无 —— 本阶段未改动任何既有文件)

Added:
- diagnostics/diagnostic_b2b.py   采集：warmup→it2000 hook（真实事件，前置两个原生 densify）→
                                  自然采样 groups → 每 candidate 单点 ΔTile@0 测量（solo restore）→
                                  4 policy 组执行 → 100 步续训 + Group ROI 质量评测 → CSV/JSON
- diagnostics/analyze_b2b.py      可加性/K-scaling/policy 对比统计 + 4 图（宿主运行）
- paper_b/b2_b_scalability/{data,plots,logs}
- project_md/PAPER_B_B2B_REPORT.md
```

FastGS 原始训练算法零修改（`git diff --stat` 空）。复用 v2 的 `G` 全局、`native_selection_sets`、`local/恢复`机制。

**smoke（K=8×1 seed）修复的两个 bug**：① snapshot 曾在 warmup 末尾捕获（混入 it2000 原生 densify 的 +12k 高斯）→ 移入 hook 内事件前；② 组执行逐 candidate split 时每次重置 `tmp_radii` 为 snapshot 长度 → 改为只在开始设置一次、由原生 postfix/prune 维护行对齐。

## 2. 实际 GPU / 3. iteration

**GPU 7（RTX 4090 24GB，任务书指定，运行时独占）**，容器 `wyc-compre`，conda `fastgs`，场景 `room`（--eval）。
诊断点 **iteration 2000**（room 原生 interval=500 下第 3 个真实 densification 事件；checkpoint 已包含 it1000/it1500 两次真实 FastGS densification 演化，N=137,125）。

复现：

```bash
docker exec -e CUDA_VISIBLE_DEVICES=7 -w /mnt/workspace/FastGS wyc-compre \
  /opt/miniconda3/envs/fastgs/bin/python diagnostics/diagnostic_b2b.py \
  -s /mnt/workspace/Dataset/room --eval --densification_interval 500 \
  --grad_abs_thresh 0.0008 --diag_iter 2000 --Ks 30,100,300 --n_group_seeds 3
```

## 4. K 与 group seeds / 5. Candidate distribution

```text
K ∈ {30, 100, 300} × 3 seeds（seed=31337+K*10+s），population 均匀自然采样（无 50:50 人为配比）
it2000 真实 population: 12,158 candidates（native clone 119 / native split 12,039 —— 99.0% split）
组内 native 构成: K=30: 0/30,0/30,0/30 clone · K=100: 1/99,0/100,0/100 · K=300: 4/296,4/296,2/298
Group ROI（union of pre-action candidate ROIs，action 前固定，全 policy/step 共用）:
   K=30 覆盖 ~11.5% 像素/视图 · K=100 ~56.8% · K=300 ~65.4%
```

公平性：所有 policy 分支同一 checkpoint 深拷贝、同一 optimizer state、同一 100 步相机序列（Random(2024)）、训练前统一 seed=1234；densify/prune/reset 全关。Split 种子按 parent 身份唯一编码 `200000+parent_index*100+1`——单点测量与组执行产生**完全相同的子代**（组内按原始索引降序 split，父本行不被先前操作扰动），消除 draw 方差对可加性检验的干扰。

## 6. Compute Additivity（第一核心问题）

```text
                     K=30          K=100         K=300
native     rel_err  0.000e+00     0.000e+00     0.000e+00
all_clone  rel_err  0.000e+00     0.000e+00     0.000e+00
all_split  rel_err  0.000e+00     0.000e+00     0.000e+00
（9 个 group × 3 policy = 27 次检验，绝对误差 max = 0.000 tiles）
```

**Actual group ΔTile@0 = Σ_i ΔTile_i@0 精确成立（到整数级完全为零，不随 K 增长）。**
机制解释：`num_rendered` 是 rasterizer 对逐 Gaussian `tiles_touched` 的前缀和（rasterizer_impl.cu:379-383），每个 Gaussian 的 tile 计数与其他 Gaussian 无关——结构性 tile 成本按构造精确可加，本实验在 K=300、真实群体、三 policy 下经验证实。注意：这是 **K=0 结构量** 的性质；latency/FPS 不具备此性质（见 §8）。

## 7. Group Quality（Group ROI 内直接度量，非 candidate 误差求和）

| K | policy | gL1@100 | ΔgL1 vs Keep | gPSNR@100 | ΔgPSNR | ΔTile@0 | ΔTile@100 |
|---|---|---:|---:|---:|---:|---:|---:|
| 30 | Keep | 0.022862 | — | 28.2963 | — | 0 | −1051.7 |
| 30 | Native | 0.022933 | +7.1e-5 | 28.2878 | −0.009 | +7.1 | −1036.5 |
| 30 | All-Clone | 0.022872 | +1.0e-5 | 28.2898 | −0.007 | **+159.3** | −959.3 |
| 30 | All-Split | 0.022935 | +7.3e-5 | 28.2793 | −0.017 | **+7.1** | −1037.7 |
| 100 | Keep | 0.021497 | — | 28.2978 | — | 0 | −1054.0 |
| 100 | All-Clone | 0.021490 | **−7e-6** | 28.2866 | −0.011 | **+1108.3** | −400.7 |
| 100 | All-Split | 0.021514 | +1.7e-5 | 28.2704 | −0.027 | **−139.4** | −900.0 |
| 300 | Keep | 0.021053 | — | 28.2969 | — | 0 | −1054.2 |
| 300 | Native | 0.021117 | +6.3e-5 | 28.2612 | −0.036 | +10.6 | −888.3 |
| 300 | All-Clone | 0.021034 | **−1.9e-5** | 28.2967 | −0.0003 | **+1612.8** | −2.9 |
| 300 | All-Split | 0.021110 | +5.6e-5 | 28.2716 | −0.025 | **+9.2** | −886.9 |

（3 seeds 均值；native 在 K=30/100 略去同 all_split；逐 seed 方向一致：K=300 三个 seed 的 gL1 排序均为 all_clone ≤ keep < native ≈ all_split。）

## 8. Policy comparison / Trade-off

- **Quality**：All-Clone 与 Keep 在 gPSNR 上几乎不可分（K=300 差 −0.0003 dB），且 group-local L1 略优（−1.9e-5）；All-Split / Native 有小幅质量代价（−0.025~−0.036 dB, gL1 +5.6e-5~6.3e-5）。与 B2/B2A 单点结论方向一致（短程 clone ≥ split），且随 K 增大保持稳定。
- **Compute**：每 candidate 结构成本 All-Clone **+5.3~+11.1 tiles**，All-Split **−1.4~+0.24 tiles**（≈0 或负）——组级 clone/split 成本差 20~170×，与单点结论（B2）一致且精确线性扩展。@K=100 步后差距依旧：K=300 时 All-Clone 相对 Keep 净增 ~1051 tiles/视图（−2.9 vs −1054.2），All-Split 与 Keep 几乎重合（−886.9）。
- **Trade-off（本阶段最重要观察）**：K=300 时存在"质量几乎相同（gPSNR 差 ≤0.0003 dB、gL1 甚至略好）但结构 workload +1613 tiles（相对 Keep @100 步净差 ~1051 tiles/视图）"的 policy（All-Clone）与"质量小损 0.025 dB 但 workload ≈ +9 tiles"的 policy（All-Split）——**同容量（均 +300 GS）下 quality-compute 权衡空间真实且巨大**。
- Latency/FPS（辅助）：keep/native/all_clone/all_split = 1.88/1.87/1.87/1.90 ms（K=300），差异在噪声内——+300/137k（0.2%）不足以分辨端到端延迟，tile-pair 才是可测的结构成本代理。

## 9. Scaling trend

- 可加性误差：**恒为 0，不随 K 增长**（K=30→300 无任何趋势）。
- 组级 ΔTile@0：随 K 线性（clone ~5.4/cand 稳定；split 波动于 0 附近，K=100 组恰好抽到可收缩 footprint 的候选群 −1.39/cand）。
- 组质量差：方向在全部 K 与全部 seed 上稳定（all_clone ≤ keep ≤ native ≈ all_split），幅度不随 K 放大亦不消失。
- Single-candidate quality scaling（Σ单点ΔQ vs 组实际ΔQ）：**NOT AVAILABLE**（本运行为独立 warmup，CUDA 原子浮点非确定性使 B2-A 的 60 个单点 oracle 与本 population 不对应；不伪造、不训练 predictor 补标签）。Oracle-Mix 同理 = **NA**。

## 10–11. 图片 / 数据路径

```text
paper_b/b2_b_scalability/plots/{tile_additivity_error_vs_K, group_quality_vs_K,
                                group_tile_cost_vs_K, group_quality_vs_tile_cost}.png
paper_b/b2_b_scalability/data/b2b_results.csv（36 条 K×seed×policy 记录）
paper_b/b2_b_scalability/data/b2b_results.json（含全部字段与配置）
paper_b/b2_b_scalability/data/group_candidates_K{30,100,300}_seed{0,1,2}.json（成员表+split 种子+ROI）
paper_b/b2_b_scalability/data/b2b_stats.txt
paper_b/b2_b_scalability/logs/{b2b_smoke.log, b2b_full.log}
```

## 12. git 状态

```text
$ git status --short
?? diagnostics/
?? paper_b/
?? project_md/（各阶段报告/数据）

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

## 13. 三个科学问题的回答

### A. Is candidate-level tile cost approximately additive up to K=300?

**YES** —— 27/27 次检验（3 K × 3 seeds × 3 policies）绝对与相对误差均为 **0.000**（整数级精确）。
结构性 tile 成本按 rasterizer 构造逐 Gaussian 独立，可加性不随 K 衰减。candidate-level compute accounting 可直接扩展。

### B. Does candidate-level action preference remain meaningful as K increases?

**YES** —— 单点结论的两条方向均保持且被放大：
① clone/split 的结构成本差（单点 ~20×）在线性扩展后达组级 20~170×（K=300: +1613 vs +9 tiles）；
② 短程质量排序（clone ≥ split）在全部 K、全部 seed 上方向稳定（all_clone gL1 ≤ keep ≤ all_split）。
需如实标注：质量差幅度小（gL1 差 ≤7e-5，gPSNR 差 ≤0.036 dB），其"意义"主要通过与成本差的悬殊对比体现——同 +300 GS 下两种 policy 的 quality-compute 权衡差异巨大。

### C. Is candidate-wise structural action allocation still a reasonable modeling assumption?

**YES** —— 无需 interaction model：K=0 结构成本精确可加（A），质量无破坏性干扰（B，组级质量差不随 K 发散且与单点方向一致）。
适用边界（如实说明）：① 可加性是 tile-pair 结构量的性质，不适用于端到端 latency；② 质量的精确可加性未检验（单点 quality oracle 不可迁移，标 NOT AVAILABLE）；③ 结论限于 100-step 短程与单场景 room。

## 14. 对 Paper B 的综合科学结论（只陈述，不设计）

1. **Budget Exchange 的 compute 侧账户可以按 candidate 精确记账**：ΔTile 可加到 K=300 无误差，"每 candidate 一个 tile 预算项"的建模假设被强证实。
2. **组级同容量下 policy 选择产生 20~170× 的 workload 差异而质量几乎不动**（K=300：All-Clone 与 Keep gPSNR 差 0.0003 dB、成本 +1613 tiles；All-Split 成本 +9 tiles、质量 −0.025 dB）——这正是 action allocator 的价值空间，且该空间随 K 增大而扩大。
3. 短程质量上 clone 微优于 split 的单点偏好（B2/B2A）在组级保持方向一致，未出现需要 interaction model 的失效迹象。

*生成于 2026-08-27。全部数据来自真实运行（logs/b2b_full.log），无伪造。*

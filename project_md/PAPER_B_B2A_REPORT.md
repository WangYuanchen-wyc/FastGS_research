# Paper B — B2-A 报告：Split 随机性 + Residual–Gaussian 几何对齐

> 在 Diagnostic V2 的 60 个真实 FastGS candidate 上：每个 candidate 执行 Keep×1、Clone×1、
> Split×5（candidate-specific seeds），并加入 pre-action parent 投影几何与
> residual↔parent / residual↔split-separation 对齐描述子。
> 仍为 Diagnostic：**未实现** predictor / allocator / Budget Exchange / lineage / rollback。

## 1. 修改文件

```text
Modified:
- diagnostics/common.py            新增 projected_gaussian_geometry()：numpy 复刻 rasterizer
                                   computeCov3D+computeCov2D (EWA)，返回投影中心/主次轴方向/
                                   投影各向异性/extent/3σ 半径，并与原生 radii 自校验
- diagnostics/diagnostic_v2.py     build_candidate() 扩展：每视图 parent 投影几何、
                                   residual_parent_alignment(|cos|)、residual_extent_ratio、
                                   residual_centroid_offset、proj_anisotropy（均为 action 前量）
                                   （仅为新增字段，v2 既有逻辑不变）

Added:
- diagnostics/diagnostic_b2a.py    B2A 采集：Keep×1+Clone×1+Split×5（seed=90000+cid*100+r）、
                                   每次 split 后 K=0 捕获两 child 屏幕分离方向 →
                                   residual_split_alignment；输出 b2a_results.{json,csv}
- diagnostics/analyze_b2a.py       统计（随机性/对齐相关/winner/tile）+ 5 图（宿主运行）
- paper_b/b2_a_split_alignment/    data/ plots/ logs/ 全部实验产物
- project_md/PAPER_B_B2A_REPORT.md 本报告
```

FastGS 原始训练算法零修改（`git diff --stat` 空）。

## 2. 实际 GPU

**GPU 7（RTX 4090 24GB，任务书指定，运行时 0 MiB 独占）**，容器 `wyc-compre`，conda `fastgs`。场景 `/mnt/workspace/Dataset/room`（--eval）。复现：

```bash
docker exec -e CUDA_VISIBLE_DEVICES=7 -w /mnt/workspace/FastGS wyc-compre \
  /opt/miniconda3/envs/fastgs/bin/python diagnostics/diagnostic_b2a.py \
  -s /mnt/workspace/Dataset/room --eval --densification_interval 500 \
  --grad_abs_thresh 0.0008 --diag_iters 1000,1500,2000 --n_cand 20 --n_probe 8
```

## 3–4. Candidate 数量与 Split repeats

```text
candidate: 60（it1000/1500/2000 × 20；每批 native clone 10 + native split 10）
split repeats: 每 candidate 5 次，seed = 90000 + candidate_id(0..59)*100 + repeat(0..4)
   （cid0: 90000–90004 … cid59: 95900–95904，无任何 candidate 复用同一 draw）
每 candidate 分支: Keep×1 + Clone×1 + Split×5（共 7 × 100 步受控续训）
batch 状态: it1000 N=112,627 (45/9,775) · it1500 N=122,446 (62/14,679) · it2000 N=137,166 (106/11,844)
有效 local ROI: 60/60（均 8/8 视图）；有效 demand ROI: 60/60；alignment 视图覆盖: 60/60
```

公平性（同 B1/V2 协议）：同一 checkpoint 深拷贝 restore、同一 optimizer state、同一 100 步相机序列（Random(2024)）、训练前统一重置 RNG（seed=1234）；**唯一变化量 = split child 初始化采样**；split 仍用原生 `densify_and_split_fastgs`（N=2、/1.6）。

> 注：warmup 与 B2 独立重跑，由于 CUDA atomicAdd 浮点非确定性，两次运行的候选集略有差异
> （如 it1000 split set 9,775 vs 9,806）——属原生训练固有性质，两个运行各自内部自洽。

## 5. Split stochasticity（ΔQ = demand-local L1 reduction @100）

```text
split_dQ_std            : median +0.000207  mean +0.000462  max +0.004254
split_stochastic_range  : median +0.000579  mean +0.001303  max +0.013000
|mean ΔQ_split − ΔQ_clone| : median +0.000294  mean +0.000669

单次 split:   std > |gap| 的 candidate = 23/60 = 38.3%；median std/gap = 0.70
5-repeat 均值: 有效 std(=std/√5) > |gap| = 18.3%；median eff_std/gap = 0.32（信噪比≈3:1）
```

**结论：单次 split 不足以做 candidate 级评判（38% 候选噪声>信号）；取 5 次重复平均后随机性充分受控（82% 候选信号>噪声，中位信噪比 3.1）。**

## 6. Residual–Split directional alignment（本次最重要 descriptor）

对每次 split repeat：投影两 child 中心 → separation 方向 → 与 residual 主方向夹角 |cos|。

```text
pooled（300 点 = 60 cand × 5 repeats）: Pearson +0.004   Spearman +0.009
within-candidate Spearman（每 cand 5 点）: mean −0.005  median +0.100
positive / negative within-candidate ratio: 55.0% / 41.7%
```

**结论：split 采样方向与 residual 主方向的对齐程度与该次 split 的 100 步局部质量收益无关系（清晰的 NO）。机制上合理：child 偏移采自 parent 3D 协方差（近各向同性组合投影），且 100 步优化可大幅重定位 child，初始分离方向被冲销。**

## 7. Residual–Parent alignment 与 quality_action_gap（= mean ΔQ_split@100 − ΔQ_clone@100）

全量 n=60 / 过滤组 n=43（剔除几何自校验失配 >5px 的退化 sliver 候选，见 §9 注）：

| descriptor | Pearson(60) | Spearman(60) | Pearson(43) | Spearman(43) |
|---|---:|---:|---:|---:|
| scale_max | +0.031 | +0.088 | +0.049 | +0.057 |
| scale_anisotropy | +0.121 | +0.153 | +0.078 | +0.091 |
| **residual_anisotropy_mean** | **+0.156** | **+0.194** | **+0.149** | **+0.249** |
| proj_anisotropy_mean | +0.036 | +0.129 | +0.045 | +0.059 |
| residual_parent_alignment_mean | −0.048 | +0.018 | −0.048 | +0.056 |
| residual_extent_ratio_mean | −0.056 | −0.066 | +0.007 | −0.043 |
| residual_centroid_offset_mean | −0.065 | −0.120 | −0.016 | −0.084 |
| footprint_mean | +0.043 | +0.096 | — | — |

**residual_anisotropy 是唯一稳定强于 scale 类的描述子（过滤组 Spearman +0.249 vs scale_max +0.057），但绝对强度仍弱。parent 对齐 |cos|、extent 比、centroid 偏移基本无信息。**

## 8. Action winner（split 用 5-repeat 平均质量）与 native 一致率

```text
Keep best : 17/60 = 28.3%      Clone best : 19/60 = 31.7%      Split best : 24/60 = 40.0%
native vs oracle agreement: 25/60 = 41.7%（B2 单次 split 口径为 46.7%）
conditional agreement（oracle 选了 action 的 43 例）: 25/43 = 58.1%
confusion (行=native, 列=oracle):      keep  clone  split
native clone                             7     12     11
native split                            10      7     13
```

用 5-repeat 平均消除 split 运气后，native 一致率反而降至 41.7% —— B2 的 46.7% 中约 5 个百分点来自单次 split 的有利抽样。**原生 scale 启发式与短程局部 oracle 的偏离是系统性的。**

### Tile cost（附，@K=0 结构性）

```text
ΔTile_clone@0 : median +2.39  mean +19.02  max +555.9（60/60 为正）
ΔTile_split@0 : median +1.25  mean −1.37   min −118.4（可为负）
split tile std@0（repeat 间）: median +0.33 —— tile 成本对 split 随机性不敏感
```

## 9. 图片 / 数据路径

```text
paper_b/b2_a_split_alignment/plots/
  split_stochasticity_hist.png            split_gain_variance_vs_action_gap.png
  split_gain_vs_residual_split_alignment.png
  quality_gap_vs_residual_parent_alignment.png
  quality_gap_vs_residual_extent_ratio.png
paper_b/b2_a_split_alignment/data/b2a_results.csv（candidate 级聚合，60 行）
paper_b/b2_a_split_alignment/data/b2a_results.json（candidate × view × repeat 全量明细，1.2MB）
paper_b/b2_a_split_alignment/data/b2a_stats.txt
paper_b/b2_a_split_alignment/logs/b2a_smoke.log  b2a_full.log
```

> 注（几何自校验）：投影 2D 协方差 3σ 半径 vs 原生 `radii` 的失配中位数 1.38px（it1000 批 1.11px），
> 但在极端各向异性投影（proj_aniso>8 的退化 sliver）上有重尾（max 158px）；方向/对齐描述子不受影响，
> 与半径相关的相关性（extent_ratio/centroid_offset）同时报告了过滤组（n=43）。

## 10. git 状态

```text
$ git status --short
?? diagnostics/
?? paper_b/
?? project_md/（B1/B2/B2A 新增报告与数据）

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

## 11. 三个科学问题的回答

### A. Is Split stochasticity small enough for candidate-level action evaluation?

**YES（有条件）** —— 单次 split draw：**不可靠**（38.3% candidate 的 repeat 间 std 超过 Clone−Split 偏好幅度）；**取 ≥5 次重复平均后：可以**（有效噪声仅 18.3% candidate 超过信号，中位信噪比 3.1:1）。任何后续 candidate 级 action 评估必须使用重复平均协议。

### B. Does better residual–split directional alignment produce better Split quality?

**NO** —— pooled ρ=+0.009（n=300），within-candidate Spearman 中位 +0.10（正/负比 55%/42%，与掷硬币无异）。split 初始分离方向与 residual 的对齐程度不携带可测的短程质量信息。

### C. Does residual–Gaussian geometry provide more action information than simple scale alone?

**WEAK** —— residual_anisotropy 稳定强于全部 scale 类描述子（过滤组 Spearman +0.249 vs scale_max +0.057 / scale_anisotropy +0.091），方向上支持"残差几何携带 scale 之外的信息"；但绝对相关性仍弱（|ρ|<0.25，n=43/60 小样本），且 residual↔parent 对齐类描述子（|cos|、extent 比、centroid 偏移）全部无信息。不足以单独支撑 action 决策。

## 12. 对 Paper B 的综合科学结论（只陈述，不设计）

1. **方法论**：split 的固有随机性（std 中位 2.1e-4，与 clone/split 偏好同量级）此前被单次评测掩盖——未来所有 action 对比必须重复平均（本实验定标：5 次）。
2. **负面结果（重要）**：residual–split 方向对齐无预测力；residual–parent 对齐/extent 比/centroid 偏移无预测力。Paper B 若依赖"几何对齐指导 split 方向"的直觉，需要放弃或改用其他形式（例如显式控制 child 放置并做更长 horizon）。
3. **正面线索**：residual_anisotropy 与 quality_action_gap 的相关性在两个口径下均为最强（+0.19/+0.25）；native 一致率在去随机化后降至 41.7%（keep 最佳占 28.3%）——action 选择 headroom 真实存在。
4. **计算侧复确认**：ΔTile_clone@0 60/60 为正、显著大于 split（median +2.4 vs +1.25，mean +19.0 vs −1.4），且 split 的 tile 成本对随机 draw 稳定（std 中位 0.33）——compute 侧信号与 B2 一致且更干净。

*生成于 2026-08-27。全部数据来自真实运行（logs/b2a_full.log），无伪造。*

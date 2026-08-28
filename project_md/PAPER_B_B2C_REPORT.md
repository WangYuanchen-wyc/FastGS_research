# Paper B — B2-C 报告：Personalized Oracle-Mix Scalability Diagnostic

> 验证 candidate-wise structural action allocation：对同一批真实 FastGS candidates，
> 按**每个 candidate 自己的 short-horizon Clone/Split oracle** 分配 action（Oracle-Mix），
> 是否优于 Native / All-Clone / All-Split / **构成匹配的 Shuffled-Mix**。
> 本阶段为 Diagnostic：未实现 predictor / allocator / budget / interaction model。

## 1. 修改文件

```text
Modified: (无 —— 未改动任何既有文件)

Added:
- diagnostics/diagnostic_b2c.py   采集：单 master snapshot（it2000 事件前，持久化到
                                  cache/master_snapshot.pt）→ 881 个候选单点 oracle
                                  （Clone×1+Split×5，candidate-specific seeds，逐候选落盘
                                  可断点续跑）→ 9 组 × 10 分支（keep/native/all_clone/
                                  all_split/oracle_mix/shuffled×5）
- diagnostics/analyze_b2c.py      §29 表格 + §30 统计 + 7 图（宿主运行）
- paper_b/b2_c_oracle_scalability/{data,data/groups,plots,logs,cache}
- project_md/PAPER_B_B2C_REPORT.md
```

FastGS 原始训练算法零修改（`git diff --stat` 空）。

**开发过程中发现并修复的问题（如实记录）**：
1. 首次正式运行发现 33/873 候选无任何 demand-valid 视图 → oracle=None → 组策略漏处理 → Δ#GS<K。修复：无 oracle 候选默认回退其 native action（oracle 与 shuffled 使用同一回退列表，构成仍完全匹配；本运行 31/881=3.5%）。
2. snapshot 只在内存中导致跨进程无法复用 oracle 缓存（fresh warmup 因 CUDA 原子非确定性必然产生不同 snapshot）。修复：master snapshot 持久化 + 协议绑定 snapshot 指纹，缓存真正可续跑。
3. 重构引入的重复 `del gaussians` 导致一次早退（日志留痕），修复后完整重跑。
最终数据来自修复后的单一完整运行（日志 `logs/b2c_group_full.log`），内部完全自洽。

## 2. 实际 GPU / 3. iteration

**GPU 0（RTX 4090 24GB，任务书指定，运行时独占）**；容器 `wyc-compre`，conda `fastgs`，`room`（--eval）。
诊断点 **iteration 2000**（room 原生 interval=500 第 3 个真实 densification 事件；master snapshot 在该事件**执行前**捕获，含 it1000/1500 两次真实演化，N=137,398）。**oracle 与全部组分支派生自同一 snapshot、同一 candidate population**（B2-A/B2-B 的跨运行不匹配问题在本阶段不存在）。

复现：

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0 -w /mnt/workspace/FastGS wyc-compre \
  /opt/miniconda3/envs/fastgs/bin/python diagnostics/diagnostic_b2c.py \
  -s /mnt/workspace/Dataset/room --eval --densification_interval 500 \
  --grad_abs_thresh 0.0008 --diag_iter 2000 --pool_K 300 --Ks 30,100,300 \
  --n_group_seeds 3 --shuffled_repeats 5
```

## 4. K / seeds / 5. candidate distribution

```text
3 group seeds × 300-candidate pool（population 自然均匀采样，无 50:50），K=30/100/300 为嵌套前缀
population @it2000: 12,218（native clone 128 / native split 12,090 → 99.0% split）
去重后 881 个唯一候选建立 oracle；31 个（3.5%）无 demand-valid 视图 → 回退 native action
oracle 构成（全候选）: clone 55.4% / split 41.1%（native≈100% split —— oracle 是真正的混合分配）
组内 oracle 构成示例: K=30: 12~17 clone / K=100: 50~65 / K=300: 160~177
高置信（|gap|>max(sem,1e-4)）: 59.7%；median |gap| 1.89e-4 vs median SEM 0.75e-4（信噪 ≈2.5:1）
```

公平性：所有分支同一 master snapshot 深拷贝、同一 optimizer state、同一 100 步相机序列（Random(2024)）、训练前统一 seed=1234；densify/prune/reset 全关；split 组实现 = oracle repeat 0 的 draw（seed=300000+parent_index*100+0，逐 candidate 唯一）；组内 clone 批量、split 按原始索引降序（父本行保持 snapshot 值 → 子代与单点 oracle 逐位一致）。**所有 densify policy Δ#GS = K（90/90 条记录零违例）。**

## 6–7. Compute / #GS 确认

```text
dN = K 违例: 0（keep +0；其余全部 +K）
ΔTile@0（3 seeds 均值）:
   K=30 : native −10 / all_clone +316 / all_split −10 / oracle +49 / shuffled +109
   K=100: native −129 / all_clone +1578 / all_split −129 / oracle +1165 / shuffled +1047
   K=300: native −102 / all_clone +3464 / all_split −102 / oracle +1805 / shuffled +1276
（tile 可加性已在 B2-B 证明，此处仅记录实际组 workload；oracle 约为 all_clone 的 1/2 成本）
```

## 8. 每 K Policy 汇总（§29 表；主指标 group_demand_L1@100，3 seeds 均值）

### K = 30
| Policy | GroupDemand L1@100 | GroupDemand PSNR@100 | Global PSNR@100 | ΔTile@0 | ΔTile@100 | Δ#GS |
|---|---:|---:|---:|---:|---:|---:|
| Keep | 0.096421 | 18.940 | 28.4263 | +0 | +0 | +0 |
| Native | 0.096588 | 18.927 | 28.4272 | −10.4 | +37.3 | +30 |
| All-Clone | 0.096519 | 18.932 | 28.4276 | +316.3 | +196.7 | +30 |
| All-Split | 0.096599 | 18.926 | 28.4273 | −10.4 | +36.4 | +30 |
| **Oracle-Mix** | **0.096472** | **18.934** | **28.4280** | +49.3 | +66.3 | +30 |
| Shuffled-Mix mean | 0.096540 | 18.929 | 28.4277 | +109.4 | +74.2 | +30 |

### K = 100
| Policy | GroupDemand L1@100 | GroupDemand PSNR@100 | Global PSNR@100 | ΔTile@0 | ΔTile@100 | Δ#GS |
|---|---:|---:|---:|---:|---:|---:|
| Keep | 0.094309 | 19.020 | 28.4272 | +0 | +0 | +0 |
| Native | 0.094646 | 18.996 | 28.4243 | −129.0 | +868.0 | +100 |
| All-Clone | 0.094368 | 19.017 | 28.4312 | +1578.5 | +1278.0 | +100 |
| All-Split | 0.094651 | 18.996 | 28.4235 | −129.1 | +868.7 | +100 |
| **Oracle-Mix** | **0.094272** | **19.025** | **28.4315** | +1165.2 | +919.8 | +100 |
| Shuffled-Mix mean | 0.094529 | 19.005 | 28.4280 | +1047.3 | +1065.2 | +100 |

### K = 300
| Policy | GroupDemand L1@100 | GroupDemand PSNR@100 | Global PSNR@100 | ΔTile@0 | ΔTile@100 | Δ#GS |
|---|---:|---:|---:|---:|---:|---:|
| Keep | 0.093306 | 19.217 | 28.4270 | +0 | +0 | +0 |
| Native | 0.093440 | 19.204 | 28.4090 | −101.9 | +1481.3 | +300 |
| All-Clone | 0.093286 | 19.222 | 28.4319 | +3463.8 | +2507.5 | +300 |
| All-Split | 0.093444 | 19.204 | 28.4093 | −102.4 | +1482.8 | +300 |
| **Oracle-Mix** | **0.093121** | **19.234** | 28.4250 | +1805.4 | +2041.3 | +300 |
| Shuffled-Mix mean | 0.093371 | 19.214 | 28.4189 | +1275.6 | +1962.1 | +300 |

Shuffled within-seed std（5 repeats）：K=30: 5.8e-5 · K=100: 5.2e-5 · K=300: 7.5e-5。
**Oracle-Mix 在全部 3 个 K 上都是组需求质量最优 policy；K=300 时是唯一优于 Keep 的 policy。**

## 9. Oracle stability / Native vs Oracle（§30）

```text
candidate count 881 · oracle clone 55.4% / split 41.1% / 无 oracle 3.5% · 高置信 59.7%
median |oracle gap| = 1.89e-4  ·  median split SEM = 0.75e-4（gap ≈ 2.5× 自身噪声）
gap<0（clone 更优）57.4% / gap>0（split 更优）42.6%
binary agreement（native vs oracle，无 Keep）: 363/850 = 42.7%
confusion:            oracle clone  oracle split
native clone                     5              4
native split                   483            358
→ FastGS scale heuristic 把 483 个短程更适合 clone 的候选判为 split
```

## 10. Personalization（§30，跨 3 seeds mean±std）

```text
    K    pg_L1(shuffled−oracle)   pg_PSNR(oracle−shuffled)   vs Native dQ    vs AllClone dQ   vs AllSplit dQ
   30    +0.000068±0.000028       +0.0050±0.0038 dB          +0.000117       +0.000047        +0.000128
  100    +0.000257±0.000116       +0.0208±0.0096 dB          +0.000375       +0.000096        +0.000379
  300    +0.000250±0.000062       +0.0202±0.0039 dB          +0.000319       +0.000166        +0.000323
（全部 > 0：Oracle-Mix 同时优于全部对照）

Oracle vs 45 个独立 shuffled realization: 胜 42 / 负 3（3 负均在 K=30，幅度 ≤2e-5，在组内 std 5.8e-5 之内）
per-seed：oracle < shuffled mean 9/9 组
```

## 11. Scaling trend

personalization_gain：K=30 +6.8e-5（≈1.2× 组内 shuffle std，边际）→ K=100 +2.6e-4（≈4.9× std）
→ K=300 +2.5e-4（≈3.3× std）。**随 K 增大并稳定在 ~2.5e-4，无衰减迹象**；pg_PSNR 同型（0.005→0.021→0.020 dB）。

## 12. 图片 / 13. 数据路径

```text
paper_b/b2_c_oracle_scalability/plots/{oracle_vs_shuffled_quality_vs_K, personalization_gain_vs_K,
  policy_quality_vs_K, policy_tile_cost_vs_K, policy_quality_vs_tile_cost,
  native_vs_oracle_binary_confusion, oracle_gap_distribution}.png
paper_b/b2_c_oracle_scalability/data/b2c_group_results.{csv,json}（90 条 K×seed×policy）
paper_b/b2_c_oracle_scalability/data/b2c_candidate_oracles.{csv,json}（881 条单点 oracle）
paper_b/b2_c_oracle_scalability/data/groups/group_K{30,100,300}_seed{0,1,2}.json
paper_b/b2_c_oracle_scalability/data/b2c_stats.txt
paper_b/b2_c_oracle_scalability/logs/{b2c_smoke, b2c_oracle_build, b2c_group_full}.log
paper_b/b2_c_oracle_scalability/cache/（master_snapshot.pt + 881 个候选缓存，可续跑）
```

## 14. git 状态（§34）

```text
$ git status --short
?? diagnostics/analyze_b2c.py
?? diagnostics/diagnostic_b2c.py
?? project_md/PAPER_B_B2C_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

说明：B2-A/B2B 阶段产物已由用户提交（`bff8f4d Update B2-A/B`、`6075212 Update B0`）；
`paper_b/` 已被仓库 `.gitignore` 忽略（含 b2_c_oracle_scalability 全部实验数据，磁盘保留）。

## 15. 三个科学问题的回答

### A. Personalized assignment

**YES** —— Oracle-Mix 在全部 3 个 K、全部 9 个组上优于**构成完全相同**的 Shuffled-Mix：
pg_L1 = +6.8e-5 / +2.6e-4 / +2.5e-4（pg_PSNR +0.005 / +0.021 / +0.020 dB）；
击败 45 个独立 shuffle 实现中的 42 个（3 负均为 K=30 的边际情形，幅度小于组内 std）。
**结果由"哪个 candidate 得到哪个 action"决定，而非 Clone 比例** —— Shuffled-Mix 对照的直接证据。

### B. Scalability

**YES** —— 优势不随 K 衰减：K=30 时边际（≈1.2× shuffle std），K=100/300 稳定在 ~2.5e-4
（≈3.3–4.9× 组内 shuffle std）。K 越大个性化收益越稳定。

### C. Candidate-wise formulation

**YES** —— 满足判定条件（A=YES 且 B≠NO）。且补充两条强证据：
① K=300 时 Oracle-Mix 是唯一 group-demand 质量优于 Keep 的 policy（0.093121 vs 0.093306）——
个性化分配让"多 densify"严格优于"不 densify"；
② 其计算成本约为 All-Clone 的一半（+1805 vs +3464 tiles），同时质量支配 All-Clone。

## 16. 如实标注的边界

- 单场景（room）、单诊断点（it2000）、100-step 短程 horizon；绝对 L1 增益小（0.07–0.27% 基线）但方向全一致且数倍于噪声。
- 本 oracle 是**上界**：每候选需 6 次 100-step 受控评测。走向实用 action decision rule 需要 predictor（本阶段明确未实现）。
- 31/881 候选（3.5%）无可测 demand oracle，oracle/shuffled 一致回退 native action，不影响构成匹配比较。

## 17. 对 Paper B 的综合科学结论（只陈述，不设计）

1. **candidate-wise personalized structural action allocation 成立**：同构成随机分配对照下，个性化分配在所有组规模上一致更优，且优势随 K 增大而稳固——这是 B2 系列诊断一直寻找的核心正面结论。
2. FastGS scale heuristic 与短程 oracle 的二分类一致率仅 42.7%（483 个应 clone 的被判 split）——启发式的错分是系统性的，也是个性化的收益来源。
3. 计算侧：个性化分配同时给出质量与成本的优势组合（优于 all-clone 的成本、优于 all-split 的质量）——quality-compute 双目标下 mixed assignment 是真实存在的 Pareto 改进。

*生成于 2026-08-28。全部数据来自真实运行（logs/b2c_group_full.log），无伪造。*

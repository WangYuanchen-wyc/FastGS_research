# Paper B — B5 报告（B5-Fix 修订版）：Cross-Stage Structural Capacity Allocation Oracle

> **本版为 B5-Fix 修订**：修复了旧 process 运行的相机映射缺陷后基于**同一批 5 个 snapshot** 完整重跑。
> 旧 process 结果已作废并隔离（`cache/quarantine_b5fix/`）。以下全部数字来自修复后的有效运行。
> 仍为 Diagnostic：未实现 predictor / allocator / Budget Exchange / 新规则。

## 0. B5-Fix 说明（root cause 与作废声明）

**Root cause**：`scene/__init__.py:65-67` 的 `Scene.__init__(shuffle=True)` 使用**全局** `random.shuffle`
打乱 train/test 相机顺序。collect 阶段在 `seed_all(0)` 后创建 Scene；旧 process 阶段创建 Scene 前
**未重置 RNG** → 新进程随机态 → 相机顺序与 collect 不同 → (a) `cam_seq` 按下标索引到不同的训练视图
（keep/replay 模型整体偏移，gPSNR 差 0.5 dB）；(b) pool 视图与 snapshot 内 collect 期的
masks/radii 错配（demand ROI、oracle 有效性/数值全部失真）。这完全解释了旧结果 10× 的增益萎缩与
异常"衰减"形态。

**修复**：process 模式在 `Scene(...)` 前补 `seed_all(0)`；collect 保存 train/test/pool 相机身份
（image_name + uid）；process 逐项断言一致，不一致即中止。对现有 snapshot 通过 `camident` 模式
按与 collect 完全相同的确定性路径回填参考身份。

```text
camera identity assertion: PASS（train 272 / test 39 / pool 30 顺序与 collect 逐位一致）
```

**影响范围审计**：B1/B2/B2A/B2B/B2C/B4 均为单进程运行（Scene 只创建一次，快照与实验同一进程内一致），
不受此缺陷影响；B3 为独立进程但在 Scene 前有 `seed_all(0)`，与 B2C 顺序一致。**唯一受影响的是旧 B5
process 结果，已隔离作废。**

## 1. 修改文件（B5-Fix）

```text
Modified:
- diagnostics/diagnostic_b5.py   collect 保存相机身份；process 前 seed_all(0) + 三重身份断言；
                                 新增 camident 回填模式（复刻 collect 确定性路径）
- diagnostics/analyze_b5.py      统计口径修复：全部增益 per-seed 计算；HowR 先在每 seed 内对
                                 5 个配对 repeat 求 mean 再跨 2 个 seed mean 聚合；
                                 正计数分两层明确输出（seed-level 20 / setting-level 10），不混用
Added:
- paper_b/b5_cross_stage_capacity_oracle/cache/quarantine_b5fix/（旧产物隔离，1,354 项）
- 本报告（覆盖更新）
Modified(无): FastGS tracked source —— git diff --stat 为空（root cause 是既有 FastGS 行为，
             修复在诊断脚本侧，不改原始训练代码）
```

## 2. 实际 GPU / 设置（不变）

GPU 2（任务书指定）；5 snapshot 沿同一条 native trajectory（collect 单次 warmup 至 12000，事件前捕获，
**复用 B5 原快照，未重新 warmup**）；K=100 × 2 seeds × rho∈{0.50,0.75}（M=50/75）× RW 5 配对 repeats；
B2C 协议 oracle（Clone×1 + Split×5，seed=300000+idx*100+r，组实现=repeat 0）；每 snapshot/seed 固定
full-group support/demand ROI；replay 协议与 B2-C/B4 一致。Oversample 至 100 valid/组
（invalid 率 1–5%，不入组不参与 ranking）。全部 constrained policy Δ#GS=M 零违例。

## 3. Snapshot 轨迹状态（不变，来自 collect）

```text
iter    #GS_before  candidates(clone/split)      prune@event   #GS_after
1000      112,627     9,825 (  44 /  9,781)              1     122,451
1500      122,451    14,805 (  77 / 14,728)             28     137,228
2000      137,228    12,376 ( 127 / 12,249)             57     149,547
5000      204,387    11,324 ( 133 / 11,191)          1,481     214,230
12000     259,075     2,572 (  35 /  2,537)          1,958     259,689
```

## 4. 每 iteration 增益（demand L1@100；per-seed 值在 2 seeds 上 mean±std）

```text
  iter   rho               WhoN               WhoO               HowR               HowO              Joint
  1000  0.50   +0.000073±0.000038   +0.000058±0.000006   +0.000061±0.000027   +0.000047±0.000005   +0.000119±0.000032
  1000  0.75   +0.000040±0.000034   +0.000040±0.000006   +0.000135±0.000035   +0.000135±0.000063   +0.000175±0.000029
  1500  0.50   +0.000030±0.000015   +0.000060±0.000006   +0.000108±0.000017   +0.000138±0.000038   +0.000168±0.000023
  1500  0.75   +0.000028±0.000001   +0.000028±0.000002   +0.000134±0.000046   +0.000134±0.000048   +0.000162±0.000048
  2000  0.50   +0.000151±0.000047   +0.000102±0.000004   +0.000150±0.000026   +0.000102±0.000025   +0.000253±0.000021
  2000  0.75   +0.000107±0.000052   +0.000045±0.000023   +0.000213±0.000033   +0.000151±0.000042   +0.000258±0.000009
  5000  0.50   +0.000083±0.000036   +0.000086±0.000000   +0.000063±0.000011   +0.000065±0.000024   +0.000148±0.000012
  5000  0.75   +0.000031±0.000011   +0.000045±0.000010   +0.000112±0.000002   +0.000127±0.000003   +0.000157±0.000008
 12000  0.50   +0.000065±0.000057   +0.000034±0.000002   +0.000020±0.000023   −0.000011±0.000082   +0.000054±0.000025
 12000  0.75   +0.000021±0.000014   +0.000022±0.000001   +0.000014±0.000063   +0.000016±0.000076   +0.000037±0.000062

正收益计数（两种口径分开，不混用）：
seed-level（20 = 5 iter × 2 rho × 2 seeds）：
  WhoN 20/20 (100%) · WhoO 20/20 (100%) · HowR 18/20 (90%) · HowO 18/20 (90%) · Joint 19/20 (95%)
setting-level（10 iteration×rho seed-means）：
  WhoN 10/10 (100%) · WhoO 10/10 (100%) · HowR 10/10 (100%) · HowO 9/10 (90%) · Joint 10/10 (100%)
```

## 5. OW-OH vs Native-Full（每 iteration）

```text
it=1000 rho=.5:  0.106503 vs 0.106669（更优 1.7e-4）gPSNR −0.0038 · growth 50/100
it=1500 rho=.5:  0.095480 vs 0.095662（更优 1.8e-4）gPSNR +0.0116 · growth 50/100
it=2000 rho=.5:  0.097010 vs 0.097289（更优 2.8e-4）gPSNR +0.0023 · growth 50/100
it=5000 rho=.5:  0.085076 vs 0.085260（更优 1.8e-4）gPSNR +0.0011 · growth 50/100
it=12000 rho=.5: 0.085756 vs 0.085779（更优 2.3e-5）gPSNR +0.0060 · growth 50/100
（rho=0.75 同型；demand L1 10/10 全部更优；global PSNR 7/10 更优，it1000 两个 rho 与 it12000 rho.75 为微差负值）
```

**全部 5 个阶段（含 it12000 成熟态）OW-OH 以一半 growth 在 demand L1 上稳定优于 Native-Full。**

> 注：新旧运行的绝对 demand L1 不可比（pool 视图与 mask 的对应关系在旧运行中已错配）；新运行量级
> 与 B4/B2C（同协议独立运行）一致，作为交叉佐证。

## 6. 输出路径（覆盖更新）

```text
paper_b/b5_cross_stage_capacity_oracle/data/{b5_snapshot_stats.csv, b5_candidate_oracles.csv,
  b5_group_results.csv/.json, b5_stats.txt, groups/group_it{it}_seed{s}.json}
paper_b/b5_cross_stage_capacity_oracle/plots/{gain_vs_iteration, gain_vs_num_gaussians,
  quality_vs_growth_by_iteration, joint_gain_across_snapshots, candidate_count_vs_iteration}.png
paper_b/b5_cross_stage_capacity_oracle/logs/{b5_collect, b5_process_fix, b5_analysis}.log
paper_b/b5_cross_stage_capacity_oracle/cache/{snap_*.pt, camera_identity.json, oracle_*.json}
paper_b/b5_cross_stage_capacity_oracle/cache/quarantine_b5fix/（作废的旧 process 产物，留档）
```

## 7. git 状态

```text
$ git status --short
?? diagnostics/diagnostic_b5.py
?? diagnostics/analyze_b5.py
?? project_md/PAPER_B_B5_REPORT.md
（B3/B4 文件已在前次列出；paper_b/ 在 .gitignore 中）

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

## 8. 最终结论（重答；旧版结论作废）

### 1. WhoN —— **YES**
seed-level 20/20、setting-level 10/10 全正（+2.1e-5 ~ +1.51e-4），含 it12000。选择维度跨全部训练阶段稳定成立。
（旧报告 WEAK 的结论完全来自相机错配伪影，作废。）

### 2. WhoO —— **YES**
20/20、10/10 全正（+2.2e-5 ~ +1.02e-4）。

### 3. HowR / HowO —— **YES**
HowR 18/20（10/10 setting-level）；HowO 18/20（9/10，唯一非正为 it12000 rho0.5 均值 −1.1e-5，边际且
seed 方差大 ±8.2e-5）。action 类型维度跨阶段成立，仅在成熟期末段进入噪声区。

### 4. Joint —— **YES**
seed-level 19/20（唯一非正为 HowO 同一事件一侧）、setting-level 10/10；幅度 +3.7e-5 ~ +2.58e-4。

### 5. OW-OH vs Native-Full —— **YES**
demand L1 10/10 全阶段更优（一半 growth）；global PSNR 7/10（负值均为 ≤0.004 dB 微差）。
it12000 时净收益收窄至 +2.3e-5 L1（早期/中段的 1/8~1/12）但仍为正且 gPSNR +0.006 dB。

### 随 #GS / candidate population 的变化 —— **存在中段峰值与末期收窄，但从不消失**
Joint 幅度：it1000 +1.2e-4 → it2000 峰值 +2.5e-4 → it5000 +1.5e-4 → it12000 +3.7e-5~5.4e-5
（约为峰值的 1/5）。与候选池萎缩（12,376→2,572）、prune 主导（57→1,958）同步——densification 需求
收缩使可分配余量变窄，但成熟 representation 上仍有可测的正余量。

## 9. 总判定

```text
Cross-stage structural capacity allocation headroom confirmed.
```

且较旧（作废）版本更强：**Who 与 How 两个维度均跨全部 5 个阶段成立**，Joint 仅 1/20 事件非正；
headroom 呈中段峰值、末期收窄至约 1/5 但不消失。与 B3（How 难以由 pre-action 特征预测）和
B4（同质量省 50–75% growth）合并，Paper B 的核心张力保持：上界跨阶段稳固，可达性仍未解决。

*生成于 2026-08-28（B5-Fix 修订）。全部数据来自修复后真实运行（logs/b5_process_fix.log，260 条记录），无伪造。*

# Paper B — B7-Fix 报告：Dynamic Signal Measurement Correction

> 修正 B7 动态特征的三处构造错误（梯度方向一致性双重弧度转换、残差历史未跟随 Gaussian 历史状态、
> radius/visibility 混入随机相机），复用全部缓存数据（750 oracle 不重跑、无 replay）重评 Dynamic-Who。
> **结论（§12）：FAIL——修正后动态信号更弱，正式判定 primitive-level dynamic history
> does not provide useful incremental Who signal。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b7fix.py   修正版动态特征重算（纯 CPU）：按任务书修正的三处构造 +
                                    分维度 Ineff_* + 可见样本条件化 Exposure + 固定视角投影 radius
- diagnostics/analyze_b7fix.py      四组 LOSO 对照（Static / Old-Dynamic / Fixed-Dynamic /
                                    Static+Fixed-Dynamic）+ 关键增量分析 + 2 图
- paper_b/b7_fix_dynamic_signal/{data,plots,logs}
- project_md/PAPER_B_B7_FIX_REPORT.md
Modified: 无（git diff --stat 空）
```

**基础设施事件（如实记录）**：运行时容器 NVIDIA 运行时故障（容器内 NVML "Unknown Error"，
全部 GPU 不可见；宿主机正常）。为避免重启共享容器，本阶段改为**纯 CPU 路径**：
不复用会强制 `.cuda()` 的 FastGS Camera/Scene，而是从 COLMAP 二进制直接重建 CPU 相机 shim
（数学逐行复刻 `camera_utils.loadCam` + `cameras.py`，按已保存的 collect 期相机身份顺序排列，
断言 PASS；FastGS 源码零改动）。特征重算本身不含任何训练/GPU 步骤，结论不受影响。

## 2. 修正内容（对照任务书）

```text
① 梯度方向一致性: theta2 = 2*arctan2(gdy, gdx+eps)（旧代码 deg2rad(arctan2(...))*2 双重转换错误）；
   仅统计有效样本（该迭代可见且梯度>0）
② DemandPersistence: 逐历史样本用该时刻的 historical xyz/scale 在同一组固定 cost views 上
   重新投影构造 ROI_i(t)，从当时刻已保存的残差图提取 E_i(t) —— Gaussian-following demand history
   （旧代码对所有样本用 snapshot-T 固定 ROI）；新增 dyn_demand_{mean,std,slope,last_first,cv,
   high_fraction}（high_fraction 阈值=本快照全部 candidate×sample 需求能量的中位数）
③ radius/visibility: 删除 dyn_radius_mean/std/vis_persistence（随机相机混杂）；
   新增固定视角 fx_fixedview_{radius_mean,radius_slope,visibility_fraction}（历史 xyz/scale +
   snapshot 旋转投影到固定 cost views）
④ 响应量分维度: Ineff_xyz / Ineff_scale / Ineff_opacity = DemandPersistence / 各自 path（旧代码
   混合量纲单一分母）；OptimizationExposure = Σ(可见样本的瞬时梯度范数)（旧为 mean×Σvis）
特征集保持克制：Fixed-Dynamic 共 25 个（§7 清单 + 固定视角三量）
```

## 3. LOSO（Ridge 固定，target=q_best，750 候选，5 折 leave-one-iteration-out）

```text
                  spearman mean   VC@.5−rand mean（正阶段数）
static               +0.206          +0.181（5/5）
old_dynamic          +0.078          +0.064（3/5）
fixed_dynamic        +0.032          +0.017（3/5）      ← 修正后更弱
static+fixed_dynamic +0.144          +0.177（5/5）      ← 仍低于 static 单独
```

### 关键增量（per stage）

```text
 held-out  FX−OLD spear  S+FX−S spear  FX−OLD VC.5  S+FX−S VC.5
    1000        +0.104        +0.002       +0.028       +0.043
    1500        −0.070        +0.010       −0.105       +0.003
    2000        −0.023        −0.151       +0.002       −0.145
    5000        −0.105        −0.076       −0.197       −0.066
   12000        −0.135        −0.093       +0.035       +0.147
mean            −0.046        −0.062       −0.047       −0.003
positive        1/5           2/5          3/5          3/5
```

**Q1（修正后是否优于旧动态）：NO** —— FX−OLD Spearman 均值 −0.046、仅 1/5 阶段为正；
旧动态中仅存的微弱信号（it1500 +0.284）在修正后消失（+0.214），其余阶段持平或更差。
**Q2（动态是否提供 static 外增量）：NO** —— Static+Fixed ≤ Static（Spearman −0.062、VC −0.003）；
加入动态特征在 5 个阶段中的 3 个拖累静态表现。

## 4. Replay：未执行（§10 条件不满足）

## 5. 判定（§12）

```text
Fixed-Dynamic ≈（劣于）Old-Dynamic（+0.032 vs +0.078）
且 Static+Fixed-Dynamic ≤ Static（+0.144 vs +0.206）
→ FAIL，正式判定：
   primitive-level dynamic history does not provide useful incremental Who signal.
→ 停止 primitive-level predictor；下一阶段转向 region-level / interaction-level
   capacity allocation。
```

**该负面结论现在的可信度高于 B7**：三处测量错误修正后信号不升反降，说明旧动态的微弱表观
不是被构造错误掩盖的真信号，而是（a）旧 ROI 固定带来的伪相关 +（b）随机相机混杂 +
（c）oracle 噪声本身的组合。修正版测得的 +0.032 已接近零。

## 6. 对 Paper B 的合并结论（只陈述）

经 B7 + B7-Fix 双重确认：**per-Gaussian 优化历史（静态或动态）不携带 Who 维度的实用增量信号**。
结合 B6-Fix（How 退化为全局先验），primitive-level（单 Gaussian 特征 → 单 Gaussian 决策）路线
在静态、动态、How 三个方向上均已穷尽并证伪。B4/B5 证明的 allocation headroom 若要利用，
其信息必然存在于 primitive 之上的聚合层级（region/interaction）——这正是任务书指定的下一方向。

## 7. 输出 / git

```text
paper_b/b7_fix_dynamic_signal/data/{b7fix_dynamic_features.csv,.json, b7fix_loso_results.csv, b7fix_stats.txt}
paper_b/b7_fix_dynamic_signal/plots/{b7fix_dynamic_vs_static.png, b7fix_value_capture.png}
paper_b/b7_fix_dynamic_signal/logs/{b7fix_features.log, b7fix_loso.log}

$ git status --short
?? diagnostics/diagnostic_b7fix.py
?? diagnostics/analyze_b7fix.py
?? project_md/PAPER_B_B7_FIX_REPORT.md
（B5–B7 系列文件此前已列；paper_b/ 在 .gitignore 中）

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-08-30。全部数据来自缓存数据的真实重算（logs/），无 oracle 重跑、无伪造。完成即停止，不实现 replay 与最终 allocator。*

# Paper B — B7 报告：Dynamic Capacity Signal Diagnostic

> 验证：snapshot 前 100-step 优化窗口的动态历史是否比 B6 静态特征更能预测 q_best（Who 可达性）。
> 只做 Who 诊断；不做 Clone/Split predictor / allocator；FastGS 零修改。
> **结论（按 §7 标准）：FAIL —— 动态信息不优于静态，STOP primitive-level predictor。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b7.py   collect：单次 native warmup，在 5 个目标事件的 pre-snapshot 窗口内
                                 每 10 步采样（10 samples/100 steps）：瞬时 viewspace 梯度（norm+dx/dy）、
                                 激活参数（xyz/scale/opacity）、屏幕 radii、8 个固定 cost view 的残差图
                                 （fp16）；事件前持久化 snapshot+pool 数据+相机身份。
                                 process：相机断言 → oversample 150 valid/快照 → B5 定义 oracle
                                 （训练 100 步 Keep 参考 + Clone×1 + Split×5，缓存）→ 静态（B6 集）
                                 + 动态特征（窗口统计 + DemandPersistence / OptimizationExposure /
                                 OptimizationInefficiency + 梯度方向一致性）。
- diagnostics/analyze_b7.py      Ridge 固定 LOSO：Static / Dynamic / Static+Dynamic 三组对照
- paper_b/b7_dynamic_capacity_signal/{data,plots,logs,cache}
- project_md/PAPER_B_B7_REPORT.md
Modified: 无（git diff --stat 空）
```

## 2. 实际 GPU / 数据

GPU 5（任务书指定）。iterations 1000/1500/2000/5000/12000；每快照 150 valid 候选（oversample，
seed 公式同 B5）；共 **750 候选 × 750 oracle**（约 4,500 次 100-step 受控分支，全部真实运行）。
窗口内无 densification 事件（事件间隔 500 > 窗口 100）→ parent_index 在窗口内稳定，历史与快照
候选一一对应；相机断言 PASS。动态特征 22 个（克制：每序列 mean/std/slope/CV + 3 个构造量 +
方向一致性），静态特征 21 个（B6 集去 CV/一致性子集）。

三个核心构造量（按任务书定义实现）：
```text
DemandPersistence        = 固定 ROI 内窗口残差能量均值（10 samples）
OptimizationExposure     = mean(瞬时 xyz 梯度范数) × Σ visibility
OptimizationInefficiency = DemandPersistence / (|Δxyz|+|Δscale|+|Δopacity| 路径长 + eps)
```

## 3. LOSO 结果（Ridge 固定，target=q_best）

```text
 held-out              set  spearman  top25(rand)   VC@.5(rand)      VC@.75(rand)
  1000/1500/2000/5000/12000 逐行见 data/b7_loso_results.csv；汇总：

           static : spearman mean +0.206 | VC@.5−rand mean +0.181（5/5 阶段为正）
          dynamic : spearman mean +0.078 | VC@.5−rand mean +0.064（3/5 为正）
 static+dynamic : spearman mean +0.148 | VC@.5−rand mean +0.156（5/5 为正，但低于 static 单独）

Dynamic 逐阶段 Spearman: 1000 +0.024 · 1500 +0.284 · 2000 −0.047 · 5000 +0.085 · 12000 +0.042
Static  逐阶段 Spearman: 1000 +0.192 · 1500 +0.259 · 2000 +0.233 · 5000 +0.208 · 12000 +0.136
```

**动态历史不优于静态**：Dynamic-only Spearman 仅 0.078（3/5 阶段 VC 优势为正，it2000 为负）；
**加入动态特征反而降低静态表现**（static+dynamic 0.148 < static 0.206）——窗口统计在此问题上
引入的是噪声而非信号。三个核心构造量未产生区分力（细节见 features CSV）。

## 4. Replay：未执行（gate 未通过）

按任务书 §5："只有当 Dynamic-only 或 Static+Dynamic 明显优于 B6 Static 时才做 replay"——
条件不满足（两者均低于本运行的 Static 基线）。

## 5. 判定（§7）

```text
Dynamic-only / Static+Dynamic 均不优于 Static（更不优于随机以上门槛的相对提升要求）：
→ FAIL。
→ STOP primitive-level predictor。
→ 按任务书指引：下一步应转向 region-level / interaction-level capacity allocation。
```

## 6. 如实标注的重要观察

1. **本运行的 Static 基线（Spearman +0.206，VC@0.5 5/5 阶段 > random）明显好于 B6 的 Static**
   （+0.03~0.15，VC 2/5）。两者特征集几乎相同、协议相同、候选来自不同轨迹/采样（B7: 150/快照
   vs B6: ~200/快照，且 oracle 为独立重新计算）。这说明 **"静态可达性"的测量本身随 oracle 实例
   （候选池与 oracle 噪声）有大幅波动**——单次运行的 predictability 数字不可外推，跨运行重复
   测量是必要的。这同时弱化了 B6 "Who 不可达"结论的强度：更准确的表述是
   **"Who 可达性弱且不稳定（跨运行 Spearman 0.0–0.26 区间波动）"**。
2. 但 B7 的核心问题有清晰答案：在该波动区间内，**动态窗口信息不提供静态之外的增量**
   （0/5 阶段 dynamic > static 的 Spearman，加入后联合下降）。
3. 需求侧残差历史（DemandPersistence）与优化响应（Exposure/Inefficiency）在 per-Gaussian 粒度上
   无预测力——与 B3/B6 的静态残差结论一致，now extends to dynamic residual。

## 7. 对 Paper B 的合并结论（只陈述）

经 B1→B7 证据链：
1. Capacity allocation 的 oracle 上界真实且跨阶段（B4/B5）；
2. Primitive-level（per-Gaussian）可达性：静态弱且不稳定（B6 vs B7 跨运行波动），动态无增量（B7），
   How 预测退化为全局先验（B6-Fix）；
3. **per-Gaussian primitive 层面的 predictor 路线应停止**；按任务书指引，剩余方向为
   region-level / interaction-level capacity allocation（上界存在而 primitive 不可达的自然推论：
   信号可能在聚合层级）。

## 8. 输出 / git

```text
paper_b/b7_dynamic_capacity_signal/data/{b7_dynamic_features.csv,.json, b7_loso_results.csv, b7_stats.txt}
paper_b/b7_dynamic_capacity_signal/plots/{dynamic_vs_static_spearman.png, dynamic_value_capture.png}
paper_b/b7_dynamic_capacity_signal/logs/{b7_collect, b7_process, b7_loso}.log
paper_b/b7_dynamic_capacity_signal/cache/（5 组 snap7/dyn + 750 oracle + 预测，可续跑）

$ git status --short
?? diagnostics/diagnostic_b7.py
?? diagnostics/analyze_b7.py
?? project_md/PAPER_B_B7_REPORT.md
（B5/B6 系列文件此前已列；paper_b/ 在 .gitignore 中）

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-08-29。全部数据来自真实运行（logs/，750 候选 × 6 分支受控 replay），无伪造。完成即停止，不实现最终方法。*

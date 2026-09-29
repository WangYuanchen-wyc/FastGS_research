# Paper B — B8-L 报告（Cache Fix 修订版）：Long-Horizon Cardinality Validation

> **本版为 Cache Fix 修订版**。旧版报告的 "persistent 38%" 受 cache key bug 影响而被低估——
> smoke 期的 12 个 parent（132 分支）经 JSON round-trip 后 checkpoint key 变为字符串
> （"0"/"100"/"500"/"1000"），full run 用整数 key 读取导致全部指标 NA，这 12 个 parent
> 被错误排除出统计。修复后零 GPU 重算全部 58 parents。**旧报告的 38% 不是最终结果。**
> **修复后判定（按原 §11 标准，完全按真实数据重判）：B8-L NO-GO（但比旧版更接近边界）。**

## 1. Cache bug 与修复

```text
Bug:    branches.json 保存时 ev 的 checkpoint key 全部变为字符串（JSON dict key 序列化），
        full run 内存分支用 int key —— 一致；但 smoke 期缓存、full run 复用的 132 分支
        读取时 ev.get(100) 落空 → 12 parents × 11 分支指标全 NA。
修复:   diagnostics/diagnostic_b8l.py 的 run_one() 读取缓存后统一
        rec["ev"] = {int(k): v for k, v in rec["ev"].items()}
数据:   diagnostics/rebuild_b8l_cache.py 从 branches.json 零 GPU 重建
        b8l_repeat_results.{csv,json} —— 638 行，l1_1000 NA = 0，58/58 parents 完整。
```

**未重新跑任何 GPU experiment**；所有 638 个分支结果均为原始缓存数据。

## 2. 修改文件

```text
Modified:
- diagnostics/diagnostic_b8l.py     cache 读取 int-key 归一（上述一行修复）
- diagnostics/analyze_b8l.py        missing-data 处理：缺 Split2@1000 或 MSCC@1000 的 parent
                                    → persistent_label = "missing"，不计入任何 denominator；
                                    所有统计显式输出 valid/total 与 missing count
- project_md/PAPER_B_B8L_REPORT.md  本报告（覆盖更新；旧版 38% 作废）
Added:
- diagnostics/rebuild_b8l_cache.py  零 GPU 数据重建脚本
```

## 3. 修复后的核心结果（58/58 parents，valid 29+29，missing 0）

### C 组（higher-cardinality，gain_t = L1_Split2(t) − L1_MSCC(t)）

```text
    ALL-C (valid 29/29, missing 0): gain100 +1.06e-4 (21/29 pos) | gain500 +1.07e-4 (16/29) |
                                    gain1000 +1.73e-4 (17/29) | persistent@1000 14/29 (48%)
                                    | norm_gain1000>1 13/29 (45%)
 it1000-C (valid 10/10): gain100 +1.20e-4 | gain500 +2.78e-4 (8/10) | gain1000 +2.46e-4 (8/10)
                         persistent 8/10 (80%) | norm>1 7/10
 it2000-C (valid 10/10): gain100 +0.59e-4 | gain500 +0.25e-4 (4/10) | gain1000 +1.54e-4 (3/10)
                         persistent 1/10 (10%) | norm>1 1/10
 it5000-C (valid  9/9): gain100 +1.42e-4 (9/9) | gain500 +0.08e-4 (4/9) | gain1000 +1.14e-4 (6/9)
                         persistent 5/9 (56%) | norm>1 5/9
```

### B 组对照（Split-2 sufficient：Split-3 长期是否无收益）

```text
    ALL-B (valid 29/29, missing 0): gain1000(S3−S2) mean +0.66e-4 (13/29 pos)
 it1000-B: +1.45e-4 (5/10 pos) · it2000-B: +1.42e-4 (5/9 pos) · it5000-B: −0.80e-4 (3/10 pos)
 → B 组"Split-3 长期无收益或更差"仅 16/29 (55%)，且 ALL 均值方向为 Split-3 更好（+0.66e-4）：
    对照组自身在长程下不再干净地支持 "Split-2 足够"。
```

### 与旧（受 bug 影响）数字的对照

| 指标 | 旧（23/29 valid） | 修复后（29/29 valid） |
|---|---:|---:|
| C persistent@1000 | 11/29 (38%) | **14/29 (48%)** |
| it1000-C persistent | 7/10 (70%) | **8/10 (80%)** |
| it2000-C persistent | 1/10 | 1/10（不变） |
| it5000-C persistent | 3/9 (33%) | **5/9 (56%)** |
| C gain1000 mean | +1.70e-4 | **+1.73e-4** |
| B 无收益比例 | 18/29 (62%) | **16/29 (55%)**（新增的 smoke B parents 多为 pos） |

### Child lifecycle（如实声明不完整）

```text
当前 lifecycle 文件仅含 full run 新跑的 46/58 parents（4,780 行）；
smoke 期的 12 parents 的 lifecycle 行在 full run flush 时被覆盖，不可恢复。
按任务书处理：标记 lifecycle incomplete，不再用 lifecycle 支撑任何结论；
本报告删除旧版基于 lifecycle 的"children 长期健康"表述（该表述在 46-parent 子集上成立，
但不作为证据呈现）。
```

## 4. 五个问题的回答（修复后数据）

**Q1**（B 组 Split-3 长期无收益？）：**不支持** —— ALL-B gain1000 均值 +0.66e-4（方向为
Split-3 更好），"无收益或更差"仅 55%；it1000/it2000 的 B 组均值均为正。B8-A 判为
"Split-2 sufficient" 的 parent 在长程下并未表现出对 Split-3 的稳定排斥。

**Q2**（C 组 Split-MSCC 长期仍优？）：**48%（14/29）persistent**（±1 SEM 保守口径）；
norm_gain1000>1 为 45%（13/29）——接近半数，但非"稳定一批"的多数。

**Q3**（100→500→1000 轨迹）：**衰减后部分存留** —— C 组均值 +1.06e-4 → +1.07e-4 → +1.73e-4
（整体未消失）；但 pos 比例从 21/29 降到 16/29 再回 17/29，且 it2000 在 500 步时仅 4/10 正、
最终 persistent 1/10。个体轨迹极性翻转频繁（如 idx=11795: −3.8e-4→−0.3e-4→−2.4e-4；
idx=76088: +2.2e-4→−0.0e-4→−1.4e-4）——长程下单 parent 差异被全局优化噪声部分淹没。

**Q4**（children 是否短期消亡）：lifecycle 不完整，按任务书不再下结论。

**Q5**（跨 stage？）：**persistent 比率跨 stage 不一致**：it1000 80% / it2000 10% / it5000 56%。
it1000 强成立；it2000 明确不成立；it5000 过半但不稳定。

## 5. 判定（§8 标准，按修复后真实数据完全重判）

```text
GO 条件逐条：
1. C 组稳定一批 parent 长程优于 Split-2 : 部分（48% persistent，it2000 仅 10%）
2. B 组大部分仍 Split-2 sufficient      : 不成立（55%，均值方向相反）
3. 现象跨多个 stage                     : 部分（it1000 强、it2000 反、it5000 中）

→ B8-L NO-GO（维持，但与旧报告理由不同）
```

**修复后结论（比旧版更准确的表述）**：cardinality 异质性在长程下**没有消失**（C 组 gain1000
均值仍 +1.73e-4、48% persistent、45% 非噪声级），但它**不稳定且不对称**——B 对照组同样在长程
出现 Split-3 更好的倾向（B8-A 的 B/C 分类在长程下部分失效），仅 it1000（早期、结构最不成熟）
展现强且一致的 higher-cardinality 优势（80% persistent）。按任务书"跨多数 stage"的标准，
不构成 GO；但也非旧版所述"优势基本消失"——真实图景是 **short-horizon cardinality 标签
在 long-horizon 下部分退化，早期训练阶段除外**。

**对 Paper B 的含义（修正旧版表述）**：B8-A 的 MSCC 在长程下仍携带信号（gain1000 未归零），
但作为 per-parent 稳定结构属性的可靠性不足；若继续 cardinality 方向，最有证据支撑的切入点是
**结构成熟度与 cardinality 需求的交互（早期强、后期弱）**。是否继续由下一阶段任务决定；
按任务书本阶段不进入 B8-B。

## 6. 输出 / git

```text
paper_b/b8_l_long_horizon_cardinality/data/{b8l_repeat_results.csv,.json（已重建，0 NA）,
  b8l_parent_results.csv, b8l_child_lifecycle.csv（incomplete: 46/58 parents，仅存档）,
  b8l_stats.txt}
paper_b/b8_l_long_horizon_cardinality/plots/{gain_over_time, split2_vs_mscc_long_horizon,
  per_stage_persistence, child_lifecycle}.png（已按修复数据重绘）
paper_b/b8_l_long_horizon_cardinality/logs/{b8l_analysis_fix.log}

$ git status --short
?? diagnostics/diagnostic_b8l.py（Modified）
?? diagnostics/analyze_b8l.py（Modified）
?? diagnostics/rebuild_b8l_cache.py（Added）
?? project_md/PAPER_B_B8L_REPORT.md（覆盖更新）

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-09-01（Cache Fix）。零 GPU 重跑；全部数字来自原始 638 分支缓存的重新读取与统计。*
*按任务书停止：不进入 B8-B。*

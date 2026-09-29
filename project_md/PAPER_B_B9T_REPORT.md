# Paper B — B9-T 报告：Temporal Split Demand

> 验证 stronger split（parent→3 children vs FastGS 原生 2）的价值是否随训练阶段变化。
> Model-level 完整训练（5 strategies × 2 seeds × 30k iters，GPU 7，约 5.5h GPU 时间）。
> **判定（§12）：B9-T NO-GO —— 假设的 temporal pattern（早期强、后期弱）不成立；
> 实际模式更复杂：所有 Split-3 变体都加速收敛（~10-19%），但 final 质量差异不显著，
> 且 Late-3 而非 Early-3 拥有最佳 final PSNR。**

## 1. 修改文件

```text
Added:
- diagnostics/diagnostic_b9t.py   完整训练：native densify wrapper（Split-N 可配），
                                 5 strategies × 2 seeds × 30000 iters 全程，
                                 每 500 iters 记录 train/test PSNR/SSIM/LPIPS + #GS + wall-clock
- diagnostics/analyze_b9t.py      time-to-quality（Q90/95/99）+ final metrics + PSNR-vs-#GS + 6 图
- paper_b/b9_t_temporal_split_demand/{data,plots,cache,logs}
- project_md/PAPER_B_B9T_REPORT.md
Modified: 无（git diff --stat 空；Split-N 调用原生 densify_and_split_fastgs 的 N 参数）
```

## 2. 实验设置

```text
Strategies: native（全 Split-2） · early3（it1000-5500 用 Split-3）· middle3（5500-10000）
            · late3（10000-14500）· always3（全程 Split-3）
Seeds: 2（如实注明：非 3 seeds，按任务书降级为诊断级）
完整训练: 30000 iterations（FastGS 原生终点），每 500 iters 记录
除 split cardinality 外全部原生一致：VCD metric / prune / opacity reset / final prune /
optimizer schedule（3 段步频）/ camera sampling（同 seed）
Stage 划分（基于真实 schedule，事件 1000-14500 步长 500）：
  Early = 1000-5500（10 events） · Middle = 5500-10000（9 events） · Late = 10000-14500（9 events）
  每事件平均 split parents：~67k（Early）· ~67k（Middle）· ~65k（Late）
```

## 3. Final Metrics（30k iters，2 seeds mean±sd）

```text
 strategy      PSNR              SSIM            LPIPS           #GS      time(s)
   native   32.153±0.228   0.9238±0.0008   0.2129±0.0099    207,120     2073
   early3   31.968±0.177   0.9247±0.0006   0.2098±0.0098    236,108     1863
  middle3   32.083±0.233   0.9248±0.0004   0.2101±0.0098    231,044     2009
    late3   32.176±0.218   0.9251±0.0006   0.2097±0.0099    230,412     2016
  always3   32.162±0.313   0.9263±0.0008   0.2054±0.0103    271,837     1998
```

**Final PSNR 差异全部在 seed 标准差（±0.18-0.31）内**——任何策略都不显著优于/劣于 native。
SSIM/LPIPS 有微弱倾向：always3 最好（SSIM +0.0025 / LPIPS −0.0075），但同样不显著。

## 4. Time-to-Quality（核心加速指标）

```text
 strategy    Q90 (s)     Q95 (s)     Q99 (s)     Q95 加速比
   native    487         916         1427         —
   early3    418(-14%)   739(-19%)   1238(-13%)   1.24×
  middle3    480(-1%)    817(-11%)   1517(+6%)    1.12×
    late3    479(-2%)    822(-10%)   1283(-10%)   1.11×
  always3    459(-6%)    782(-15%)   1525(+7%)    1.17×
```

**所有 Split-3 变体在 Q95 上都快于 native（10-19%）**，其中 early3 最快（-19%）。
但 **这不是 early-specific 的**——middle3/late3/always3 也有 10-15% 的加速。

## 5. Representation Cost

```text
#GS 相对 native: early3 +14.0% · middle3 +11.6% · late3 +11.2% · always3 +31.2%
```

always3 以 31% 额外 GS 换取 +0.01 dB PSNR 和 15% 加速——收益主要来自"更多参数+更快达到"
而非"更好的结构"。early3 以 14% 额外 GS 换取 19% 加速但 **final PSNR 反而低 0.19 dB**。

## 6. 关键发现（如实）

1. **无 clean temporal pattern**：假设是"early 强、late 弱"，但 late3 拥有最好的 final PSNR
   （+0.02 dB）且加速比与 early3 几乎相同（10% vs 19%——差距不大）。三个 stage 窗口的
   Split-3 变体表现高度相似，差异在 seed 噪声内。

2. **加速来自更多 Gaussians，不来自"更好的结构"**：
   - 所有 Split-3 的 PSNR-vs-#GS 曲线与 native 几乎重合（见 psnr_vs_gs.png）
   - 加速主要因为更多初始 capacity 让优化更早进入有效区间

3. **Early3 的悖论**：Q95 最快（739s vs native 916s）但 final PSNR 最低（-0.19 dB）——
   早期 Split-3 加速了收敛但 **可能过度参数化了早期结构**，导致后续优化的 final 质量略差。
   这与 B8-L 的 it1000 70-80% persistent 形成有趣对照（B8-L 中早期 parent 确实从更高 N 获益），
   但 model-level 全局效果是负的——**局部增益不等于全局收益**。

4. **opacity reset 处的暂态**（it3000/12000 PSNR 骤降至 ~9.5 再恢复）：所有策略一致，
   是 FastGS 原生行为（reset 后模型需要数百步恢复），不影响比较。

## 7. GO / NO-GO（§12 标准）

```text
1. Early-3 明显更快达到相同质量            : Y（Q95 −19%）——但所有 Split-3 变体都加速
2. Middle/Late-3 收益明显减弱              : N（late3 加速 10% 且 final PSNR 最佳）
3. Always-3 无同比例长期收益且明显增 #GS    : Y（+31% GS 换 +0.01 dB）
4. 多 seeds 稳定                           : 部分（方向一致，幅度在噪声内）
5. 完整训练指标可见                        : Y

→ B9-T NO-GO（核心条件 2 失败：temporal pattern 不是"早期强后期弱"）
```

**核心结论**：Split-3 的加速效果**与训练阶段基本无关**（所有窗口都 ~10-19%），
加速来自更多 Gaussians 而非 stage-dependent structural need。假设的
"早期结构未成熟时 stronger split 更有价值"在 model-level 全局训练中**不成立**——
虽然 B8-A/B8-L 在局部 ROI 确实观察到了早期 parent 的 higher-N 优势（it1000 80% persistent），
但当应用于**全部**早期 split candidates 时，model-level 效果反而为负。

## 8. 对 Paper B 的含义（只陈述）

1. **cardinality 主线的证据链闭环**：B8-A（100-step GO）→ B8-L（1000-step NO-GO）→
   B9-T（30k-step NO-GO，无 temporal pattern）——split cardinality 的局部收益
   在全局完整训练尺度上不转化为持续质量优势或 stage-dependent 需求。

2. **唯一保留的正面信号**：所有 Split-3 变体都加速收敛 ~10-19%——这是"更多 Gaussians 加速
   拟合"的 generic effect（PSNR-vs-#GS 曲线重合佐证），不是 cardinality-specific 信号。

3. Paper B 应**停止 temporal-cardinality 主线**。下一阶段需要真正不同的信息源
   （multi-view structural evidence、或完全不同的问题形式）。

## 9. 输出 / git

```text
paper_b/b9_t_temporal_split_demand/data/{training_curves.csv, final_metrics.csv,
  split_event_stats.csv, b9t_stats.txt}
paper_b/b9_t_temporal_split_demand/plots/{psnr_vs_iteration, psnr_vs_time, gs_vs_iteration,
  psnr_vs_gs, final_quality_cost, time_to_quality}.png
paper_b/b9_t_temporal_split_demand/logs/{b9t_smoke, b9t_full, b9t_analysis}.log

$ git status --short
?? diagnostics/diagnostic_b9t.py
?? diagnostics/analyze_b9t.py
?? project_md/PAPER_B_B9T_REPORT.md

$ git diff --stat
（空 —— FastGS tracked source 零修改）
```

*生成于 2026-09-02。全部数据来自真实运行（10 × 30k-iter 完整训练，600 条 curve 记录），无伪造。*
*如实注明：2 seeds（诊断级），非 3 seeds——按任务书允许但不构成最终结论的充分统计量。*
*按任务书停止：不实现正式方法。*

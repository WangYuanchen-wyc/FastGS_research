# Paper B — B21-V1b 报告：Long-Horizon & Event Disentangling

> 严格回答三问：(1) 后期 densification 低收益是否只是 +400 太短？(2) 新增 Gaussian 本身有没有用？(3) Full event 效果差是否被伴随操作抵消？
> GPU 5→7（GPU 5 被外部进程占满后切换），Room，3 seeds × 30k，5 个预固定 event（1500 sanity + 5000/7000/11000/13000 low），四分支成对反事实（深度快照 + RNG 对齐 + 固定 ROI），horizons +100/+400/+1000/+2000，窗口内 native events/opacity reset 四分支同调度。
> **判定：Case A — Slow Benefit。V1 在 +400 看到的低收益主要是瞬态恢复伪影：Full-Event 在 +2000 追平 Skip-All（不再为负）。按预注册优先级，暂不能进入 Error-Aware V2。**

## 1. Event Decomposition（依据 FastGS 代码，`event_decomposition.csv`）

| 组件 | A Skip | B Add-Only | C Side-Only | D Full |
|---|---|---|---|---|
| creation：clone/split 追加（optimizer state 扩展 + 梯度累加器清零 + split 父点移除） | ✗ | ✓ | ✗ | ✓ |
| prune：opacity<0.005 / radii>20 / scale>0.1·extent 候选的 multinomial 半删 | ✗ | ✗ | ✓ | ✓ |
| opacity clamp：min(opacity, 0.8) | ✗ | ✗ | ✓ | ✓ |

C 分支的实际净增点数：seed0 上为 **−17 / −61,223 / −70,759 / −75,481**（it=5000–11000）——mid/late event 的伴随操作一次就要删除 6–7.5 万个点，量级远超 creation（+2k–10k）。

## 2. 核心数据（dPSNR vs Skip-All，正 = 该分支优于跳过）

seed 2（完整形态，其余两 seed 同构，见 `b21v1b_stats.txt`）：

| it | hz | B Add-Only | C Side-Only | D Full |
|---|---|---:|---:|---:|
| 5000 | +400 | −0.12 | −0.44 | −0.15 |
| 5000 | +1000 | +0.01 | −0.19 | −0.01 |
| 5000 | +2000 | **+0.10** | **−0.75** | **+0.21** |
| 7000 | +1000 | +0.21 | −1.12 | +0.17 |
| 7000 | +2000 | +0.02 | −0.23 | −0.03 |
| 11000 | +1000 | −0.00 | −0.28 | −0.03 |
| 11000 | +2000 | +0.00 | −1.15 | +0.01 |
| 13000 | +1000 | +0.12 | −1.49 | +0.12 |
| 13000 | +2000 | +0.02 | −0.37 | +0.01 |

Per-seed 均值（low events，+2000）：Add−Skip = −0.023 / −0.011 / **+0.036**；Full−Skip = +0.012 / +0.042 / **+0.054**；Side−Skip = **−0.973 / −0.536 / −0.627**。

Sanity（it=1500，Add−Skip @+2000）：**+0.359 / +0.041 / +0.364**——早期容量有真实价值（2/3 seeds 明显）。

## 3. 六问简答

1. **Full 的低收益只是见效慢吗？** 主要是。Full−Skip 在 +400 为负（−0.01~−0.53，creation 冲击 + prune/clamp 瞬态代价），到 +1000/+2000 追平并小幅转正。Q1 在 3/3 seeds、12/12 cells 成立。**但注意：是"追平到 0 附近"，不是反超**——+2000 时 Full 相对 Skip 的优势 ≤0.05 dB。
2. **Add-Only 本身有效吗？** 在预注册 0.1 dB 口径下**无明确收益**：+2000 的 Add−Skip 介于 −0.02~+0.04 dB（+1000 时 +0.07~+0.09，仍 <0.1）。Q2 在 3/3 seeds 成立。新增 4k–20k 个 Gaussian 的长窗口净回报 ≈ 0（微正）。
3. **Side-Effect-Only 有负作用吗？** 有且很大：Side−Skip @+2000 = **−0.54~−0.97 dB**（3 seeds）。且这是"prune+clamp 无 creation 补充"政策在窗口内被后续 native events 持续放大的累计效果。
4. **Full 与 Add 的差异来自哪里？** Full ≈ Add + 被部分抵消的 side damage：Full@+2000 略高于 Add（+0.01~+0.09 dB 均值差），远好于 Side 单独的 −0.5~−1.0——**creation 的收益在 Full event 内抵消了大部分 side-effect 损失**，两者不是简单相加（Side 单独 −0.6，加 creation 后 Full 反而 +0.05）。
5. **跨 seeds / events 一致？** 一致——Q1 3/3（12/12 cells）、Q2 3/3、Side 的负作用 3/3；唯 seed 间数值幅度有差（Side −0.54~−0.97）。
6. **下一步？** 按预注册优先级 A > C > B 且 Q1 成立：**Case A — Slow Benefit，暂不进入 Error-Aware V2**。同时应如实记录：Q2 的证据（Add-Only 长窗口 ≈0 净收益）与 Case B 方向一致但未达预注册门槛（+1000 均值 0.07–0.09 dB 恰在 0.1 门槛之下；+2000 归零）。

## 4. 诚实备注

- **本轮曾发现并修复关键执行 bug**：初版分支窗口循环漏写 `gaussians.optimizer_step(cur)`，四分支在窗口内只 backward 不 step（参数冻结；probe 实证：grads 非零、exp_avg 恒定）。修复后全部数据重跑；本报告数字全部来自修复后运行。初版错误数据未进入任何报告。
- **Q1 的"追平"是回到均势而非反超**：+2000 时 Full 对 Skip 的优势从未达到 +0.1 dB。因此 Case A 的准确表述是：**+400 的负收益是瞬态伪影，但这些 event 的长窗口净收益本身也 ≈ 0**——既不能断言"低收益 event 无价值"（Case B 未达门槛），也不能断言"有价值"。
- Side-Only 的 −0.5~−1.0 dB 部分来自窗口内后续 native events 在"只删不增"政策下的累计放大，不纯是单次 event 的 side effect；单次 side effect 的即时代价见 +100/+400（−0.5~−4.4 dB）。
- 实现自检（快照逐位恢复、前 100 次相机采样一致、同调度）全部通过，非 INVALID。

## 5. 产物

- `paper_b/b21_v1b_event_disentangling/data/event_decomposition.csv`
- `paper_b/b21_v1b_event_disentangling/data/branch_metrics{,_s*}.csv` — 240 行分支×horizon
- `paper_b/b21_v1b_event_disentangling/data/long_horizon_roi{,_s*}.csv`
- `paper_b/b21_v1b_event_disentangling/data/seed_event_summary.csv`、`b21v1b_stats.txt`
- 脚本：`diagnostics/diagnostic_b21v1b.py`、`diagnostics/analyze_b21v1b.py`；日志：`b21v1b_full.log`

# Paper B — B2 Multi-Candidate Local Action Oracle 报告

> 在 B1 已验证的 Keep/Clone/Split 受控协议上扩展：多 candidate（FastGS 真实候选）、
> candidate-supported 局部质量（support ROI + demand ROI）、action-specific tile cost
> （gaussian_tile_delta @K=0/@K=100）、pre-action candidate/residual descriptors。
> 本阶段仍为 Diagnostic：**未实现** predictor / allocator / budget / lineage / rollback。

## 0. 执行环境与偏离说明

| 项 | 值 |
|---|---|
| 容器 / 环境 | `wyc-compre` , `/opt/miniconda3/envs/fastgs`（torch 1.12.1+cu116） |
| 场景 | `/mnt/workspace/Dataset/room`（MipNeRF360, 311 cams, --eval, 5 个固定 test 全局 probe） |
| **GPU 偏离** | 任务指定 `CUDA_VISIBLE_DEVICES=2`，但正式运行时 GPU2 被他人任务占用（turbovla-libero, 18.6GB, PID 3726138，B1 之后启动）。为保证独占与可复现，改用**完全空闲的 GPU5**（0 MiB 起跑）。其余配置与任务书一致。 |
| 诊断点 | iteration 1000 / 1500 / 2000（room 原生 interval=500 下均为真实 densification 事件；1000 为首个事件，1500/2000 的 checkpoint 已包含此前原生事件的真实演化） |
| 复现命令 | `docker exec -e CUDA_VISIBLE_DEVICES=5 -w /mnt/workspace/FastGS wyc-compre /opt/miniconda3/envs/fastgs/bin/python diagnostics/diagnostic_v2.py -s /mnt/workspace/Dataset/room --eval --densification_interval 500 --grad_abs_thresh 0.0008 --diag_iters 1000,1500,2000 --n_cand 20 --n_probe 8 --out_tag action_diag_v2` |

## 1. 协议要点（B1 全部保留）

- 三分支同一 checkpoint（逐分支深拷贝 restore，含双 optimizer state）、同一 100 步相机序列（`Random(2024)`）、同一 RNG 控制（action 前 seed=1234 / 训练前重置 seed=1234）；100 步内参数优化开启，densification / pruning / opacity reset 全部关闭。
- Clone/Split 复用 FastGS 原生 `densify_and_clone_fastgs` / `densify_and_split_fastgs`（单 parent mask），未重实现。
- ROI 在 action 前固定：parent 投影 3σ 屏幕半径（渲染器原生 `radii`）+ 1 tile(16px) margin，clamp 到图像内；Keep/Clone/Split 与 step 0/50/100 共用同一 ROI，不随 children footprint 改变。
- Demand ROI = Candidate ROI ∩ FastGS 原生高残差 mask（`get_loss(render,gt) > loss_thresh`，与 VCD metric_map 同一定义，action 前计算并冻结）。
- Candidate 采样：stratified（importance 分位数分箱，箱内随机）+ 固定 seed（`cand_seed·1000+iteration`），clone/split 各半（一侧不足从另一侧补足），非全 top-importance。
- 每 candidate 固定 local probe = 前 30 个训练相机池中前 8 个可见视图（radii>0 且 ROI 有效，确定性无 RNG）；cost views 同一集合。
- `num_rendered`（Gaussian–tile pairs）经包级 `_C` 运行期代理获取，未改 CUDA。
- tile cost 命名 `gaussian_tile_delta`（非 Actual FPS Cost）；@K=0 为结构性工作量、@K=100 为短程已实现工作量，分开统计。

## 2. Smoke test（通过）

room / iter 1000 / 2 candidates（各取 clone、split 候选集中位 importance 者），`--force_one_each`：

| field | Candidate 1 (native clone, idx 39895) | Candidate 2 (native split, idx 30243) |
|---|---|---|
| importance / scale_max / aniso | 10 / 0.0036 / 1.26 | 45 / 0.053 / 1.78 |
| visible / local / demand views | 6 / 6 / 6 | 6 / 6 / 6 |
| ΔQ_support_clone / ΔQ_support_split @100 | +9.7e-6* / — | −4.5e-5 / +2.2e-5 |
| **ΔQ_demand_clone @100** | **+5.1e-5** | −4.8e-5 |
| **ΔQ_demand_split @100** | −9.8e-5 | **+4.0e-6** |
| **ΔTile_clone @0** | **+2.00** | **+16.50** |
| **ΔTile_split @0** | **+1.00** | **−0.17** |
| oracle winner (= native) | clone ✓ | split ✓ |

*c00: keep/clone/split local_err@100 = 0.036615 / 0.036629 / 0.036564（support 口径 clone 略差、split 略好，demand 口径 clone 更好——两口径可分叉，故 demand 为主指标）。smoke 验证项全部通过：ROI/Demand ROI 生成正确、三分支局部指标输出、ΔTile@0 输出、100 步训练正常、JSON/CSV 正确保存（`project_md/action_diag_smoke.json/.csv`）。

## 3. 数据量

```text
总 candidate 数:          60  (iter 1000/1500/2000 × 20)
native clone 数:          30
native split 数:          30
有效 local ROI 数:        60/60（全部 ≥1 个有效 ROI 视图，实际均 8/8）
有效 demand ROI 数:       60/60（全部含 ≥1 个 demand 视图）
batch 状态:               it1000 N=112,627 (clone set 45 / split set 9,806)
                          it1500 N=122,478 (80 / 14,641)
                          it2000 N=137,181 (123 / 12,091)
```

---
# Paper B — B2 Multi-Candidate Local Action Oracle 报告

- 数据源: `project_md/action_diag_v2.json`（真实运行，无伪造）
- scene: `/mnt/workspace/Dataset/room` , global probe: test
- 总 candidate 数: **60** (native clone: 30, native split: 30)
- 有效 local ROI candidate 数: 60
- 有效 demand ROI candidate 数 (≥1 valid demand view): 60

## A. Quality 分布（ΔQ = Err_keep − Err_action，正值 = 优于 Keep；主指标 demand-local L1@100）

```text
ΔQ_demand_clone  : n=60 mean=+0.000000 median=+0.000016 std=0.000992 min=-0.006561 max=+0.002521
ΔQ_demand_split  : n=60 mean=-0.000312 median=-0.000058 std=0.001385 min=-0.009045 max=+0.002018
ΔQ_support_clone : n=60 mean=+0.000024 median=+0.000002 std=0.000226 min=-0.000639 max=+0.001455
ΔQ_support_split : n=60 mean=-0.000080 median=-0.000034 std=0.000179 min=-0.000591 max=+0.000344
quality_action_gap (split−clone, demand): n=60 mean=-0.000312 median=-0.000078 std=0.000800 min=-0.003222 max=+0.001449
  gap>0 (split更好): 40.0%   gap<0 (clone更好): 60.0%   |gap|<1e-4 (无差): 23.3%
```

## B. Action winner（oracle = demand-local L1@100 最小者；demand 无效时回退 support-local）

```text
Keep best  : 16/60 = 26.7%
Clone best : 23/60 = 38.3%
Split best : 21/60 = 35.0%
```

## C. FastGS native action vs short-horizon oracle

```text
agreement rate: 28/60 = 46.7%
confusion (row=native, col=oracle winner):
                  keep   clone   split
native clone          9      14       7
native split          7       9      14
```

## D/E. quality_action_gap 相关性（n=60）

| descriptor | n | Pearson | Spearman |
|---|---:|---:|---:|
| scale_max | 60 | +0.158 | +0.125 |
| scale_anisotropy | 60 | +0.190 | +0.154 |
| residual_energy_mean | 60 | +0.163 | +0.123 |
| residual_anisotropy_mean | 60 | -0.035 | -0.124 |
| footprint_mean | 60 | +0.116 | +0.107 |
| importance_score | 60 | +0.103 | +0.211 |

## F. Compute（gaussian_tile_delta = ΔTile，K=0 结构性 / K=100 已实现）

```text
ΔTile_clone@0 : n=60 mean=+7.149524 median=+2.464286 std=15.737296 min=+1.125000 max=+117.875000
ΔTile_split@0 : n=60 mean=+1.120159 median=+1.250000 std=2.242548 min=-10.000000 max=+10.125000
tile_action_gap@0 (split−clone): n=60 mean=-6.029365 median=-1.312500 std=16.648102 min=-125.375000 max=+0.125000
ΔTile_clone@100: n=60 mean=+3.773036 median=+1.732143 std=8.518083 min=-3.625000 max=+57.000000
ΔTile_split@100: n=60 mean=+0.796052 median=+0.875000 std=5.653080 min=-32.750000 max=+12.000000
Clone net Δ#GS: all==1 ? True  (mean=+1.000)
Split net Δ#GS: all==1 ? True  (mean=+1.000)
```

```text
latency keep/clone/split ms (light protocol): 2.222 / 2.116 / 2.039
```

```text
global PSNR@100 keep/clone/split: 27.4731 / 27.4729 / 27.4731
```

- plots/quality_gap_histogram.png
- plots/native_vs_oracle_confusion.png
- plots/quality_gap_vs_scale.png
- plots/quality_gap_vs_residual_anisotropy.png
- plots/delta_tile_clone_vs_split.png
- plots/quality_vs_tile_cost.png

---

## 4. 补充统计与解读（诚实归因）

```text
条件一致率（oracle 判定"应执行某个 action"的 44 例中 native 判对的）: 28/44 = 63.6%
ΔTile_clone@0 最小值: +1.125（60/60 全为正 —— clone 结构性必增 tile 工作量）
ΔTile_split@0 为负的 candidate: 4/60（split 可以净降低 tile 工作量）
median |quality_action_gap| = 3.3e-4 L1 ≈ keep 基线 demand error (0.0928) 的 0.36%
```

1. **质量信号可测但偏小**：B1 中全图指标完全不可分辨（≤0.002 dB）；换 demand-local L1 后单高斯动作差异进入可测范围（|gap| 中位数 0.36% 基线，max ~10%）。这是协议升级的直接收益。
2. **Clone 短程略优、Split 短程略差**（demand 口径 mean gap −3.1e-4 偏向 clone）：与 B1 机制解释一致——split 丢弃父本 1000+ 步优化成果与 Adam 动量，100 步内未完全恢复；clone 无破坏性。但 23.3% 的 candidate 两 action 无实质差异（|gap|<1e-4）。
3. **Keep 最佳占 26.7%**：约 1/4 的 FastGS 真实候选在短程内"不动最好"，说明原生候选集中存在过度操作空间（Paper B 的 headroom 证据之一）。
4. **原生 action 与短程 oracle 一致率仅 46.7%**（含 keep）；在 oracle 判定应执行 action 的子集中为 63.6%。注意：oracle 是 100 步局部 L1 口径，native 优化的是长程全局目标，低一致率≠native 错误，但确实说明"短程局部收益"与"scale 启发式"选择了相当不同的候选。
5. **相关性全部弱**（|Pearson|≤0.19, |Spearman|≤0.21）：scale_max / scale_anisotropy / residual_energy / residual_anisotropy / footprint / importance 都不足以解释 quality_action_gap——原始 scale 启发式**不**足以解释最佳 action（对 Paper B 是必要非充分证据：需要更丰富 descriptor，但本次简单 descriptor 也未捕获）。
6. **计算信号强且方向一致（本次最稳的发现）**：同为 net +1 GS，ΔTile_clone@0 全部为正（median +2.5, mean +7.1, max +117.9），ΔTile_split@0 接近零且可为负（median +1.25, mean +1.1, 4/60 为负），tile_action_gap@0 mean −6.0——**clone 复制完整 tile footprint，split 收缩 footprint**，结构性 workload 差异显著。@K=100 已实现工作量差异收窄（+3.8 vs +0.8）但仍同向。
7. latency（轻量协议）与 global PSNR 差异在噪声内，作辅助参考，不下结论。

## 5. 修改文件

```text
Modified: (无 —— FastGS tracked source 零修改)

Added:
- diagnostics/__init__.py            包标记
- diagnostics/common.py              共享工具：seed/clone_tree/CProxy/原生训练步/投影/ROI/局部指标/残差描述子
- diagnostics/diagnostic_v2.py       B2 采集主脚本（warmup 带诊断钩子 + 多 candidate 三分支 + JSON/CSV 输出）
- diagnostics/analyze_v2.py          统计 A-F + Pearson/Spearman + 6 图 + 自动报告（宿主 python3 运行）
- project_md/action_diag_v2.json/.csv/_report.md   正式数据（60 candidates）
- project_md/action_diag_smoke.json/.csv           smoke 数据
- project_md/b2_smoke.log / b2_full.log            运行日志
- project_md/plots/*.png                          6 张分析图
```

## 6. 结论

**Engineering: DIAGNOSTIC V2 PASS**（smoke 两处小 bug 修复后通过；60/60 candidate 全有效；全部产物落盘。唯一偏离：GPU2 被外部任务占用改用 GPU5，已记录。）

**Scientific signal: PROMISING**（依据：① 同为 net +1 GS 时 clone/split 的结构性 tile 成本差异稳定可测且方向一致（60/60 clone 为正）；② demand-local 协议首次把单高斯动作质量差异拉入可测范围，且 26.7% 候选短程内 keep 最佳、原生一致率仅 46.7%，存在 action 选择 headroom；③ 但质量 gap 幅度小（中位 0.36% 基线）、所有简单 descriptor 相关性弱、单场景 n=60、100 步短程——若仅以质量-descriptor 相关性评判则为 WEAK。综合判定 PROMISING，需更多场景/更长 horizon 复核。）

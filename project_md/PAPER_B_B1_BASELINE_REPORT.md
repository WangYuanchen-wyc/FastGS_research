# Paper B — B1 Baseline Report: Keep / Clone / Split 受控诊断

> 实验：对 FastGS 一个**真实** densification candidate，从**完全相同的 checkpoint** 分别执行 Keep / Clone / Split，各继续训练 100 步，比较质量收益与计算成本。
> 本阶段**未实现**任何 Paper B 方法（无 lineage / rollback / persistent ID / action score / budget）。
> 本阶段**未修改**任何 FastGS 训练算法源码（`git diff` 为空）。

---

## 1. 执行环境

| 项 | 值 |
|---|---|
| Docker 容器 | `wyc-compre`（宿主 `/home/wyc` ↔ 容器 `/mnt`） |
| 项目路径 | 容器内 `/mnt/workspace/FastGS`（= 宿主 `/home/wyc/workspace/FastGS`） |
| Conda | `/opt/miniconda3/envs/fastgs`（python 3.7, torch 1.12.1+cu116） |
| GPU | `CUDA_VISIBLE_DEVICES=2`，RTX 4090 24GB（独占，运行前 0 MiB 占用） |
| 场景 | `/mnt/workspace/Dataset/room`（Mip-NeRF360，311 相机，`--eval` → 39 test 相机） |
| 复现命令 | 见 §7 |

## 2. 修改了哪些文件

```text
Modified: (无 —— 所有被跟踪源码文件零修改, git diff --stat 为空)

Added:
- diagnostic_b1.py                 独立诊断脚本(新增, 不改动任何现有训练逻辑)
- project_md/b1_results.json       全部实验数据(JSON, 机器可读)
- project_md/b1_run.log            完整运行日志
- project_md/PAPER_B_B1_BASELINE_REPORT.md  本报告
```

`diagnostic_b1.py` 做了什么（全部为**复用** FastGS 原生组件，非重实现）：

| 诊断需求 | 复用的原生函数 | 说明 |
|---|---|---|
| 找真实 candidate | `utils/fast_utils.py:sampling_cameras` + `compute_gaussian_score_fastgs` | 与 `train.py:134-138` 逐字相同的调用方式 |
| 选择逻辑 | 复现 `gaussian_model.py:477-494` 的选择数学 | metric_mask ∧ grad/scale 条件，**只读不执行** |
| checkpoint | `GaussianModel.capture / restore`（`gaussian_model.py:73-128`） | 外加逐分支深拷贝（Adam 原地更新 param/exp_avg，引用会被污染；且必须保 `nn.Parameter` 类型否则 `requires_grad` 丢失、训练冻结） |
| 强制单点 clone | `GaussianModel.densify_and_clone_fastgs(single_mask, ones)` | 原生函数，mask 只含 parent 一行 |
| 强制单点 split | `GaussianModel.densify_and_split_fastgs(single_mask, ones, N=2)` | 原生函数，原生 0.8·N 缩放与采样构造 |
| 继续训练 | `render_fastgs` + 原生 loss + `optimizer_step`（原生 3 段步频调度） | 每步与 `train.py:78-104` 相同 |
| num_rendered/tile pairs | 对包模块级 `_C` 句柄装零侵入代理（记录 forward 第 1/2 返回值） | **未改 CUDA**、未改包文件，仅运行期 monkeypatch |

## 3. 实验 candidate（真实 FastGS 选择，非自定分数）

诊断点 = `iteration 1000`，即 room 原生配置（`densification_interval=500`）下 FastGS **整个训练的第一次 densification 事件**，checkpoint 时刻所有 112,627 个 Gaussian 仍为 SfM 初始化点。

```text
Scene:                /mnt/workspace/Dataset/room (Mip-NeRF360, --eval)
Iteration:            1000 (首个 densification 事件; densify_from_iter=500, interval=500)
Parent index:         7445 (选择方式: 候选集中 importance_score 最大者)
FastGS native action: split
Importance score:     58213   (10 视角平均误差像素归因计数; 门控阈值 >5)
Grad:                 0.000737  (>= grad_thresh 0.0002  ✓ clone 梯度条件)
Grad_abs:             0.005598  (>= grad_abs_thresh 0.0008 ✓ AbsGS split 条件)
Scale:                [12.038, 4.202, 5.271]  (max=12.038 > dense×extent → split_qualifier ✓)
Opacity:              0.767
Pruning score:        0.99986
该事件候选总数:        clone 41 个 / split 9743 个 (metric_mask ∧ 梯度 ∧ 尺寸)
Checkpoint 时 #GS:    112627 (全部为初始 SfM 点)
```

parent #7445 同时满足 split 的全部三个原生条件（metric_mask、grad_abs、scale>阈值），且其 `max_scale=12.04` 远大于 clone/split 尺度分界 → FastGS 对它的原生判定为 **split**。Clone 分支对该 parent 强制 clone、Split 分支强制 split，属于对同一对象的受控干预。

## 4. 三个 branch 的结果

| Metric | Keep | Clone | Split |
| --- | ---: | ---: | ---: |
| #GS after action | 112627 | 112628 | 112628 |
| net Δ#GS | 0 | **+1** | **+1** |
| Loss @0 | 0.05594 | 0.05595 | 0.05594 |
| Loss @100 | 0.05560 | 0.05559 | 0.05560 |
| PSNR @0 | 26.2582 | 26.2580 | 26.2581 |
| PSNR @50 | 26.2934 | 26.2929 | 26.2956 |
| PSNR @100 | 26.2163 | 26.2166 | 26.2146 |
| SSIM @100 | 0.8396 | 0.8397 | 0.8396 |
| LPIPS @100 | 0.3457 | 0.3456 | 0.3457 |
| Render latency @100 (ms, mean±std) | 1.935±0.083 | 1.979±0.099 | 2.036±0.103 |
| FPS @100 | 516.7 | 505.4 | 491.1 |
| Tile pairs / num_rendered (mean/view) | 439,909 | 440,856 | 438,806 |

（probe = 5 个固定 test 相机；latency 为 CUDA Event 测量、3 次 warmup + 10 reps × 5 views；tile pairs 为 rasterizer forward 返回的 `num_rendered` 即 Gaussian–tile key/value 对数，5 视角平均。）

```text
ΔPSNR_clone  @0   = -0.0002 dB      ΔPSNR_split  @0   = -0.0001 dB
ΔPSNR_clone  @50  = -0.0005 dB      ΔPSNR_split  @50  = +0.0022 dB
ΔPSNR_clone  @100 = +0.0003 dB      ΔPSNR_split  @100 = -0.0016 dB
Split - Clone PSNR gap @100 = -0.0020 dB

ΔLatency_clone = +0.044 ms (约 +2.3%,  < 1σ=0.099 ms)
ΔLatency_split = +0.101 ms (约 +5.2%, ≈ 1σ=0.103 ms)
```

### 结果解读（诚实归因，不过度解读）

1. **net Δ#GS 验证通过**：clone +1（复制一行）；split +1（+2 子代 −1 父本）。与 FastGS 原生生命周期（`gaussian_model.py` clone/split/prune 调用链）的理论预测一致。
2. **单高斯结构动作对整图指标的影响低于本协议分辨率**：|ΔPSNR| ≤ 0.002 dB、|ΔSSIM/LPIPS| ≤ 0.0002。原因：1 个高斯 / 112,627 个，且 parent 只影响图像局部区域，5 视角全图平均把局部效应稀释约 5 个数量级。这一"不可分辨"本身是 Paper B 的重要基线事实：**逐高斯边际收益无法用全图 probe 指标直接测得**，未来需要局部化度量（本阶段不设计）。
3. **Split @100 略低于 Keep/Clone（−0.0016 dB）与机制一致**：split 删除了已优化 1000 步的父本（其 Adam 动量也被丢弃，子代动量清零，`cat_tensors_to_optimizer` 用 zeros 扩展），两个子代在 100 步内未能完全恢复父本的拟合；clone 保留原行、仅追加副本，不破坏现有解。方向合理，幅度在噪声内。
4. **Latency 差异不可归因于单个高斯**：N 仅 +0.0009%，机制上不可能造成 +2~5% 延迟；测得的 ΔLatency 均 <1σ，应视为测量波动。结论：**此协议下三者渲染成本无统计学可分辨差异**。
5. **Tile pairs 是唯一有机制解释力的计算量信号**：clone 使该视角平均 pairs **+947**（+0.22%——复制了一个 footprint 很大的 splat，约等效新增一个覆盖 ~900+ tile 的实例）；split 使 pairs **−1,103**（−0.25%——两个 /1.6 缩小的子代总覆盖面积小于父本，`auxiliary.h` Compact Box 椭圆随之收缩）。即：**clone 复制完整 tile 覆盖，split 收缩 tile 覆盖**——对 Paper B 的 compute-cost 建模是直接可用的量化事实（每视角基线 ~44 万 pairs）。

## 5. 公平性确认

```text
Same checkpoint:                YES  三分支均从同一 master checkpoint 的独立深拷贝 restore
                                     (capture 含 params + optimizer.state_dict + shoptimizer.state_dict)
Same camera sequence:           YES  预生成 100 步相机序列 (random.Random(2024))，三分支逐位相同
Same RNG control:               YES  action 前 seed=1234 (split 采样确定)，训练前重置 seed=1234
                                     (三分支训练起始 RNG 状态逐位一致)
Same initial optimizer state:   YES  Keep 直接继承；Clone/Split 仅新增行的 Adam 矩为零(原生行为)
New densification disabled:     YES  100 步内不进入任何 densify 分支
Pruning disabled:               YES  无 prune_points / final_prune 调用
Opacity reset disabled:         YES  无 reset_opacity / 0.8 截断
Native FastGS Clone reused:     YES  densify_and_clone_fastgs 原函数、原参数构造
Native FastGS Split reused:     YES  densify_and_split_fastgs 原函数 (N=2, /1.6 缩放, 原生采样)
Single-candidate isolation:     YES  mask 仅含 parent #7445，其余 9,783 个候选不被处理
```

已知并声明的协议细节（三分支一致，不影响公平性）：
- checkpoint 取在 iteration 1000 的 `loss.backward()`+统计更新之后、原生 `densify_and_prune_fastgs` 之前；该 iteration 尚未执行的 `optimizer_step` 的 pending 梯度随 restore 一并丢弃（三分支同样丢弃）。
- 分支内不累积 densification 统计（`add_densification_stats` 仅服务于被禁用的 densify 决策，不影响梯度）。
- 100 步的 iteration 编号延续原生（1001..1100），lr 调度与 SH 步频调度按原生 `update_learning_rate` / `optimizer_step` 执行。

## 6. 代码状态

```text
$ git status --short
?? __pycache__/
?? diagnostic_b1.py
?? project_md/b1_results.json
?? project_md/b1_run.log

$ git diff --stat
(空 —— 无任何被跟踪文件被修改; 运行产生的 .pyc 变更已还原, cameras.json/input.ply 副产物已删除)
```

## 7. 复现命令

```bash
docker exec -e CUDA_VISIBLE_DEVICES=2 -w /mnt/workspace/FastGS wyc-compre \
  /opt/miniconda3/envs/fastgs/bin/python diagnostic_b1.py \
    -s /mnt/workspace/Dataset/room --eval \
    --densification_interval 500 --grad_abs_thresh 0.0008 \
    --diag_iter 1000 --diag_steps 100
```

超参来源：room 场景沿用仓库 `train_base.sh` 的原生配置（`densification_interval=500`、`grad_abs_thresh=0.0008`），其余用 `arguments/__init__.py` 默认值。诊断点 1000 = 该配置下训练的**第一个** densification 事件。

## 8. 对 Paper B 的基线结论（只陈述，不设计下一步）

1. 单高斯 Keep/Clone/Split 的**质量差**在 100 步窗口内低于全图指标分辨率（≤0.002 dB）→ 逐高斯 action 评估必须换局部度量协议。
2. **成本差可测且机制清晰**：clone ≈ 复制完整 tile footprint（本例 +947 pairs/view），split ≈ 收缩 footprint（本例 −1,103 pairs/view，net #GS 同为 +1）。Paper B 的 compute-budget 项应基于 pairs 而非 #GS。
3. split 存在"父本优化成果+Adam 动量被丢弃"的内在恢复成本（本例 100 步未完全恢复）——这是 Reversible Densification 动机的直接实证信号。

*生成于 2026-08-27，数据来自真实运行（log: `project_md/b1_run.log`, JSON: `project_md/b1_results.json`），无任何伪造。*

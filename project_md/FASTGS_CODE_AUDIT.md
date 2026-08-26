# FastGS 代码审计报告 — Paper B 代码地图

> 审计对象：`/home/wyc/workspace/FastGS`（main 分支，commit `44e02a5`）
> 论文：*FastGS: Training 3D Gaussian Splatting in 100 Seconds*（arXiv 2511.04283, CVPR 2026）
> 用途：为 Paper B（Lineage-aware Reversible Densification）提供准确的代码级基线地图。
> **本文档只做分析，未修改任何训练算法。**
>
> 官方声明的代码血统（README.md:250）：built upon **3DGS** + **Taming-3DGS** + **Speedy-Splat** + **Abs-GS**。
> git 分支 `3dgsaccel_abs`、`3dgsaccel_abs_cb` 印证了演进顺序：先接 AbsGS，再加 Compact Box。

---

## 0. 文件地图与贡献归属总览

```text
FastGS/
├── train.py                       # 训练主循环：VCD/VCP 的调度入口（本文档 B/C 节）
├── arguments/__init__.py          # 全部超参：loss_thresh / grad_abs_thresh / grad_thresh / dense / mult
├── utils/fast_utils.py            # ★VCD/VCP 打分核心：sampling_cameras + compute_gaussian_score_fastgs
├── scene/gaussian_model.py        # ★densify_and_prune_fastgs / final_prune_fastgs / 生命周期函数
├── gaussian_renderer/__init__.py  # render_fastgs：mult、metric_map、get_flag、(N,4) screenspace_points
└── submodules/
    ├── diff-gaussian-rasterization_fastgs/     # ★CUDA 光栅化器（VCD 计数 + Compact Box + AbsGS 梯度）
    │   ├── diff_gaussian_rasterization_fastgs/__init__.py   # PyTorch autograd 接口
    │   ├── rasterize_points.cu                # C++ glue：metricCount 分配、mult 传递
    │   └── cuda_rasterizer/
    │       ├── auxiliary.h        # ★★duplicateToTilesTouched = SNUGBOX 椭圆 + t = mult·t（Compact Box 核心）
    │       ├── forward.cu         # preprocessCUDA(tiles_touched 计数) + renderCUDA(atomicAdd metricCount)
    │       ├── backward.cu        # PerGaussianRenderCUDA（Speedy-Splat 反向）+ fabs 绝对梯度（AbsGS）
    │       └── rasterizer_impl.cu # duplicateWithKeys / identifyTileRanges / bucket 机制
    ├── fused-ssim/                # 继承自 Speedy-Splat 工具链（快速 SSIM）
    └── simple-knn/                # vanilla 3DGS 原有
```

**贡献归属速查**（详见 E 节，防止误归因）：

| 组件 | 归属 | 关键代码位置 |
|---|---|---|
| VCD（多视角一致加密打分 + metric_mask 门控） | **FastGS 原创**（打分脚手架借自 Taming） | `utils/fast_utils.py:45`、`scene/gaussian_model.py:494` |
| VCP（多视角一致剪枝打分 + final_prune_fastgs） | **FastGS 原创**（budget 块借自 Taming） | `gaussian_model.py:505-518, 533-540`、`train.py:153-158` |
| Compact Box（β=mult 缩小 SNUGBOX 椭圆） | **FastGS 原创**（椭圆/分片脚手架借自 Speedy-Splat） | `cuda_rasterizer/auxiliary.h:338` |
| 绝对梯度 densification | **AbsGS**（继承） | `backward.cu:594-596`、`gaussian_model.py:485,490` |
| 稀疏像素反向、SNUGBOX/AccuTile、dc/sh 分离 | **Speedy-Splat**（继承） | `backward.cu:402`、`auxiliary.h`、`forward.cu:24` |
| metric_map 渲染计分套路、budget 剪枝、SparseGaussianAdam、0.8 opacity 截断 | **Taming-3DGS**（继承） | `fast_utils.py`、`gaussian_model.py:505-518,520-522` |
| 梯度累积→clone/split→prune 基本框架 | **vanilla 3DGS**（继承） | `gaussian_model.py:431-531` |

---

## 1. FastGS vs Vanilla 结构对比图

```text
======================== Vanilla 3DGS 训练迭代 ========================
每 iteration:
  render() ──> L1+DSSIM loss ──> backward
  [每 100 iter, 500~15000]:
      xyz_gradient_accum (N,1)                       # 每 iter 累积
      densify_and_clone(grad > 0.0002 ∧ small)       # 直接 clone
      densify_and_split(grad > 0.0002 ∧ large)       # 直接 split (N=2, /1.6)
      prune_points(α<0.005 ∨ max_radii2D>20 ∨ scale>0.1·extent)
  [每 3000 iter]: reset_opacity(0.01)
  optimizer.step()  每 iter

======================== FastGS 训练迭代 (train.py) ======================
每 iteration:
  render_fastgs(mult) ──> L1+fused_SSIM ──> backward
      │  screenspace_points 变为 (N,4)：[:, :2]=普通梯度, [:, 2:]=|绝对梯度|(AbsGS)
  [每 iter < 15000]: add_densification_stats 同时累积两套梯度 (gaussian_model.py:528)
  [每 densification_interval (默认100, train_base.sh=500) iter, 500~15000]:
      ① sampling_cameras: 随机抽 10 个训练相机           (fast_utils.py:10)
      ② compute_gaussian_score_fastgs(DENSIFY=True):     (fast_utils.py:45)
           每视角: 渲染 → L1 误差图 > loss_thresh → metric_map
                   → 带 metric_map 二次渲染 → CUDA atomicAdd 每高斯计数
           => importance_score (N,), pruning_score (N,)
      ③ densify_and_prune_fastgs:                        (gaussian_model.py:468)
           metric_mask = importance_score > 5             ★VCD 门控
           clone: metric_mask ∧ grad(0.0002) ∧ scale≤dense·extent
           split: metric_mask ∧ |grad|(0.0012, AbsGS) ∧ scale>dense·extent
           prune 候选(α<0.005∨过大) --multinomial 50%预算加权采样--> prune  ★VCP-阶段1
           全体 opacity 截断到 0.8
  [每 3000 iter]: reset_opacity(0.01)                    # 同 vanilla
  [每 3000 iter, 15000<iter<30000]:                      # ★VCP-阶段2 (vanilla 没有)
      ④ compute_gaussian_score_fastgs(DENSIFY=False) → pruning_score
      ⑤ final_prune_fastgs: prune(α<0.1 ∨ pruning_score>0.9)
  optimizer_step(): 3 段式稀疏步频调度 (1/16/32/64)       # FastGS 自己的调度
```

---

## A. Vanilla 3DGS baseline 在 FastGS 代码中的对应

Vanilla 的 densification 流水线 `gradient accumulation → densify_and_clone → densify_and_split → opacity/size pruning` 在 FastGS 中**全部保留但被改名/被门控**：

| Vanilla 3DGS | FastGS 对应函数 | 位置 | 差异 |
|---|---|---|---|
| `add_densification_stats`（梯度累积） | `add_densification_stats` | `gaussian_model.py:528-531` | **双通道累积**：`xyz_gradient_accum`（`grad[:, :2]`，vanilla 同款）+ `xyz_gradient_accum_abs`（`grad[:, 2:]`，AbsGS 绝对梯度） |
| `densify_and_clone` | `densify_and_clone_fastgs` | `gaussian_model.py:455-466` | 签名多了 `metric_mask`；选中条件 = `metric_mask ∧ filter`（VCD 门控）；额外搬运 `tmp_radii` |
| `densify_and_split` | `densify_and_split_fastgs` | `gaussian_model.py:431-453` | 同上多 `metric_mask`；`selected_pts_mask[:mask.shape[0]] = mask` 处理"分数张量比当前 N 短"的 padding；搬运 `tmp_radii` |
| `densify_and_prune`（编排） | `densify_and_prune_fastgs` | `gaussian_model.py:468-526` | ①选择阈值拆分：clone 用 `grad_thresh=0.0002`（vanilla 同值），split 用 `grad_abs_thresh=0.0012`（AbsGS）；②插入 `metric_mask`；③prune 变为 budget 加权采样；④尾部把全体 opacity 截断到 0.8（Taming 配方） |
| `prune_points`（opacity/size 剪枝） | `prune_points`（原样保留）+ 两个调用点 | `gaussian_model.py:364-381`；调用点 `518`（densify 阶段）与 `540`（final 阶段） | 函数本体与 vanilla 一致（含 `tmp_radii` 扩展） |
| `reset_opacity` | `reset_opacity` | `gaussian_model.py:279-282` | 与 vanilla 相同（min(α, 0.01)） |

Vanilla 里被**删除**的：`densify_and_clone/densify_and_split`（无后缀版本）不存在于本仓库；vanilla 的 `render()` 被 `render_fastgs()` 替换（多了 `mult`、`get_flag`、`metric_map` 参数，返回 `accum_metric_counts`）。

**Vanilla tile 边界（对照 D 节用）**：vanilla `preprocessCUDA` 计算 `radius = ceil(3·sqrt(λ_max))`，用 `getRect(center, radius)` 取覆盖 3σ 圆的**轴对齐正方形 tile 矩形**，`tiles_touched = 矩形面积`，`duplicateWithKeys` 为矩形内每个 tile 复制一个 (tile|depth, gaussian_id) 对。FastGS 仓库中 `getRect` 仍保留在 `auxiliary.h:50-72`（两个重载），但 **CUDA 主流程已不再调用它**（见 D 节）。

---

## B. VCD（View-Consistent Densification）全链路追踪

### B.1 调用图

```text
train.py:132  iteration % densification_interval == 0 且 500 < iter < 15000
  │
  ├─ train.py:134-135   my_viewpoint_stack = scene.getTrainCameras().copy()
  │                      camlist = sampling_cameras(stack)          [fast_utils.py:10]
  │                      → 随机 pop 出 num_cams=10 个相机（硬编码 10）
  │
  ├─ train.py:138       importance_score, pruning_score =
  │                      compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
  │                      [fast_utils.py:45-105]  ★VCD/VCP 打分核心（整个调用在 train.py:108 的 torch.no_grad() 内）
  │   │
  │   └─ 对 10 个视角逐个循环 (fast_utils.py:73)：
  │       ├─ (a) 渲染①   render_image = render_fastgs(view, gaussians, pipe, bg, args.mult)["render"]
  │       │              [gaussian_renderer/__init__.py:18]；此时 metric_map=None → 零图，get_flag=None→False
  │       │              输出 render_image: (3,H,W)
  │       ├─ (b) photometric_loss = compute_photometric_loss(view, render_image)
  │       │              [fast_utils.py:27-31] = 0.8·L1 + 0.2·(1−fused_SSIM)，标量 ()
  │       ├─ (c) l1_loss_norm = get_loss(render_image, gt_image)
  │       │              [fast_utils.py:21-25]：逐像素 |Δ| 沿通道均值 → (H,W) → min-max 归一化 → (H,W)
  │       ├─ (d) metric_map = (l1_loss_norm > args.loss_thresh).int()
  │       │              (H,W) int；loss_thresh 默认 0.1（garden 用 0.06）      ★误差掩码阈值
  │       ├─ (e) 渲染②   render_pkg = render_fastgs(view, gaussians, pipe, bg, args.mult,
  │       │                          get_flag=True, metric_map=metric_map)
  │       │              accum_metric_counts = render_pkg["accum_metric_counts"]  (P,) int32
  │       │              [CUDA 端：forward.cu:401-407 atomicAdd，见 B.2]
  │       ├─ (f) full_metric_counts += accum_metric_counts        (P,)   [仅 DENSIFY=True]
  │       └─ (g) full_metric_score += photometric_loss · accum_metric_counts   (P,)
  │
  ├─ fast_utils.py:99   pruning_score = (full_metric_score − min)/(max − min)   (P,) ∈ [0,1]
  ├─ fast_utils.py:102  importance_score = floor(full_metric_counts / 10)       (P,) int
  │
  └─ train.py:139-145   gaussians.densify_and_prune_fastgs(..., importance_score, pruning_score)
      │
      ├─ gaussian_model.py:494   metric_mask = importance_score > 5     (P,) bool   ★VCD 核心门控
      ├─ gaussian_model.py:484   grad_qualifiers     = ‖accum/denom‖₂ ≥ grad_thresh(0.0002)      # clone 用
      ├─ gaussian_model.py:485   grad_qualifiers_abs = ‖accum_abs/denom‖₂ ≥ grad_abs_thresh(0.0012) # split 用(AbsGS)
      ├─ gaussian_model.py:486   clone_qualifiers = max_axis(scale) ≤ dense·extent   # dense=0.001, extent=cameras_extent
      ├─ gaussian_model.py:487   split_qualifiers = max_axis(scale) > dense·extent
      ├─ gaussian_model.py:489   all_clones = clone_qualifiers ∧ grad_qualifiers
      ├─ gaussian_model.py:490   all_splits = split_qualifiers ∧ grad_qualifiers_abs
      ├─ gaussian_model.py:496   densify_and_clone_fastgs(metric_mask, all_clones)
      └─ gaussian_model.py:497   densify_and_split_fastgs(metric_mask, all_splits)
```

### B.2 CUDA 端每高斯 metric 累积（渲染②的内部路径）

```text
gaussian_renderer/__init__.py:84  render_fastgs(..., get_flag=True, metric_map=metric_map)
  └─ :40-56   GaussianRasterizationSettings(..., mult=mult, get_flag=get_flag, metric_map=metric_map)
  └─ :93-102  rasterizer(means3D, means2D, dc, shs, ...) → 返回 accum_metric_counts
      └─ diff_gaussian_rasterization_fastgs/__init__.py:63-103
           get_flag=None→False；args 含 metric_map/mult/get_flag
           _C.rasterize_gaussians(...) 返回第 9 个返回值 metricCount
      └─ rasterize_points.cu:103-111
           if(get_flag): metricCount = torch.full({P}, 0, int32)；accum_metric_counts_ptr = data
      └─ rasterize_points.cu:123-152 → cuda_rasterizer/rasterizer_impl.cu:292-460 Rasterizer::forward
      └─ rasterizer_impl.cu:437-458 → cuda_rasterizer/forward.cu:441-484 FORWARD::render(..., metric_map, get_flag, metricCount)
      └─ forward.cu:274-438 renderCUDA kernel：
           :401-407   if (get_flag)
                        if (metric_map[pix_id] == 1)
                          atomicAdd(&metricCount[collected_id[j]], 1);
```

**计数语义（精确）**：`accum_metric_counts[g]` = 该视角下，高斯 g **实际参与混合**（通过 `power≤0`、`alpha≥1/255`、`test_T≥0.0001` 三道剔除，forward.cu:380-395）且所在像素被 `metric_map` 标记为高误差的**像素-高斯 incidence 计数**。不是"多少个视角标记过它"，而是"平均每视角被记到几次像素级误差归因"。因此：

- `importance_score = floor(Σ₁⁰ᵥ counts_v / 10)` = 平均每视角的误差像素归因次数；
- `metric_mask = importance_score > 5`（`gaussian_model.py:494`）＝ 平均每视角对 >5 个高误差像素有实质贡献的高斯才允许 clone/split；
- 分母 `len(camlist)=10` 硬编码于 `sampling_cameras` 的 `num_cams = 10`（`fast_utils.py:13`）。

### B.3 张量清单

| 步骤 | tensor | shape | dtype | 产生处 |
|---|---|---|---|---|
| 渲染① | render_image | (3,H,W) | float32 | `fast_utils.py:75` |
| 视角损失 | photometric_loss | () | float32 | `fast_utils.py:27` |
| 误差图 | l1_loss_norm | (H,W) | float32 | `fast_utils.py:22` |
| 误差掩码 | metric_map | (H,W)（CUDA 侧按 H·W 一维寻址） | int | `fast_utils.py:82` |
| CUDA 计数 | accum_metric_counts | (P,) | int32 | `rasterize_points.cu:109` / `forward.cu:405` |
| 跨视角累计 | full_metric_counts / full_metric_score | (P,) | int/float | `fast_utils.py:90-97` |
| 输出 | importance_score / pruning_score | (P,) | int/float | `fast_utils.py:99,102` |
| 门控 | metric_mask | (P,) | bool | `gaussian_model.py:494` |

> 注意 P 是**打分时刻**的高斯数；clone/split 内部用 `selected_pts_mask[:mask.shape[0]] = mask`（split, `gaussian_model.py:436`）和 `padded_importance[:scores.shape[0]]`（prune, `:513`）把旧长度张量 pad 到新长度——**新增高斯天然落在 pad 区（不计分、不被剪）**。这个 padding 模式对 Paper B 的 lineage 张量同样必须遵守。

### B.4 VCD 阈值汇总

| 阈值 | 默认值 | 出处 |
|---|---|---|
| 误差图二值化 `loss_thresh` | 0.1（garden 0.06） | `arguments/__init__.py:94` |
| VCD 门控 `importance_score > 5` | 5（硬编码） | `gaussian_model.py:494` |
| clone 梯度阈 `grad_thresh` | 0.0002（= vanilla） | `arguments/__init__.py:98` |
| split 绝对梯度阈 `grad_abs_thresh` | 0.0012（场景 0.0002~0.002） | `arguments/__init__.py:95` |
| 尺度分界 `dense`（即 percent_dense）×extent | 0.001 | `arguments/__init__.py:99` |
| densification_interval | 100（train_base.sh 用 500） | `arguments/__init__.py:87` |
| densify 窗口 | (500, 15000) | `arguments/__init__.py:89-90` |

---

## C. VCP（View-Consistent Pruning）全链路追踪

`pruning_score` 与 `importance_score` 在 `compute_gaussian_score_fastgs` 中**同源同时计算**（B 节），但消费点有两个阶段。

### C.1 阶段 1 — densification 期内的 budget 剪枝（`gaussian_model.py:499-518`）

```text
gaussian_model.py:499  prune_mask = (get_opacity < 0.005)                      # (N,) vanilla 同款
:500-503               ∨ max_radii2D > 20 (max_screen_size, 仅 iter>3000)
                       ∨ max(scale) > 0.1·extent
:505                   scores = 1 − pruning_score
:506-507               to_remove = Σprune_mask;  remove_budget = int(0.5·to_remove)
:511-515               padded_importance = zeros(N);  [:len(scores)] = 1/(1e-6 + scores)
                       sampled_indices = multinomial(padded_importance, remove_budget, replacement=False)
:517                   final_prune = prune_mask ∧ selected_pts_mask
:518                   prune_points(final_prune)
```

- **与 vanilla opacity pruning 的区别 1**：vanilla 在每个 densification 事件直接 `prune_points(prune_mask)` 全量剪掉；FastGS 只从候选中**按权重无放回抽样 50%** 剪掉，权重 `1/(1e-6+1−pruning_score)` 使 **pruning_score 高（误差归因大）的候选更优先被剪**，新出生高斯（pad 区权重 0）绝不剪。
- **与 vanilla 的区别 2**：剪枝信号从"单视角 opacity/size"升级为"10 视角误差归因分数"。
- 代码注释（:509）"The budget is not necessary for our method"——该 budget 机制本身是 Taming-3DGS 的遗产（论文摘要亦称 FastGS "dispensing with the budgeting mechanism"）；FastGS 的实际剪枝主力是阶段 2。
- 紧接着 `:520-522` 把全体 opacity 截断到 0.8（`inverse_sigmoid(min(α, 0.8))`，Taming 配方，vanilla 无此步）。

### C.2 阶段 2 — 收敛后的 final 剪枝（`train.py:150-158` + `gaussian_model.py:533-540`）

```text
train.py:153  iteration % 3000 == 0 且 15_000 < iteration < 30_000
              （实际触发：18000, 21000, 24000, 27000；30k 被开区间排除）
  ├─ train.py:154-157  camlist = sampling_cameras(...)
  │                     _, pruning_score = compute_gaussian_score_fastgs(..., DENSIFY=False)
  │                     [DENSIFY=False ⇒ 不累计 full_metric_counts ⇒ importance_score=None，渲染次数同为 10]
  └─ train.py:158       gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)
        ├─ gaussian_model.py:537  prune_mask = (get_opacity < 0.1)      # 比 densify 期严 20 倍
        ├─ gaussian_model.py:538  scores_mask = pruning_score > 0.9     # (N,) bool   ★VCP 硬阈值
        ├─ gaussian_model.py:539  final_prune = prune_mask ∨ scores_mask
        └─ gaussian_model.py:540  prune_points(final_prune)             # 确定性、无 budget、无抽样
```

**语义**：15k 后模型基本收敛，仍带高 `pruning_score`（持续对高误差像素有贡献却修不好误差）的高斯被视为冗余/噪声直接删除；opacity 阈值也从 0.005 收紧到 0.1。这是 FastGS 控制高斯总数、实现 100s 训练的主要手段之一。

### C.3 两阶段对照表

| | 阶段1（densify 期） | 阶段2（final） |
|---|---|---|
| 触发 | 每 densification_interval，500~15000 | 每 3000，15000~30000 |
| 打分调用 | `DENSIFY=True`（双分数） | `DENSIFY=False`（仅 pruning_score） |
| opacity 阈 | 0.005 | 0.1 |
| VCP 用法 | 候选内加权抽样（50% budget），`1/(1−score)` 权重 | 硬阈值 `score > 0.9` 直接剪 |
| 确定性 | 随机（multinomial） | 确定 |
| vanilla 对应 | vanilla 每 100 iter 全量剪 | **无对应，纯 FastGS 新增** |

---

## D. Compact Box（CB）CUDA 修改定位

### D.1 Vanilla tile bounding vs FastGS

| | Vanilla 3DGS | FastGS |
|---|---|---|
| 边界形状 | 3σ 圆的外接**正方形**（轴对齐 tile 矩形） | SNUGBOX **精确椭圆**，且椭圆阈值被 β=mult 缩小 |
| 计算位置 | `preprocessCUDA` 内 `getRect`，`duplicateWithKeys` 内再算一遍 | `preprocessCUDA` 与 `duplicateWithKeys` 都调 `duplicateToTilesTouched` |
| 每 tile 精度 | 矩形内**所有** tile 都发射 (Gaussian,tile) 对 | `processTiles`（AccuTile）沿短轴逐片扫描，**每片只发射椭圆真正跨过的 tile** |
| 依赖参数 | 无（3σ 固定） | `mult`（β）：`t = mult · t` |

### D.2 CB 核心代码（唯一的 FastGS 算术修改就一行）

`submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/auxiliary.h`：

```cpp
// :318  duplicateToTilesTouched(p, con_o, grid, mult, idx, off, depth, keys, values)
//   ---- SNUGBOX Code ----        ← 椭圆脚手架来自 Speedy-Splat
:329  float disc = con_o.y*con_o.y − con_o.x*con_o.z;
:332  if (con_o.x <= 0 || con_o.z <= 0 || disc >= 0) return 0;     // 病态椭圆剔除
:337  float t = 2.0f * log(con_o.w * 255.0f);   // SNUGBOX 阈值: α·G = 1/255 的等值线
:338  t = mult * t;                             // ★★★ beta in Compact Box —— FastGS 唯一算术修改 ★★★
:340-355  x_term/y_term → bbox_argmin/argmax → computeEllipseIntersection 得椭圆外接框
:358-365  rect_min/rect_max = 椭圆外接框覆盖的 tile 范围
:377  return processTiles(...)                  // AccuTile 逐片发射 key/value
```

`processTiles`（`auxiliary.h:201-315`）：沿 y（或 x，取跨度小者）逐 BLOCK 尺寸切片，用解析椭圆-直线交点（`computeEllipseIntersection`, `:183-198`）求每片内椭圆的精确 v 向范围，只给**真正相交的 tile** 写 `gaussian_keys_unsorted[off] = (tile_id<<32)|depth_bits; gaussian_values_unsorted[off] = gaussian_idx`（`:293-309`）；传入 `nullptr` 时仅计数不写（给 preprocess 用）。

### D.3 `mult` 进入 renderer 的完整路径

```text
arguments/__init__.py:100   self.mult = 0.5   # CLI --mult；T&T/DB 场景用 0.7（train_base.sh:10-13）
  ↓ train.py:96 / fast_utils.py:75,84 / render.py:37
render_fastgs(cam, pc, pipe, bg, mult, ...)           [gaussian_renderer/__init__.py:18]
  ↓ :51  GaussianRasterizationSettings(..., mult = mult, ...)
GaussianRasterizationSettings NamedTuple 字段 mult   [diff_gaussian_rasterization_fastgs/__init__.py:185]
  ↓ :87  args 元组携带 raster_settings.mult
RasterizeGaussiansCUDA(..., const float mult, ...)    [rasterize_points.cu:73]
  ↓ :144 传给 CudaRasterizer::Rasterizer::forward
Rasterizer::forward(..., const float mult, ...)       [cuda_rasterizer/rasterizer_impl.cu:313]
  ├─ :362 传给 FORWARD::preprocess → preprocessCUDA(..., mult, ...)   [forward.cu:176]
  │    └─ forward.cu:246  tiles_count = duplicateToTilesTouched(point_image, con_o, grid, mult, 0,0,0, nullptr, nullptr)
  │                       // 计数模式：tiles_touched[idx] = tiles_count；==0 则该高斯直接剔除（不渲染）
  └─ :393 duplicateWithKeys<<<>>>(P, mult, ...)       [rasterizer_impl.cu:120]
       └─ :143  duplicateToTilesTouched(points_xy[idx], con_o[idx], grid, mult, idx, off, depth, keys, values)
                // 写模式：只发射椭圆(×β)覆盖的 tile 的 key/value 对
```

### D.4 如何减少 Gaussian–tile pairs

1. **椭圆代替正方形**（SNUGBOX，继承）：对角向拉长的高斯，3σ 外接方形包含大量几乎无贡献的角部 tile；精确椭圆 + AccuTile 分片后每片只保留真实相交 tile。
2. **β=mult 缩小椭圆**（FastGS 原创）：`t' = mult·t` 把"可见贡献"等值线从 `α·G = 1/255` 提到约 `(1/255)^mult`（mult=0.5、α≈1 时 ≈ 1/16），高斯在屏幕上的有效半径按 `√mult` 缩短，边缘 tile 全部不再发射 key。`num_rendered`（排序对总数，`rasterizer_impl.cu:383`）随之下降，直接减少 radix-sort 与正/反向逐 tile 遍历开销。
3. **代价与副作用**：被裁掉的 tile 上既没有前向混合也没有反向梯度（backward 不接收 mult，梯度只沿已发射 pair 回流，`backward.cu`）；同时 VCD 的 `metricCount` 只统计已发射 pair（`forward.cu:401-407`），因此 **mult 会轻微影响 VCD 打分的归因覆盖范围**——Paper B 若复用打分机制需注意这一点。另外 `radii[idx]`（visibility_filter / max_radii2D 用）仍是 vanilla 3σ 半径（`forward.cu:240,262`），与 CB 无关。

### D.5 哪些文件属于 CB 的"真正实现"

| 文件 | 角色 |
|---|---|
| `cuda_rasterizer/auxiliary.h` | **核心**：`t = mult·t`（:338）+ `duplicateToTilesTouched` + `processTiles` + `computeEllipseIntersection`（后三者是 Speedy-Splat 椭圆/分片脚手架，FastGS 复用） |
| `cuda_rasterizer/forward.cu` | `preprocessCUDA` 用 CB 计数 `tiles_touched` 并剔除 0-tile 高斯（:244-248） |
| `cuda_rasterizer/rasterizer_impl.cu` | `duplicateWithKeys`（:120-149）经 CB 发射 key/value |
| `rasterize_points.cu` / `ext.cpp` / `rasterize_points.h` | mult 的 C++ 参数管线 |
| `diff_gaussian_rasterization_fastgs/__init__.py` | `GaussianRasterizationSettings.mult` 字段 |
| `gaussian_renderer/__init__.py` + `arguments/__init__.py` + `train.py`/`render.py` | Python 侧 mult 传递与 CLI |

辅助但非 CB 本体：`identifyTileRanges`（`rasterizer_impl.cu:155-180`，Speedy-Splat 版本，兼容无效 key 的守卫代码）、`getRect`（`auxiliary.h:50-72`，vanilla 遗留，主流程已不用）。

---

## E. 继承组件清单（勿归为 FastGS 原创 VCD/VCP/CB）

### E.1 Abs-GS 绝对梯度

| 环节 | 位置 |
|---|---|
| 反向核内 `fabs` 累积：`Register_dL_dmean2D_z += fabs(tmp_x); _w += fabs(tmp_y)` | `backward.cu:594-596`（寄存器声明 `:470-471`，atomicAdd `:609-610`） |
| `dL_dmeans2D` 分配为 `{P,4}`（vanilla 是 {P,2}） | `rasterize_points.cu:198` |
| `screenspace_points = zeros((N,4))`（前 2 通道普通梯度，后 2 通道绝对梯度） | `gaussian_renderer/__init__.py:27` |
| 双累积器 `xyz_gradient_accum` / `xyz_gradient_accum_abs` | `gaussian_model.py:529-530` |
| **消费**：split 门控用绝对梯度 `grad_abs_thresh=0.0012`；clone 门控仍用普通梯度 0.0002 | `gaussian_model.py:484-485,489-490` |
| README 自证："--grad_abs_thresh Absolute gradient (**same as Abs-GS**) threshold for split" | `README.md:142` |

### E.2 Speedy-Splat 组件

1. **SNUGBOX 精确椭圆 + AccuTile 分片 tile 分配**：`auxiliary.h:183-387`（多处注释 "built upon Speedy-Splat"）。注意：**没有这一层就没有 CB 的载体**，但 β=mult 才是 FastGS 的增量。
2. **稀疏像素逐高斯反向** `PerGaussianRenderCUDA`（warp-per-splat + bucket 状态）：`backward.cu:402-619`；配套 bucket 机制 `perTileBucketCount`（`rasterizer_impl.cu:183`）、`SampleState`（`:247-253`）、前向快照 `sampled_T/sampled_ar`（`forward.cu:363-369`）。`BACKWARD::render`（`backward.cu:855-899`）**只**启动此核；`backward.cu:623` 起的旧版逐像素 `renderCUDA` 是未启动的死代码。
3. **dc / shs 分离缓冲**（SH 稀疏化）：`computeColorFromSH(..., dc, shs, ...)`（`forward.cu:24`）、渲染接口 `dc=` 参数（`gaussian_renderer/__init__.py:88,96`）。
4. **fused-ssim 子模块**（`submodules/fused-ssim`）：train.py:18 `fast_ssim`。

### E.3 Taming-3DGS 组件

1. **多视角误差图打分脚手架**：`sampling_cameras` / `compute_photometric_loss` / `get_loss` / metric_map 二次渲染 / `accum_metric_counts` 原子计数 / `pruning_score` min-max 归一化——`utils/fast_utils.py` 的整体结构与 Taming 的 `compute_gaussian_scores` 同源。FastGS 的增量在于：把该分数同时用作**加密门控（VCD 的 metric_mask）**并设计了 final 阶段的 VCP 硬阈值剪枝。
2. **budget 剪枝块**：`gaussian_model.py:505-518`（multinomial 50%、`1/(1e-6+1−score)` 权重、padding 模式），Taming `densify_and_prune_taming_3dgs` 的直接移植（含 "The budget is not necessary..." 注释）。
3. **SparseGaussianAdam**：`gaussian_model.py:25`（import）、`gaussian_model.py:211`、CUDA `cuda_rasterizer/adam.cu` + Python 包装 `diff_gaussian_rasterization_fastgs/__init__.py:245-271`。默认配置 `optimizer_type="default"` 不走此路径（train_base.sh 亦为 default）。
4. **0.8 opacity 截断**（densification 尾部）：`gaussian_model.py:520-522`，Taming 训练配方。
5. `visibility_filter = (radii > 0).nonzero()` 返回**索引**而非布尔（`gaussian_renderer/__init__.py:108`），Taming 风格。

### E.4 FastGS 自有的调度类修改（非 vanilla，也非直接照搬）

- `optimizer_step`（`gaussian_model.py:225-244`）：3 段式步频——≤15k：主优化器每 iter、SH 优化器每 16 iter；15k~20k：二者每 32 iter；>20k：每 64 iter。注释明言"goal is similar to the sparse Adam of taming 3dgs"（省 step 而非省梯度计算）。
- 双优化器：`shoptimizer` 单独以 `highfeature_lr/20` 优化 `_features_rest`（`gaussian_model.py:205,209`）；`capture/restore` 相应扩展（`:73-128`）。
- `train_base.sh` 将 `densification_interval` 从 100 提到 500（省去 4/5 的打分开销），靠 VCD 门控补偿稀疏事件下的选择质量。

---

## F. Densification 生命周期（Paper B 最关键）

**通用背景**：高斯的全部状态由 6 个按行对齐的 `nn.Parameter`（`_xyz (N,3)`、`_features_dc (N,1,3)`、`_features_rest (N,15,3)`、`_opacity (N,1)`、`_scaling (N,3)`、`_rotation (N,4)`）+ 每参数两份 Adam 矩（`exp_avg`/`exp_avg_sq`）+ 辅助张量（`xyz_gradient_accum (N,1)`、`xyz_gradient_accum_abs (N,1)`、`denom (N,1)`、`max_radii2D (N,)`、`tmp_radii (N,)`）构成。**身份 = 行下标，没有任何持久 ID。**

### F.1 Clone 生命周期

```text
入口: densify_and_prune_fastgs (gaussian_model.py:496)
  └─ densify_and_clone_fastgs(metric_mask, all_clones)            [gaussian_model.py:455]
       ① selected_pts_mask = metric_mask ∧ all_clones             (N,) bool   [:456]
       ② 张量拷贝（原始激活前空间值，逐行 gather）:                [:458-464]
            new_xyz        = self._xyz[mask]              (M,3)
            new_features_dc= self._features_dc[mask]      (M,1,3)
            new_features_rest = self._features_rest[mask] (M,15,3)
            new_opacities = self._opacity[mask]           (M,1)
            new_scaling   = self._scaling[mask]           (M,3)
            new_rotation  = self._rotation[mask]          (M,4)
            new_tmp_radii = self.tmp_radii[mask]          (M,)
       ③ densification_postfix(new_*)                            [gaussian_model.py:409]
            ├─ cat_tensors_to_optimizer                          [:383]
            │    对 optimizer 与 shoptimizer 的每个 param group:
            │      param       = cat(param, new)                 (N+M, ·)   ← 新高斯追加在末尾
            │      exp_avg     = cat(exp_avg, zeros_like(new))   ← ★子代 Adam 矩清零
            │      exp_avg_sq  = cat(exp_avg_sq, zeros_like(new))
            │      （重建 nn.Parameter 并重新挂回 optimizer.state，del 旧键）
            ├─ 模型张量重新指向优化器内的参数                    [:418-423]
            ├─ tmp_radii = cat(tmp_radii, new_tmp_radii)         [:425]
            └─ ★清零所有统计: xyz_gradient_accum/abs、denom、
               max_radii2D 全部 zeros((N+M,·))                    [:426-429]
       ④ 回到 densify_and_prune_fastgs:520-522
            opacity = inverse_sigmoid(min(α, 0.8))
            └─ replace_tensor_to_optimizer("opacity")            [:327]
                 exp_avg/exp_avg_sq 也被整体清零（:332-333）
       ⑤ self.tmp_radii = None                                    [:523-524]
```

**要点**：parent 的行保持原下标不动；child 追加在尾部；child 的 Adam 矩从零开始（继承属性值、不继承动量）；**每次 densification 事件把全体（含老高斯）的梯度统计与 max_radii2D 清零**。

### F.2 Split 生命周期

```text
入口: densify_and_prune_fastgs (gaussian_model.py:497)
  └─ densify_and_split_fastgs(metric_mask, all_splits, N=2)       [gaussian_model.py:431]
       ① n_init_points = 当前 N（注意：clone 步刚发生在前，N 已含新 clone）
          mask = metric_mask ∧ filter                              (N_old,) bool
          selected_pts_mask = zeros(N, bool); [:N_old] = mask      (N,) bool  ← padding 模式  [:434-436]
       ② 子代属性生成（全部 .repeat(N=2) → 2M 行）:                [:438-448]
            stds      = get_scaling[mask].repeat(2,1)              (2M,3)
            samples   = Normal(0, stds)                            (2M,3)   ← 父本局部系采样
            rots      = build_rotation(_rotation[mask]).repeat(2,1,1)
            new_xyz   = R·samples + parent_xyz.repeat(2,1)         (2M,3)
            new_scaling = log( get_scaling[mask].repeat(2,1) / 1.6 )   (2M,3)  ← /1.6 = 0.8·N
            new_rotation/features/opacity = parent.repeat(2,·)
            new_tmp_radii = tmp_radii[mask].repeat(2)
       ③ densification_postfix(new_*)                              [:450]   ← 2M 个子代追加尾部，Adam 矩清零
       ④ prune_filter = cat(selected_pts_mask,                     (N+2M,) bool
                           zeros(2M, bool))                        [:452]
            → 前段 True = 父本（原下标处），尾段 False = 子代
       ⑤ prune_points(prune_filter)                                [:453]
            → 删除全部父本行；子代行因 compaction 前移
```

**要点**：子代不是"替换"父本位置，而是**先追加、后删父**；父本属性（含 raw opacity/scaling/rotation/SH/tmp_radii）被复制，Adam 矩归零；随机种子来自 `torch.normal`（未固定）。

### F.3 Prune 生命周期

```text
三个调用点:
  (a) densify_and_split_fastgs:453   prune_filter（删父本）
  (b) densify_and_prune_fastgs:518   final_prune（budget 加权抽样结果）
  (c) final_prune_fastgs:540         final_prune（α<0.1 ∨ pruning_score>0.9）
       │  注意 (c) 单独发生（15k 后），此时 self.tmp_radii 已是 None
  └─ prune_points(mask)  —— mask 语义 = "要删的行"        [gaussian_model.py:364]
       ① valid_points_mask = ~mask                     (N,) bool
       ② _prune_optimizer(valid_points_mask)            [:342]
            对 optimizer + shoptimizer 的每个 group:
              exp_avg    = exp_avg[valid]               ← 幸存者的动量【保留】
              exp_avg_sq = exp_avg_sq[valid]
              param      = param[valid]（重建 nn.Parameter）
       ③ 模型 6 张量 = 对应优化器组张量                 [:368-373]
       ④ xyz_gradient_accum / _abs / denom / max_radii2D = [valid]   [:375-379]
       ⑤ if self.tmp_radii is not None: tmp_radii = tmp_radii[valid] [:380-381]
```

**索引变化**：布尔掩码索引做 compaction——所有幸存高斯按原相对顺序重新编号为 `0..N_new−1`。**任何按行下标缓存的外部映射在 prune 后全部失效**。

---

## G. Paper B 未来插入点分析（只分析，不实现）

### G.1 关键结论：当前代码没有稳定 Gaussian identity，只有 tensor index

证据：

1. **没有任何 id 字段**：`GaussianModel` 的全部 per-Gaussian 状态就是 F 节列出的张量集合，`save_ply`/`load_ply`（`gaussian_model.py:246-325`）也只写属性列，无 id。
2. **三个操作都会改写下标语义**：clone 追加（尾部新下标）、split 删父（尾部子代前移）、prune compaction（全体重排）。一个高斯训练全程的下标是不断变化的。
3. **历史被周期性抹除**：`densification_postfix`（`gaussian_model.py:426-429`）每次 densification 事件把梯度累积/denom/max_radii2D 清零——任何寄存在这些缓冲里的"历史"活不过一个事件。
4. **唯一跨代传递的东西**：raw 属性拷贝（clone）、属性+缩小 scaling（split）、`tmp_radii`（两者都有）、以及——注意——**Adam 矩不遗传**（子代/克隆一律清零，`cat_tensors_to_optimizer:395-396`）。

因此 Paper B 的 `persistent_gaussian_id` 必须是**独立于上述任何缓冲的第七类张量**，并且要镜像 optimizer 张量的三条生命周期通路（见 G.3）。

### G.2 插入点地图（按事件前后）

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 训练循环 train.py                                                             │
│  :130 add_densification_stats          （每个 iter，无索引变化，无需插入）      │
│  :139 densify_and_prune_fastgs ─────────────────────────────────┐            │
│  :158 final_prune_fastgs ────────────────────────────────┐      │            │
└──────────────────────────────────────────────────────────┼──────┼────────────┘
                                                           ▼      ▼
                    ┌──────────────────────────────────────────────────────┐
                    │ gaussian_model.py                                     │
                    │  densify_and_prune_fastgs:468                         │
                    │   ├─ :496 densify_and_clone_fastgs:455                │
                    │   │    [clone 前] :457 捕获 parent_id 快照             │
                    │   │    [clone 后] :466 (postfix 返回后) 尾部 = 子代     │
                    │   │            └─ 内部 densification_postfix:409      │
                    │   │                 (append 通路: 需为 id 张量扩展)     │
                    │   ├─ :497 densify_and_split_fastgs:431                │
                    │   │    [split 前] :434-436 parent 快照（属性+id）       │
                    │   │    [split 后] :450 (postfix 后/prune 前) 登记子代   │
                    │   │               :453 prune_points(删父)              │
                    │   ├─ :505-518 budget 剪枝 → :518 prune_points          │
                    │   └─ (阶段2) final_prune_fastgs:533                    │
                    │        [prune 前] :540 之前捕获被删 id                   │
                    │        └─ prune_points:364                             │
                    │             [prune 前] :365 mask 内即被删集合            │
                    │             [prune 后] :381 之后 compaction 已完成       │
                    └──────────────────────────────────────────────────────┘
```

### G.3 各机制的具体插入建议（含函数/行号）

**1) `persistent_gaussian_id`（int64 per-Gaussian 张量）**

| 生命周期通路 | 必须镜像的函数 | 具体插入点 |
|---|---|---|
| 初始化 | `create_from_pcd` / `load_ply` | `gaussian_model.py:190` 之后赋 `arange(N_init)`；`load_ply` 需在 PLY 增加 id 属性 |
| append（clone/split 子代） | `densification_postfix` | **不能**走 `:426-429` 的清零逻辑；需在 `:425`（tmp_radii cat 处）追加 `self._gaussian_id = cat(id, new_ids)`，new_ids 来自全局单调计数器（clone: 在 `:457-464` gather 处同步生成；split: 仿 `new_rotation` 的 `.repeat(N,1)` 模式在 `:444` 附近生成） |
| prune（compaction） | `prune_points` | `:375` 一排掩码索引处加 `self._gaussian_id = self._gaussian_id[valid_points_mask]`（与 `tmp_radii` 的 `:380-381` 完全同构——**tmp_radii 就是现成的模板**，但它每次 densify 后被置 None，id 张量不能置 None、也不能被 `densification_postfix` 清零） |
| checkpoint | `capture`/`restore` | `:73-128` 元组需加一项，否则断点续训丢身份 |
| 落盘 | `save_ply`/`load_ply` | `construct_list_of_attributes:246` 加 `id` 列 |

**2) `event_id`（全局事件计数器）**
- 递增点：`densify_and_clone_fastgs:455` 入口、`densify_and_split_fastgs:431` 入口、`prune_points:364` 入口（或简化为在 `densify_and_prune_fastgs:468` 与 `final_prune_fastgs:533` 两个编排函数入口各 +1，把 clone/split/prune 合并视为一个复合事件——**推荐后者**，与 `torch.cuda.empty_cache():526` 的事件边界一致）。
- 传递：作为参数传入两个 densify 函数，写进 lineage 记录。

**3) parent snapshot（clone 前 / split 前）**
- clone 前：`gaussian_model.py:457`（`selected_pts_mask` 刚算完、任何张量未动）——快照 `(parent_id, 关键属性, iteration, event_id)`。此处属性张量随后只读不写，安全。
- split 前：`gaussian_model.py:434-436`——**必须在 `:450` postfix 之前**完成快照，因为 `:453` 会把父本行物理删除；postfix 之后父本虽仍在（还没 prune）但已被复制出子代，语义上"split 事件"的原子快照点应取 `:437`（生成子代前）。

**4) child lineage（clone 后 / split 后）**
- clone 后：`gaussian_model.py:466`（`densification_postfix` 返回处）——新克隆占据尾部区间 `[N_before, N_before+M)`，`N_before = n_init_points_of_this_call`；登记 `(child_id → parent_id, event_id)`。此处用**行区间**定位子代是安全的（下一步 split 的 postfix 之前不会再有改动；且同函数内 split 的 padding 逻辑 `:436` 已假设"clone 追加在尾部"这一事实）。
- split 后：`gaussian_model.py:450` 返回处（**prune 前**）登记 `(2M 个 child_id → parent_id, 事件参数 σ/R)`；若在 `:453` 之后登记，尾部区间已因删父前移，须改为纯 id 引用（不用下标）。**推荐 prune 前登记**。

**5) prune 前 / prune 后**
- prune 前：`gaussian_model.py:365`——`mask` 参数本身就是被删集合（注意三个调用点传入的都是"删除语义"掩码）；快照 `self._gaussian_id[mask]` + 当时的属性行，即得"墓碑记录"（Paper B 未来 collapse/回滚所需）。`final_prune_fastgs:540` 调用点同理，且该阶段 `tmp_radii is None`，快照代码要容忍这一点。
- prune 后：`prune_points` 末尾（`:381` 之后）只需做一件事——**若 Paper B 维护任何 id→index 的外部索引表，必须在此处按 `valid_points_mask` 重映射**；否则无需动作。

**6) 额外风险清单（Paper B 实现时必须核对）**
- `replace_tensor_to_optimizer`（`gaussian_model.py:327`）：opacity 截断会**整体重建** opacity 参数及其 Adam 矩，但不改变行数/顺序——id 张量无需处理，但若 Paper B 把 id 放进 optimizer 参数组则会被误清零。
- `densify_and_prune_fastgs:505-518` 的 padding 模式：新出生高斯位于 `scores.shape[0]` 之后的 pad 区。任何"按打分张量长度对齐"的新逻辑都要遵守同样约定。
- `restore`（`:108`）从 checkpoint 恢复时不经过任何 densification 通路——id 计数器也必须一并持久化，否则续训后 id 撞号。
- split 的随机性（`torch.normal`, `:440`）未固定种子：Paper B 若要"确定性回放 lineage"，需要另行固定 RNG 状态（vanilla 亦然）。
- VCD/VCP 打分张量（importance/pruning_score）长度是**打分时刻的 N**，在 clone→split→prune 的复合事件中逐渐过期——若未来给 lineage 记录附上"出生时的分数"，应在 `compute_gaussian_score_fastgs` 返回处（`fast_utils.py:105`）随事件一起归档，而不是事后按下标回查。

### G.4 一句话结论

> 当前 FastGS（以及它的全部上游）中，**高斯身份 = 张量行号**，且该行号在每次 densification 事件中都会失效重建；唯一贯穿生命周期的 per-Gaussian 附件是 `tmp_radii` 的处理模式（append 时 cat、prune 时掩码索引、事件后清理）。Paper B 的 lineage 张量应以 `tmp_radii` 为模板、但取消"事件后清理"并避开 `densification_postfix:426-429` 的清零区，同时镜像 optimizer 的 append/prune 两条通路与 capture/restore、save/load_ply 四个持久化点。

---

## H. 附：阈值与超参速查表

| 参数 | 值 | 定义处 | 使用处 |
|---|---|---|---|
| `densify_from_iter` / `densify_until_iter` | 500 / 15000 | `arguments/__init__.py:89-90` | `train.py:127,132` |
| `densification_interval` | 100（base.sh: 500；big.sh: 100） | `:87` | `train.py:132` |
| `opacity_reset_interval` | 3000 | `:88` | `train.py:133,147`（同时决定 size_threshold=20 生效线） |
| `loss_thresh` | 0.1 | `:94` | `fast_utils.py:82` |
| `grad_thresh`（clone，普通梯度） | 0.0002 | `:98` | `gaussian_model.py:484` |
| `grad_abs_thresh`（split，AbsGS 绝对梯度） | 0.0012（场景 0.0002–0.002） | `:95` | `gaussian_model.py:485` |
| `dense`（= percent_dense） | 0.001（场景 0.003–0.015） | `:99` | `gaussian_model.py:486-487` |
| `mult`（Compact Box β） | 0.5（T&T/DB: 0.7） | `:100` | renderer → `auxiliary.h:338` |
| VCD 门控 | `importance_score > 5`（硬编码） | — | `gaussian_model.py:494` |
| 打分相机数 | 10（硬编码） | `fast_utils.py:13` | — |
| densify 期 prune | α<0.005；radii>20；scale>0.1·extent；budget=50% | — | `gaussian_model.py:499-518` |
| opacity 截断（densify 后） | 0.8 | — | `gaussian_model.py:520-522` |
| final prune | 每 3000 iter（15k,30k 开区间）；α<0.1 ∨ pruning_score>0.9 | — | `train.py:153`、`gaussian_model.py:533-540` |
| optimizer 步频 | ≤15k: 1/16；15–20k: 32；>20k: 64 | — | `gaussian_model.py:225-244` |
| split 子代缩放 | /(0.8·N)=1.6，N=2 | — | `gaussian_model.py:443` |

*生成于 2026-08-26，基于工作区 `/home/wyc/workspace/FastGS` 静态代码审计；未执行训练，未改动任何算法文件。*

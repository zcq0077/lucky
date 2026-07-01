# Map-Conditioned Fork-Aware TrAISformer Implementation Notes

本文档记录当前仓库中已经实现的 Map-Conditioned Fork-Aware TrAISformer 方案，包括设计动机、代码改动、运行方式、关键配置、验证方法和后续实验建议。

## 1. 实现目标

原始 TrAISformer 使用 AIS 历史序列：

```text
x_t = [lat_t, lon_t, sog_t, cog_t]
```

并通过 sparse four-hot 分类头分别预测：

```text
lat, lon, sog, cog
```

这种结构保留了多模态生成能力，但模型只能从历史轨迹序列本身学习，并不知道当前位置附近的航路结构。当前实现的目标是给模型加入训练集统计得到的航路先验：

```text
map_feature(lat, lon)
```

让模型在训练和推理时都能查询当前位置所在网格的航路信息，从而改善：

```text
1. 连续转弯区域的预测稳定性；
2. 航路分叉区域的方向选择；
3. 普通直行航段和复杂航段的区分；
4. 对局部高转向率区域的建模能力。
```

本实现不替换原始 four-hot 预测头，也不引入强导航约束或强重采样规则。热图只是作为条件表示加入 Transformer 输入 embedding。

## 2. 当前实现范围

本次已经实现：

```text
1. 新增 map_prior.py，用训练集构建航路先验热图；
2. 在 config_trAISformer.py 中加入 map prior 和 turn intent 配置；
3. 在 models.py 中加入 map_encoder，并将 map embedding 加到 token embedding；
4. 在 models.py 中加入可选 turn intent 辅助任务；
5. 在 trAISformer.py 中用 Data["train"] 构建并注册 map prior；
6. 在 trainers.py 中修正 next-step 训练 mask；
7. 新增 .gitignore，忽略 __pycache__ 和 results。
```

本次没有实现：

```text
1. 强导航约束 loss；
2. 候选点硬评分；
3. 强制重采样；
4. 复杂 map-aware sampling；
5. obstacle 通道默认启用。
```

这些功能容易压平真实大转弯或破坏多模态生成，因此不作为第一版主方案。

## 3. 热图先验构建

新增文件：

```text
map_prior.py
```

主函数：

```python
build_map_prior(l_data, config, savedir=None)
```

输入为训练集轨迹列表 `Data["train"]`，输出为：

```text
features          shape: [H, W, C]
direction_probs   shape: [H, W, K]
visit_count       shape: [H, W]
turn_rate         shape: [H, W]
branch_score      shape: [H, W]
direction_entropy shape: [H, W]
density_norm      shape: [H, W]
```

默认网格大小：

```python
map_prior_lat_size = 120
map_prior_lon_size = 120
```

默认方向桶数量：

```python
map_direction_bins = 36
```

即每 10 度一个方向桶。

### 3.1 density_norm

对训练集中每个轨迹点按归一化经纬度投影到网格：

```text
cell = grid_index(lat_norm, lon_norm)
visit_count[cell] += 1
```

之后对访问次数做空间平滑和 `log1p` 归一化，得到：

```text
density_norm
```

它表示历史上该区域是否经常被船舶经过。

### 3.2 turn_rate

对每条轨迹取连续三点：

```text
p_{t-1}, p_t, p_{t+1}
```

先把归一化经纬度还原成真实经纬度：

```python
lat = lat_min + lat_norm * (lat_max - lat_min)
lon = lon_min + lon_norm * (lon_max - lon_min)
```

然后计算真实地理 bearing：

```text
heading_in  = bearing(p_{t-1} -> p_t)
heading_out = bearing(p_t -> p_{t+1})
```

转向角：

```text
turn_angle = circular_abs_diff(heading_out, heading_in)
```

如果：

```python
turn_angle >= map_turn_angle_threshold_deg
```

则认为当前位置发生一次明显转向。默认阈值：

```python
map_turn_angle_threshold_deg = 25.0
```

最终：

```text
turn_rate[cell] = turn_count[cell] / valid_turn_base_count[cell]
```

为了避免噪声，访问数低于：

```python
map_min_count = 10
```

的 cell 会被抑制。

### 3.3 branch_score

分叉不等同于转弯。分叉看的是同一区域是否存在多个主要离开方向。

对每个轨迹点计算离开方向：

```text
heading_out = bearing(p_t -> p_{t+1})
```

并统计方向直方图：

```text
direction_hist[cell, k] += 1
```

得到方向概率：

```text
direction_probs[cell, k]
```

然后对方向概率做环形平滑，寻找局部峰值。只有满足以下条件的峰才算有效主方向：

```text
1. 峰值概率 >= map_branch_peak_prob_threshold；
2. 不同峰之间角度间隔 >= map_branch_min_separation_deg。
```

默认：

```python
map_branch_peak_prob_threshold = 0.12
map_branch_min_separation_deg = 35.0
```

得到主峰数量：

```text
branch_peak_count[cell]
```

最终定义：

```text
branch_score = clamp((branch_peak_count - 1) / 2, 0, 1)
```

含义：

```text
0.0: 只有一个主方向，不像分叉；
0.5: 两个主方向，可能是分叉；
1.0: 三个或更多主方向，强分叉或交汇区域。
```

### 3.4 direction_entropy

方向熵定义为：

```text
entropy[cell] = -sum_k p_k log(p_k) / log(K)
```

它表示该网格离开方向的分散程度。

注意：方向熵只作为辅助通道，不单独判断分叉。普通弯道也可能有较高方向熵，因此实际分叉判断以多峰方向为主。

### 3.5 默认 feature 通道

当前默认使用 4 个通道：

```text
features = [
    density_norm,
    turn_rate,
    branch_score,
    direction_entropy,
]
```

当前没有默认启用 obstacle 通道：

```python
map_use_obstacle_features = False
```

原因是低密度区域不一定是障碍物，也可能只是数据稀疏区。第一版先不让模型学习这个不稳定信号。

如果后续确实要启用 obstacle 通道，需要同时设置：

```python
map_use_obstacle_features = True
map_prior_channels = 6
```

启用后 feature 通道会变为：

```text
features = [
    density_norm,
    turn_rate,
    branch_score,
    direction_entropy,
    obstacle_proximity,
    obstacle_turn_score,
]
```

## 4. 模型融合方式

修改文件：

```text
models.py
```

原始模型中，四个属性分别 embedding 后拼接：

```python
lat_embeddings = self.lat_emb(inputs[:, :, 0])
lon_embeddings = self.lon_emb(inputs[:, :, 1])
sog_embeddings = self.sog_emb(inputs[:, :, 2])
cog_embeddings = self.cog_emb(inputs[:, :, 3])

token_embeddings = torch.cat(
    (lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),
    dim=-1,
)
```

然后加位置编码：

```python
embeddings = token_embeddings + position_embeddings
```

当前实现中，如果：

```python
use_map_prior = True
```

则额外查询 map feature：

```python
map_features = self.query_map_features(inputs_real)
```

再通过轻量 MLP 映射到 Transformer embedding 维度：

```python
self.map_encoder = nn.Sequential(
    nn.Linear(map_prior_channels, config.n_embd),
    nn.GELU(),
    nn.Dropout(map_emb_pdrop),
    nn.Linear(config.n_embd, config.n_embd),
    nn.LayerNorm(config.n_embd),
)
```

最终加法融合：

```python
embeddings = embeddings + map_emb_w * map_embeddings
```

默认：

```python
map_emb_w = 0.10
map_emb_pdrop = 0.10
```

选择加法融合的原因：

```text
1. 不改变 Transformer 输入维度；
2. 不改变原始 four-hot 预测头；
3. use_map_prior=False 时可以退回原始模型；
4. 比 concat 更稳定，第一版风险更小。
```

## 5. Map Prior 注册方式

模型新增函数：

```python
register_map_prior(features, direction_probs=None)
```

它会把 `features` 注册为非持久化 buffer：

```python
self.register_buffer("map_prior_features", features, persistent=False)
```

使用 `persistent=False` 的原因是：

```text
1. map prior 可以由训练集重新构建；
2. 不把大数组写进 checkpoint；
3. 避免 checkpoint 和当前配置/数据不匹配。
```

因此每次运行 `trAISformer.py` 时，如果开启 `use_map_prior`，都会重新构建并注册一次 map prior。

## 6. Turn Intent 辅助任务

修改文件：

```text
models.py
```

配置开关：

```python
use_turn_intent_head = True
turn_intent_loss_w = 0.03
```

新增预测头：

```python
self.turn_head = nn.Linear(config.n_embd, 5)
```

5 类转向意图：

```text
0: straight
1: slight_left
2: slight_right
3: sharp_left
4: sharp_right
```

标签自动从真实轨迹生成。对连续三点：

```text
p_{t-1}, p_t, p_{t+1}
```

计算：

```text
heading_in  = bearing(p_{t-1} -> p_t)
heading_out = bearing(p_t -> p_{t+1})
delta = signed_circular_diff(heading_out, heading_in)
```

注意：当前 bearing 定义为从北开始顺时针角度，因此：

```text
delta > 0 表示右转；
delta < 0 表示左转。
```

标签规则：

```text
abs(delta) < straight_threshold -> straight
delta <= -sharp_threshold       -> sharp_left
delta >= sharp_threshold        -> sharp_right
delta < -straight_threshold     -> slight_left
delta > straight_threshold      -> slight_right
```

默认阈值：

```python
turn_straight_threshold_deg = 10.0
turn_sharp_threshold_deg = 35.0
```

辅助 loss：

```python
loss = main_loss + turn_intent_loss_w * turn_loss
```

默认权重较小：

```python
turn_intent_loss_w = 0.03
```

这样可以让模型显式学习转向模式，同时尽量不压制原始 four-hot 主任务。

## 7. Padding Mask 处理

当前数据中，大量轨迹短于训练窗口。如果处理不当，padding 区域会被模型当成真实轨迹点，影响训练和 turn intent 标签。

本次修正了训练 mask：

```python
masks = masks[:, 1:].to(self.device)
```

原因是 `with_targets=True` 时：

```python
inputs  = x[:, :-1, :]
targets = x[:, 1:, :]
```

loss 对应的是 target 位置，因此 mask 也应该对齐到：

```text
target mask = masks[:, 1:]
```

turn intent 标签也会检查三点有效性，只在连续位置都非 padding、且相邻点有有效位移时计算辅助 loss。

## 8. 训练入口修改

修改文件：

```text
trAISformer.py
```

模型创建后，如果开启 map prior：

```python
if getattr(cf, "use_map_prior", False):
    import map_prior

    prior = map_prior.build_map_prior(Data["train"], cf, savedir=cf.savedir)
    model.register_map_prior(prior["features"], prior["direction_probs"])
```

注意：这里只使用：

```text
Data["train"]
```

构建热图，不使用 valid/test，避免数据泄漏。

运行后会在结果目录保存：

```text
map_prior.npz
map_prior_components.png
```

`map_prior_components.png` 用于快速检查 density、turn_rate、branch_score、direction_entropy 是否合理。

## 9. 关键配置

当前配置位于：

```text
config_trAISformer.py
```

Map prior 配置：

```python
use_map_prior = True
map_prior_lat_size = 120
map_prior_lon_size = 120
map_direction_bins = 36
map_min_count = 10
map_turn_angle_threshold_deg = 25.0
map_branch_peak_prob_threshold = 0.12
map_branch_min_separation_deg = 35.0
map_smooth_iter = 1
map_direction_smooth_iter = 1
map_use_obstacle_features = False
map_prior_channels = 6 if map_use_obstacle_features else 4
map_emb_w = 0.10
map_emb_pdrop = 0.10
map_prior_plot = True
```

Turn intent 配置：

```python
use_turn_intent_head = True
turn_intent_loss_w = 0.03
turn_straight_threshold_deg = 10.0
turn_sharp_threshold_deg = 35.0
turn_ignore_first = True
```

实验目录名会自动加入：

```text
-map
-turn
```

例如：

```text
results/ct_dma-pos-pos_vicinity-10-40-map-turn-...
```

## 10. 运行方式

推荐使用已经安装 PyTorch 的 Conda 环境：

```powershell
cd D:\工作\1_code\TrAISformer-main
conda activate dst-mamba
python trAISformer.py
```

如果不想显式 activate：

```powershell
cd D:\工作\1_code\TrAISformer-main
conda run -n dst-mamba python trAISformer.py
```

运行流程：

```text
1. 加载 train/valid/test 数据；
2. 过滤低速开始前轨迹点；
3. 创建 AISDataset；
4. 用 Data["train"] 构建 map_prior；
5. 注册 map_prior 到模型；
6. 训练模型；
7. 保存 best checkpoint；
8. 评估 test 集；
9. 保存 prediction_error.png。
```

## 11. 验证过的检查

已经通过以下检查：

```powershell
python -m py_compile config_trAISformer.py models.py trainers.py trAISformer.py map_prior.py
```

在 `dst-mamba` 环境下也通过：

```powershell
conda run -n dst-mamba python -m py_compile config_trAISformer.py models.py trainers.py trAISformer.py map_prior.py
```

Smoke test 已验证：

```text
1. map_prior 能构建出 shape 为 [120, 120, 4] 的 features；
2. model forward 能正常输出 logits；
3. loss 为有限值；
4. trainers.sample 能正常生成预测序列；
5. optimizer 分组能覆盖新增 map_encoder 和 turn_head 参数。
```

## 12. 推荐消融实验

为了判断改动是否真正有效，建议至少跑以下实验：

```text
A. 原始 baseline
   use_map_prior = False
   use_turn_intent_head = False

B. Map Embedding only
   use_map_prior = True
   use_turn_intent_head = False

C. Map Embedding + Turn Intent
   use_map_prior = True
   use_turn_intent_head = True

D. Map Embedding + different map_emb_w
   map_emb_w = 0.05 / 0.10 / 0.30

E. Turn intent weight ablation
   turn_intent_loss_w = 0.00 / 0.02 / 0.03 / 0.05
```

建议不要只看 overall ADE/FDE，还要单独评估：

```text
overall ADE/FDE
turn-segment ADE/FDE
branch-region ADE/FDE
straight-segment ADE/FDE
```

原因是 map prior 的收益主要集中在复杂转向和分叉区域，整体 ADE/FDE 可能被大量普通直行航段稀释。

## 13. 复现论文指标时的注意事项

如果发现结果比论文差，不应优先归因于设备不同。设备一般主要影响训练速度，只有在显存不足导致 batch size、模型大小或训练轮数改变时，才会明显影响结果。

更常见原因包括：

```text
1. 代码输出单位是 km，论文表格可能是 nautical mile；
2. 数据预处理和 train/valid/test 划分不完全一致；
3. 采样参数 n_samples、top_k、r_vicinity 不一致；
4. 预测 horizon 和 init_seqlen 不一致；
5. 随机种子、PyTorch/CUDA 版本带来的小幅波动；
6. 原评估代码中存在经纬度范围硬编码，需要检查。
```

README 中说明：

```text
代码输出是 km，论文中数值换算成了 nautical mile。
```

换算：

```text
1 nautical mile = 1.852 km
nmi = km / 1.852
```

另外，当前评估脚本中存在硬编码：

```python
v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
```

而配置中实际 ROI 为：

```python
lat_min = 55.5
lat_max = 58.0
lon_min = 10.3
lon_max = 13.0
```

更合理的写法应使用模型自身范围：

```python
v_ranges = torch.tensor([model.lat_range, model.lon_range, 0, 0]).to(cf.device)
v_roi_min = torch.tensor([model.lat_min, model.lon_min, 0, 0]).to(cf.device)
```

这会直接影响 haversine 误差计算，建议后续单独修正并重新评估 baseline。

## 14. 风险和调参建议

### 14.1 map_emb_w 太大

风险：

```text
模型过度依赖训练集热图，泛化变差。
```

建议：

```python
map_emb_w = 0.05 或 0.10
```

如果验证集稳定提升，再尝试：

```python
map_emb_w = 0.30
```

### 14.2 map_min_count 太低

风险：

```text
低样本 cell 的 branch_score 或 turn_rate 受噪声影响。
```

当前默认：

```python
map_min_count = 10
```

如果热图噪声仍然明显，可以尝试：

```python
map_min_count = 20
```

### 14.3 turn_intent_loss_w 太大

风险：

```text
辅助任务压制 four-hot 主任务。
```

当前默认：

```python
turn_intent_loss_w = 0.03
```

建议不要一开始超过：

```python
turn_intent_loss_w = 0.10
```

### 14.4 obstacle 通道误导

风险：

```text
低密度区域不一定是障碍，也可能只是训练数据少。
```

因此当前默认：

```python
map_use_obstacle_features = False
```

只有在确认 density 图和真实海岸线/障碍区域吻合后，再考虑启用。

## 15. 推荐下一步

建议按以下顺序继续：

```text
1. 先跑 baseline：
   use_map_prior=False
   use_turn_intent_head=False

2. 修正评估中的 v_ranges/v_roi_min 硬编码；

3. 跑 Map Embedding only；

4. 跑 Map Embedding + Turn Intent；

5. 对比 overall、turn、branch、straight 分段误差；

6. 检查 map_prior_components.png 是否符合真实航路结构；

7. 如果复杂航段提升但直行航段退化，降低 map_emb_w 或 turn_intent_loss_w；

8. 如果整体无提升但复杂区域有提升，单独报告复杂区域指标。
```

第一版的核心目标不是用强规则把轨迹硬拉到航道上，而是让 Transformer 在训练阶段学习到局部航路结构先验。这种方式更稳，也更适合保留 TrAISformer 原本的多模态生成能力。

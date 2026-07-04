# Qwen 在线船长概率选择器融合方案

本文档整理一种不依赖“Qwen 打标签”的融合方案：把 Qwen 当作在线语义编码器，让它像经验船长一样参与 TrAISformer 的概率选择过程。核心目标是让 Qwen 不直接预测经纬度，而是根据历史轨迹和地图语义生成航行决策语义向量，再把这个语义向量转成方向概率偏置，修正 TrAISformer 原本输出的 lat/lon/sog/cog logits。

## 1. 总体思想

TrAISformer 原本做的是：

```text
历史 AIS 轨迹
    ↓
Transformer
    ↓
lat/lon/sog/cog logits
    ↓
softmax 概率
    ↓
采样未来点
```

加入 Qwen 后，希望变成：

```text
历史 AIS 轨迹 + 地图语义摘要
    ↓
Qwen 语义编码器
    ↓
船长语义向量 qwen_vec
    ↓
Captain Probability Selector
    ↓
方向概率偏置 direction_bias
    ↓
修正 TrAISformer logits
    ↓
最终概率分布
    ↓
采样未来点
```

这里 Qwen 的作用不是输出一段文字，也不是输出具体坐标，而是提供高层航行判断：当前更应该直行、左转、右转、大幅左转还是大幅右转。TrAISformer 仍然负责具体的网格概率预测。

## 2. 为什么不让 Qwen 直接输出全部网格概率

当前配置中位置网格大约是：

```text
lat_size = 250
lon_size = 270
二维位置组合 = 250 * 270 = 67500
```

让 Qwen 直接判断 67500 个位置概率不现实，也不稳定。更合理的是让 Qwen 只判断少量高层方向：

```text
straight
slight_left
slight_right
sharp_left
sharp_right
```

然后把这 5 类方向偏置投影到局部 lat/lon 候选网格上。这样既保留大模型的语义理解，又不破坏 TrAISformer 的数值预测能力。

## 3. 融合位置

推荐同时支持两种融合位置。

### 3.1 Embedding 融合

把 Qwen 语义向量投影到 TrAISformer 的 embedding 维度，然后加到每个时间步的 token embedding 上：

```text
token_emb = lat_emb + lon_emb + sog_emb + cog_emb + pos_emb
qwen_emb = Linear(qwen_vec)
final_emb = token_emb + qwen_emb_w * qwen_emb
```

这个方式让 Qwen 影响 Transformer 内部特征学习，适合训练阶段。

优点：

```text
实现简单
稳定
不会直接强改概率
可以让模型自己学会如何利用 Qwen 语义
```

缺点：

```text
对采样概率的影响不够直接
需要重新训练后才明显发挥作用
```

### 3.2 Logits 概率偏置融合

让 Qwen 语义向量经过一个小 MLP，输出方向偏置：

```text
qwen_vec
    ↓
Captain Selector MLP
    ↓
[straight, slight_left, slight_right, sharp_left, sharp_right]
```

然后把方向偏置加到 TrAISformer 的 lat/lon logits 上：

```text
final_logits = traisformer_logits + lambda_qwen * qwen_bias_logits
final_probs = softmax(final_logits)
```

这个方式让 Qwen 像“船长概率选择器”一样直接影响最终概率。

优点：

```text
作用直接
适合解释“Qwen 辅助概率选择”
对分叉、转弯、复杂航道更有针对性
```

缺点：

```text
需要小心控制 lambda_qwen
如果 Qwen 语义方向错误，会把概率推偏
```

第一版建议同时实现 embedding 融合和 logits 融合，但默认打开较弱的 logits 融合：

```text
qwen_emb_w = 0.05
qwen_bias_w = 0.05 或 0.10
```

## 4. 方向偏置如何映射到 lat/lon logits

TrAISformer 当前是分别输出 lat logits 和 lon logits，不是直接输出二维网格 logits。因此最精确的二维网格修正会比较重。第一版建议做轻量近似。

### 4.1 轻量版：对 lat/lon 边缘 logits 加方向趋势

假设当前位置为：

```text
lat_now_idx
lon_now_idx
```

对所有候选 lat index 和 lon index 计算相对方向趋势：

```text
候选 lat 大于当前 lat：向北趋势
候选 lat 小于当前 lat：向南趋势
候选 lon 大于当前 lon：向东趋势
候选 lon 小于当前 lon：向西趋势
```

结合当前航向，把“左转/右转/直行”转换成对 lat logits 和 lon logits 的加权偏置。

这个版本实现轻，速度快，但方向几何不够精细。

### 4.2 推荐版：局部二维候选窗口内构造方向 bias

由于采样本来使用 `pos_vicinity`，只在当前点附近采样，因此可以只考虑局部窗口：

```text
lat_idx ± r_vicinity
lon_idx ± r_vicinity
```

对窗口内每个二维候选点计算：

```text
candidate_bearing = bearing(current_point, candidate_point)
delta = candidate_bearing - current_heading
```

根据 delta 分类：

```text
abs(delta) <= 10°             -> straight
10° < delta <= 35°            -> slight_right
-35° <= delta < -10°          -> slight_left
delta > 35°                   -> sharp_right
delta < -35°                  -> sharp_left
abs(delta) > 120°             -> reverse penalty
```

然后得到一个局部二维 bias map：

```text
bias_2d[lat_i, lon_j] = direction_bias[class(delta)]
```

难点是当前模型分别采样 lat 和 lon。可以把二维 bias 压回边缘：

```text
lat_bias[i] = max_j bias_2d[i, j]
lon_bias[j] = max_i bias_2d[i, j]
```

然后：

```text
lat_logits = lat_logits + lambda_qwen * lat_bias
lon_logits = lon_logits + lambda_qwen * lon_bias
```

这个版本更符合“向哪里拐”的几何含义，也不需要重写整个模型为二维位置分类。

### 4.3 更高级版：二维联合位置采样

长期可以把 lat/lon 改成联合二维采样：

```text
P(lat, lon) = P(lat) * P(lon) * exp(qwen_bias_2d)
```

然后在局部窗口里直接从二维概率分布采样。这个版本最合理，但改动会比第一版大。建议等轻量版有效后再做。

## 5. 真在线方案

真在线指 Qwen 在训练和推理时实时参与前向计算。

### 5.1 训练流程

```text
batch seqs
    ↓
根据每条轨迹的历史段生成 prompt
    ↓
Qwen 实时编码 prompt
    ↓
得到 qwen_vec
    ↓
TrAISformer forward(seqs, qwen_vec)
    ↓
生成原始 logits
    ↓
Captain Selector 生成 qwen 方向 bias
    ↓
final_logits = logits + lambda_qwen * bias
    ↓
计算原始轨迹预测 loss
    ↓
反向传播更新 TrAISformer、qwen_proj、captain_selector
```

第一版建议冻结 Qwen：

```text
Qwen 参数 requires_grad = False
Qwen forward 使用 torch.no_grad()
只训练 TrAISformer 和融合层
```

这样 Qwen 参与训练流程，但不参与梯度更新。它像一个在线外部语义传感器。

### 5.2 推理流程

```text
输入前 18 个历史点
    ↓
生成 prompt
    ↓
Qwen 编码一次，得到 qwen_vec
    ↓
sample() 每一步都带 qwen_vec
    ↓
每一步用 qwen direction bias 修正 logits
    ↓
采样未来轨迹
```

第一版不建议每预测一步重新调用 Qwen，因为太慢。先在初始历史段调用一次 Qwen，并在整个预测窗口复用 qwen_vec。

### 5.3 真在线优点

```text
最符合“大模型在线引导”
推理时面对新轨迹可以即时编码
不需要预先生成标签或缓存文件
```

### 5.4 真在线缺点

```text
训练非常慢
Qwen 和 TrAISformer 同时占显存
batch size 可能需要大幅降低
如果 Qwen 出现 CPU offload，速度会明显下降
实验复现成本较高
```

### 5.5 适用场景

真在线适合：

```text
小批量实验
最终 demo
验证 Qwen 是否能实时辅助概率选择
```

不适合作为一开始的大规模训练方式。

## 6. 缓存在线方案

缓存在线指提前计算 Qwen 语义向量，训练和推理时直接读取 qwen_vec。注意它缓存的是 Qwen hidden embedding，不是标签，也不是答案。

### 6.1 为什么缓存仍然合理

如果满足：

```text
Qwen 冻结
prompt 固定
输入历史轨迹固定
```

那么：

```text
实时 Qwen(prompt) == 读取缓存 qwen_vec
```

从数学上看，缓存只是省掉重复前向计算，不改变模型使用 Qwen 的方式。

### 6.2 缓存生成流程

```text
训练集 / 验证集 / 测试集
    ↓
对每条轨迹取 init_seqlen 历史段
    ↓
生成 history-only prompt
    ↓
Qwen 编码
    ↓
取最后一层 hidden state pooling
    ↓
保存 qwen_vec
```

建议保存格式：

```text
qwen_cache/
  train_qwen_vecs.npz
  valid_qwen_vecs.npz
  test_qwen_vecs.npz
```

每条记录至少包含：

```text
trajectory_id
mmsi
time_start
qwen_vec
prompt_hash
```

其中 `prompt_hash` 用来防止 prompt 改了但缓存没更新。

### 6.3 训练流程

```text
Dataset 读取 seqs
    ↓
根据 trajectory_id 读取 qwen_vec
    ↓
TrAISformer forward(seqs, qwen_vec)
    ↓
Captain Selector 生成方向 bias
    ↓
修正 logits
    ↓
正常轨迹预测 loss
```

这里没有 Qwen 实时计算，所以训练速度接近普通 TrAISformer。

### 6.4 推理流程

对于测试集评估：

```text
提前缓存 test qwen_vec
评估时直接读取
```

对于真实新轨迹：

```text
第一次遇到轨迹时实时调用 Qwen
保存 qwen_vec 到缓存
后续复用
```

这就是比较实用的“在线 + 缓存”混合模式。

### 6.5 缓存在线优点

```text
训练快
显存压力小
可复现
不需要标签
仍然让 Qwen 语义向量参与模型 forward 和 logits 修正
```

### 6.6 缓存在线缺点

```text
如果 prompt 设计不好，需要重新生成缓存
如果想让 Qwen 随预测过程动态更新，就不能只缓存初始向量
严格说训练时不是每个 batch 实时跑 Qwen
```

## 7. Prompt 设计

无论真在线还是缓存在线，prompt 都必须只包含历史信息，不能包含真实未来。

推荐 prompt 不要太长，使用结构化摘要：

```text
You are a maritime navigation semantic encoder.
Use only the historical AIS summary and map statistics.
Do not predict exact coordinates.
Encode whether the vessel is likely to continue straight, turn left, turn right, or follow a complex channel.

Historical AIS summary:
- current latitude:
- current longitude:
- current heading:
- recent mean heading:
- recent mean turn:
- recent speed trend:
- current normalized SOG:
- current normalized COG:

Map statistics:
- density:
- turn_rate:
- branch_score:
- direction_entropy:
- dominant route directions:
```

不要让 Qwen 输出文本答案，只取 hidden state。

## 8. Qwen hidden state 怎么取

推荐使用最后一个有效 token 的 hidden state：

```text
qwen_hidden = last_hidden_state[batch, last_token, :]
```

也可以尝试 mean pooling：

```text
qwen_hidden = mean(last_hidden_state over non-padding tokens)
```

第一版建议用 last token hidden state，简单稳定。

然后投影：

```text
qwen_vec = LayerNorm(qwen_hidden)
qwen_emb = Linear(qwen_hidden_dim, traisformer_n_embd)
```

如果 Qwen hidden 维度较大，可以先降维：

```text
qwen_hidden_dim -> 512 -> n_embd
```

## 9. 训练哪些参数

第一版建议：

```text
Qwen: 冻结
qwen_proj: 训练
captain_selector: 训练
TrAISformer: 训练
map_encoder: 按当前配置训练
```

不建议第一版训练 Qwen。等整个流程有效后，再考虑 LoRA：

```text
第二阶段：冻结 Qwen + 训练融合层
第三阶段：LoRA 微调 Qwen 的少量参数
第四阶段：候选轨迹奖励或 GRPO
```

## 10. 配置建议

建议新增配置：

```python
use_qwen_semantic_encoder = False
qwen_model_path = r"D:\Jason1982\wsl\Models\Qwen3-4B-Instruct-2507"
qwen_freeze = True
qwen_use_cache = True
qwen_cache_dir = "./qwen_cache/"
qwen_fusion = "gated_add"  # "add", "gated_add", "logit_bias", "both"
qwen_emb_w = 0.05
qwen_bias_w = 0.10
qwen_direction_bins = 5
qwen_prompt_max_points = 18
qwen_recompute_each_step = False
```

第一版推荐：

```python
use_qwen_semantic_encoder = True
qwen_freeze = True
qwen_use_cache = True
qwen_fusion = "both"
qwen_emb_w = 0.05
qwen_bias_w = 0.05
```

## 11. 需要新增或修改的文件

建议新增：

```text
qwen_semantic_encoder.py
generate_qwen_embedding_cache.py
```

建议修改：

```text
config_trAISformer.py
datasets.py
models.py
trainers.py
trAISformer.py
```

### 11.1 qwen_semantic_encoder.py

职责：

```text
加载本地 Qwen
冻结参数
根据 AIS 历史和地图特征构造 prompt
返回 qwen hidden vector
支持 batch encode
支持 GPU/CPU 自动选择
```

### 11.2 generate_qwen_embedding_cache.py

职责：

```text
遍历 train/valid/test
为每条轨迹生成 qwen_vec
保存 npz 或 pt
记录 prompt_hash
```

### 11.3 datasets.py

职责：

```text
返回 seq, mask, seqlen, mmsi, time_start, qwen_vec
```

如果没有 qwen_vec，则返回全零向量和 qwen_mask=0。

### 11.4 models.py

职责：

```text
增加 qwen_proj
增加 qwen_gate
增加 captain_selector
forward 接收 qwen_vec
embedding 融合
logits bias 融合
```

### 11.5 trainers.py

职责：

```text
从 batch 中取 qwen_vec
传入 model
```

真在线模式下，trainer 还需要调用 qwen_encoder 实时生成 qwen_vec。

### 11.6 sample()

职责：

```text
采样时接收 qwen_vec
每一步预测 logits 时都传入 model
使用同一个 qwen_vec 修正所有预测步
```

## 12. 损失函数

第一版不需要额外标签 loss，只用原始轨迹预测 loss：

```text
loss = lat_loss + lon_loss + sog_loss + cog_loss
```

Qwen 通过 logits bias 影响预测概率。如果偏置方向有助于降低真实轨迹的 cross entropy，那么融合层会被训练得更有用。

可选正则：

```text
qwen_bias_l2 = mean(direction_bias^2)
loss += qwen_bias_reg_w * qwen_bias_l2
```

这个正则可以防止 Qwen bias 一开始过强。

## 13. 重要风险和解决方案

### 13.1 Qwen 语义错了会带偏概率

解决：

```text
qwen_bias_w 从 0.05 开始
使用 gated fusion
加入 bias L2 正则
做消融实验
```

### 13.2 真在线太慢

解决：

```text
优先使用缓存在线
只在 init_seqlen 调用一次 Qwen
降低 prompt 长度
冻结 Qwen
```

### 13.3 显存不足

解决：

```text
使用缓存在线
降低 batch_size
Qwen 使用 fp16/bf16
必要时使用更小 Qwen
```

### 13.4 数据泄漏

解决：

```text
Qwen prompt 只包含历史轨迹和训练集构建的地图先验
不能包含真实未来
测试集缓存也只能用历史段生成
```

### 13.5 Qwen 向量维度过大

解决：

```text
Linear 降维
LayerNorm
Dropout
qwen_emb_w 小权重融合
```

## 14. 推荐实验顺序

第一组，验证 Qwen 融合是否可用：

```text
Baseline
Baseline + Map Prior
Baseline + Map Prior + Qwen Embedding Add
Baseline + Map Prior + Qwen Logit Bias
Baseline + Map Prior + Qwen Embedding Add + Logit Bias
```

第二组，验证在线方式：

```text
Cached Qwen
True Online Qwen
Cached Qwen + realtime inference Qwen
```

第三组，验证复杂区域：

```text
整体 ADE/FDE
转弯区域 ADE/FDE
分叉区域 ADE/FDE
高 direction_entropy 区域 ADE/FDE
可视化 best-of-16
top-1 轨迹合理性
```

## 15. 最推荐的第一版实现

建议不要一开始就真在线大规模训练，而是先做：

```text
冻结 Qwen
缓存 qwen_vec
训练 qwen_proj + captain_selector + TrAISformer
embedding add + logits bias 两种融合都支持
qwen_bias_w = 0.05
```

这个版本满足：

```text
不需要 Qwen 打标签
Qwen 语义向量参与模型 forward
Qwen 方向偏置参与概率修正
训练速度可接受
显存压力可控
实验可复现
```

等这个版本有效后，再开启真在线：

```text
推理时实时调用 Qwen
训练时小 batch 真在线验证
最后考虑 LoRA 或强化学习
```

## 16. 一句话总结

这个方案可以让 Qwen 作为在线船长概率选择器：它不负责输出最终坐标，而是根据历史轨迹和地图语义生成航行方向偏置，直接修正 TrAISformer 的预测 logits，让模型在转弯、分叉和复杂航道中更倾向于合理方向。

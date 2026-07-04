# Qwen 轻量在线概率选择器使用说明

当前实现的是“缓存在线”轻量版，不再使用 `captain_qwen.py` 生成的 JSON 标签。Qwen 的作用变成：

```text
历史轨迹 + 地图摘要
    ↓
Qwen hidden embedding
    ↓
TrAISformer embedding 融合 + logits 概率偏置
```

Qwen 不输出坐标、不输出 JSON、不参与反向传播；它只提供语义向量。训练时更新的是 TrAISformer、`qwen_projector` 和 `captain_selector`。

## 第一步：生成 Qwen embedding cache

先激活能加载 Qwen 的环境，例如：

```powershell
conda activate lucky
```

生成全部 train/valid/test 缓存：

```powershell
python generate_qwen_embedding_cache.py --phase all --device cuda --batch-size 4 --overwrite
```

如果显存压力大，把 batch size 降低：

```powershell
python generate_qwen_embedding_cache.py --phase all --device cuda --batch-size 1 --overwrite
```

生成后会得到：

```text
qwen_cache/ct_dma_train_qwen_vecs.npz
qwen_cache/ct_dma_valid_qwen_vecs.npz
qwen_cache/ct_dma_test_qwen_vecs.npz
```

这些文件保存的是 Qwen hidden vector，不是标签。

## 第二步：打开 Qwen 融合

在 `config_trAISformer.py` 中设置：

```python
use_qwen_semantic_encoder = True
qwen_use_cache = True
qwen_fusion = "both"
qwen_emb_w = 0.05
qwen_bias_w = 0.05
```

如果只是想先跑 baseline，就保持：

```python
use_qwen_semantic_encoder = False
```

## 第三步：重新训练

开启 Qwen 后，实验名会自动带上 `-qwen`，不会覆盖原来的 checkpoint。

```powershell
python trAISformer.py
```

注意：如果 `retrain=False`，程序会尝试加载已有 `-qwen` checkpoint。第一次跑 Qwen 版本时应该设置：

```python
retrain = True
```

## 旧 JSON 标注版本

`captain_qwen.py` 现在是 legacy 工具，主训练路径不会读取 `captain_labels/*.jsonl`。

当前轻量版使用的是：

```text
generate_qwen_embedding_cache.py
qwen_semantic_encoder.py
qwen_cache/*.npz
```

## 当前实现的融合方式

模型里有两种融合：

```text
Embedding 融合：
qwen_vec -> qwen_projector -> 加到 token embedding

Logits 融合：
qwen_vec -> captain_selector -> 5 个方向分数
方向分数 -> lat/lon logits 偏置
```

最终概率来自：

```text
final_logits = TrAISformer logits + qwen_bias_w * Qwen direction bias
final_probs = softmax(final_logits)
```

建议先从 `qwen_bias_w = 0.05` 开始，确认不会把概率强行推偏后再调大。

# Qwen 船长语义接入说明

本文档说明当前项目中 Qwen 的接入方式。当前实现的是第一阶段：让本地 Qwen 作为“船长语义教师”，根据 AIS 历史轨迹、地图先验特征和训练阶段可见的真实未来趋势，生成结构化的航行语义标签。后续可以把这些标签蒸馏进 TrAISformer 的 Captain Head，让模型在训练和推理阶段使用这些语义判断。

## 当前已经实现的内容

新增脚本：

```text
captain_qwen.py
```

新增配置：

```text
qwen_model_path = r"D:\Jason1982\wsl\Models\Qwen3-4B-Instruct-2507"
qwen_captain_labels_dir = "./captain_labels/"
qwen_captain_label_phase = "train"
qwen_captain_max_samples = 1000
qwen_captain_future_steps = 24
qwen_captain_temperature = 0.1
qwen_captain_top_p = 0.8
qwen_captain_max_new_tokens = 512
```

脚本会做以下事情：

1. 按 `trAISformer.py` 相同方式读取并过滤 AIS 数据。
2. 只用训练集构建地图先验，避免测试集信息泄漏。
3. 从轨迹片段中提取历史航向、速度趋势、当前位置、地图密度、转弯强度、分叉强度、方向熵、主导方向等信息。
4. 在训练/验证标签生成时，可额外加入真实未来方向，只用于离线标签生成。
5. 调用本地 Qwen，要求它输出严格 JSON。
6. 对 Qwen 输出做数值校验。如果它明显违背未来转角或地图统计，会自动纠偏并在 `postprocess` 字段中记录。

## 输出标签含义

每条标签大致包含：

```json
{
  "scene_type": "turning_area",
  "is_branch": false,
  "intent": "sharp_left",
  "turn_strength": "large",
  "main_channel_following": false,
  "direction_bias": {
    "straight": -0.2,
    "left": 0.8,
    "right": -0.5
  },
  "confidence": 0.97,
  "reason": "High turn_rate and future_intent_hint indicate sharp left turn.",
  "postprocess": []
}
```

关键字段解释：

- `scene_type`：当前区域语义，例如主航道、分叉、转弯区、复杂方向区。
- `is_branch`：是否判断为分叉区域。
- `intent`：船舶高层转向意图，例如直行、轻微左转、大幅右转。
- `turn_strength`：转向强度。
- `main_channel_following`：是否更像沿主航道航行。
- `direction_bias`：后续可转成采样概率偏置，增强左转、右转或直行方向。
- `postprocess`：如果 Qwen 输出和数值事实冲突，这里会记录自动修正。

## 运行方式

先激活环境：

```powershell
conda activate dst-mamba
```

只检查数据和提示词，不加载 Qwen：

```powershell
python captain_qwen.py --dry-run --phase train --max-samples 1 --output .\results\_qwen_captain_dry_run.jsonl
```

真实调用 Qwen 生成 1 条样例：

```powershell
python captain_qwen.py --phase train --max-samples 1 --max-new-tokens 256 --output .\results\_qwen_captain_one_label.jsonl
```

批量生成训练标签：

```powershell
python captain_qwen.py --phase train --max-samples 1000 --output .\captain_labels\captain_labels_train_qwen.jsonl --resume
```

如果想随机抽样而不是优先抽分叉/转弯/复杂区域：

```powershell
python captain_qwen.py --phase train --max-samples 1000 --sampling random --output .\captain_labels\captain_labels_train_qwen_random.jsonl
```

## 数据泄漏注意事项

训练集或验证集生成标签时，可以使用真实未来轨迹，因为这些标签只是用于离线训练 Captain Head。

测试集绝对不能使用真实未来轨迹。测试时如果只是想让 Qwen 根据历史和地图判断，需要加：

```powershell
python captain_qwen.py --phase test --max-samples 100 --no-true-future --output .\captain_labels\captain_labels_test_qwen_no_future.jsonl
```

不过更推荐的最终推理方式是：训练阶段用 Qwen 生成标签，推理阶段不用 Qwen，而是让模型内部的 Captain Head 输出语义判断。

## 为什么要加数值校验

第一次真实测试时，Qwen 曾经把一个 `turn_rate=0.8542`、未来方向明显大幅左转的样本判断成直行。这说明大模型虽然能做语义解释，但不能无条件相信。

因此当前脚本加入了两层保护：

1. 在输入中加入 `derived_hints`，直接告诉 Qwen 根据未来转角和地图统计得到的确定性提示。
2. 在解析输出后再次校验。如果 Qwen 的 `intent` 和真实未来转角明显冲突，脚本会自动把标签改成几何事实对应的方向。

这样 Qwen 更像“语义解释器”，而不是完全替代数值判断。

## 后续接入训练的建议

下一步可以实现：

```text
captain_labels.jsonl
        ↓
AISDataset 读取语义标签
        ↓
TrAISformer 增加 Captain Head
        ↓
训练 scene_type / intent / direction_bias
        ↓
推理时用 Captain Head 修正 lat/lon 采样概率
```

推荐消融实验：

```text
Baseline
Baseline + Map Prior
Baseline + Map Prior + Turn Intent
Baseline + Map Prior + Qwen Captain Labels
Baseline + Map Prior + Qwen Captain Labels + Guided Sampling
```

重点看分叉、转弯、高方向熵区域的误差，而不只看整体平均误差。

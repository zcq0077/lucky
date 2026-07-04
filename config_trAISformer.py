# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration flags to run the main script.
"""

import os
import pickle
import torch


class Config():
    # 训练总开关
    # True：重新训练模型，适合第一次跑新结构、改了模型结构、打开 Qwen/地图/转向模块后使用。
    # False：不训练，只加载 savedir 下已有 model.pt 做测试；如果没有对应 checkpoint 会报错。
    retrain = True

    # TensorBoard 日志开关
    # True：需要看训练曲线时开启；False：普通命令行训练即可。
    tb_log = False

    # 训练设备
    # 有 NVIDIA GPU 且 CUDA 可用时用 cuda:0；没有 GPU 或显存不够时改成 cpu，但会非常慢。
    device = torch.device("cuda:0")
#     device = torch.device("cpu")
    
    # 基础训练参数
    # max_epochs：最大训练轮数；实际可能被 early stopping 提前停止。
    # batch_size：显存不够时调小，例如 16 或 8。
    # n_samples：测试时采样多少条候选轨迹，论文式 best-of-N 通常用 16。
    max_epochs = 50
    batch_size = 32
    n_samples = 16

    # 早停模块
    #===================================================
    # True：验证集长期不提升就提前停止，推荐开启，避免过拟合和浪费时间。
    # False：一定训练满 max_epochs，适合做严格固定 epoch 对比。
    early_stopping = True

    # 连续多少个 epoch 验证集没有明显提升就停止；小数据/调参时可设 4-8。
    early_stop_patience = 4

    # 验证损失至少下降多少才算“有提升”。
    early_stop_min_delta = 1e-4

    # 至少训练多少 epoch 后才允许早停，避免刚开始波动就停。
    early_stop_min_epochs = 10

    # 测试集轨迹可视化模块
    #===================================================
    # True：测试结束后从测试集抽取轨迹，画历史/真实/预测图，推荐开启。
    # False：只输出数值误差，不生成直观轨迹图。
    test_visualize = True

    # 可视化抽取多少条测试轨迹。
    test_visualize_n = 10

    # 每条轨迹画多少条预测候选。默认等于 n_samples，也就是 best-of-16 可视化。
    test_visualize_pred_samples = n_samples

    # 抽样随机种子，固定后每次画的测试样本一致，方便对比。
    test_visualize_seed = 42
    
    # 轨迹长度设置
    # init_seqlen：给模型看的历史点数。
    # max_seqlen：训练截断的最大轨迹长度。
    # min_seqlen：过滤太短轨迹的阈值。
    init_seqlen = 18
    max_seqlen = 120
    min_seqlen = 36
    
    # 数据集名称；当前代码主要按 ct_dma 的范围和网格配置写。
    dataset_name = "ct_dma"

    if dataset_name == "ct_dma": #==============================
   
        # 网格大小
        # lat/lon/sog/cog 分别被离散成这些类别，原模型用分类方式预测下一个点。
        # 当 mode == "grad" 或 "pos_grad" 时，sog/cog 字段会被替换成 dlat/dlon。
        lat_size = 250
        lon_size = 270
        sog_size = 30
        cog_size = 72

        # 各字段 embedding 维度，四个相加得到 n_embd。
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 128
        n_cog_embd = 128
    
        # ct_dma 区域经纬度范围，用于归一化坐标还原成真实经纬度。
        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13

    
    #===========================================================================
    # 模型预测模式与采样模块
    # 当前建议保持 mode="pos"，也就是直接预测下一个位置网格。
    # sample_mode="pos_vicinity" 会限制采样在当前位置附近，避免跳到很远的网格，推荐开启。
    mode = "pos"  #"pos", "pos_grad", "mlp_pos", "mlpgrid_pos", "velo", "grid_l2", "grid_l1", 
                            # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
    sample_mode =  "pos_vicinity" # "pos", "pos_vicinity" or "velo"

    # top_k：每个维度只保留概率最高的 k 个候选；None 表示不截断。
    top_k = 10 # int or None 

    # r_vicinity：pos_vicinity 的邻域半径；太小可能限制真实转弯，太大可能采样发散。
    r_vicinity = 40 # int

    # 地图先验模块：Map-conditioned route prior
    #===================================================
    # True：根据训练集构建密度/转弯/分叉/方向复杂度地图，并作为额外 embedding 输入模型。
    # 推荐在当前增强实验中开启；如果要跑原始 baseline，需要设为 False。
    use_map_prior = True

    # 地图先验网格大小，越大越细但统计更稀疏。
    map_prior_lat_size = 120
    map_prior_lon_size = 120

    # 航向方向直方图分桶数，36 表示每 10 度一个方向桶。
    map_direction_bins = 36

    # 访问次数低于该阈值的网格不可靠，会屏蔽转弯/分叉/方向熵特征。
    map_min_count = 10

    # 相邻航向变化超过该角度，统计为转弯事件。
    map_turn_angle_threshold_deg = 25.0

    # 方向分布局部峰值超过该概率，才可能算作一个分支方向。
    map_branch_peak_prob_threshold = 0.12

    # 两个方向峰至少相隔多少度，才认为是不同分支。
    map_branch_min_separation_deg = 35.0

    # 空间平滑次数，降低噪声；过大可能抹掉细小航道。
    map_smooth_iter = 1
    map_direction_smooth_iter = 1

    # 障碍/低密度特征
    # False：推荐默认关闭。AIS 低密度不等于障碍物，容易引入噪声。
    # True：只有当你确认低密度区域确实代表不可航行/障碍附近时再开启。
    map_use_obstacle_features = False
    map_obstacle_density_quantile = 0.10
    map_obstacle_proximity_radius = 5

    # 地图特征通道数；打开 obstacle 后为 6，否则为 4。
    map_prior_channels = 6 if map_use_obstacle_features else 4

    # 地图 embedding 融合权重和 dropout；效果不稳时先调小 map_emb_w。
    map_emb_w = 0.10
    map_emb_pdrop = 0.10

    # True：保存 map_prior_components.png，方便检查地图先验是否合理。
    map_prior_plot = True

    # 转向意图辅助任务模块
    #===================================================
    # True：额外训练一个转向分类头，让模型学习直行/轻左/轻右/急左/急右。
    # 推荐在地图增强和 Qwen 融合实验中开启；如果要跑严格原始 baseline，需要设为 False。
    use_turn_intent_head = True

    # 转向辅助 loss 权重；太大可能干扰主任务，通常 0.01-0.05。
    turn_intent_loss_w = 0.03

    # 小于 straight 阈值视为直行；大于 sharp 阈值视为急转。
    turn_straight_threshold_deg = 10.0
    turn_sharp_threshold_deg = 35.0

    # True：忽略第一个点的转向标签，因为缺少完整前后航向。
    turn_ignore_first = True

    # Qwen 在线语义编码/船长概率选择器模块
    #===================================================
    # 本模块不使用 JSON 标签。它读取 qwen_cache/*.npz 中的 Qwen hidden embedding，
    # 再通过 qwen_projector 融合到 Transformer embedding，并通过 captain_selector 修正 lat/lon logits。
    qwen_model_path = r"D:\Jason1982\wsl\Models\Qwen3-4B-Instruct-2507"

    # True：启用 Qwen 语义向量融合，当前要跑“大模型辅助概率选择器”时开启。
    # False：完全不使用 Qwen cache，适合跑 baseline 或只比较地图/转向模块。
    use_qwen_semantic_encoder = False

    # True：Qwen 本体冻结，只使用已缓存的 hidden embedding，压力小、推荐。
    # 当前轻量版不训练 Qwen 参数。
    qwen_freeze = True

    # True：使用预生成的 qwen_cache，推荐开启。
    # False：真在线每个 batch 调 Qwen，当前轻量训练路径未启用，会报提示。
    qwen_use_cache = True

    # Qwen embedding 缓存目录。开启 use_qwen_semantic_encoder 前必须先生成 train/valid/test 三个 npz。
    qwen_cache_dir = "./qwen_cache/"

    # Qwen3-4B-Instruct hidden size 是 2560；如果换模型，需要和缓存维度一致。
    qwen_hidden_size = 2560

    # Qwen 融合方式：
    # add：只加到 embedding；
    # gated_add：门控后加到 embedding；
    # logit_bias：只修正 lat/lon logits；
    # both：embedding + logits 两种都用，当前推荐。
    qwen_fusion = "both"  # "add", "gated_add", "logit_bias", or "both"

    # Qwen embedding 融合强度；刚开始建议小一点，避免大模型语义向量压过原轨迹特征。
    qwen_emb_w = 0.05

    # Qwen 方向概率偏置强度；越大越像“船长强行改概率”，建议从 0.05 起。
    qwen_bias_w = 0.05

    # Qwen projector dropout，防止过拟合 Qwen 向量。
    qwen_emb_pdrop = 0.10

    # logits 偏置影响的局部半径，默认和 r_vicinity 一致。
    qwen_logit_bias_radius = r_vicinity

    # 生成 Qwen prompt 时使用多少个历史点，默认等于 init_seqlen。
    qwen_prompt_max_points = init_seqlen

    # 生成 Qwen cache 时的批大小；显存不够时用 1，显存够可尝试 2/4。
    qwen_cache_batch_size = 4

    # 旧版 Qwen JSON 打标签模块，仅供 captain_qwen.py 备用
    #===================================================
    # 注意：当前主训练路径不读取 captain_labels/*.jsonl。
    # 如果你走现在的“Qwen 概率选择器”路线，这一块不用管。
    qwen_captain_labels_dir = "./captain_labels/"
    qwen_captain_label_phase = "train"
    qwen_captain_max_samples = 1000
    qwen_captain_future_steps = 24
    qwen_captain_temperature = 0.1
    qwen_captain_top_p = 0.8
    qwen_captain_max_new_tokens = 512
    
    # Blur loss 模块
    #===================================================
    # True：对分类概率做平滑，降低离散网格中心误差的影响，原论文代码常用，推荐开启。
    # False：严格普通交叉熵分类，不做模糊平滑。
    blur = True

    # True：平滑卷积核可学习；False：固定均值平滑。默认 False 更稳定。
    blur_learnable = False

    # blur loss 权重和重复平滑次数。
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0
    
    # 数据路径配置
    #===================================================
    # 数据文件位于 ./data/ct_dma/，通常不用改。
    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"
    
    
    # Transformer 模型结构参数
    #===================================================
    # n_head/n_layer 越大模型越强但更吃显存；当前保持原代码设置。
    n_head = 8
    n_layer = 8

    # 总输出类别数和 embedding 总维度。
    full_size = lat_size + lon_size + sog_size + cog_size
    n_embd = n_lat_embd + n_lon_embd + n_sog_embd + n_cog_embd

    # GPT/Transformer dropout。
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    # 优化器与学习率配置
    #===================================================
    # learning_rate：主学习率；如果打开 Qwen 后 loss 不稳定，可以尝试 3e-4。
    learning_rate = 6e-4 # 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights

    # 学习率衰减：warmup 后 cosine decay，推荐开启。
    lr_decay = True
    warmup_tokens = 512*20 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)

    # DataLoader worker 数；Windows 上如果多进程加载有问题，可以改成 0。
    num_workers = 4 # for DataLoader
    
    # 实验命名标签：打开地图/转向/Qwen 后，会自动在 results 文件夹名中加入 -map/-turn/-qwen。
    map_tag = "-map" if use_map_prior else ""
    turn_tag = "-turn" if use_turn_intent_head else ""
    qwen_tag = "-qwen" if use_qwen_semantic_encoder else ""
    filename = f"{dataset_name}"\
        + f"-{mode}-{sample_mode}-{top_k}-{r_vicinity}"\
        + f"{map_tag}{turn_tag}{qwen_tag}"\
        + f"-blur-{blur}-{blur_learnable}-{blur_n}-{blur_loss_w}"\
        + f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}"\
        + f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}"\
        + f"-head-{n_head}-{n_layer}"\
        + f"-bs-{batch_size}"\
        + f"-lr-{learning_rate}"\
        + f"-seqlen-{init_seqlen}-{max_seqlen}"
    savedir = "./results/"+filename+"/"
    
    ckpt_path = os.path.join(savedir,"model.pt")   

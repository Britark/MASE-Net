# config.py
#LOLv2_real需要把patch_size改为2，win_size改为4
def default_config():
    """返回默认配置字典"""
    model_config = {
        'loss_weights': {
            'reconstruction_weight': 1.5230,
            'auxiliary_weight': 0.1067,
            'psnr_weight': 0.0981,
            'ssim_weight': 0.15,
            'lab_color_weight': 0.012,
            'perceptual_weight': 0.2318
        },
        'image_size': {
            'batch_size': 32,
            'height': 400,  # 默认图像高度
            'width': 600,  # 默认图像宽度
            'patch_size': 4,
            'win_size': 2
        },
        'feature_extractor': {
            'patch_size': 4,
            'in_channels': 3,
            'embed_dim': 128,
            'win_size': 2
        },
        'emb_gen': {
            'input_dim': 128,  # 与特征提取器的embed_dim一致
            'hidden_dim': 256,  # 通常设置为input_dim的2倍
            'output_dim': 128,  # 设置为与input_dim相同
            'dropout_rate': 0.05
        },
        'qk_gen': {
            'dim': 128,  # 与emb_gen的output_dim保持一致
            'num_heads': 3,  # 按要求设置为3
            'dropout_rate': 0.05  # 按要求设置为0.1
        },
        'kv_gen': {
            'embed_dim': 128,
            'dropout_rate': 0.05
        },
        'multi_attention_1': {  # 键名已改为multi_attention_1
            'embed_dim': 128,  # 输入维度
            'num_heads': 2,  # 注意力头数量
            'dropout': 0.05,  # dropout概率
            'bias': True,  # 是否使用偏置
            'q_noise': 0.005,  # 量化噪声强度
            'qn_block_size': 8,  # 量化噪声分块大小
            'num_expert': 3,  # 专家数量
            'use_attention_gate': True,  # 是否使用注意力门控
            'cvloss': 0.05,  # 专家负载均衡损失系数
            'aux_loss': 0.01,  # 辅助损失
            'zloss': 0.001,  # 门控输出正则化损失系数
            'sample_topk': 1,  # 每个样本选择topk专家进行前向计算
            'noisy_gating': False,  # 是否在门控函数中加入噪声
            'use_pos_bias': False  # 是否使用位置编码
        },
        'moe_1': {
            'input_size': 128,  # 输入特征维度，与attn_output_1的最后一个维度一致
            'head_size': 128,  # 输出维度，与input_size保持一致
            'hidden_sizes': [256, 512],  # 多层FFN: 256→512→1024→256
            'num_experts': 6,  # 专家数量
            'k': 3,  # 每个token选择的专家数量
            'need_merge': False,  # 是否需要合并专家输出
            'cvloss': 0.05,  # 专家负载均衡损失系数
            'aux_loss': 0.01,  # 辅助损失
            'zloss': 0.001,  # 门控输出正则化损失系数
            'bias': True,  # 是否使用偏置
            'noisy_gating': False  # 是否在门控函数中加入噪声
        },
        'multi_attention_2': {  # 新增的multi_attention_2配置
            'embed_dim': 128,  # 输入维度
            'num_heads': 2,  # 注意力头数量
            'dropout': 0.05,  # dropout概率
            'bias': True,  # 是否使用偏置
            'q_noise': 0.005,  # 量化噪声强度
            'qn_block_size': 8,  # 量化噪声分块大小
            'num_expert': 3,  # 专家数量
            'use_attention_gate': False,  # 不使用注意力门控
            'cvloss': 0.05,  # 专家负载均衡损失系数
            'aux_loss': 0.01,  # 辅助损失
            'zloss': 0.001,  # 门控输出正则化损失系数
            'sample_topk': 1,  # 每个样本选择topk专家进行前向计算
            'noisy_gating': False,  # 是否在门控函数中加入噪声
            'use_pos_bias': False  # 是否使用位置编码
        },
        'moe_2': {
            'input_size': 128,  # 输入特征维度，与attn_output_1的最后一个维度一致
            'head_size': 128,  # 改为与input_size一致，支持多层FFN
            'hidden_sizes': [256, 512],  # 多层FFN: 256→512→768→256
            'num_experts': 3,  # 专家数量
            'k': 2,  # 每个token选择的专家数量
            'need_merge': True,  # 是否需要合并专家输出
            'cvloss': 0.05,  # 专家负载均衡损失系数
            'aux_loss': 0.01,  # 辅助损失
            'zloss': 0.001,  # 门控输出正则化损失系数
            'bias': True,  # 是否使用偏置
            'noisy_gating': False  # 是否在门控函数中加入噪声
        },
        'multi_attention_3': {  # 新增的multi_attention_2配置
            'embed_dim': 128,  # 输入维度
            'num_heads': 2,  # 注意力头数量
            'dropout': 0.05,  # dropout概率
            'bias': True,  # 是否使用偏置
            'q_noise': 0.05,  # 量化噪声强度
            'qn_block_size': 8,  # 量化噪声分块大小
            'num_expert': 3,  # 专家数量
            'use_attention_gate': False,  # 不使用注意力门控
            'cvloss': 0.05,  # 专家负载均衡损失系数
            'aux_loss': 0.01,  # 辅助损失
            'zloss': 0.001,  # 门控输出正则化损失系数
            'sample_topk': 1,  # 每个样本选择topk专家进行前向计算
            'noisy_gating': False,  # 是否在门控函数中加入噪声
            'use_pos_bias': True  # 是否使用位置编码
        },
        'moe_3': {
            'input_size': 128,  # 输入特征维度，与attn_output_1的最后一个维度一致
            'head_size': 128,  # 改为与input_size一致，支持多层FFN
            'hidden_sizes': [256, 512, 256],  # 多层FFN: 256→512→1024→512→256
            'num_experts': 3,  # 专家数量
            'k': 2,  # 每个token选择的专家数量
            'need_merge': True,  # 是否需要合并专家输出
            'cvloss': 0.05,  # 专家负载均衡损失系数
            'aux_loss': 0.01,  # 辅助损失
            'zloss': 0.001,  # 门控输出正则化损失系数
            'bias': True,  # 是否使用偏置
            'noisy_gating': False  # 是否在门控函数中加入噪声
        },
        'multi_attention_4': {  # 新增的multi_attention_2配置
            'embed_dim': 128,  # 输入维度
            'num_heads': 2,  # 注意力头数量
            'dropout': 0.05,  # dropout概率
            'bias': True,  # 是否使用偏置
            'q_noise': 0.005,  # 量化噪声强度
            'qn_block_size': 8,  # 量化噪声分块大小
            'num_expert': 3,  # 专家数量
            'use_attention_gate': False,  # 不使用注意力门控
            'cvloss': 0.05,  # 专家负载均衡损失系数
            'aux_loss': 0.01,  # 辅助损失
            'zloss': 0.001,  # 门控输出正则化损失系数
            'sample_topk': 1,  # 每个样本选择topk专家进行前向计算
            'noisy_gating': False,  # 是否在门控函数中加入噪声
            'use_pos_bias': False  # 是否使用位置编码
        },
        'isp_1': {
            'latent_dim': 128,  # 输入维度，与模型输出维度一致
            'param_dim': 1,  # ISP参数维度
            'hidden_dims': [96, 64, 32],  # 隐藏层维度，逐层扩展
            'dropout': 0.05
        },
        'isp_2': {
            'latent_dim': 128,
            'param_dim': 576,
            'hidden_dims': [256, 384, 512, 576],
            'dropout': 0.05
        },
        'isp_4': {
            'latent_dim': 128,
            'param_dim': 1,
            'hidden_dims': [96, 64, 32],
            'dropout': 0.05
        }
}

    return model_config
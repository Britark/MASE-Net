# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp  # 添加混合精度支持

import feature_extractor
from feature_extractor import FeatureExtractor
import config
from config import default_config
import emb_gen
from MoA import MultiheadAttention, MoE
from utils import *
from decoder import *
from ISP import *
from losses import *


class Model(nn.Module):
    """主模型类，组装各个预定义的模块"""

    def __init__(self, config=None):
        """
        使用配置字典初始化模型
        Args:
            config: 包含所有模型参数的配置字典，如果为None则使用默认配置
        """
        super(Model, self).__init__()
        self.expert_indices = None
        # 如果没有提供配置，使用默认配置
        if config is None:
            config = default_config()

        # 从config读取图像尺寸参数，设置H和W
        self.patch_size = config['image_size']['patch_size']
        self.win_size = config['image_size']['win_size']

        # 添加损失权重超参数
        loss_weights_config = config.get('loss_weights', {})
        self.reconstruction_weight = loss_weights_config.get('reconstruction_weight', 1.0)  # 重建损失权重
        self.auxiliary_weight = loss_weights_config.get('auxiliary_weight', 0.1)  # 辅助损失权重

        # 获取特征提取器配置
        fe_config = config.get('feature_extractor', {})

        # 实例化特征提取器
        self.feature_extractor = FeatureExtractor(
            patch_size=fe_config.get('patch_size', 2),#patch_size=4 win_size=5&& patch_size=8 win_size=7
            in_channels=fe_config.get('in_channels', 3),
            embed_dim=fe_config.get('embed_dim', 128),
            win_size=fe_config.get('win_size', 5)
        )

        # 获取EispGeneratorFFN配置
        emb_gen_config = config.get('emb_gen', {})

        # 实例化EispGeneratorFFN
        self.emb_gen = emb_gen.EispGeneratorFFN(
            input_dim=emb_gen_config.get('input_dim', 128),
            hidden_dim=emb_gen_config.get('hidden_dim', 512),
            output_dim=emb_gen_config.get('output_dim', 128),
            dropout_rate=emb_gen_config.get('dropout_rate', 0.1)
        )

        # 添加线性层，将e_isp的最后一个维度映射为其4倍大
        self.e_isp_expand = nn.Linear(
            emb_gen_config.get('output_dim', 128),
            emb_gen_config.get('output_dim', 128) * 4
        )

        # 获取QKIspGenerator配置
        qk_gen_config = config.get('qk_gen', {})

        # 实例化QKIspGenerator
        self.qk_gen = emb_gen.QKIspGenerator(
            dim=qk_gen_config.get('dim', 128),
            num_heads=qk_gen_config.get('num_heads', 4),
            dropout_rate=qk_gen_config.get('dropout_rate', 0.1)
        )

        # 获取KVFeatureGenerator配置
        kv_gen_config = config.get('kv_gen', {})

        # 实例化KVFeatureGenerator
        self.kv_gen = emb_gen.KVFeatureGenerator(
            embed_dim=kv_gen_config.get('embed_dim', 128),
            dropout_rate=kv_gen_config.get('dropout_rate', 0.1)
        )

        # 获取MultiheadAttention配置
        multi_attn_1_config = config.get('multi_attention_1', {})

        # 实例化MultiheadAttention
        self.multi_attention_1 = MultiheadAttention(
            embed_dim=multi_attn_1_config.get('embed_dim', 128),
            num_heads=multi_attn_1_config.get('num_heads', 2),
            dropout=multi_attn_1_config.get('dropout', 0.1),
            bias=multi_attn_1_config.get('bias', True),
            q_noise=multi_attn_1_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_1_config.get('qn_block_size', 8),
            num_expert=multi_attn_1_config.get('num_expert', 4),
            head_dim=multi_attn_1_config.get('head_dim', 64),
            use_attention_gate=multi_attn_1_config.get('use_attention_gate', True),
            cvloss=multi_attn_1_config.get('cvloss', 0.05),
            aux_loss=multi_attn_1_config.get('aux_loss', 1),
            zloss=multi_attn_1_config.get('zloss', 0.001),
            sample_topk=multi_attn_1_config.get('sample_topk', 0),
            noisy_gating=multi_attn_1_config.get('noisy_gating', True),
            use_pos_bias=multi_attn_1_config.get('use_pos_bias', False)
        )

        # LayerNorm for attention residual
        self.attn_norm = nn.LayerNorm(multi_attn_1_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_1_config = config.get('moe_1', {})

        # 创建MoE实例
        self.moe_1 = MoE(
            input_size=moe_1_config.get('input_size', 128),
            head_size=moe_1_config.get('head_size', 512),
            num_experts=moe_1_config.get('num_experts', 8),#16
            k=moe_1_config.get('k', 4),#8
            need_merge=moe_1_config.get('need_merge', False),
            cvloss=moe_1_config.get('cvloss', 0.05),#0.01
            aux_loss=moe_1_config.get('aux_loss', 1),#10
            zloss=moe_1_config.get('zloss', 0.001),#0.005
            bias=moe_1_config.get('bias', False),
            activation=nn.GELU(),
            noisy_gating=moe_1_config.get('noisy_gating', True)
        )

        # 专门为moe_output_1定义的归一化层
        self.moe_1_norm = nn.LayerNorm(moe_1_config.get('input_size', 128))

        # 获取MultiheadAttention2配置
        multi_attn_2_config = config.get('multi_attention_2', {})

        # 实例化MultiheadAttention2
        self.multi_attention_2 = MultiheadAttention(
            embed_dim=multi_attn_2_config.get('embed_dim', 128),
            num_heads=multi_attn_2_config.get('num_heads', 2),
            dropout=multi_attn_2_config.get('dropout', 0.1),
            bias=multi_attn_2_config.get('bias', True),
            q_noise=multi_attn_2_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_2_config.get('qn_block_size', 8),
            num_expert=multi_attn_2_config.get('num_expert', 4),
            head_dim=multi_attn_2_config.get('head_dim', 64),
            use_attention_gate=multi_attn_2_config.get('use_attention_gate', False),
            cvloss=multi_attn_2_config.get('cvloss', 0.05),
            aux_loss=multi_attn_2_config.get('aux_loss', 1),
            zloss=multi_attn_2_config.get('zloss', 0.001),
            sample_topk=multi_attn_2_config.get('sample_topk', 0),
            noisy_gating=multi_attn_2_config.get('noisy_gating', True),
            use_pos_bias=multi_attn_2_config.get('use_pos_bias', False)
        )

        # LayerNorm for attention2 residual
        self.attn2_norm = nn.LayerNorm(multi_attn_2_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_2_config = config.get('moe_2', {})

        # 创建MoE实例
        self.moe_2 = MoE(
            input_size=moe_2_config.get('input_size', 128),
            head_size=moe_2_config.get('head_size', 512),
            num_experts=moe_2_config.get('num_experts', 8),
            k=moe_2_config.get('k', 2),
            need_merge=moe_2_config.get('need_merge', True),
            cvloss=moe_2_config.get('cvloss', 0.05),
            aux_loss=moe_2_config.get('aux_loss', 1),
            zloss=moe_2_config.get('zloss', 0.001),
            bias=moe_2_config.get('bias', False),
            activation=nn.GELU(),
            noisy_gating=moe_2_config.get('noisy_gating', True)
        )

        # 专门为moe_output_1定义的归一化层
        self.moe_2_norm = nn.LayerNorm(moe_2_config.get('input_size', 128))

        # 实例化PatchNeighborSearcher
        self.patch_neighbor_searcher = PatchNeighborSearcher()

        multi_attn_3_config = config.get('multi_attention_3', {})

        # 实例化MultiheadAttention3
        self.multi_attention_3 = MultiheadAttention(
            embed_dim=multi_attn_3_config.get('embed_dim', 128),
            num_heads=multi_attn_3_config.get('num_heads', 2),
            dropout=multi_attn_3_config.get('dropout', 0.1),
            bias=multi_attn_3_config.get('bias', True),
            q_noise=multi_attn_3_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_3_config.get('qn_block_size', 8),
            num_expert=multi_attn_3_config.get('num_expert', 8),
            head_dim=multi_attn_3_config.get('head_dim', 64),
            use_attention_gate=multi_attn_3_config.get('use_attention_gate', False),
            cvloss=multi_attn_3_config.get('cvloss', 0.05),
            aux_loss=multi_attn_3_config.get('aux_loss', 1),
            zloss=multi_attn_3_config.get('zloss', 0.001),
            sample_topk=multi_attn_3_config.get('sample_topk', 0),
            noisy_gating=multi_attn_3_config.get('noisy_gating', True),
            use_pos_bias=multi_attn_3_config.get('use_pos_bias', True)
        )

        # LayerNorm for attention2 residual
        self.attn3_norm = nn.LayerNorm(multi_attn_3_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_3_config = config.get('moe_3', {})

        # 创建MoE实例
        self.moe_3 = MoE(
            input_size=moe_3_config.get('input_size', 128),
            head_size=moe_3_config.get('head_size', 512),
            num_experts=moe_3_config.get('num_experts', 8),
            k=moe_3_config.get('k', 2),
            need_merge=moe_3_config.get('need_merge', True),
            cvloss=moe_3_config.get('cvloss', 0.05),
            aux_loss=moe_3_config.get('aux_loss', 1),
            zloss=moe_3_config.get('zloss', 0.001),
            bias=moe_3_config.get('bias', False),
            activation=nn.GELU(),
            noisy_gating=moe_3_config.get('noisy_gating', True)
        )

        # 归一化层
        self.moe_3_norm = nn.LayerNorm(moe_3_config.get('input_size', 128))

        multi_attn_4_config = config.get('multi_attention_4', {})

        # 实例化MultiheadAttention4（自注意力）
        self.multi_attention_4 = MultiheadAttention(
            embed_dim=multi_attn_4_config.get('embed_dim', 128),
            num_heads=multi_attn_4_config.get('num_heads', 2),  # 可以适当增加头数
            dropout=multi_attn_4_config.get('dropout', 0.1),
            bias=multi_attn_4_config.get('bias', True),
            q_noise=multi_attn_4_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_4_config.get('qn_block_size', 8),
            num_expert=multi_attn_4_config.get('num_expert', 8),
            head_dim=multi_attn_4_config.get('head_dim', 64),
            use_attention_gate=multi_attn_4_config.get('use_attention_gate', False),
            cvloss=multi_attn_4_config.get('cvloss', 0.05),
            aux_loss=multi_attn_4_config.get('aux_loss', 1),
            zloss=multi_attn_4_config.get('zloss', 0.001),
            sample_topk=multi_attn_4_config.get('sample_topk', 0),
            noisy_gating=multi_attn_4_config.get('noisy_gating', True),
            use_pos_bias=multi_attn_4_config.get('use_pos_bias', False)  # 自注意力中可以不使用位置偏置
        )

        # LayerNorm for attention2 residual
        self.attn4_norm = nn.LayerNorm(multi_attn_4_config.get('embed_dim', 128))

        # 获取ISP解码器配置
        isp_1_config = config.get('isp_1', {})
        isp_2_config = config.get('isp_2', {})
        isp_3_config = config.get('isp_3', {})
        isp_4_config = config.get('isp_4', {})

        # 实例化四个ISP解码器
        self.isp_1 = ISPDecoder(
            latent_dim=isp_1_config.get('latent_dim', 128),
            param_dim=isp_1_config.get('param_dim', 1),
            hidden_dims=isp_1_config.get('hidden_dims', [256, 128, 64, 32])
        )

        self.isp_2 = ISPDecoder(
            latent_dim=isp_2_config.get('latent_dim', 128),
            param_dim=isp_2_config.get('param_dim', 3),
            hidden_dims=isp_2_config.get('hidden_dims', [256, 128, 64, 32])
        )

        self.isp_3 = ISPDecoder(
            latent_dim=isp_3_config.get('latent_dim', 128),
            param_dim=isp_3_config.get('param_dim', 9),
            hidden_dims=isp_3_config.get('hidden_dims', [256, 128, 64, 32])
        )

        self.isp_4 = ISPDecoder(
            latent_dim=isp_4_config.get('latent_dim', 128),
            param_dim=isp_4_config.get('param_dim', 7),
            hidden_dims=isp_4_config.get('hidden_dims', [256, 128, 64, 32])
        )

        # 添加一个新的解码器用于最终去噪
        self.isp_final = ISPDecoder(
            latent_dim=128,  # 与其他解码器相同
            param_dim=7,  # 与第一次去噪相同的参数维度
            hidden_dims=[256, 128, 64, 32]
        )

        # 在初始化过程中配置CUDA优化
        if torch.cuda.is_available():
            # 启用cuDNN自动调优
            torch.backends.cudnn.benchmark = True
            # 让算法行为确定
            torch.backends.cudnn.deterministic = False

        # 初始化去噪锐化器以避免重复创建
        self.denoising_sharpening = DenoisingSharpening()

    def forward(self, x, target=None):
        """
        定义模型的前向传播过程

        Args:
            x: 输入的低光照图像
            target: 目标正常光照图像。训练模式下必须提供，否则无法计算损失
            compute_loss: 是否计算损失。如果为None，则根据self.training判断

        Returns:
            如果compute_loss为True: (enhanced_image, loss) - 增强后的图像和计算的损失
            如果compute_loss为False: enhanced_image - 仅返回增强后的图像
        """
        # 使用混合精度计算
        with amp.autocast(enabled=torch.cuda.is_available()):
            # 如果训练模式下没有提供target，则抛出错误
            if self.training and target is None:
                raise ValueError("训练模式下必须提供target图像")

            batch_size, channels, height, width = x.shape

            # 确保输入张量连续
            x = x.contiguous()

            # 特征提取
            features, h_windows, w_windows = self.feature_extractor(x)

            window_size = self.patch_size * self.win_size  # 计算窗口大小

            # 重塑输入图像为[batch_size, channels, h_windows, window_size, w_windows, window_size]格式
            # 使用reshape而不是view以保证内存连续性
            x_reshaped = x.reshape(batch_size, channels, h_windows, window_size, w_windows, window_size)

            # 调整维度顺序为[batch_size, h_windows, w_windows, window_size, window_size, channels]
            x_reordered = x_reshaped.permute(0, 2, 4, 3, 5, 1).contiguous()

            # 重塑为[batch_size, h_windows * w_windows, window_size, window_size, channels]
            x_win = x_reordered.reshape(batch_size, h_windows * w_windows, window_size, window_size, channels)

            # EispGeneratorFFN处理
            e_isp = self.emb_gen(features)

            # 确保e_isp连续性，优化线性层计算
            e_isp_cont = e_isp.contiguous()
            e_isp_big = self.e_isp_expand(e_isp_cont)

            # QKIspGenerator处理
            q_isp, k_isp = self.qk_gen(e_isp)

            # KVFeatureGenerator处理
            k_features, v_features = self.kv_gen(features)

            # MultiheadAttention处理attn_output_1[batch_size, patches, seq_len, input_size]
            attn_output_1, _, atten_aux_loss_1 = self.multi_attention_1(
                query=q_isp,
                key=k_features,
                value=v_features,
                k_isp=k_isp,
                need_weights=False,
                before_softmax=False,
                need_head_weights=False
            )

            # 归一化 & 残差连接
            attn_output_1 = self.attn_norm(attn_output_1) + q_isp

            # MoE处理
            moe_output_1, moe_aux_loss_1 = self.moe_1(
                x=attn_output_1,
                sample_topk=0,
                multiply_by_gates=True
            )

            # 归一化处理moe_output_1
            moe_output_1 = self.moe_1_norm(moe_output_1)  # 归一化操作
            moe_output_1 = self.moe_1.concat(moe_output_1, e_isp_big)
            moe_output_1 = moe_output_1.squeeze(2)

            # MultiheadAttention2处理
            attn_output_2, _, atten_aux_loss_2 = self.multi_attention_2(
                query=moe_output_1,
                key=moe_output_1,
                value=moe_output_1,
                k_isp=None,
                need_weights=False,
                before_softmax=False,
                need_head_weights=False
            )

            # 归一化处理attn_output_2
            attn_output_2 = self.attn2_norm(attn_output_2) + moe_output_1

            # MOE处理
            moe_output_2, moe_aux_loss_2 = self.moe_2(
                x=attn_output_2,
                sample_topk=0,
                multiply_by_gates=True
            )

            # 归一化处理moe_output_2+残差连接
            moe_output_2 = self.moe_2_norm(moe_output_2) + attn_output_2

            # PatchNeighborSearcher 模块
            neighbor_features = self.patch_neighbor_searcher(moe_output_2, h_windows, w_windows)

            # MultiheadAttention3处理
            attn_output_3, _, atten_aux_loss_3 = self.multi_attention_3(
                query=moe_output_2,
                key=neighbor_features,
                value=neighbor_features,
                k_isp=None,
                need_weights=False,
                before_softmax=False,
                need_head_weights=False
            )

            # 归一化处理attn_output_3
            attn_output_3 = self.attn3_norm(attn_output_3) + moe_output_2

            # MoE处理
            moe_output_3, moe_aux_loss_3 = self.moe_3(
                x=attn_output_3,
                sample_topk=0,
                multiply_by_gates=True
            )

            # 归一化处理moe_output_2
            moe_output_3 = self.moe_3_norm(moe_output_3) + attn_output_3

            # 自注意力处理
            attn_output_4, _, atten_aux_loss_4 = self.multi_attention_4(
                query=moe_output_3,
                key=moe_output_3,  # 自注意力：key和query相同
                value=moe_output_3,  # 自注意力：value和query相同
                k_isp=None,
                need_weights=False,
                before_softmax=False,
                need_head_weights=False
            )

            # 残差连接和归一化
            attn_output_4 = self.attn4_norm(attn_output_4) + moe_output_3

            # 1. 合并前两个维度
            bsz, patches, k, head_dim = attn_output_4.shape
            attn_output_4 = attn_output_4.reshape(-1, k, head_dim)  # [bsz*patches, tgt_len, head_dim]

            # 2. 将原始的第三个维度（tgt_len）放到第一个位置
            attn_output_4 = attn_output_4.transpose(0, 1)  # [tgt_len, bsz*patches, head_dim]

            # 直接从转置后的张量中获取四个专家嵌入
            isp_expert_1 = attn_output_4[0]
            isp_expert_2 = attn_output_4[1]
            isp_expert_3 = attn_output_4[2]
            isp_expert_4 = attn_output_4[3]

            # 分别输入到四个解码器中
            isp_final_1 = self.isp_1(isp_expert_1)  # [num_windows, param_dim_1]
            isp_final_2 = self.isp_2(isp_expert_2)
            isp_final_3 = self.isp_3(isp_expert_3)
            isp_final_4 = self.isp_4(isp_expert_4)

            # 获取配置中的param_dim参数
            param_dim_1 = self.isp_1.param_dim  # 通过解码器输出层获取param_dim
            param_dim_2 = self.isp_2.param_dim
            param_dim_3 = self.isp_3.param_dim
            param_dim_4 = self.isp_4.param_dim

            isp_final_1 = isp_final_1.reshape(-1, h_windows * w_windows, param_dim_1)
            isp_final_2 = isp_final_2.reshape(-1, h_windows * w_windows, param_dim_2)
            isp_final_3 = isp_final_3.reshape(-1, h_windows * w_windows, 3, 3)
            isp_final_4 = isp_final_4.reshape(-1, h_windows * w_windows, param_dim_4)

            enhanced_image = x_win

            # 按顺序应用四个增强方法
            # 4. 去噪与锐化
            enhanced_image_1 = self.denoising_sharpening(images=enhanced_image, params=isp_final_4)

            # 1. LIME增强
            enhanced_image_2 = patch_gamma_correct(L=enhanced_image_1, gamma_increment=isp_final_1)

            # 2. 白平衡
            enhanced_image_3 = white_balance(predicted_gains=isp_final_2, I_source=enhanced_image_2)

            # 3. 色彩校正矩阵
            enhanced_image_4 = apply_color_correction_matrices(I_source=enhanced_image_3, pred_matrices=isp_final_3)

            # 使用新的解码器生成最终去噪参数
            isp_final_5 = self.isp_final(isp_expert_4)
            isp_final_5 = isp_final_5.reshape(-1, h_windows * w_windows, param_dim_4)

            # 应用最终去噪和锐化
            enhanced_image_final = self.denoising_sharpening(images=enhanced_image_4, params=isp_final_5)

            # 将增强后的窗口图像重构回原始图像形状
            enhanced_image_final = enhanced_image_final.permute(0, 1, 4, 2, 3).contiguous()  # [B, P, C, H, W]
            enhanced_image_final = enhanced_image_final.reshape(
                batch_size, h_windows, w_windows, channels, window_size, window_size
            )
            enhanced_image_final = enhanced_image_final.permute(0, 3, 1, 4, 2,
                                                                5).contiguous()  # [B, C, h_win, win_size, w_win, win_size]
            enhanced_image_final = enhanced_image_final.reshape(batch_size, channels,
                                                                h_windows * window_size, w_windows * window_size)

            # 训练模式下才计算损失并返回
            if self.training:
                # 计算总的辅助损失
                auxiliary_loss = (atten_aux_loss_1 + moe_aux_loss_1 + atten_aux_loss_2 + \
                                  moe_aux_loss_2 + atten_aux_loss_3 + moe_aux_loss_3 + atten_aux_loss_4) / 7

                # 使用目标图像计算重建损失
                l1_loss = L1ReconstructionLoss()

                reconstruction_loss = l1_loss(enhanced_image_final, target)

                # 使用权重系数组合不同的损失
                loss = (self.reconstruction_weight * reconstruction_loss +
                        self.auxiliary_weight * auxiliary_loss)

                return enhanced_image_final, loss, reconstruction_loss
            else:
                # 测试模式，只返回增强后的图像
                return enhanced_image_final
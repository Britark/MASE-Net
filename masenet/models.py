# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp  # 添加混合精度支持
import kornia.losses  # 添加kornia损失函数

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
from torchvision.models import vgg16


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
        self.perceptual_weight = loss_weights_config.get('perceptual_weight', 0.1)
        self.psnr_weight = loss_weights_config.get('psnr_weight', 0.28)  # PSNR损失权重 (降低)
        self.ssim_weight = loss_weights_config.get('ssim_weight', 0.3)  # SSIM损失权重 (降低)
        self.lab_color_weight = loss_weights_config.get('lab_color_weight', 0.02)  # LAB色彩损失权重

        # 获取特征提取器配置
        fe_config = config.get('feature_extractor', {})

        # 实例化特征提取器
        self.feature_extractor = FeatureExtractor(
            patch_size=fe_config.get('patch_size', 2),  # patch_size=4 win_size=5&& patch_size=8 win_size=7
            in_channels=fe_config.get('in_channels', 3),
            embed_dim=fe_config.get('embed_dim', 128),
            win_size=fe_config.get('win_size', 5)
        )

        # 获取EispGeneratorFFN配置
        emb_gen_config = config.get('emb_gen', {})

        # 获取QKIspGenerator配置（提前获取以使用num_heads）
        qk_gen_config = config.get('qk_gen', {})

        # 实例化EispGeneratorFFN
        self.emb_gen = emb_gen.EispGeneratorFFN(
            input_dim=emb_gen_config.get('input_dim', 128),
            hidden_dim=emb_gen_config.get('hidden_dim', 512),
            output_dim=emb_gen_config.get('output_dim', 128),
            dropout_rate=emb_gen_config.get('dropout_rate', 0.1)
        )

        # 添加线性层，将e_isp的最后一个维度映射为其num_heads倍大
        self.e_isp_expand = nn.Linear(
            emb_gen_config.get('output_dim', 128),
            emb_gen_config.get('output_dim', 128) * qk_gen_config.get('num_heads', 3)
        )

        # 实例化QKIspGenerator
        self.qk_gen = emb_gen.QKIspGenerator(
            dim=qk_gen_config.get('dim', 128),
            num_heads=qk_gen_config.get('num_heads', 3),
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
        embed_dim_1 = multi_attn_1_config.get('embed_dim', 128)
        num_heads_1 = multi_attn_1_config.get('num_heads', 2)
        self.multi_attention_1 = MultiheadAttention(
            embed_dim=embed_dim_1,
            num_heads=num_heads_1,
            dropout=multi_attn_1_config.get('dropout', 0.1),
            bias=multi_attn_1_config.get('bias', True),
            q_noise=multi_attn_1_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_1_config.get('qn_block_size', 8),
            num_expert=multi_attn_1_config.get('num_expert', 4),
            head_dim=embed_dim_1 // num_heads_1,  # 自动计算head_dim
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
            head_size=moe_1_config.get('head_size', 256),
            hidden_sizes=moe_1_config.get('hidden_sizes', None),  # 支持多层FFN
            num_experts=moe_1_config.get('num_experts', 3),
            k=moe_1_config.get('k', 1),
            need_merge=moe_1_config.get('need_merge', False),
            cvloss=moe_1_config.get('cvloss', 0.05),
            aux_loss=moe_1_config.get('aux_loss', 0.01),
            zloss=moe_1_config.get('zloss', 0.001),
            bias=moe_1_config.get('bias', True),
            activation=nn.GELU(),
            noisy_gating=moe_1_config.get('noisy_gating', False)
        )

        # 专门为moe_output_1定义的归一化层
        self.moe_1_norm = nn.LayerNorm(moe_1_config.get('input_size', 128))

        # 获取MultiheadAttention2配置
        multi_attn_2_config = config.get('multi_attention_2', {})

        # 实例化MultiheadAttention2
        embed_dim_2 = multi_attn_2_config.get('embed_dim', 128)
        num_heads_2 = multi_attn_2_config.get('num_heads', 1)
        self.multi_attention_2 = MultiheadAttention(
            embed_dim=embed_dim_2,
            num_heads=num_heads_2,
            dropout=multi_attn_2_config.get('dropout', 0.1),
            bias=multi_attn_2_config.get('bias', True),
            q_noise=multi_attn_2_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_2_config.get('qn_block_size', 8),
            num_expert=multi_attn_2_config.get('num_expert', 3),
            head_dim=embed_dim_2 // num_heads_2,  # 自动计算head_dim
            use_attention_gate=multi_attn_2_config.get('use_attention_gate', False),
            cvloss=multi_attn_2_config.get('cvloss', 0.05),
            aux_loss=multi_attn_2_config.get('aux_loss', 0.01),
            zloss=multi_attn_2_config.get('zloss', 0.001),
            sample_topk=multi_attn_2_config.get('sample_topk', 1),
            noisy_gating=multi_attn_2_config.get('noisy_gating', False),
            use_pos_bias=multi_attn_2_config.get('use_pos_bias', False)
        )

        # LayerNorm for attention2 residual
        self.attn2_norm = nn.LayerNorm(multi_attn_2_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_2_config = config.get('moe_2', {})

        # 创建MoE实例
        self.moe_2 = MoE(
            input_size=moe_2_config.get('input_size', 128),
            head_size=moe_2_config.get('head_size', 256),
            hidden_sizes=moe_2_config.get('hidden_sizes', None),  # 支持多层FFN
            num_experts=moe_2_config.get('num_experts', 3),
            k=moe_2_config.get('k', 1),
            need_merge=moe_2_config.get('need_merge', True),
            cvloss=moe_2_config.get('cvloss', 0.05),
            aux_loss=moe_2_config.get('aux_loss', 0.01),
            zloss=moe_2_config.get('zloss', 0.001),
            bias=moe_2_config.get('bias', True),
            activation=nn.GELU(),
            noisy_gating=moe_2_config.get('noisy_gating', False)
        )

        # 专门为moe_output_1定义的归一化层
        self.moe_2_norm = nn.LayerNorm(moe_2_config.get('input_size', 128))

        # 实例化PatchNeighborSearcher
        self.patch_neighbor_searcher = PatchNeighborSearcher()

        multi_attn_3_config = config.get('multi_attention_3', {})

        # 实例化MultiheadAttention3
        embed_dim_3 = multi_attn_3_config.get('embed_dim', 128)
        num_heads_3 = multi_attn_3_config.get('num_heads', 1)
        self.multi_attention_3 = MultiheadAttention(
            embed_dim=embed_dim_3,
            num_heads=num_heads_3,
            dropout=multi_attn_3_config.get('dropout', 0.1),
            bias=multi_attn_3_config.get('bias', True),
            q_noise=multi_attn_3_config.get('q_noise', 0.05),
            qn_block_size=multi_attn_3_config.get('qn_block_size', 8),
            num_expert=multi_attn_3_config.get('num_expert', 3),
            head_dim=embed_dim_3 // num_heads_3,  # 自动计算head_dim
            use_attention_gate=multi_attn_3_config.get('use_attention_gate', False),
            cvloss=multi_attn_3_config.get('cvloss', 0.05),
            aux_loss=multi_attn_3_config.get('aux_loss', 0.01),
            zloss=multi_attn_3_config.get('zloss', 0.001),
            sample_topk=multi_attn_3_config.get('sample_topk', 1),
            noisy_gating=multi_attn_3_config.get('noisy_gating', False),
            use_pos_bias=multi_attn_3_config.get('use_pos_bias', True)
        )

        # LayerNorm for attention2 residual
        self.attn3_norm = nn.LayerNorm(multi_attn_3_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_3_config = config.get('moe_3', {})

        # 创建MoE实例
        self.moe_3 = MoE(
            input_size=moe_3_config.get('input_size', 128),
            head_size=moe_3_config.get('head_size', 256),
            hidden_sizes=moe_3_config.get('hidden_sizes', None),  # 支持多层FFN
            num_experts=moe_3_config.get('num_experts', 3),
            k=moe_3_config.get('k', 1),
            need_merge=moe_3_config.get('need_merge', True),
            cvloss=moe_3_config.get('cvloss', 0.05),
            aux_loss=moe_3_config.get('aux_loss', 0.01),
            zloss=moe_3_config.get('zloss', 0.001),
            bias=moe_3_config.get('bias', True),
            activation=nn.GELU(),
            noisy_gating=moe_3_config.get('noisy_gating', False)
        )

        # 归一化层
        self.moe_3_norm = nn.LayerNorm(moe_3_config.get('input_size', 128))

        multi_attn_4_config = config.get('multi_attention_4', {})

        # 实例化MultiheadAttention4（自注意力）
        embed_dim_4 = multi_attn_4_config.get('embed_dim', 128)
        num_heads_4 = multi_attn_4_config.get('num_heads', 1)
        self.multi_attention_4 = MultiheadAttention(
            embed_dim=embed_dim_4,
            num_heads=num_heads_4,
            dropout=multi_attn_4_config.get('dropout', 0.1),
            bias=multi_attn_4_config.get('bias', True),
            q_noise=multi_attn_4_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_4_config.get('qn_block_size', 8),
            num_expert=multi_attn_4_config.get('num_expert', 3),
            head_dim=embed_dim_4 // num_heads_4,  # 自动计算head_dim
            use_attention_gate=multi_attn_4_config.get('use_attention_gate', False),
            cvloss=multi_attn_4_config.get('cvloss', 0.05),
            aux_loss=multi_attn_4_config.get('aux_loss', 0.01),
            zloss=multi_attn_4_config.get('zloss', 0.001),
            sample_topk=multi_attn_4_config.get('sample_topk', 1),
            noisy_gating=multi_attn_4_config.get('noisy_gating', False),
            use_pos_bias=multi_attn_4_config.get('use_pos_bias', False)
        )

        # LayerNorm for attention2 residual
        self.attn4_norm = nn.LayerNorm(multi_attn_4_config.get('embed_dim', 128))

        # 获取ISP解码器配置
        isp_1_config = config.get('isp_1', {})
        isp_2_config = config.get('isp_2', {})
        isp_4_config = config.get('isp_4', {})

        # 实例化三个ISP解码器（移除isp_3，去噪锐化使用固定参数）
        self.isp_1 = ISPDecoder(
            latent_dim=isp_1_config.get('latent_dim', 128),
            param_dim=isp_1_config.get('param_dim', 1),
            hidden_dims=isp_1_config.get('hidden_dims', [256, 128, 64, 32]),
            dropout=isp_1_config.get('dropout', 0.1),
            use_identity_init=True  # 保持接近0的初始化
        )

        self.isp_2 = ISPDecoder(
            latent_dim=isp_2_config.get('latent_dim', 128),
            param_dim=isp_2_config.get('param_dim', 576),
            hidden_dims=isp_2_config.get('hidden_dims', [256, 384, 512, 576]),
            dropout=isp_2_config.get('dropout', 0.1),
            use_identity_init=True  # 保持接近0的初始化
        )

        self.isp_4 = ISPDecoder(
            latent_dim=isp_4_config.get('latent_dim', 128),
            param_dim=isp_4_config.get('param_dim', 1),
            hidden_dims=isp_4_config.get('hidden_dims', [256, 128, 64, 32]),
            dropout=isp_4_config.get('dropout', 0.1),
            use_identity_init=False,  # 使用随机初始化
            bias_init=1.0  # 饱和度参数初始化为1.0，让网络默认预测更强的增强
        )

        vgg_model = vgg16(pretrained=True).features[:16]
        for param in vgg_model.parameters():
            param.requires_grad = False

        self.perceptual_loss = PerceptualLoss(vgg_model)
        self.perceptual_loss.eval()

        # 在初始化过程中配置CUDA优化
        if torch.cuda.is_available():
            # 启用cuDNN自动调优
            torch.backends.cudnn.benchmark = True
            # 让算法行为确定
            torch.backends.cudnn.deterministic = False

        # 初始化去噪锐化器以避免重复创建
        self.denoising_sharpening = DenoisingSharpening()
        self.color_transform = ColorTransform(
            residual_scale=0.5,    # 增大残差连接的缩放因子
            use_residual=True      # 使用残差连接
        )
        
        # 添加单隐藏层MLP用于处理isp_final_2
        self.isp_final_2_mlp = nn.Sequential(
            nn.Linear(9, 9),
            nn.GELU(),
            nn.Linear(9, 9)
        )
        
        # 添加可学习的残差连接权重
        self.isp_final_2_residual_weight = nn.Parameter(torch.tensor(0.5))
        
        # 使用与ISP解码器相同的初始化方法
        with torch.no_grad():
            # 第一层
            nn.init.xavier_uniform_(self.isp_final_2_mlp[0].weight, gain=0.3)
            nn.init.zeros_(self.isp_final_2_mlp[0].bias)
            # 输出层
            nn.init.xavier_uniform_(self.isp_final_2_mlp[2].weight, gain=0.3)
            nn.init.zeros_(self.isp_final_2_mlp[2].bias)

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
            attn_output_4 = self.attn4_norm(attn_output_4) + moe_output_3 + 1e-6

            # 1. 合并前两个维度
            bsz, patches, k, head_dim = attn_output_4.shape
            attn_output_4 = attn_output_4.reshape(-1, k, head_dim)  # [bsz*patches, tgt_len, head_dim]

            # 2. 将原始的第三个维度（tgt_len）放到第一个位置
            attn_output_4 = attn_output_4.transpose(0, 1)  # [tgt_len, bsz*patches, head_dim]

            # 从转置后的张量中获取三个专家嵌入（移除isp_expert_3）
            isp_expert_1 = attn_output_4[0]
            isp_expert_2 = attn_output_4[1]
            isp_expert_4 = attn_output_4[2]  # 跳过index 2，直接使用index 3作为饱和度专家

            # 分别输入到三个解码器中
            isp_final_1 = self.isp_1(isp_expert_1)  # [num_windows, param_dim_1 = 1] gamma参数
            isp_final_2 = self.isp_2(isp_expert_2)  # [num_windows, param_dim_2 = 9] 联合校正参数
            isp_final_4 = self.isp_4(isp_expert_4)  # [num_windows, param_dim_4 = 1] 饱和度参数

            # 获取配置中的param_dim参数
            param_dim_1 = self.isp_1.param_dim
            param_dim_2 = self.isp_2.param_dim
            param_dim_4 = self.isp_4.param_dim

            isp_final_1 = isp_final_1.reshape(-1, h_windows * w_windows, param_dim_1)

            # 像素级变换矩阵参数重塑
            window_size = self.patch_size * self.win_size  # 8
            isp_final_2 = isp_final_2.reshape(batch_size, h_windows * w_windows, 576)
            isp_final_2 = isp_final_2.reshape(batch_size, h_windows, w_windows, 64, 9)
            isp_final_2 = isp_final_2.reshape(batch_size, h_windows, w_windows, 8, 8, 9)
            isp_final_2 = isp_final_2.permute(0, 1, 3, 2, 4, 5)
            isp_final_2 = isp_final_2.reshape(batch_size, h_windows * 8, w_windows * 8, 9)
            
            # 通过MLP处理isp_final_2，并加上可学习的残差连接
            isp_final_2_original = isp_final_2
            isp_final_2_processed = self.isp_final_2_mlp(isp_final_2)
            isp_final_2 = isp_final_2_original + self.isp_final_2_residual_weight * isp_final_2_processed

            isp_final_4 = isp_final_4.reshape(-1, h_windows * w_windows, param_dim_4)

            # 将原始图像从[batch_size, 3, H, W]转换为[batch_size, H, W, 3]
            enhanced_image = x.permute(0, 2, 3, 1)  # [batch_size, H, W, 3]

            # 将sRGB空间转换到线性空间 (L^2.2)
            #enhanced_image_final = torch.pow(enhanced_image, 2.2)

            # 按顺序应用3个增强方法

            # 1. 联合变换矩阵
            enhanced_image_final = self.color_transform(I_source=enhanced_image,
                                                               pred_transform_params=isp_final_2)

            # 3.应用最终去噪和锐化（使用固定参数）
            enhanced_image_final = self.denoising_sharpening(images=enhanced_image_final)

            # 4. gamma增强
            global_gamma = isp_final_1.mean(dim=1, keepdim=True)  # [B, 1, 1]
            enhanced_image_final = gamma_correct(L=enhanced_image_final, gamma_increment=global_gamma)

            # 从线性空间转换回sRGB空间 (L^(1/2.2))
            #enhanced_image_final = torch.pow(enhanced_image_final, 1.0 / 2.2)

            # 对所有windows取平均，得到每个图的全局饱和度参数
            global_saturation = isp_final_4.mean(dim=1)  # [B, 1]
            enhanced_image_final = contrast_enhancement(image = enhanced_image_final, alpha = global_saturation)

            # 训练模式下才计算损失并返回
            if self.training:
                # 计算总的辅助损失
                auxiliary_loss = (atten_aux_loss_1 + moe_aux_loss_1 + atten_aux_loss_2 + \
                                  moe_aux_loss_2 + atten_aux_loss_3 + moe_aux_loss_3 + atten_aux_loss_4) / 7

                # 使用目标图像计算重建损失 (使用Charbonnier损失提升鲁棒性)
                charbonnier_loss = CharbonnierLoss(epsilon=1e-3)
                reconstruction_loss = charbonnier_loss(enhanced_image_final, target)

                # 感知损失
                perceptual = self.perceptual_loss(enhanced_image_final, target)

                # LAB色彩损失 - 增加预防性检查
                def safe_lab_loss(pred, target):
                    """安全的LAB损失计算，避免CUDA索引越界"""
                    try:
                        # 预检查：确保输入有效
                        if pred.numel() == 0 or target.numel() == 0:
                            return torch.tensor(0.0, device=pred.device, requires_grad=True)
                        
                        if pred.shape != target.shape:
                            return torch.tensor(0.0, device=pred.device, requires_grad=True)
                            
                        # 确保数值在安全范围内
                        pred_safe = torch.clamp(pred, 0.0, 1.0)
                        target_safe = torch.clamp(target, 0.0, 1.0)
                        
                        # 检查是否包含NaN或Inf
                        if torch.isnan(pred_safe).any() or torch.isinf(pred_safe).any():
                            return torch.tensor(0.0, device=pred.device, requires_grad=True)
                        if torch.isnan(target_safe).any() or torch.isinf(target_safe).any():
                            return torch.tensor(0.0, device=pred.device, requires_grad=True)
                        
                        # 尝试LAB损失计算
                        lab_color_loss = LABColorLoss(color_weight=1.0)
                        return lab_color_loss(pred_safe, target_safe)
                        
                    except Exception as e:
                        # 如果仍然失败，返回简单的L1损失
                        return F.l1_loss(pred, target, reduction='mean') * 0.01
                
                color_loss = safe_lab_loss(enhanced_image_final, target)


                # 计算PSNR损失 (添加数值稳定性检查)
                psnr_value = kornia.metrics.psnr(enhanced_image_final, target, max_val=1.0)
                # 限制PSNR值范围，避免极端情况
                psnr_value = torch.clamp(psnr_value, min=0.0, max=50.0)
                # 使用更稳定的损失计算方式
                psnr_loss = torch.max(torch.tensor(0.0, device=psnr_value.device),
                                      (50.0 - psnr_value) / 50.0)

                # 计算SSIM损失 (添加数值稳定性检查)
                ssim_value = kornia.metrics.ssim(enhanced_image_final, target, window_size=11, max_val=1.0)
                ssim_value = torch.clamp(ssim_value, min=0.0, max=1.0)  # 限制SSIM值范围
                ssim_loss = 1.0 - ssim_value.mean()  # SSIM越大损失越小

                # 检查损失是否为nan或inf
                if torch.isnan(psnr_loss) or torch.isinf(psnr_loss):
                    psnr_loss = torch.tensor(0.0, device=psnr_loss.device)
                if torch.isnan(ssim_loss) or torch.isinf(ssim_loss):
                    ssim_loss = torch.tensor(0.0, device=ssim_loss.device)

                # 计算每种损失的加权值
                weighted_reconstruction = self.reconstruction_weight * reconstruction_loss
                weighted_perceptual = self.perceptual_weight * perceptual
                weighted_auxiliary = self.auxiliary_weight * auxiliary_loss
                weighted_psnr = self.psnr_weight * psnr_loss
                weighted_ssim = self.ssim_weight * ssim_loss
                weighted_color = self.lab_color_weight * color_loss
                
                # 使用权重系数组合不同的损失 (L1+感知+辅助+PSNR+SSIM+LAB色彩+饱和度匹配)
                loss = (weighted_reconstruction + weighted_perceptual + weighted_auxiliary +
                        weighted_psnr + weighted_ssim + weighted_color)
                
                # 打印每种损失的加权值
                print(f"Loss Details - Reconstruction: {weighted_reconstruction.item():.6f}, "
                      f"Perceptual: {weighted_perceptual.item():.6f}, "
                      f"Auxiliary: {weighted_auxiliary.item():.6f}, "
                      f"PSNR: {weighted_psnr.item():.6f}, "
                      f"SSIM: {weighted_ssim.item():.6f}, "
                      f"Color: {weighted_color.item():.6f}, "
                      f"Total: {loss.item():.6f}")

                return enhanced_image_final, loss, reconstruction_loss
            else:
                # 测试模式，只返回增强后的图像
                return enhanced_image_final
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch.cuda.amp import custom_fwd, custom_bwd, autocast

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

"""
    此文件所有的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch
"""

@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,  # 输入维度
            num_heads,  # top-k个注意力头数
            dropout=0.0,  # dropout概率
            bias=False,  # 线性层偏置
            q_noise=0.0,  # 量化噪声强度（用于模型参数 量化，默认0不启用）
            qn_block_size=8,  # 量化噪声的分块大小（与q_noise配合使用）
            num_expert=4,  # 专家数量（用于稀疏混合专家MoE结构）
            head_dim=24,  # 每个注意力头的维度（若设置会覆盖num_heads计算逻辑）num_heads = embed_dim//head_dim
            use_attention_gate=False,  # 是否使用注意力门控，默认不走
            cvloss=0,  # 专家负载均衡损失系数（MoE中平衡专家使用的损失项）
            aux_loss=0,  # 辅助损失
            zloss=0,  # 门控输出正则化损失系数（防止门控值过小）
            sample_topk=0,  # 每个样本选择topk专家进行前向计算（0表示全选）
            noisy_gating=False,  # 是否在门控函数中加入噪声（增强MoE鲁棒性）
            use_pos_bias=False,  # 是否使用位置编码的标志位
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 模型定义维度
        self.kdim = embed_dim  # Key的维度，等于embed_dim
        self.vdim = embed_dim  # 同上

        self.num_heads = num_heads  # 每个注意力头的维度embed_dim // num_heads
        # 使用fairseq定制的Dropout模块
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = head_dim  # 每个头的维度也就是Q⋅Wq_i后的维度也就是计算注意力机制的时候的维度
        self.scaling = self.head_dim ** -0.5  # 缩放因子1/√d_k

        # linearK，这个变换也就是KWk中的Wk，个专家共享相同Wk
        # q_noise量化噪声比例 (0.0~1.0)
        # qn_block_size噪声块大小 (如8表示8x8块)

        # 每个专家的linearQi，也就是Wq_i
        self.q_proj = MoELinearWrapper(
            input_size=embed_dim,
            head_size=self.head_dim,
            num_experts=num_expert,
            k=self.num_heads,
            cvloss=cvloss,
            aux_loss=aux_loss,
            zloss=zloss,
            noisy_gating=noisy_gating
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, self.head_dim, bias=bias), q_noise, qn_block_size
        )
        # linearV，这个变换也就是VWv中的Wv，个专家共享相同Wk
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, self.head_dim, bias=bias), q_noise, qn_block_size
        )

        self.use_attention_gate = use_attention_gate

        self.sample_topk = sample_topk

        # 位置编码相关属性
        self.use_pos_bias = use_pos_bias
        self.pos_bias = None  # 位置编码嵌入表
        self.pos_indices = None  # 位置索引

        if use_pos_bias:
            self.init_pos_bias()

        self.reset_parameters()  # 初始化模型的权重

        self.skip_embed_dim_check = False  # 是否跳过嵌入维度检查

        # 启用CUDNN基准测试以提高性能
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def reset_parameters(self):
        # 当Q/K/V维度相同时，使用缩放版Xavier均匀初始化k_proj(Wk)，v_proj(Wv)权重（经验性优化）
        # 缩放因子 1/sqrt(2) 用于平衡多头注意力的合并结果
        std_k = 1.0 / math.sqrt(self.kdim)
        std_v = 1.0 / math.sqrt(self.vdim)
        nn.init.normal_(self.k_proj.weight, 0, std_k)
        nn.init.normal_(self.v_proj.weight, 0, std_v)

        # nn.init.xavier_uniform_(self.out_proj.weight)
        # if self.out_proj.bias is not None:
        #     nn.init.constant_(self.out_proj.bias, 0.0)

    def init_pos_bias(self):
        """
        初始化二维相对位置编码，专为中心位置对周围8个位置的注意力设计

        每个位置学习一个向量，维度等于注意力头的数量
        """
        # 创建位置偏置嵌入层，8个固定的相对位置
        self.pos_bias = nn.Embedding(8, self.num_heads)

        # 直接使用从0到7的索引表示8个位置关系
        self.pos_indices = torch.arange(8)

        # 只有当缓冲区不存在时才注册
        if not hasattr(self, 'pos_indices'):
            self.register_buffer('pos_indices', self.pos_indices)

        # 为了清晰，可以添加位置到方向的映射注释
        # 0: 左上, 1: 上, 2: 右上, 3: 左, 4: 右, 5: 左下, 6: 下, 7: 右下

    def apply_pos_bias(self, attn_weights):
        """
        将位置偏置应用到注意力分数矩阵上

        参数:
            attn_weights: 注意力分数矩阵，形状为[bsz * patches, heads, tgt_len, src_len]

        返回:
            添加了位置偏置的注意力分数矩阵
        """
        # 确保pos_indices在正确的设备上
        device = attn_weights.device
        self.pos_indices = self.pos_indices.to(device)

        # 从嵌入层获取位置偏置
        bias = self.pos_bias(self.pos_indices)

        # 步骤1: 调整维度顺序，使heads维度在前
        bias = bias.permute(1, 0)

        # 步骤2: 将每个位置编码重复4次，确保每连续4个位置共享同一编码
        bias = bias.repeat_interleave(3, dim=1)

        # 步骤3: 添加维度，为批次和序列长度做准备
        bias = bias.unsqueeze(0).unsqueeze(2)

        # 步骤4: 将bias广播到与attn_weights相同的维度
        bias_broadcasted = bias.expand_as(attn_weights)

        # 步骤5: 添加到注意力分数上
        return attn_weights + (bias_broadcasted / self.scaling)

    def forward(
            self,
            query,
            key: Optional[Tensor],  # 可选
            value: Optional[Tensor],  # 可选
            k_isp: Optional[Tensor] = None,
            need_weights: bool = True,  # 是否返回多头平均注意力分数
            before_softmax: bool = False,  # 是否返回softmax前的原始分数
            need_head_weights: bool = False,  # 是否返回每个头的独立注意力分数
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """

        Args:
            need_weights (bool, 可选):
                是否返回注意力权重（经过多头平均后的结果，默认为False）。
            before_softmax (bool, 可选):
                是否返回经过softmax前的原始注意力权重和值。
            need_head_weights (bool, 可选):
                是否返回每个注意力头的权重（会强制启用need_weights）。
                默认返回所有注意力头的平均权重。
        """
        # 使用自动混合精度计算
        with autocast(enabled=torch.cuda.is_available()):
            # 是否返回每个头的独立注意力分数
            if need_head_weights:
                need_weights = True

            bsz, patches, tgt_len, embed_dim = query.size()  # 返回query大小（目标序列长度（ISP超参数）、批次(图片数)、块数、嵌入维度）
            src_len = tgt_len  # 默认源序列长度等于目标序列长度（自注意力场景，MoA2）
            # 检查query的嵌入维度是否符合预期（仅在启用检查时）
            if not self.skip_embed_dim_check:
                assert (
                        embed_dim == self.embed_dim
                ), f"query dim {embed_dim} != {self.embed_dim}"  # query输入维度不等于模型定义维度
            assert list(query.size()) == [bsz, patches, tgt_len, embed_dim]  # 检查query的尺寸是否符合标准
            # 处理key的存在情况（非自注意力时传入）
            if key is not None:
                key_bsz, key_p, src_len, _ = key.size()
                if not torch.jit.is_scripting():  # 脚本模式外进行维度验证（TorchScript会跳过这些检查）
                    assert key_bsz == bsz  # key和query的批大小必须相同
                    assert key_p == patches
                    assert value is not None  # 由key必须要有value
                    assert (key_bsz, key_p, src_len) == value.shape[:3]  # key和value的前两个尺度必须一样

            # 如果是自注意力机制，计算q(qWq_i), k(KWk), v(VWv)，辅助损失
            # qkv都由query生成
            """
                q的结构
                [
                    [样本1的专家1映射, 样本1的专家3映射],  # 样本1
                    [样本2的专家2映射, 样本2的专家3映射],  # 样本2
                    [样本3的专家1映射, 样本3的专家2映射],  # 样本3
                ]
            """

            assert key is not None and value is not None
            # 确保张量连续以优化计算
            query_cont = query.contiguous()
            key_cont = key.contiguous()
            value_cont = value.contiguous()

            q, aux_loss = self.q_proj.map(
                query_cont, k_isp=k_isp, sample_topk=self.sample_topk,
                attention_gate=self.use_attention_gate
            )  # q 形状为[batch_size, p, tgt_len, k, head_size]
            k = self.k_proj(key_cont)  # k 的形状为 [bsz, p, src_len, head_dim]
            v = self.v_proj(value_cont)  # v 的形状为 [bsz, p, src_len, head_dim]
            q *= self.scaling  # q乘缩放因子1/√d_k，之后直接和k点积就行，softmax((q/√d_k)k)

            assert k is not None
            assert k.size(2) == src_len

            '''
            计算自注意力分数
            q的形状是(b,p,i,k,e)：批次(图片数)、块数、目标序列长度（ISP超参数）、注意力头数量、每个头的嵌入维度
            k的形状是(b,p,j,e)：批次(图片数)、块数、源序列长度（每块的token数）、嵌入维度
            'bpike,bpje->bpkij'输出张量的维度：bpkij，批次(图片数)、块数、注意力头数量、目标序列长度、源序列长度
            '''
            attn_weights = torch.einsum('bpike,bpje->bpkij', q, k)

            # 应用位置编码（如果启用）
            if self.use_pos_bias:
                # 获取当前形状
                bsz, patches, num_heads, tgt_len, src_len = attn_weights.shape  # [bsz, patches, self.num_heads, tgt_len, src_len]

                # 将attn_weights重新排列为形状[bsz*patches, heads, tgt_len, src_len]
                attn_weights = attn_weights.reshape(bsz * patches, self.num_heads, tgt_len, src_len)

                # 应用位置编码
                attn_weights = self.apply_pos_bias(attn_weights)

                # 恢复原始形状 [bsz, patches, self.num_heads, tgt_len, src_len]
                attn_weights = attn_weights.reshape(bsz, patches, self.num_heads, tgt_len, src_len)

            bsz, patches, num_heads, tgt_len, src_len = attn_weights.shape

            attn_weights = attn_weights.contiguous().reshape(bsz * patches * self.num_heads, tgt_len,
                                                             src_len)  # 将结果张量重塑为维度(批次大小×块数×注意力头数量，目标长度，源长度)

            assert list(attn_weights.size()) == [bsz * patches * self.num_heads, tgt_len, src_len]

            # 如果需要返回softmax前的原始分数，返回qk/√d_k、和v
            if before_softmax:
                return attn_weights, v

            # 对注意力权重在最后一个维度上应用softmax函数，使权重归一化为概率分布
            attn_weights_float = utils.softmax(attn_weights, dim=-1)

            attn_weights = attn_weights_float.type_as(attn_weights)  # 将浮点型的注意力权重转换回原始注意力权重的数据类型
            # 对注意力权重应用dropout，随机丢弃张量元素，增强模型的鲁棒性。在推理阶段，dropout会被关闭，所有权重都会被保留。
            attn_probs = self.dropout_module(attn_weights)

            # softmax(qk/√d_k)·v
            assert v is not None
            # 重塑为原始形状以便einsum操作
            attn_probs_reshaped = attn_probs.reshape(bsz, patches, self.num_heads, tgt_len, src_len)
            attn = torch.einsum('bpkij,bpje->bpike', attn_probs_reshaped,
                                v)  # [bsz, patches, tgt_len, self.num_heads, self.head_dim]
            # 注意力计算后的形状
            assert list(attn.size()) == [bsz, patches, tgt_len, self.num_heads, self.head_dim]

            attn = self.q_proj.reduce(
                attn)  # 加权求和得到MoA输出 原代码此处为attn = self.q_proj.reduce(attn).transpose(0, 1)，形状最终为[bsz, patches, tgt_len, self.head_dim]

            # 重塑注意力权重为[批次大小, 注意力头数量, 目标长度, 源长度]并转置
            if need_weights:
                attn_weights = attn_weights_float.reshape(
                    bsz, patches, self.num_heads, tgt_len, src_len
                ).permute(2, 0, 1, 3, 4)  # [self.num_heads, bsz, patches, tgt_len, src_len]
                if not need_head_weights:
                    # average attention weights over heads
                    attn_weights = attn_weights.mean(dim=0)  # 沿着头维度对注意力权重求平均
            else:
                attn_weights: Optional[Tensor] = None

        return attn, attn_weights, aux_loss  # 返回MoA注意力最终计算结果、注意力权重矩阵、辅助损失


class MoE(nn.Module):
    """ 使用一层FFN作为专家的稀疏门控。
        参数:
        input_size: 整数 - 输入的大小
        output_size: 整数 - 输出的大小
        num_experts: 整数 - 专家的数量
        hidden_size: 整数 - 专家的隐藏层大小
        noisy_gating: 布尔值 - 是否使用噪声门控
        k: 整数 - 对每个批次元素使用多少个专家
    """

    def __init__(self, input_size, head_size, num_experts, k, need_merge=False, cvloss=0, aux_loss=0, zloss=0,
                 bias=False, activation=None, noisy_gating=True, hidden_sizes=None):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating  # 是否使用噪声门控
        self.num_experts = num_experts  # 专家总数量
        self.input_size = input_size  # 输入维度
        self.head_size = head_size  # 注意力头大小
        self.need_merge = need_merge
        self.saved_top_k_indices = None
        self.token_expert_indices = None  # 用于存储每个token选择的专家索引
        
        # 支持多层FFN的专家
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias, hidden_sizes)  # 并行专家层，支持多层FFN
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)  # 输出层保持单层
        
        self.k = min(k, self.num_experts)
        self.cvloss = cvloss  # 变异系数损失
        self.aux_loss = aux_loss  # 切换损失
        self.zloss = zloss  # Z_loss
        self.activation = activation
        # 门控权重矩阵
        # 门控权重矩阵
        # ✅ 更强的初始化
        self.w_atten_gate = nn.Parameter(torch.randn(num_experts, num_experts) * 0.2, requires_grad=True)
        self.w_gate = nn.Parameter(torch.randn(input_size, num_experts) * 0.2, requires_grad=True)

        # 使用LeCun Normal替代Xavier
        std_gate = 1.0 / math.sqrt(input_size)
        std_atten = 1.0 / math.sqrt(num_experts)
        nn.init.normal_(self.w_gate, 0, std_gate)
        nn.init.normal_(self.w_atten_gate, 0, std_atten)

        if noisy_gating:
            self.w_noise = nn.Parameter(torch.randn(input_size, num_experts) * 0.01, requires_grad=True)

    def cv_squared(self, x):
        """ 样本的变异系数平方。
            作为损失函数时，鼓励正分布更加均匀。
            添加epsilon以保证数值稳定性。
            对于空张量返回0。
            Args:
                x: 一个张量
            Returns:
                一个标量
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean() ** 2 + eps)  # 计算变异系数的平方: 方差除以均值的平方

    # 计算变异系数
    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    # 1. 对输入的概率张量 probs 按照维度0求和，得到各专家的累积概率。
    # 2. 使用 F.normalize 进行归一化操作，将求和后的结果转换为和为1的分布，归一化时使用 L1 范数（p=1）沿 dim=0 进行归一化。
    # 3. 调用 cv_squared 函数计算归一化后概率分布的系数平方。

    def auxiliary_loss(self, probs, freqs):
        """修改为支持浮点freqs参数，保持梯度流"""
        # 确保都是浮点数进行计算，避免类型转换截断梯度
        probs_sum = F.normalize(probs.sum(0), p=1, dim=0)

        # ★ 关键新增一行：显式转 dtype，避免 F.normalize() 报错
        freqs = freqs.to(probs.dtype)  # 或 freqs = freqs.float()

        freqs_norm = F.normalize(freqs, p=1, dim=0)
        loss = probs_sum * freqs_norm
        return loss.sum() * self.num_experts

    # 1. 对输入的logits 进行指数运算，转换为未归一化的概率量度。并沿着 dim=1 求和，得到每个样本对应的归一化因子
    # 2. 对求和的结果取对数，再平方平方。
    # 3. 计算上述值的均值，作为 z loss 的最终值。
    def compute_zloss(self, logits):
        # 通过使用log_softmax优化数值稳定性
        log_sum = torch.logsumexp(logits, dim=1)
        return torch.mean(log_sum ** 2)

    def atten_gating(self, Q_isp, K_isp, sample_topk=1, noise_epsilon=1e-3):
        """
        前向传播

        参数:
            Q_isp (Tensor): 形状为 [batches, patches, 1, input_dim] 的张量
            K_isp (Tensor): 形状为 [batches, patches, N, input_dim] 的张量
            k (int): top-k中的k值，表示保留的权重数量

        返回:
            gate_weights (Tensor): 形状为 [-1, N] 的张量，
                                  表示门控权重，每个向量中只有k个非零元素
        """
        # 使用混合精度计算
        with autocast(enabled=torch.cuda.is_available()):
            # 获取输入形状
            batches, patches, N, input_dim = K_isp.shape
            scale = 1.0 / math.sqrt(input_dim)
            # 计算注意力分数：Q_isp与K_isp的点积
            # Q_isp的形状是[batches, patches, 1, input_dim]
            # K_isp的形状是[batches, patches, N, input_dim]
            # 确保输入张量是连续的
            Q_isp = Q_isp.contiguous()
            K_isp = K_isp.contiguous()

            attention_scores = torch.matmul(Q_isp, K_isp.transpose(-1, -2))  # [batches, patches, 1, N]

            # 移除多余的维度
            attention_scores = attention_scores.squeeze(2)  # [batches, patches, N]

            attention_scores = attention_scores.reshape(-1, N)  # [-1, N]

            attention_scores = attention_scores * scale

            attention_scores = torch.nan_to_num(
                attention_scores,
                nan=0.0,
                posinf=20.0,  # 上限可根据需要调到30、40
                neginf=-20.0
            )

            clean_logits = attention_scores
            # 如果启用了噪声门控并且当前处于训练模式，则添加噪声
            if self.noisy_gating and self.training:
                # 计算输入 x 与噪声权重矩阵 self.w_noise 的矩阵乘积，得到原始噪声标准差
                raw_noise_stddev = attention_scores @ self.w_atten_gate
                # 通过 softplus 激活函数保证噪声标准差为正值，并加上一个极小值 noise_epsilon
                noise_stddev = torch.nan_to_num(
                    F.softplus(raw_noise_stddev) + noise_epsilon,
                    nan=noise_epsilon,
                    posinf=1e3, neginf=noise_epsilon
                )
                # 生成与 clean_logits 相同形状的随机噪声并按噪声标准差进行缩放，然后与 clean_logits 相加
                noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
                # 使用带噪声的 logits
                logits = noisy_logits
            else:
                logits = clean_logits
            # 应用softmax获取注意力权重

            logits = torch.nan_to_num(
                logits,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0
            )

            attention_weights = F.softmax(logits, dim=1)  # [-1, N]

            # 展平attention_weights并保存为probs
            probs = attention_weights

            # 如果训练阶段，且sample_topk > 0启用混合选择策略
            if sample_topk > 0:
                # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
                # top_k_gates = torch.gather(probs, 1, top_k_indices)
                assert sample_topk <= self.k
                # 先在所有专家中选择topk中的topk - sample_topk个
                _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
                masked_probs = probs + 1e-6  # 添加极小值避免零概率
                masked_probs[torch.arange(probs.size(0)).unsqueeze(
                    1), top_km1_indices] = 0  # 创建掩码概率：已选位置置零防止重复选择
                k_indices = torch.multinomial(masked_probs, sample_topk)  # 从剩余概率中采样 sample_topk 个专家
                # 合并确定性和采样得到的专家索引
                top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
                top_k_gates = torch.gather(probs, 1, top_k_indices)
            else:
                top_k_gates, top_k_indices = probs.topk(self.k, dim=1)  # 常规top-k选择（确定性选择）

            # 门控值归一化
            top_k_gates = top_k_gates / \
                          (top_k_gates.sum(dim=1, keepdim=True) + 1e-6)

            # 创建全零张量用于存储门控权重
            zeros = torch.zeros_like(probs)  # ✅ 移除requires_grad，让autograd自动处理
            zeros = zeros.requires_grad_(probs.requires_grad)  # ✅ 明确设置
            gates = zeros.scatter(1, top_k_indices, top_k_gates)

            # 统计每个专家被选中的样本数（用于负载均衡计算）
            freqs_float = gates.sum(0)  # 用概率和
            self.expert_size = (gates > 0).sum(0).long()  # 只是为了 debug/打印，可保留
            # 门控值和对应索引展平为[batch_size * k]
            top_k_gates = top_k_gates.flatten()  # 门控值[0.5, 0.3, 0.7, 0.9, 0.6, 0.8] 3 个样本 × 2 个专家
            top_k_experts = top_k_indices.flatten()  # 门控索引[1, 3, 2, 4, 1, 2]3 个样本 × 2 个专家

            # 以下三行是关键代码，不能修改，保持原始实现逻辑
            # 过滤零值门控（保留有效路由信息）
            nonzeros = top_k_gates.nonzero().squeeze(-1)
            top_k_experts_nonzero = top_k_experts[nonzeros]  # 门控索引[1, 3, 2, 4, 1, 2]
            # topk个专家按ID排序，优化后续计算效率
            _, _index_sorted_experts = top_k_experts_nonzero.sort(0)  # 排序后的门控索引
            self.index_sorted_experts = nonzeros[_index_sorted_experts]

            """
            top_k_indices = [[1, 3],  # 第一个样本选择了专家 1 和 3
                            [2, 4],  # 第二个样本选择了专家 2 和 4
                            [1, 2]]  # 第三个样本选择了专家 1 和 2
            top_k_gates = [[0.5, 0.3],  # 第一个样本选择专家 1 和 3 的门控值分别是 0.5 和 0.3
                        [0.7, 0.9],  # 第二个样本选择专家 2 和 4 的门控值分别是 0.7 和 0.9
                        [0.6, 0.8]]  # 第三个样本选择专家 1 和 2 的门控值分别是 0.6 和 0.8
            """
            self.batch_index = self.index_sorted_experts.div(self.k,
                                                             rounding_mode='trunc')  # 每个专家的样本池（每个专家处理哪些样本？样本池是样本原始索引组成的），再把样本池按照专家索引排序，[专家1样本池，专家2样本池，...]
            self.batch_gates = top_k_gates[
                self.index_sorted_experts]  # 每个专家的样本门控值池（每个专家处理的样本对应的门控值），再把样本门控值池按照专家索引排序，[专家1样本门控值池，专家2样本门控值池，...]
            # 计算损失
            loss = 0
            # 变异系数损失：鼓励均匀使用各专家
            loss += self.cvloss * self.compute_cvloss(gates)
            # 辅助损失
            loss += self.aux_loss * self.auxiliary_loss(probs, freqs_float)
            # zloss
            loss += self.zloss * self.compute_zloss(logits)

            return loss

    def top_k_gating(self, x, sample_topk=1, noise_epsilon=1e-3):
        """
            噪声 top-k 门控。
            参见论文：https://arxiv.org/abs/1701.06538

            参数:
              x: 输入张量，形状为 [-1, input_size]
              sample_topk: 训练阶段采样选取的专家数量（0 表示不采样）
              noise_epsilon: 防止数值不稳定的小浮点数
            返回:
              loss: 综合了 cv loss、switch loss 及 z loss 的最终损失值
        """
        # 使用混合精度计算
        with autocast(enabled=torch.cuda.is_available()):
            # 确保输入张量连续以优化计算
            x = x.contiguous()

            d = x.size(1)
            scale = 1.0 / math.sqrt(d)
            # 计算输入 x 与门控权重矩阵 self.w_gate 的矩阵乘积，得到干净的 logits
            clean_logits = x @ self.w_gate
            clean_logits = clean_logits * scale
            clean_logits = torch.clamp(clean_logits, min=-10.0, max=10.0)
            # ===== DEBUG: 查看 clamp 前后 logits 分布 =====

            # 如果启用了噪声门控并且当前处于训练模式，则添加噪声
            if self.noisy_gating and self.training:
                # 计算输入 x 与噪声权重矩阵 self.w_noise 的矩阵乘积，得到原始噪声标准差
                raw_noise_stddev = x @ self.w_noise
                # 通过 softplus 激活函数保证噪声标准差为正值，并加上一个极小值 noise_epsilon
                noise_stddev = torch.nan_to_num(
                    F.softplus(raw_noise_stddev) + noise_epsilon,
                    nan=noise_epsilon,
                    posinf=1e3, neginf=noise_epsilon
                )
                # 生成与 clean_logits 相同形状的随机噪声并按噪声标准差进行缩放，然后与 clean_logits 相加
                noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
                # 使用带噪声的 logits
                logits = noisy_logits
            else:
                logits = clean_logits

            logits = torch.nan_to_num(
                logits,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0
            )

            probs = F.softmax(logits, dim=1)  # [-1, N]
            # ✅ 调试位置2: 检查概率分布

            # 如果训练阶段，且sample_topk > 0启用混合选择策略
            if sample_topk > 0:
                # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
                # top_k_gates = torch.gather(probs, 1, top_k_indices)
                assert sample_topk <= self.k
                # 先在所有专家中选择topk中的topk - sample_topk个
                _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
                masked_probs = probs + 1e-6  # 添加极小值避免零概率
                masked_probs[torch.arange(probs.size(0)).unsqueeze(
                    1), top_km1_indices] = 0  # 创建掩码概率：已选位置置零防止重复选择
                k_indices = torch.multinomial(masked_probs, sample_topk)  # 从剩余概率中采样 sample_topk 个专家
                # 合并确定性和采样得到的专家索引
                top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
                top_k_gates = torch.gather(probs, 1, top_k_indices)
            else:
                top_k_gates, top_k_indices = probs.topk(self.k, dim=1)  # 常规top-k选择（确定性选择）
            # 门控值归一化
            top_k_gates = top_k_gates / \
                          (top_k_gates.sum(dim=1, keepdim=True) + 1e-6)

            # 保存为实例变量，以便在forward和concat中使用
            self.saved_top_k_indices = top_k_indices

            gates = torch.zeros_like(probs)  # 创建与 probs 同形的零张量
            gates.scatter_(1, top_k_indices, top_k_gates)  # 关键修改：in-place 操作保留计算图

            # 统计每个专家被选中的样本数（用于负载均衡计算）
            expert_size_int = (gates > 0).sum(0).long()  # int64, 和 input 行数严格对齐
            self.expert_size = expert_size_int  # ParallelLinear 会用到

            # 2) 浮点计数 —— 门控概率之和，用于负载均衡损失，能传播梯度
            freqs_float = gates.sum(0)  # float32

            # 门控值和对应索引展平为[batch_size * k]
            top_k_gates = top_k_gates.flatten()  # 门控值[0.5, 0.3, 0.7, 0.9, 0.6, 0.8] 3 个样本 × 2 个专家
            top_k_experts = top_k_indices.flatten()  # 门控索引[1, 3, 2, 4, 1, 2]3 个样本 × 2 个专家

            # 以下三行是关键代码，不能修改，保持原始实现逻辑
            # 过滤零值门控（保留有效路由信息）
            nonzeros = top_k_gates.nonzero().squeeze(-1)
            top_k_experts_nonzero = top_k_experts[nonzeros]  # 门控索引[1, 3, 2, 4, 1, 2]
            # topk个专家按ID排序，优化后续计算效率
            _, _index_sorted_experts = top_k_experts_nonzero.sort(0)  # 排序后的门控索引
            self.index_sorted_experts = nonzeros[_index_sorted_experts]

            self.batch_index = self.index_sorted_experts.div(self.k,
                                                             rounding_mode='trunc')  # 每个专家的样本池（每个专家处理哪些样本？样本池是样本原始索引组成的），再把样本池按照专家索引排序，[专家1样本池，专家2样本池，...]
            self.batch_gates = top_k_gates[
                self.index_sorted_experts]  # 每个专家的样本门控值池（每个专家处理的样本对应的门控值），再把样本门控值池按照专家索引排序，[专家1样本门控值池，专家2样本门控值池，...]
            # 计算损失
            loss = 0
            # 变异系数损失：鼓励均匀使用各专家
            loss += self.cvloss * self.compute_cvloss(gates)
            # 辅助损失
            loss += self.aux_loss * self.auxiliary_loss(probs, freqs_float)
            # zloss
            loss += self.zloss * self.compute_zloss(logits)
            # 在函数末尾添加

            return loss

    """
        MoE层前向传播过程，实现动态路由和专家计算

        参数:
          x: 输入张量，形状为 [bsz, patches, tgt_len, self.num_heads, self.head_dim]
          sample_topk: 训练时采样的专家数量（0表示禁用采样，正常topk选择）
          multiply_by_gates: 是否用门控值加权专家输出

        返回:
          y: 经过专家聚合的输出，形状同输入x
          loss: 门控机制计算的总损失
    """

    def forward(self, x, sample_topk=1, multiply_by_gates=True):
        """
        前向传播输入是[bsz, patches, tgt_len, self.input_size]
        """
        # 使用混合精度计算
        with autocast(enabled=torch.cuda.is_available()):
            # 输入数据预处理
            bsz, patches, length, emb_size = x.size()  # 获取输入张量的维度：批次大小、块数量、序列长度、注意力头数量和每个头的嵌入维度
            x = x.contiguous()
            x = x.reshape(-1, emb_size)  # 将四维张量重塑为二维张量，新形状为[bsz*p*length*k, emb_size]

            # 计算门控权重更新到self属性，并返回loss
            loss = self.top_k_gating(x, sample_topk=sample_topk)
            # expert_inputs的每一项的索引对应专家索引，值是索引对应专家的输入数据
            expert_inputs = x[self.batch_index]
            # 并行计算所有专家的FFN前向传播（现在内部已处理激活函数）
            h = self.experts(expert_inputs, self.expert_size)  # 多层FFN，内部已处理激活
            # 并行计算所有专家输出层的前向传播
            expert_outputs = self.output_experts(h, self.expert_size)  # 输出层：head_size -> input_size
            # 门控加权
            # batch_gates: 每个路由项对应的门控权重，形状 [num_selected]
            # 通过[:, None]扩展维度实现逐元素相乘
            if multiply_by_gates:
                expert_outputs = expert_outputs * self.batch_gates[:, None]
            # 输出聚合
            # 创建全零基础张量用于聚合结果
            if self.need_merge:
                zeros = torch.zeros((bsz * patches * length, self.input_size),
                                    dtype=expert_outputs.dtype, device=expert_outputs.device)
                # index_add操作：将专家输出累加到对应位置
                # 参数说明：
                #   dim=0 - 按行累加
                #   index=self.batch_index - 目标行索引
                #   source=expert_outputs - 需要添加的数据
                y = zeros.index_add(0, self.batch_index,
                                    expert_outputs)  # [batch_size*seq_len, output_dim]，包含了每个专家的输出向量相加（而不是合并）
                y = y.contiguous()
                y = y.reshape(bsz, patches, length, self.input_size)
            else:
                # need_merge=False 分支：使用 index_add 安全填充专家输出
                # 1) 计算总行数并创建全零扁平张量
                total_rows = bsz * patches * length * self.k
                zeros = torch.zeros(
                    (total_rows, self.input_size),
                    dtype=expert_outputs.dtype,
                    device=expert_outputs.device
                )

                # 2) 重建 expert_positions（在原代码中已计算，用于指示在 k 维度的位置）
                expert_positions = torch.zeros(
                    len(self.index_sorted_experts),
                    dtype=torch.long,
                    device=expert_outputs.device
                )
                expert_counts = torch.zeros(
                    (bsz * patches * length),
                    dtype=torch.long,
                    device=expert_outputs.device
                )
                for i, sample_idx in enumerate(self.batch_index):
                    expert_positions[i] = expert_counts[sample_idx]
                    expert_counts[sample_idx] += 1

                # 3) 计算扁平后每行的索引：batch_index * k + expert_positions
                row_indices = self.batch_index * self.k + expert_positions  # shape [num_selected]

                # 4) 用 index_add 将每个 expert_outputs 累加到对应行
                y_flat = zeros.index_add(0, row_indices, expert_outputs)

                # 5) 最后 reshape 回 [bsz, patches, length, k, input_size]
                y = y_flat.view(bsz, patches, length, self.k, self.input_size)

                # 记录每个token选择的专家索引，用于concat方法
                self.token_expert_indices = self.saved_top_k_indices.reshape(bsz, patches, length, self.k)

            return y, loss

    def map(self, x, k_isp=None, sample_topk=1, attention_gate=False):
        """
            MoE映射函数（适配分块输入，保持独立计算）
            参数:
                x: 输入张量，形状 [batch_size, p=64, seq_len = 1, emb_size]
                k_isp: K张量(可选)，形状 [batch_size, p=64, N, emb_size]
                sample_topk: 训练时采样的专家数量（0表示禁用采样）
                attention_gate: 是否使用注意力门控，默认为False
            返回:
                y: 每个块中每个token对应的k个专家输出，形状 [batch_size, p=64, seq_len = 1, k = 2, head_size]
                loss: 门控机制的负载均衡损失
        """
        # 使用混合精度计算
        with autocast(enabled=torch.cuda.is_available()):
            if attention_gate and k_isp is not None:
                # 使用注意力门控，保持原有维度
                bsz, p, seq_len, emb_size = x.size()  # 输入query的尺寸
                loss = self.atten_gating(x, k_isp, sample_topk=sample_topk)  # 计算辅助损失

                x = x.reshape(bsz * p * seq_len, emb_size)  # 直接合并前三个维度
            else:
                # 使用原始门控方式
                bsz, p, seq_len, emb_size = x.size()  # 输入query的尺寸
                x = x.reshape(bsz * p * seq_len, emb_size)  # 直接合并前三个维度
                loss = self.top_k_gating(x, sample_topk=sample_topk)  # 计算辅助损失

            # 门控后的索引信息
            # 此时x已经展平，直接使用
            # 此时对象已存储路由信息：
            #   self.batch_index: 每个专家处理的原始token索引 [num_selected]
            #   self.expert_size: 每个专家被选中的次数 [num_experts]
            #   self.index_sorted_experts: 按专家ID排序后的所有样本索引 [num_selected]
            expert_inputs = x[self.batch_index]  # 这个张量包含几部分，每一部分包含某个专家所处理的所有样本token
            expert_outputs = self.experts(expert_inputs, self.expert_size)  # 继承自nn.Module，自动调用forward方法

            # 输出聚合
            # 创建全零基础张量
            zeros = torch.zeros(
                (bsz * p * seq_len * self.k, self.head_size),
                dtype=expert_outputs.dtype,
                device=expert_outputs.device
            )
            # 按排序后的索引填充专家输出（确保同专家计算连续）
            y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
            y = y.reshape(bsz, p, seq_len, self.k, self.head_size)  # 保持原始seq_len


            return y, loss  # 返回top-k个专家的qWq_i，下一步送入注意力点积运算以及辅助损失

    def reduce(self, x, multiply_by_gates=True):
        # 使用混合精度计算
        with autocast(enabled=torch.cuda.is_available()):
            x = x.contiguous()
            bsz, patches, length, k, head_dim = x.size()

            # 重塑x为二维张量
            x = x.reshape(-1, head_dim)

            expert_inputs = x[self.index_sorted_experts]

            expert_outputs = self.output_experts(expert_inputs, self.expert_size)

            if multiply_by_gates:
                if self.batch_gates.shape[0] != expert_outputs.shape[0]:
                    # 确保形状兼容
                    min_size = min(self.batch_gates.shape[0], expert_outputs.shape[0])
                    self.batch_gates = self.batch_gates[:min_size]
                    expert_outputs = expert_outputs[:min_size]

                expert_outputs = expert_outputs * self.batch_gates[:, None]

            # 创建输出张量
            zeros = torch.zeros((bsz * patches * length, self.input_size),
                                dtype=expert_outputs.dtype, device=expert_outputs.device)

            # 确保batch_index所有值都在有效范围内
            max_index = bsz * patches * length - 1

            if self.batch_index.max() > max_index:
                # 裁剪越界索引
                valid_mask = self.batch_index <= max_index
                valid_indices = self.batch_index[valid_mask]
                valid_outputs = expert_outputs[valid_mask]
                y = zeros.index_add(0, valid_indices, valid_outputs)
            else:
                y = zeros.index_add(0, self.batch_index, expert_outputs)

            y = y.reshape(bsz, patches, length, self.input_size)

            return y

    def concat(self, y, e_isp):
        """
        将e_isp的最后一个维度拆分成4等份，然后按顺序加到y的对应k维度上

        参数:
            y: 前向传播输出的张量，形状为 [bsz, patches, length=1, k=4, self.input_size]
            e_isp: 输入张量，形状为 [bsz, patches, embed_dim]，embed_dim能被4整除

        返回:
            result: 求和后的结果，形状为 [bsz, patches, length=1, k=4, self.input_size]
        """
        # 使用混合精度计算
        with autocast(enabled=torch.cuda.is_available()):
            bsz, patches, length, k, input_size = y.size()


            # 确认e_isp的最后一维可以被4整除
            embed_dim = e_isp.size(-1)

            # 验证每个分块大小与input_size相匹配
            split_size = embed_dim // k

            # 确保输入张量连续
            y = y.contiguous()
            e_isp = e_isp.contiguous()

            # 将e_isp最后一个维度拆分成4等份
            # [bsz, patches, embed_dim] -> [bsz, patches, 4, embed_dim//4]
            e_isp_reshaped = e_isp.reshape(bsz, patches, k, split_size)

            # 调整e_isp_reshaped的维度以匹配y
            # [bsz, patches, 4, split_size] -> [bsz, patches, 1, 4, split_size]
            e_isp_expanded = e_isp_reshaped.unsqueeze(2)

            # 直接将重塑后的e_isp与y相加
            combined = y + e_isp_expanded

            return combined


# ParallelLinear以及ParallelExperts实现了多专家并行计算的线性层
class ParallelLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, expert_size, weight, bias=None):
        # 确保输入张量连续
        input = input.contiguous()

        output_list = []  # 初始化一个空列表，用于保存每个专家计算后的输出

        # 🔧 修改这里：处理浮点expert_size
        # ❌ 原来的代码：
        # expert_size_list = expert_size.tolist()  # 如果expert_size是浮点会导致split报错

        # ✅ 修改为：
        if expert_size.dtype.is_floating_point:
            # 如果expert_size是浮点数（来自gates.sum(0)），转换为整数用于split
            expert_size_int = torch.round(expert_size).long()
            expert_size_list = expert_size_int.tolist()
        else:
            # 如果expert_size已经是整数，直接使用
            expert_size_list = expert_size.tolist()

        # 将输入张量按照 expert_size_list 指定的尺寸进行分割
        input_list = input.split(expert_size_list, dim=0)

        # 记录权重形状用于调试
        for i in range(weight.size(0)):
            if expert_size_list[i] > 0:  # 只处理有样本的专家
                if bias is not None:
                    o_i = torch.mm(input_list[i], weight[i]) + bias[i]
                else:
                    o_i = torch.mm(input_list[i], weight[i])
                output_list.append(o_i)
            else:
                # 对于没有样本的专家，添加一个空张量
                if bias is not None:
                    empty_output = torch.empty((0, weight[i].size(1)),
                                               dtype=input.dtype,
                                               device=input.device)
                    output_list.append(empty_output)
                else:
                    empty_output = torch.empty((0, weight[i].size(1)),
                                               dtype=input.dtype,
                                               device=input.device)
                    output_list.append(empty_output)

        output = torch.cat(output_list, dim=0)  # 将所有专家的输出拼接成最终的输出张量

        # 🔧 关键：保存原始的expert_size（保持梯度），而不是转换后的整数版本
        # 这样在backward时可以保持梯度流
        variables = (input, expert_size, weight, bias)  # 注意这里是原始expert_size
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input, expert_size, weight, bias = ctx.saved_tensors
        num_linears = weight.size(0)

        expert_size_list = expert_size.tolist()
        input_list = input.split(expert_size_list, dim=0)
        grad_list = grad_out.split(expert_size_list, dim=0)

        # 计算输入的梯度
        d_input_list = []
        for i in range(num_linears):
            if expert_size_list[i] > 0:
                d_input_list.append(torch.mm(grad_list[i], weight[i].t()))
            else:
                empty_grad = torch.empty((0, weight[i].size(0)),
                                         dtype=grad_out.dtype,
                                         device=grad_out.device)
                d_input_list.append(empty_grad)

        d_input = torch.cat(d_input_list, dim=0)

        # 计算权重的梯度 - ✅ 保持核心逻辑，去掉所有检查
        d_weight_list = []
        for i in range(num_linears):
            if expert_size_list[i] > 0:
                # ✅ 核心计算逻辑：input^T * grad_out
                d_weight_list.append(torch.mm(input_list[i].t(), grad_list[i]))
            else:
                # ✅ 无样本专家的零梯度
                d_weight_list.append(torch.zeros_like(weight[i]))

        d_weight = torch.stack(d_weight_list, dim=0)

        # 计算偏置的梯度 - ✅ 保持核心逻辑，去掉所有检查
        if bias is not None:
            d_bias_list = []
            for i in range(num_linears):
                if expert_size_list[i] > 0:
                    # ✅ 核心计算逻辑：按批次维度求和
                    d_bias_list.append(grad_list[i].sum(0))
                else:
                    # ✅ 无样本专家的零梯度
                    d_bias_list.append(torch.zeros_like(bias[i]))
            d_bias = torch.stack(d_bias_list, dim=0)
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False, hidden_sizes=None) -> None:
        """
            初始化并行专家模块，支持多层FFN
            参数:
            num_experts: 专家的总数量
            input_size: 每个专家的输入特征维度
            output_size: 每个专家的输出特征维度
            bias: 是否使用偏置，默认为False
            hidden_sizes: 隐藏层维度列表，如[512, 1024]表示两层隐藏层
                         如果为None则使用单层(input_size -> output_size)
        """
        super().__init__()
        
        # 构建层序列
        if hidden_sizes is None:
            # 向后兼容：单层FFN
            self.layers = nn.ModuleList([
                self._create_parallel_layer(num_experts, input_size, output_size, bias)
            ])
            self.layer_sizes = [(input_size, output_size)]
        else:
            # 多层FFN: input -> hidden1 -> hidden2 -> ... -> output
            self.layers = nn.ModuleList()
            self.layer_sizes = []
            
            # 输入层到第一个隐藏层
            prev_size = input_size
            for hidden_size in hidden_sizes:
                layer = self._create_parallel_layer(num_experts, prev_size, hidden_size, bias)
                self.layers.append(layer)
                self.layer_sizes.append((prev_size, hidden_size))
                prev_size = hidden_size
            
            # 最后一个隐藏层到输出层
            final_layer = self._create_parallel_layer(num_experts, prev_size, output_size, bias)
            self.layers.append(final_layer)
            self.layer_sizes.append((prev_size, output_size))

        self.reset_parameters()

    def _create_parallel_layer(self, num_experts, input_size, output_size, bias):
        """创建单个并行线性层"""
        w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            b = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            b = None
        return nn.ParameterDict({'weight': w, 'bias': b})

    def reset_parameters(self) -> None:
        # 使用LeCun Normal初始化（适合GELU）
        for layer in self.layers:
            w = layer['weight']
            for i in range(w.size(0)):
                fan_in = w.size(1)
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(w[i], 0, std)

    # 使用自定义的ParallelLinear进行前向计算
    def forward(self, inputs, expert_size):
        x = inputs
        # 逐层前向传播
        for i, layer in enumerate(self.layers):
            w = layer['weight']
            b = layer['bias']
            x = ParallelLinear.apply(x, expert_size, w, b)
            
            # 除了最后一层，都应用GELU激活
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        
        return x


class MoELinearWrapper(nn.Module):  # ✅ 继承nn.Module
    def __init__(self, input_size, head_size, num_experts, k, need_merge=False,
                 cvloss=0, aux_loss=0, zloss=0, bias=False,
                 activation=None, noisy_gating=True, hidden_sizes=None):
        # ✅ 必须首先调用super().__init__()
        super().__init__()

        # ✅ 然后才能创建子模块
        self.moe = MoE(
            input_size=input_size,
            head_size=head_size,
            num_experts=num_experts,
            k=k,
            need_merge=need_merge,
            cvloss=cvloss,
            aux_loss=aux_loss,
            zloss=zloss,
            bias=bias,
            activation=activation,
            noisy_gating=noisy_gating,
            hidden_sizes=hidden_sizes
        )

    def forward(self, x, sample_topk=0, multiply_by_gates=True):
        return self.moe(x, sample_topk, multiply_by_gates)

    def map(self, x, k_isp=None, sample_topk=0, attention_gate=False):
        return self.moe.map(x, k_isp, sample_topk, attention_gate)

    def reduce(self, x, multiply_by_gates=True):
        return self.moe.reduce(x, multiply_by_gates)

    def concat(self, y, e_isp):
        return self.moe.concat(y, e_isp)

    # 其他属性转发...
    @property
    def batch_index(self):
        return self.moe.batch_index

    @property
    def batch_gates(self):
        return self.moe.batch_gates

    @property
    def token_expert_indices(self):
        return self.moe.token_expert_indices

    @property
    def saved_top_k_indices(self):
        return self.moe.saved_top_k_indices

    @property
    def index_sorted_experts(self):
        return self.moe.index_sorted_experts

    @property
    def expert_size(self):
        return self.moe.expert_size
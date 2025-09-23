import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp  # 导入混合精度训练支持
import math


class EispGeneratorFFN(nn.Module):
    """
    将特征图通过FFN层映射为E_isp张量

    输入特征图形状: [batches, windows, tokens, input_dim]
    输出E_isp形状: [batches, windows, output_dim]

    每个patch独立处理，将patch内所有tokens映射为一个output_dim维度的向量
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        """
        初始化EispGeneratorFFN

        参数:
            input_dim (int): 输入特征的维度
            hidden_dim (int): FFN隐藏层的维度
            output_dim (int): 输出E_isp的维度
            dropout_rate (float): Dropout比率，默认0.1
        """
        super(EispGeneratorFFN, self).__init__()

        # FFN结构定义
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()  # 使用GELU激活函数
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # 最终映射到output_dim
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # 初始化网络参数
        self._initialize_weights()

        # 使用JIT编译加速FFN主要计算路径
        if torch.cuda.is_available():
            self.ffn_path = nn.Sequential(
                self.fc1,
                self.dropout1,
                self.activation,
                self.fc2,
                self.dropout2,
                self.activation
            )
            self.ffn_path = torch.jit.script(self.ffn_path)

    def _initialize_weights(self):
        """改进的FFN初始化策略"""
        # 针对GELU激活函数的优化初始化
        for m in [self.fc1, self.fc2]:
            # GELU的有效增益约为1.7（相比ReLU的sqrt(2)）
            fan_in = m.in_features
            std = math.sqrt(1.7 / fan_in)  # 为GELU调整
            nn.init.normal_(m.weight, 0, std)
            nn.init.zeros_(m.bias)

        # 输出层使用更小的初始化，避免梯度爆炸
        nn.init.normal_(self.fc_out.weight, 0, 0.02)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 形状为 [batches, windows, num_tokens, input_dim] 的特征图

        返回:
            E_isp (Tensor): 形状为 [batches, windows, output_dim] 的张量
        """
        # 使用混合精度计算
        with amp.autocast(enabled=torch.cuda.is_available()):
            # 获取输入形状
            batches, windows, num_tokens, input_dim = x.shape

            # 重塑为 [batches*windows, num_tokens, input_dim] 便于并行处理所有patch
            x_reshaped = x.reshape(batches * windows, num_tokens, input_dim)

            # 对输入进行归一化
            x_normalized = self.layer_norm(x_reshaped)

            # 应用FFN网络 - 使用JIT编译过的模块加速计算
            if hasattr(self, 'ffn_path'):
                x = self.ffn_path(x_normalized)
            else:
                # 应用FFN第一层
                x = self.fc1(x_normalized)
                x = self.dropout1(x)
                x = self.activation(x)

                # 应用FFN第二层
                x = self.fc2(x)
                x = self.dropout2(x)
                x = self.activation(x)

            # 对每个windows内的所有tokens进行池化操作，使用快速实现
            x_pooled = torch.mean(x, dim=1)  # [batches*windows, hidden_dim]

            # 最终映射到输出维度
            E_isp = self.fc_out(x_pooled)  # [batches*windows, output_dim]

            # 重塑回原始batch和windows维度
            E_isp = E_isp.reshape(batches, windows, -1)  # [batches, windows, output_dim]

        return E_isp


class QKIspGenerator(nn.Module):
    """
    将E_isp转换为K_isp和Q_isp

    输入E_isp形状: [batches, windows, input_dim]
    输出K_isp形状: [batches, windows, N, input_dim]
    输出Q_isp形状: [batches, windows, 1, input_dim]
    """

    def __init__(self, dim, num_heads, dropout_rate):
        """
        初始化KQGenerator

        参数:
            dim (int): 输入和输出的维度
            num_heads (int): 生成的K_isp头数量N
            dropout_rate (float): Dropout比率，默认0.1
        """
        super(QKIspGenerator, self).__init__()

        self.dim = dim
        self.num_heads = num_heads

        # 优化：使用单个大矩阵替代多个独立映射以提高并行性
        self.k_combined = nn.Linear(dim, dim * num_heads)

        # Q路径：单个映射
        self.q_projection = nn.Linear(dim, dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 层归一化
        self.norm = nn.LayerNorm(dim)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        """针对Attention的优化初始化"""
        # 使用截断正态分布，标准差根据维度调整
        # 这是Transformer中常用的初始化策略
        std = math.sqrt(2.0 / (self.dim + self.dim))  # Xavier的变种

        # K投影矩阵初始化
        nn.init.trunc_normal_(self.k_combined.weight, std=std)
        nn.init.zeros_(self.k_combined.bias)

        # Q投影矩阵初始化 - 使用稍小的std以保持稳定性
        nn.init.trunc_normal_(self.q_projection.weight, std=std * 0.8)
        nn.init.zeros_(self.q_projection.bias)

    def forward(self, E_isp):
        """
        前向传播

        参数:
            E_isp (Tensor): 形状为 [batches, windows, input_dim] 的张量

        返回:
            tuple: (K_isp, Q_isp)
                K_isp (Tensor): 形状为 [batches, windows, N, input_dim] 的张量
                Q_isp (Tensor): 形状为 [batches, windows, 1, input_dim] 的张量
        """
        # 使用混合精度计算
        with amp.autocast(enabled=torch.cuda.is_available()):
            # 应用层归一化
            E_isp_norm = self.norm(E_isp)

            # 获取输入形状
            batches, windows, input_dim = E_isp_norm.shape

            # K路径: 使用单个大矩阵进行计算以提高并行性
            k_combined = self.k_combined(E_isp_norm)  # [batches, windows, num_heads*dim]
            k_combined = self.dropout(k_combined)

            # 重塑为所需的维度
            K_isp = k_combined.view(batches, windows, self.num_heads, self.dim)  # [batches, windows, N, dim]

            # Q路径: 生成一个Q_isp向量
            Q_isp = self.q_projection(E_isp_norm)  # [batches, windows, input_dim]
            Q_isp = self.dropout(Q_isp)  # 应用dropout

            # 添加维度以形成[batches, windows, 1, input_dim]的形状
            Q_isp = Q_isp.unsqueeze(2)  # [batches, windows, 1, input_dim]

        return Q_isp, K_isp


class KVFeatureGenerator(nn.Module):
    """
            前向传播

            参数:
                feature (Tensor): 形状为 [batches, windows, tokens, embed_dim] 的张量

            返回:
                tuple: (k_feature, v_feature)
                    k_feature (Tensor): 与输入feature相同，形状为 [batches, windows, tokens, embed_dim]
                    v_feature (Tensor): 通过Wv映射后的feature，形状为 [batches, windows, tokens, embed_dim]
    """

    def __init__(self, embed_dim, dropout_rate):
        super(KVFeatureGenerator, self).__init__()

        # 优化：将K和V投影合并为一个大矩阵以提高并行性
        self.kv_projection = nn.Linear(embed_dim, embed_dim * 2)
        self.embed_dim = embed_dim

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        """针对K,V投影的优化初始化"""
        # 对于K,V投影，使用专门的初始化策略
        # 参考T5和GPT的初始化方式
        std = math.sqrt(1.0 / self.embed_dim)

        # 由于是K,V投影，使用相对保守的初始化
        nn.init.normal_(self.kv_projection.weight, 0, std)
        nn.init.zeros_(self.kv_projection.bias)

    def forward(self, feature):
        # 使用混合精度计算
        with amp.autocast(enabled=torch.cuda.is_available()):
            # 获取输入形状
            batches, windows, tokens, embed_dim = feature.shape

            # 重塑为 [batches*windows*tokens, embed_dim] 便于并行处理
            feature_flat = feature.reshape(-1, embed_dim)

            # 同时计算K和V投影
            kv_combined = self.kv_projection(feature_flat)  # [batches*windows*tokens, embed_dim*2]
            kv_combined = self.dropout(kv_combined)

            # 重塑回原始形状并分离K和V
            kv_combined = kv_combined.reshape(batches, windows, tokens, embed_dim * 2)
            k_feature, v_feature = torch.chunk(kv_combined, 2, dim=-1)  # 各自形状为 [batches, windows, tokens, embed_dim]

        return k_feature, v_feature
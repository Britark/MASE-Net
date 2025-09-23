import torch
import torch.nn as nn
import torch.cuda.amp as amp  # 引入混合精度支持


class Decoder(nn.Module):
    def __init__(self, latent_dim, param_dim, hidden_dims=[96, 120, 150, 180], dropout=0.1):
        super().__init__()

        layers = []
        input_dim = latent_dim

        # 构建MLP层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))  # 添加dropout
            input_dim = hidden_dim

        # 最终输出层
        self.main = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], param_dim)

        # JIT编译加速主要计算路径
        if torch.cuda.is_available():
            self.main = torch.jit.script(self.main)
            self.output = torch.jit.script(self.output)

    def forward(self, x):
        # 使用混合精度计算
        with amp.autocast(enabled=torch.cuda.is_available()):
            x = self.main(x)
            params = self.output(x)
        return params


class ISPDecoder(nn.Module):
    """
    专家ISP处理器

    处理非零嵌入，并将结果恢复到原始位置
    """

    def __init__(self, latent_dim, param_dim, hidden_dims=[96, 120], dropout=0.1, use_identity_init=True, bias_init=0.0):
        """
        初始化专家ISP处理器

        Args:
            latent_dim: 输入嵌入维度
            param_dim: 输出参数维度
            hidden_dims: 解码器隐藏层维度
            dropout: dropout率
            use_identity_init: 是否使用特殊的身份初始化（接近0的初始化）
            bias_init: 偏置初始化值，用于控制预测参数的基础值
        """
        super(ISPDecoder, self).__init__()

        # 创建解码器
        self.decoder = Decoder(latent_dim, param_dim, hidden_dims, dropout)
        self.param_dim = param_dim
        self.use_identity_init = use_identity_init
        self.bias_init = bias_init

        # 根据参数决定是否使用特殊初始化
        if use_identity_init:
            self._init_output_layer_identity()
        else:
            self._init_output_layer_random()

    def _init_output_layer_identity(self):
        """
        SOTA无约束初始化策略：
        - 颜色矩阵参数：初始化为接近零的小值（残差学习）
        - Gamma参数：初始化为0 (因为final_gamma = 1 + increment)
        - 饱和度参数：初始化为0 (tanh(0)=0，对应1.5倍增强)
        - 其他参数：保持小值初始化
        """
        output_layer = self.decoder.output
        
        # 使用正常的权重初始化，避免梯度消失
        nn.init.xavier_uniform_(output_layer.weight, gain=0.3)  # 增大gain
        
        # 偏置初始化策略
        if self.param_dim == 1:  # 饱和度参数
            # 强制初始化为正值，防止网络学习降低饱和度
            nn.init.constant_(output_layer.bias, 0.5)  # tanh(0.5)≈0.46，对应约2.1倍增强
        else:
            # 其他参数初始化为0，让网络从"不改变"开始学习
            nn.init.zeros_(output_layer.bias)

    def _init_output_layer_random(self):
        """
        改进的输出层初始化 - 针对tanh映射优化
        """
        output_layer = self.decoder.output

        # 使用正态分布初始化，让初始alpha有合理的分布
        nn.init.normal_(output_layer.weight, mean=0.0, std=0.3)

        # 针对饱和度参数的智能初始化
        if self.param_dim == 1:  # 饱和度参数
            # 强制初始化为正值，防止网络学习降低饱和度
            nn.init.constant_(output_layer.bias, 0.5)  # tanh(0.5)≈0.46，对应约2.1倍增强
        else:
            # 其他参数使用自定义的bias初始化值
            nn.init.constant_(output_layer.bias, self.bias_init)


    def forward(self, expert_embedding):
        """
        前向传播 - 处理专家嵌入并生成解码后的参数

        Args:
            expert_embedding: 形状为 [num_windows, embed_dim] 的张量
                              其中可能包含零嵌入表示未被选择的窗口

        Returns:
            output: 形状为 [num_windows, param_dim] 的张量
                              保持未选择位置为零
        """
        # 使用混合精度计算
        with amp.autocast(enabled=torch.cuda.is_available()):
            params = self.decoder(expert_embedding)
        return params
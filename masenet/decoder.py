import torch
import torch.nn as nn
import torch.cuda.amp as amp  # 引入混合精度支持


class Decoder(nn.Module):
    def __init__(self, latent_dim, param_dim, hidden_dims=[96, 120, 150, 180]):
        super().__init__()

        layers = []
        input_dim = latent_dim

        # 构建MLP层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
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

    def __init__(self, latent_dim, param_dim, hidden_dims=[96, 120]):
        """
        初始化专家ISP处理器

        Args:
            latent_dim: 输入嵌入维度
            param_dim: 输出参数维度
            hidden_dims: 解码器隐藏层维度
        """
        super(ISPDecoder, self).__init__()

        # 创建解码器
        self.decoder = Decoder(latent_dim, param_dim, hidden_dims)
        self.param_dim = param_dim

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
            num_windows, embed_dim = expert_embedding.shape

            # 获取解码器的输出维度
            param_dim = self.decoder.output.out_features

            # 创建输出张量，初始化为1e-6，而不是0 - 保持原有逻辑
            output = torch.zeros(num_windows, param_dim, device=expert_embedding.device)

            # 创建非零掩码：通过检查是否有非零元素 - 保持原有逻辑
            non_zero_mask = torch.any(expert_embedding != 0, dim=1)  # [num_windows]

            # 如果没有非零嵌入，直接返回零张量 - 保持原有逻辑
            if not torch.any(non_zero_mask):
                return output

            # 优化：计算非零元素的数量以预分配内存
            valid_count = non_zero_mask.sum().item()

            # 提取非零嵌入 - 保持原有逻辑
            valid_embeddings = expert_embedding[non_zero_mask]  # [valid_count, embed_dim]

            # 使用解码器处理有效嵌入
            valid_outputs = self.decoder(valid_embeddings)  # [valid_count, param_dim]

            # 将解码结果放回原始位置 - 保持原有逻辑
            output[non_zero_mask] = valid_outputs.to(output.dtype)

        return output
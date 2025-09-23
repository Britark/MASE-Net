import torch
import torch.nn as nn
import torch.cuda.amp as amp  # 导入混合精度支持


def initialize_2d_relative_position_bias(self, heads=1):
    """
    以下的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch

    初始化二维相对位置编码，专为3×3网格中心位置对周围8个位置的注意力设计

    参数:
        heads: 注意力头的数量
    """
    # 直接创建位置偏置嵌入层，8个固定的相对位置
    self.pos_bias = nn.Embedding(8, heads)

    # 使用JIT编译加速嵌入层（如果可用）
    if torch.cuda.is_available():
        self.pos_bias = torch.jit.script(self.pos_bias)

    # 创建0到7的索引张量并作为固定缓冲区注册
    pos_indices = torch.arange(8, dtype=torch.long)
    self.register_buffer('pos_indices', pos_indices)

    # 预计算并缓存转置后的位置索引形状，以备复用
    self.register_buffer('transformed_bias_shape', torch.zeros(1))

    # 为了清晰，可以保存相对位置的语义标记（可选）
    # 例如：0=左上, 1=上, 2=右上, 3=左, 4=右, 5=左下, 6=下, 7=右下


def apply_center_to_surrounding_pos_bias(self, fmap, scale):
    """
    以下的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch

    将位置偏置应用到中心对周围的注意力分数矩阵上

    参数:
        fmap: 注意力分数矩阵，形状为[batch, heads, 1, 8]
        scale: 缩放因子

    返回:
        添加了位置偏置的注意力分数矩阵
    """
    # 使用混合精度计算
    with amp.autocast(enabled=torch.cuda.is_available()):
        # 确保注意力图是连续的内存
        fmap = fmap.contiguous()

        # 检查是否已经缓存了变换后的偏置
        if hasattr(self, 'cached_bias') and self.transformed_bias_shape.item() == fmap.size(1):
            # 如果头数相同，直接使用缓存的偏置
            bias = self.cached_bias
        else:
            # 从嵌入层获取位置偏置
            bias = self.pos_bias(self.pos_indices)  # 形状为[8, heads]

            # 重排为[1, heads, 1, 8]
            bias = bias.permute(1, 0).contiguous()  # 将 [8, heads] 变为 [heads, 8]，确保连续内存
            bias = bias.unsqueeze(0).unsqueeze(2)  # 添加批次和查询维度，得到 [1, heads, 1, 8]

            # 缓存变换后的偏置以避免重复计算
            self.register_buffer('cached_bias', bias)
            self.transformed_bias_shape[0] = fmap.size(1)  # 存储当前的头数

        # 根据形状进行广播
        bias_broadcasted = bias.expand_as(fmap)

        # 应用偏置并采用高效的原位操作
        result = fmap.clone()
        result.add_(bias_broadcasted / scale)

        return result
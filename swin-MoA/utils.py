import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp  # 导入混合精度支持


class PatchNeighborSearcher(nn.Module):
    """
    以下的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch

    将扁平化的 ISP 张量 [batches, patches, length, input_size]
    恢复为 [batches, H, W, length, input_size]，
    并基于 nn.Unfold(kernel_size=3, stride=1, padding=1) 以步长 1
    搜索每个中心 patch 的 3×3 区域，剔除中心自身，
    最终返回 shape 为 [batches, patches, length*8, input_size] 的邻居特征。
    其中 patches = H * W，patches 是按行优先(flatten)。
    """

    def __init__(self):
        super().__init__()
        # 使用 Unfold 自带 padding=1 来填充外圈0
        self.unfold = nn.Unfold(kernel_size=3, stride=1, padding=1)

        # 预计算邻居索引，避免每次前向传播重复计算
        self.register_buffer('neighbor_indices', torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], dtype=torch.long))

    def forward(
            self,
            E_flat: torch.Tensor,  # [batches, patches, length, input_size]
            H: int,
            W: int
    ) -> torch.Tensor:
        """
        Args:
            E_flat: Tensor of shape [batches, patches, length, input_size]
            H: patch 网格的行数
            W: patch 网格的列数（确保 patches == H * W）
        Returns:
            neigh_features: Tensor of shape [batches, patches, length*8, input_size]
                              length*8 表示每个中心 patch 的 8 个邻居拼接后的特征长度
        """
        # 使用混合精度计算提高性能
        with amp.autocast(enabled=torch.cuda.is_available()):
            batches, patches, length, input_size = E_flat.shape
            assert patches == H * W, (
                f"patches ({patches}) 必须等于 H*W ({H}*{W}={H * W})"
            )

            # 确保输入张量是连续的，提高后续操作效率
            E_flat = E_flat.contiguous()

            # ----- 1. 恢复为网格形式 [B, H, W, L, D] -----
            E_grid = E_flat.view(batches, H, W, length, input_size)

            # ----- 2. 合并 length 和 input_size 作为通道维度，准备用 Unfold -----
            # [B, H, W, L, D] -> [B, L, D, H, W]
            X = E_grid.permute(0, 3, 4, 1, 2).contiguous()  # 确保张量连续性
            # [B, L, D, H, W] -> [B, L*D, H, W]
            X = X.reshape(batches, length * input_size, H, W)

            # ----- 3. 使用 Unfold 提取所有 3x3 补丁 (自动零填充外圈) -----
            # out: [B, (L*D)*9, H*W]
            patches_unfold = self.unfold(X)

            # ----- 4. 重塑为 [B, L*D, 9, H*W] -----
            _, c9, num_centers = patches_unfold.shape
            # c9 == L*D*9
            patches_unfold = patches_unfold.view(batches, length * input_size, 9, num_centers)

            # ----- 5. 去除中心自身 (patch index 4) 只保留 8 个邻居 -----
            # 使用预计算的邻居索引
            neigh = patches_unfold.index_select(2, self.neighbor_indices)  # [B, L*D, 8, P]

            # ----- 6. 恢复维度并 permute -----
            neigh = neigh.view(batches, length, input_size, 8, patches)
            neigh = neigh.permute(0, 4, 1, 3, 2).contiguous()  # [B, P, L, 8, D]

            # ----- 7. 合并 length 和 8 维度 -> length*8 -----
            # [B, P, L, 8, D] -> [B, P, L*8, D]
            neigh_features = neigh.reshape(batches, patches, length * 8, input_size)

            return neigh_features


class ISPParameterGenerator(nn.Module):
    """
    以下的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch

    ISP参数生成器

    根据窗口的ISP嵌入和专家索引，重组ISP嵌入到专家维度上
    """

    def __init__(self):
        super(ISPParameterGenerator, self).__init__()

    def forward(self, isp_per_win, expert_indices, num_experts):
        """
        前向传播函数 - 使用高效的并行方式处理

        Args:
            isp_per_win: 形状为 [batches, windows, k, embed_dim] 的张量
                         表示每个batch的每个window生成的k个ISP嵌入
            expert_indices: 形状为 [num_windows, k] 的张量
                           其中 num_windows = batches * windows
                           存储了每个window选择的ISP专家序号
            num_experts: 整数，表示ISP专家的总数量

        Returns:
            expert_embeddings: 形状为 [num_experts, num_windows, embed_dim] 的张量
                              表示按专家索引重组后的嵌入
        """
        # 使用混合精度计算提高性能
        with amp.autocast(enabled=torch.cuda.is_available()):
            # 获取输入张量的形状
            batches, windows, k, embed_dim = isp_per_win.shape
            num_windows = batches * windows

            # 确保输入张量是连续的
            isp_per_win = isp_per_win.contiguous()
            expert_indices = expert_indices.contiguous()

            # 重塑ISP嵌入为 [num_windows, k, embed_dim]
            isp_per_win = isp_per_win.reshape(num_windows, k, embed_dim)

            # 确保expert_indices的形状正确
            assert expert_indices.shape == (
                num_windows, k), f"专家索引形状 {expert_indices.shape} 与预期形状 ({num_windows}, {k}) 不匹配"

            # 创建输出张量，初始化为零，形状为 [num_experts, num_windows, embed_dim]
            expert_embeddings = torch.zeros(num_experts, num_windows, embed_dim,
                                            device=isp_per_win.device,
                                            dtype=isp_per_win.dtype)

            # 为每个(window, k)对创建索引
            win_indices = torch.arange(num_windows, device=expert_indices.device).unsqueeze(1).expand(-1, k)
            k_indices = torch.arange(k, device=expert_indices.device).unsqueeze(0).expand(num_windows, -1)

            # 将expert_indices展平，同时创建对应的window索引
            flat_expert_indices = expert_indices.reshape(-1)
            flat_win_indices = win_indices.reshape(-1)
            flat_k_indices = k_indices.reshape(-1)

            # 排除无效的专家索引（如果有的话）
            valid_mask = (flat_expert_indices >= 0) & (flat_expert_indices < num_experts)
            flat_expert_indices = flat_expert_indices[valid_mask]
            flat_win_indices = flat_win_indices[valid_mask]
            flat_k_indices = flat_k_indices[valid_mask]

            # 使用高级索引获取对应的嵌入
            flat_embeddings = isp_per_win[flat_win_indices, flat_k_indices]

            # 使用index_put_操作批量处理所有嵌入，无需循环
            expert_embeddings.index_put_(
                (flat_expert_indices, flat_win_indices),
                flat_embeddings,
                accumulate=False  # 不累加，后面的值会覆盖前面的
            )

            return expert_embeddings


# 使用示例
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)

    # 启用CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 创建测试数据
    batches = 2
    windows = 3
    k = 2
    embed_dim = 4
    num_experts = 4

    # 创建随机ISP嵌入
    isp_per_win = torch.randn(batches, windows, k, embed_dim)

    # 创建专家索引，确保每个窗口选择的k个专家是不同的
    expert_indices = torch.zeros(batches * windows, k, dtype=torch.long)
    for win_idx in range(batches * windows):
        # 从0到num_experts-1中随机选择k个不重复的专家索引
        expert_indices[win_idx] = torch.randperm(num_experts)[:k]

    # 打印输入数据
    print("=" * 50)
    print("输入数据:")
    print("-" * 50)
    print(f"ISP嵌入形状: {isp_per_win.shape}")
    print(isp_per_win)

    print("\n" + "-" * 50)
    print(f"专家索引形状: {expert_indices.shape}")
    print("\n专家索引内容:")
    print(expert_indices)

    print("\n" + "-" * 50)
    print(f"总专家数: {num_experts}")

    # 创建ISP参数生成器
    generator = ISPParameterGenerator()

    # 生成专家嵌入
    expert_embeddings = generator(isp_per_win, expert_indices, num_experts)

    # 打印输出数据
    print("\n" + "=" * 50)
    print("输出数据:")
    print("-" * 50)
    print(f"专家嵌入形状: {expert_embeddings.shape}")
    print(expert_embeddings)
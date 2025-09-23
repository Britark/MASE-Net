import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp  # 导入混合精度支持
import math


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


class RGB_HVI(nn.Module):
    def __init__(self, density_k, alpha_S, alpha_I):
        """
        RGB ↔ HVI 变换模块
        Args:
            density_k: [batch, height, width, 1] 每个像素的density_k参数
            alpha_S: [batch, height, width, 1] 局部对比度参数
            alpha_I: [batch, height, width, 1] 局部亮度参数（全局）
        """
        super(RGB_HVI, self).__init__()

        # 将参数保存为模块属性
        self.register_buffer('density_k', density_k)
        self.register_buffer('alpha_S', alpha_S)
        self.register_buffer('alpha_I', alpha_I)

    def HVIT(self, img):
        """
        RGB → HVI 变换
        Args:
            img: [batch, channel, height, width] RGB图像
        Returns:
            hvi: [batch, 3, height, width] HVI图像
        """
        eps = 1e-8
        pi = 3.141592653589793

        # 转换输入格式：[batch, channel, height, width] → [batch, height, width, channel]
        img = img.permute(0, 2, 3, 1)  # [batch, height, width, 3]

        # 1. 计算HSV中的基本分量
        value = img.max(dim=3, keepdim=True)[0]  # I_max = max(R', G', B') [batch, height, width, 1]
        img_min = img.min(dim=3, keepdim=True)[0]  # I_min = min(R', G', B') [batch, height, width, 1]
        delta = value - img_min  # Δ = I_max - I_min

        # 2. 色相计算
        hue = torch.zeros_like(value)  # [batch, height, width, 1]

        # 当 I_max = R' 且 Δ > ε 时：h = (G'-B')/Δ mod 6
        mask_r = (img[..., 0:1] == value) & (delta > eps)
        hue[mask_r] = ((img[..., 1:2] - img[..., 2:3]) / (delta + eps))[mask_r] % 6

        # 当 I_max = G' 且 Δ > ε 时：h = 2 + (B'-R')/Δ
        mask_g = (img[..., 1:2] == value) & (delta > eps)
        hue[mask_g] = (2.0 + (img[..., 2:3] - img[..., 0:1]) / (delta + eps))[mask_g]

        # 当 I_max = B' 且 Δ > ε 时：h = 4 + (R'-G')/Δ
        mask_b = (img[..., 2:3] == value) & (delta > eps)
        hue[mask_b] = (4.0 + (img[..., 0:1] - img[..., 1:2]) / (delta + eps))[mask_b]

        # 当 Δ ≤ ε 时：h = 0 (已经初始化为0)
        hue = hue / 6.0  # H = h/6 mod 1

        # 3. 饱和度计算：S = Δ/(I_max + ε)
        saturation = delta / (value + eps)
        saturation[value == 0] = 0

        # 4. 强度坍缩函数：C_k = (sin(π·I_max/2) + ε)^(1/k_w)
        # 注意：density_k = 1/k_w，所以公式变为 C_k = (sin(π·I_max/2) + ε)^density_k
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(self.density_k)

        # 5. 极化变换：θ = 2π · H
        # h_plane = cos(θ), v_plane = sin(θ)
        ch = (2.0 * pi * hue).cos()  # h_plane
        cv = (2.0 * pi * hue).sin()  # v_plane

        # 6. HVI构建：
        # Ĥ = C_k · S · h_plane
        # V̂ = C_k · S · v_plane
        # I = I_max
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value

        # 合并HVI通道
        hvi = torch.cat([H, V, I], dim=3)  # [batch, height, width, 3]
        hvi = hvi.permute(0, 3, 1, 2)# [batch, 3, height, width]
        return hvi

    def PHVIT(self, hvi):
        """
        HVI → RGB 逆变换
        Args:
            hvi: [batch, height, width, 3] HVI图像
        Returns:
            rgb: [batch, channel, height, width] RGB图像
        """
        eps = 1e-8
        pi = 3.141592653589793

        H, V, I = hvi[..., 0:1], hvi[..., 1:2], hvi[..., 2:3]  # 分离HVI通道

        # 限制范围
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)

        # 1. 恢复极化分量
        # V_recovered = α_I · I
        v =  I  # 扩展alpha_I维度以匹配I
        v = torch.clamp(v, 0, 1)  # 立即clamp！

        # 重新计算C_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(self.density_k)

        # h_norm = Ĥ/(C_k + ε), v_norm = V̂/(C_k + ε)
        H_norm = H / (color_sensitive + eps)
        V_norm = V / (color_sensitive + eps)
        H_norm = torch.clamp(H_norm, -1, 1)
        V_norm = torch.clamp(V_norm, -1, 1)

        # 2. 恢复色相：H_recovered = arctan2(v_norm, h_norm)/(2π) mod 1
        h = torch.atan2(V_norm + eps, H_norm + eps) / (2 * pi)
        h = h % 1

        # 🔍 关键调试：在α_S使用前后添加调试信息
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        s = s * self.alpha_S  # 简单的线性调整

        # V_recovered = clamp(V_recovered, 0, 1)
        v = torch.clamp(v, 0, 1)

        # 4. HSV → RGB 标准变换
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1.0 - s)
        q = v * (1.0 - (f * s))
        t = v * (1.0 - ((1.0 - f) * s))

        # 初始化RGB
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        # 根据hi值分配RGB (对应文档中的6种情况)
        hi0 = (hi == 0)  # 当 i = 0：(R,G,B) = (V_recovered, t, p)
        hi1 = (hi == 1)  # 当 i = 1：(R,G,B) = (q, V_recovered, p)
        hi2 = (hi == 2)  # 当 i = 2：(R,G,B) = (p, V_recovered, t)
        hi3 = (hi == 3)  # 当 i = 3：(R,G,B) = (p, q, V_recovered)
        hi4 = (hi == 4)  # 当 i = 4：(R,G,B) = (t, p, V_recovered)
        hi5 = (hi == 5)  # 当 i = 5：(R,G,B) = (V_recovered, p, q)

        r[hi0] = v[hi0];
        g[hi0] = t[hi0];
        b[hi0] = p[hi0]

        r[hi1] = q[hi1];
        g[hi1] = v[hi1];
        b[hi1] = p[hi1]

        r[hi2] = p[hi2];
        g[hi2] = v[hi2];
        b[hi2] = t[hi2]

        r[hi3] = p[hi3];
        g[hi3] = q[hi3];
        b[hi3] = v[hi3]

        r[hi4] = t[hi4];
        g[hi4] = p[hi4];
        b[hi4] = v[hi4]

        r[hi5] = v[hi5];
        g[hi5] = p[hi5];
        b[hi5] = q[hi5]

        # 合并RGB通道
        rgb = torch.cat([r, g, b], dim=3)  # [batch, height, width, 3]
        rgb = rgb * self.alpha_I

        # 转换输出格式：[batch, height, width, channel] → [batch, channel, height, width]
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = torch.clamp(rgb, 0, 1)

        return rgb

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from kornia.filters import gaussian_blur2d, bilateral_blur
import torch.cuda.amp as amp  # 引入混合精度支持


def patch_gamma_correct(L, gamma_increment, epsilon=1e-5):
    """
    以下的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch

    对每个 12×12 patch 做 γ 校正，输入的gamma_increment作为初始伽马值1的增量，
    并将最终伽马值限制在 [0.5, 1.6] 之间，输出结果限制在 [ε, 1]。

    参数:
        L: 原始低光图，shape = (bsz, patches, h, w, 3)，假设没有全零情况
        gamma_increment: 每个 patch 的 γ 增量值，shape = (bsz, patches, 1)
        epsilon: 最小值阈限
    返回:
        R: 增强结果，shape = (bsz, patches, h, w, 3)
    """
    # 使用混合精度计算
    with amp.autocast(enabled=torch.cuda.is_available()):
        bsz, patches, h, w, c = L.shape
        # 1) 计算最终的伽马值 = 初始值(1) + 增量
        final_gamma = 1.0 + gamma_increment

        # 2) 将伽马值限制在 [0.5, 1.6] 之间
        clamped_gamma = torch.clamp(final_gamma, min=0.3, max=2.0)

        # 然后将其重塑为[bsz, patches, 1, 1, 1] - 保持原有变换逻辑
        gamma_reshaped = clamped_gamma.view(bsz, patches, 1, 1, 1)

        # 最后扩展到与L相同的形状[bsz, patches, h, w, c] - 保持原有变换逻辑
        gamma_exp = gamma_reshaped.expand(bsz, patches, h, w, c)

        # 4) 幂运算 - 使用torch.pow提高效率但保持相同的计算逻辑
        R = torch.pow(L, gamma_exp)

        # 5) 限制最终结果在 [ε, 1]
        return torch.clamp(R, min=epsilon, max=1.0)


def white_balance(predicted_gains, I_source):
    """
    以下的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch

    应用白平衡增益到低光图像上，其中predicted_gains是初始白平衡参数[1,1,1]的增量

    参数:
        predicted_gains: 预测的白平衡增益增量, 形状为 [B, P, 3], 其中P是patches数量
        I_source: 原始低光图像, 形状为 [B, P, H, W, 3]

    返回:
        I_whitebalance: 白平衡增强后的图像, 形状为 [B, P, H, W, 3]
    """
    # 使用混合精度计算
    with amp.autocast(enabled=torch.cuda.is_available()):
        # 获取维度信息
        _, _, h, w, _ = I_source.shape

        # 步骤1: 计算最终的白平衡参数 = 初始参数[1,1,1] + 预测的增量
        base_gains = torch.ones_like(predicted_gains)  # 初始白平衡参数[1,1,1]
        final_gains = base_gains + predicted_gains  # 加上增量得到最终参数

        # 步骤2: 对最终白平衡参数应用特定通道的约束 - 保持原有逻辑
        processed_gains = torch.empty_like(final_gains)

        # R通道范围：0.5 ~ 2.0 (索引0)
        processed_gains[:, :, 0] = torch.clamp(final_gains[:, :, 0], 0.3, 2.5)

        # G通道范围：0.8 ~ 1.2 (索引1)
        processed_gains[:, :, 1] = torch.clamp(final_gains[:, :, 1], 0.8, 1.2)

        # B通道范围：0.5 ~ 2.0 (索引2)
        processed_gains[:, :, 2] = torch.clamp(final_gains[:, :, 2], 0.3, 2.5)

        # 步骤3: 应用白平衡增益到原始图像 - 保持原有变换逻辑
        # 将增益值扩展到与I_source相同的维度 [B, P, 3] -> [B, P, H, W, 3]
        expanded_gains = processed_gains.unsqueeze(2).unsqueeze(2).expand(-1, -1, h, w, -1)

        # 应用增益到原始图像上 (逐通道相乘)
        I_whitebalance = I_source * expanded_gains

        # 处理可能的无效值 (NaN 或 Inf)
        invalid_mask = torch.isnan(I_whitebalance) | torch.isinf(I_whitebalance)
        I_whitebalance = torch.where(invalid_mask, I_source, I_whitebalance)

        # 设置极小值，避免完全黑的像素
        min_value = 1e-5

        # 添加截断操作，确保输出在[min_value, 1]范围内
        I_whitebalance = torch.clamp(I_whitebalance, min_value, 1.0)

        return I_whitebalance


def apply_color_correction_matrices(I_source, pred_matrices):
    """
    以下的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch

    对原始图像应用色彩校正矩阵 (完全向量化实现，无循环)

    参数:
        I_source: 原始低光图像, 形状为 [B, P, H, W, 3]
        pred_matrices: 预测的色彩校正矩阵增量，形状为 [B, P, 3, 3]

    返回:
        I_color_corrected: 色彩校正后的图像, 形状为 [B, P, H, W, 3]
    """
    # 使用混合精度计算
    with amp.autocast(enabled=torch.cuda.is_available()):
        # 获取维度信息
        bsz, num_patches, h, w, _ = I_source.shape

        # 步骤1: 计算最终的色彩校正矩阵 = 初始矩阵(单位矩阵) + 预测的增量 - 保持原有变换逻辑
        # 创建单位矩阵，形状匹配pred_matrices
        identity_matrix = torch.eye(3, device=pred_matrices.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_patches,
                                                                                                     -1, -1)

        # 单位矩阵加上增量得到最终的色彩校正矩阵
        final_matrices = identity_matrix + pred_matrices

        # 步骤2: 约束CCM矩阵在合理范围内 - 保持原有逻辑
        # 创建对角元素掩码和非对角元素掩码
        diag_mask = torch.eye(3, device=pred_matrices.device).bool().unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        non_diag_mask = ~diag_mask

        # 约束对角元素范围：0.8 ~ 1.2
        constrained_matrices = torch.where(
            diag_mask.expand_as(final_matrices),
            torch.clamp(final_matrices, 0.6, 1.4),
            final_matrices
        )

        # 约束非对角元素范围：-0.2 ~ 0.2
        constrained_matrices = torch.where(
            non_diag_mask.expand_as(constrained_matrices),
            torch.clamp(constrained_matrices, -0.5, 0.5),
            constrained_matrices
        )

        # 步骤3: 应用色彩校正矩阵到原始图像 - 保持原有逻辑
        # 将图像从[B, P, H, W, 3]重塑为[B, P, H*W, 3]以便进行批量矩阵乘法
        I_reshaped = I_source.reshape(bsz, num_patches, h * w, 3)

        # 使用批量矩阵乘法优化替代einsum，保持相同的计算结果
        I_corrected = torch.matmul(I_reshaped, constrained_matrices.transpose(-1, -2))

        # 安全检查：处理可能的NaN或Inf
        invalid_mask = torch.isnan(I_corrected) | torch.isinf(I_corrected)
        I_corrected = torch.where(invalid_mask, I_reshaped, I_corrected)

        # 重塑回原始形状 [B, P, H, W, 3]
        I_color_corrected = I_corrected.reshape(bsz, num_patches, h, w, 3)

        # 添加截断操作，确保输出在极小值(1e-5)到1之间
        min_value = 1e-5
        I_color_corrected = torch.clamp(I_color_corrected, min_value, 1.0)

        return I_color_corrected


class DenoisingSharpening(nn.Module):
    """
    以下的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch

    对图像patches进行降噪和锐化，具有精细的噪声控制：
    - 扩大的双边滤波核以获得更好的平滑效果
    - 反转和指数化的噪声掩码以强调真实边缘
    - 软阈值处理通过sigmoid函数平滑地进行细节提取
    - 降低lambda范围以避免过度锐化
    - 提高tau下限以使噪声抑制曲线更陡峭
    - Skip Gate机制在噪声或细节过小时跳过锐化
    """

    def __init__(self, noise_thresh: float = 2e-3, kernel_size: int = 3, k_init: float = 100.0,
                 skip_thresh: float = 1e-4):
        super().__init__()
        self.noise_thresh = noise_thresh
        self.kernel_size = kernel_size
        # 引入可学习的k参数，用于软阈值
        self.k = nn.Parameter(torch.tensor(k_init))
        # 引入skip gate阈值
        self.skip_thresh = skip_thresh

    def forward(self, images: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        参数:
            images: [B, P, H, W, 3] 浮点值在[0,1]范围内
            params: [B, P, 7] 参数: [sigma_s, sigma_r, sigma_f, lambda, tau, gain, offset]
        返回:
            enhanced: [B, P, H, W, 3]
        """
        # 使用混合精度计算
        with amp.autocast(enabled=torch.cuda.is_available()):
            B, P, H, W, C = images.shape
            # 重塑为 [N, C, H, W] - 保持原有逻辑
            x = images.view(-1, H, W, C).permute(0, 3, 1, 2)
            p = params.view(-1, 7)

            # 将参数限制在安全范围内
            sigma_s = p[:, 0].clamp(0.2, 5.0)
            sigma_r = p[:, 1].clamp(0.01, 1.0)
            sigma_f = p[:, 2].clamp(0.2, 3.0)
            lam = p[:, 3].clamp(0.1, 2.0)
            tau = p[:, 4].clamp(0.5, 5.0)
            gain = p[:, 5].clamp(0.2, 2.0)
            offset = p[:, 6].clamp(0.01, 1.0)

            # 双边平滑
            sigma_space = torch.stack([sigma_s, sigma_s], dim=1)
            bf = bilateral_blur(
                x,
                kernel_size=(self.kernel_size, self.kernel_size),
                sigma_color=sigma_r,
                sigma_space=sigma_space,
                border_type='reflect'
            )
            # 用标量替代张量作为nan_to_num的参数
            bf = torch.nan_to_num(bf, nan=0.0, posinf=1.0, neginf=0.0)
            # 手动处理任何仍然存在的无效值
            bf = torch.where(torch.isnan(bf) | torch.isinf(bf), x, bf)

            # 高斯低通滤波用于提取细节
            sigma_g = torch.stack([sigma_f, sigma_f], dim=1)
            gf = gaussian_blur2d(
                x,
                kernel_size=(self.kernel_size, self.kernel_size),
                sigma=sigma_g,
                border_type='reflect'
            )
            # 用标量替代张量作为nan_to_num的参数
            gf = torch.nan_to_num(gf, nan=0.0, posinf=1.0, neginf=0.0)
            # 手动处理任何仍然存在的无效值
            gf = torch.where(torch.isnan(gf) | torch.isinf(gf), x, gf)

            # 提取细节
            detail = x - gf

            # 估计噪声幅度 - 保持原始代码风格
            off = offset.view(-1, 1, 1, 1)
            gn = gain.view(-1, 1, 1, 1)
            denom = (x + off).clamp(min=1e-5)
            noise_est = detail.abs() * (gn / denom)
            noise_est = noise_est.clamp(0.0, 10.0)

            # 构建反转和指数化的噪声掩码
            t = tau.view(-1, 1, 1, 1)
            exp_i = -(noise_est / t) ** 2
            exp_i = exp_i.clamp(min=-88.0, max=0.0)
            raw_mask = 1.0 - torch.exp(exp_i)
            noise_mask = raw_mask ** 2  # 强调更强的边缘

            # 计算软阈值 - 使用sigmoid函数
            k_pos = torch.abs(self.k).clamp(min=1.0)  # 确保k为正值且不为零
            detail_diff = detail.abs() - self.noise_thresh
            detail_mask = torch.sigmoid(k_pos * detail_diff)

            # 计算Skip Gate - 当噪声估计或细节过小时
            avg_noise = noise_est.mean(dim=[1, 2, 3], keepdim=True)
            avg_detail = detail.abs().mean(dim=[1, 2, 3], keepdim=True)
            should_skip = (avg_noise < self.skip_thresh) | (avg_detail < self.skip_thresh)
            # 移除这行：should_skip = should_skip.float()
            # 保持should_skip为布尔类型

            # 只锐化真实结构，应用Skip Gate
            lam = lam.view(-1, 1, 1, 1)
            sharpening_term = lam * detail * noise_mask * detail_mask
            sharpened = torch.where(
                should_skip.expand_as(bf),  # 直接使用布尔类型的should_skip
                x,  # 若应跳过，则直接使用原图
                bf + sharpening_term  # 否则进行锐化
            )

            # 确保输出在极小值到1之间
            min_value = 1e-5
            sharpened = torch.clamp(sharpened, min_value, 1.0)

            # 重新整形回原始形状 - 保持原有逻辑
            out = sharpened.permute(0, 2, 3, 1).view(B, P, H, W, C)

            return out
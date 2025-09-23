import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from kornia.filters import gaussian_blur2d, bilateral_blur
import torch.cuda.amp as amp  # 引入混合精度支持
import math


# ISP.py

def gamma_correct(L, gamma_increment, epsilon=1e-5):
    """
    全图 γ 校正：
    对整张图像应用统一的gamma值进行校正

    参数:
        L: 输入图像, shape = (B, H, W, 3)
        gamma_increment: γ 增量, shape = (B, 1, 1)
    返回:
        增强后的图像, shape = (B, H, W, 3)
    """
    # 1. 计算最终 γ
    final_gamma = 1.0 + gamma_increment  # [B, 1, 1]

    # 2. 扩展到 [B, 1, 1, 1] 以便广播到 [B, H, W, 3]
    gamma_exp = final_gamma.view(-1, 1, 1, 1)  # [B, 1, 1, 1]

    # 3. 应用幂运算并截断
    R = torch.pow(L, gamma_exp)
    return torch.clamp(R, min=epsilon, max=1.0)


class ColorTransform(nn.Module):
    """
    SOTA无约束颜色变换模块
    核心思想：依赖图像质量损失来约束，而不是硬约束矩阵参数
    """

    def __init__(self, residual_scale=0.2, use_residual=True):
        """
        初始化颜色变换器
        参数:
            residual_scale: 残差连接的缩放因子
            use_residual: 是否使用残差连接
        """
        super(ColorTransform, self).__init__()
        self.residual_scale = residual_scale
        self.use_residual = use_residual

        # 统计信息（用于调试）
        self.register_buffer('matrix_stats', torch.zeros(4))  # [mean, std, min, max]
        self.register_buffer('det_stats', torch.zeros(4))     # [mean, std, min, max]

    def forward(self, I_source, pred_transform_params):
        """
        前向传播：应用颜色变换

        参数:
            I_source: [B, H, W, 3] 输入图像
            pred_transform_params: [B, H, W, 9] 像素级变换参数

        返回:
            I_enhanced: [B, H, W, 3] 增强后的图像
        """
        with amp.autocast(enabled=torch.cuda.is_available()):
            B, H, W, C = I_source.shape

            # --------- 构建变换矩阵 ---------
            if self.use_residual:
                # 方案1：残差学习 (推荐)
                delta = pred_transform_params.view(B, H, W, 3, 3)
                identity = torch.eye(3, device=delta.device, dtype=delta.dtype).expand_as(delta)
                M = identity + self.residual_scale * delta
            else:
                # 方案2：直接预测 (更激进)
                M = pred_transform_params.view(B, H, W, 3, 3)

            # --------- 最小数值稳定性保护 ---------
            # 只做最基本的clamp，避免梯度阻断
            M = torch.clamp(M, -3.0, 3.0)

            # --------- 应用变换 ---------
            # 像素级变换：每个像素应用独立的3x3矩阵
            # I_source: [B, H, W, 3] -> [B, H, W, 1, 3]
            # M: [B, H, W, 3, 3]
            # 结果: [B, H, W, 1, 3] @ [B, H, W, 3, 3] -> [B, H, W, 1, 3]
            I_enhanced = torch.matmul(I_source.unsqueeze(-2), M).squeeze(-2)

            # --------- NaN/Inf 处理 ---------
            mask_bad = torch.isnan(I_enhanced) | torch.isinf(I_enhanced)
            if mask_bad.any():
                I_enhanced = torch.where(mask_bad, I_source, I_enhanced)

            # --------- 输出约束 ---------
            I_enhanced = torch.clamp(I_enhanced, 1e-5, 1.0)

            return I_enhanced

    def get_matrix_stats(self):
        """获取矩阵统计信息（用于调试）"""
        return {
            'matrix': {
                'mean': self.matrix_stats[0].item(),
                'std': self.matrix_stats[1].item(),
                'min': self.matrix_stats[2].item(),
                'max': self.matrix_stats[3].item()
            },
            'determinant': {
                'mean': self.det_stats[0].item(),
                'std': self.det_stats[1].item(),
                'min': self.det_stats[2].item(),
                'max': self.det_stats[3].item()
            }
        }


class DenoisingSharpening(nn.Module):
    """
    Fixed bilateral denoising:
    - 使用固定的 sigma_s=0.3, sigma_r=0.02
    - 不再接受预测参数输入
    - 仅使用双边滤波去噪，不做高斯分离或任何锐化
    """

    def __init__(
            self,
            kernel_size: int = 5,
            sigma_s: float = 1.0,
            sigma_r: float = 0.04
    ):
        super().__init__()
        self.kernel_size = kernel_size

        # 使用固定的sigma值
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r

    def forward(self, images: torch.Tensor, params: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            images: [B, H, W, 3]  浮点值在 [0,1]
            params: 忽略此参数，为了保持接口兼容性
        返回:
            out: [B, H, W, 3]      双边滤波去噪后的结果
        """
        with amp.autocast(enabled=torch.cuda.is_available()):
            B, H, W, C = images.shape

            # permute to [B,3,H,W]
            x = images.permute(0, 3, 1, 2)  # [B,3,H,W]

            # 使用固定的sigma值
            N = x.shape[0]
            sigma_s = torch.full((N,), self.sigma_s, device=x.device, dtype=x.dtype)
            sigma_r = torch.full((N,), self.sigma_r, device=x.device, dtype=x.dtype)

            # bilateral blur → out_x
            sigma_space = torch.stack([sigma_s, sigma_s], dim=1)
            out_x = bilateral_blur(
                x,
                kernel_size=(self.kernel_size, self.kernel_size),
                sigma_color=sigma_r,
                sigma_space=sigma_space,
                border_type="reflect",
            )

            # clamp & reshape back
            out_x = out_x.clamp(0.0, 1.0)
            out = out_x.permute(0, 2, 3, 1)  # [B, H, W, 3]
            return out

def smooth_abs(x, beta=0.1):
    """平滑的绝对值函数，在0附近可微"""
    return torch.where(torch.abs(x) < beta,
                      0.5 * x**2 / beta,
                      torch.abs(x))


def contrast_enhancement(image, alpha):
    """
    改进的饱和度增强函数 - 使用tanh映射，对负值友好

    参数:
        image: [B, H, W, 3] RGB格式图像，取值范围[0, 1]
        alpha: [B, 1] 饱和度增强参数

    返回:
        enhanced_image: [B, 3, H, W] 增强后的RGB图像，取值范围[1e-8, 1]
    """
    # 输入格式转换: [B, H, W, 3] -> [B, 3, H, W]
    if image.dim() == 4 and image.size(-1) == 3:
        image = image.permute(0, 3, 1, 2)

    # 确保输入格式正确
    assert image.dim() == 4 and image.size(1) == 3, "图像应为[B, 3, H, W]格式"
    assert alpha.dim() == 2 and alpha.size(1) == 1, "alpha应为[B, 1]格式"

    B, C, H, W = image.shape

    # 改进的饱和度因子计算：使用tanh映射，针对低光照优化
    # alpha=0时，tanh(0)=0，saturation_factor=1.8（默认80%增强）
    # alpha=1时，tanh(1)≈0.76，saturation_factor≈2.33
    # alpha=-1时，tanh(-1)≈-0.76，saturation_factor≈1.27
    #print(f"alpha - Mean: {alpha.mean().item():.4f}, Max: {alpha.max().item():.4f}, Min: {alpha.min().item():.4f}")

    # 使用网络预测的alpha参数进行动态饱和度调整
    base_factor = 1.5
    range_factor = 0.4
    tanh_alpha = torch.tanh(alpha)
    saturation_factor = base_factor + tanh_alpha * range_factor

    #print(f"tanh_alpha: mean={tanh_alpha.mean():.4f}, max={tanh_alpha.max():.4f}, min={tanh_alpha.min():.4f}")
    #print(f"saturation_factor: mean={saturation_factor.mean():.4f}, max={saturation_factor.max():.4f}, min={saturation_factor.min():.4f}")

    # 扩展饱和度因子的维度以匹配图像
    saturation_factor = saturation_factor.view(B, 1, 1, 1)  # [B, 1, 1, 1]

    # RGB转HSV，使用kornia库
    hsv = kornia.color.rgb_to_hsv(image)  # [B, 3, H, W]

    # 提取H, S, V通道
    h = hsv[:, 0:1]  # [B, 1, H, W] 色相
    s = hsv[:, 1:2]  # [B, 1, H, W] 饱和度
    v = hsv[:, 2:3]  # [B, 1, H, W] 明度

    # 在饱和度维度应用增强因子
    s_enhanced = s * saturation_factor

    # 将饱和度clamp到[0, 1]
    s_enhanced = torch.clamp(s_enhanced, 0, 1)

    # 重新组合HSV
    hsv_enhanced = torch.cat([h, s_enhanced, v], dim=1)  # [B, 3, H, W]

    # HSV转RGB，使用kornia库
    rgb_enhanced = kornia.color.hsv_to_rgb(hsv_enhanced)

    # 最终clamp到[1e-8, 1]
    rgb_enhanced = torch.clamp(rgb_enhanced, 1e-8, 1)

    return rgb_enhanced

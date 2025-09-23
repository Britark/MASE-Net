import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os
from pytorch_msssim import ssim

class L1ReconstructionLoss(nn.Module):
    """
    以下的patch实际上在网络中为window，但为了网络的即插即用仍然称之为patch

    L1 重建损失类，计算增强后图像与原始图像之间的 L1 距离。
    """

    def __init__(self, reduction='mean'):
        """
        初始化 L1 重建损失。

        参数:
        - reduction: 指定损失聚合方式，可选 'none', 'mean', 'sum'。默认为 'mean'。
        """
        super(L1ReconstructionLoss, self).__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, enhanced_image, original_image):
        """
        计算增强图像与原始图像之间的 L1 损失。

        参数:
        - enhanced_image: 增强后的图像，形状为 [batches, 3, H, W]
        - original_image: 原始图像，形状为 [batches, 3, H, W]

        返回:
        - loss: L1 损失值
        """
        return self.loss_fn(enhanced_image, original_image)


class PerceptualLoss(nn.Module):
    """
    基于VGG的感知损失
    使用预训练VGG网络的多层特征计算感知差异
    """

    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss) / len(loss)

class EdgeLoss(nn.Module):
    """
    曝光／边缘损失 L_e
    计算预测与目标在梯度（边缘）空间的差异，用 L1 形式：
      L_e = |∇X - ∇Y|_1
    其中 ∇ 由 Sobel 卷积近似。
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

        # Sobel 卷积核
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], dtype=torch.float32)
        ky = kx.t()
        # shape [1,1,3,3]
        self.register_buffer('sobel_x', kx.view(1,1,3,3))
        self.register_buffer('sobel_y', ky.view(1,1,3,3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   预测图 [B, C, H, W], 范围 [0,1]
            target: 目标图 [B, C, H, W], 范围 [0,1]
        Returns:
            边缘差异 L1 损失
        """
        # 先把多通道合并为单通道灰度：L = 0.299R+0.587G+0.114B
        def to_gray(x):
            if x.shape[1] == 3:
                w = torch.tensor([0.299,0.587,0.114], device=x.device).view(1,3,1,1)
                return (x * w).sum(dim=1, keepdim=True)
            return x

        p = to_gray(pred)
        t = to_gray(target)

        # Sobel 梯度
        gx_p = F.conv2d(p, self.sobel_x, padding=1)
        gy_p = F.conv2d(p, self.sobel_y, padding=1)
        gx_t = F.conv2d(t, self.sobel_x, padding=1)
        gy_t = F.conv2d(t, self.sobel_y, padding=1)

        # L1 差异
        loss = F.l1_loss(gx_p, gx_t, reduction='none') + F.l1_loss(gy_p, gy_t, reduction='none')
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'


class DetailLoss(nn.Module):
    """
    细节（structure）损失 L_d
    基于 SSIM（结构相似性），loss = 1 - SSIM
    需要安装 pytorch_msssim: pip install pytorch-msssim
    """
    def __init__(self, data_range=1.0, size_average=True):
        super().__init__()
        self.ssim = ssim
        self.data_range = data_range
        self.size_average = size_average

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   预测图 [B, C, H, W], 范围 [0,1]
            target: 目标图 [B, C, H, W], 范围 [0,1]
        Returns:
            细节损失 = 1 - SSIM(pred, target)
        """
        # pytorch_msssim.ssim 返回相似度 [0,1]
        sim = self.ssim(pred, target,
                        data_range=self.data_range,
                        size_average=self.size_average)
        return 1.0 - sim


class CharbonnierLoss(nn.Module):
    """
    Charbonnier损失函数
    鲁棒的重建损失，对outlier不敏感，特别适合图像恢复任务
    L_Charbonnier = √((pred - target)² + ε²)
    """
    def __init__(self, epsilon=1e-3, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   预测图像 [B, C, H, W], 范围 [0,1]
            target: 目标图像 [B, C, H, W], 范围 [0,1]
        Returns:
            Charbonnier损失值
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LABColorLoss(nn.Module):
    """
    LAB色彩空间损失
    专门约束色彩保真度，防止颜色偏移
    """
    def __init__(self, reduction='mean', color_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.color_weight = color_weight
        
    def rgb_to_lab(self, rgb):
        """
        RGB到LAB色彩空间转换
        rgb: [B, 3, H, W], 范围[0,1]
        """
        # 转换为线性RGB
        def srgb_to_linear(c):
            return torch.where(c <= 0.04045, c / 12.92, torch.pow((c + 0.055) / 1.055, 2.4))
        
        linear_rgb = srgb_to_linear(rgb)
        
        # RGB to XYZ (使用D65白点) - 安全创建转换矩阵
        try:
            rgb_to_xyz = torch.tensor([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ], device=rgb.device, dtype=rgb.dtype, requires_grad=False)
        except Exception:
            # 如果张量创建失败，返回原始输入
            return rgb
        
        # 安全的矩阵乘法
        try:
            B, C, H, W = linear_rgb.shape
            if H * W == 0:  # 检查空间维度
                return torch.zeros_like(rgb)
                
            linear_rgb_flat = linear_rgb.view(B, C, H * W)  # [B, 3, H*W]
            
            # 确保矩阵维度匹配
            if rgb_to_xyz.shape != (3, 3) or linear_rgb_flat.shape[1] != 3:
                return torch.zeros_like(rgb)
                
            xyz_flat = torch.bmm(rgb_to_xyz.unsqueeze(0).expand(B, -1, -1), linear_rgb_flat)
            xyz = xyz_flat.view(B, 3, H, W)
            
            # 安全的归一化到D65白点
            d65_norm = torch.tensor([0.95047, 1.0, 1.08883], device=rgb.device, dtype=rgb.dtype, requires_grad=False).view(1, 3, 1, 1)
            xyz = xyz / d65_norm
            
        except Exception:
            # 如果矩阵运算失败，返回零张量
            return torch.zeros_like(rgb)
        
        # 安全的XYZ to LAB转换
        try:
            # 数值稳定性检查
            xyz = torch.clamp(xyz, min=1e-8, max=10.0)
            
            def f(t):
                # 标准CIE LAB转换函数，增加数值稳定性
                t_safe = torch.clamp(t, min=1e-8)
                return torch.where(t_safe > 0.008856, torch.pow(t_safe, 1/3), (7.787 * t_safe) + (16/116))
            
            # 安全的通道索引
            if xyz.shape[1] < 3:
                return torch.zeros_like(rgb)
                
            fx = f(xyz[:, 0:1])  # f(X/Xn)
            fy = f(xyz[:, 1:2])  # f(Y/Yn)  
            fz = f(xyz[:, 2:3])  # f(Z/Zn)
            
            L = 116 * fy - 16           # L* = 116*f(Y/Yn) - 16
            a = 500 * (fx - fy)         # a* = 500*(f(X/Xn) - f(Y/Yn))
            b = 200 * (fy - fz)         # b* = 200*(f(Y/Yn) - f(Z/Zn))
            
            # 确保LAB值在合理范围内
            L = torch.clamp(L, 0, 100)
            a = torch.clamp(a, -128, 128)
            b = torch.clamp(b, -128, 128)
            
            result = torch.cat([L, a, b], dim=1)
            
            # 最终检查
            if result.shape != rgb.shape:
                return torch.zeros_like(rgb)
                
            return result
            
        except Exception:
            # 如果LAB转换失败，返回零张量
            return torch.zeros_like(rgb)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   预测图 [B, C, H, W], 范围 [0,1]
            target: 目标图 [B, C, H, W], 范围 [0,1]
        Returns:
            LAB色彩损失，主要约束a*b*通道
        """
        pred_lab = self.rgb_to_lab(pred)
        target_lab = self.rgb_to_lab(target)
        
        # 分离L和ab通道
        pred_a, pred_b = pred_lab[:, 1:2], pred_lab[:, 2:3]
        target_a, target_b = target_lab[:, 1:2], target_lab[:, 2:3]
        
        # 主要约束ab通道(色彩信息)
        color_loss = F.l1_loss(pred_a, target_a, reduction='none') + F.l1_loss(pred_b, target_b, reduction='none')
        
        if self.reduction == 'mean':
            return self.color_weight * color_loss.mean()
        elif self.reduction == 'sum':
            return self.color_weight * color_loss.sum()
        else:
            return self.color_weight * color_loss


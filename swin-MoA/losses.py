import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import torchvision.transforms as transforms
import os
"""
    limeloss方法没用到
"""
class LimeLoss(nn.Module):
    """
    光照图损失的模块化实现
    """

    def __init__(self, empty_threshold=1e-6):
        """
        参数:
            empty_threshold: 判断patch是否为空的阈值
        """
        super(LimeLoss, self).__init__()
        self.empty_threshold = empty_threshold
        # 下面两个属性会在 set_data() 中被赋值
        self.T_pred = None  # 形状 (bsz, patches, h, w)
        self.Lc     = None  # 原始低光图，形状 (bsz, patches, h, w, 3)
        self.valid_indices = None # (N, 2)

    def set_data(self, T_pred, Lc):
        """
        设置当前损失所需的数据。
        参数:
            T_pred: 网络预测的光照图, 形状 (bsz, patches, h, w)
            Lc:     对应的原始低光图 patch，对应形状 (bsz, patches, h, w, 3)
        """
        self.T_pred = T_pred
        self.Lc      = Lc

    def filter_valid_patches(self):
        """
        识别并过滤有效的 patch（平均值 > 阈值），并保存它们的索引。
        返回:
            valid_patches: shape (N, h, w)
        """
        bsz, num_patches, h, w = self.T_pred.shape

        # 计算每个patch的平均像素值
        patch_mean = self.T_pred.view(bsz, num_patches, -1).mean(dim=-1)  # (bsz, patches)

        # 判断哪些 patch 有效
        is_valid = patch_mean > self.empty_threshold  # (bsz, patches)

        # 获取所有有效位置的索引
        valid_indices = torch.nonzero(is_valid, as_tuple=False)  # (N, 2)
        self.valid_indices = valid_indices  # 保存下来以便 fidelity_loss 使用

        batch_idx = valid_indices[:, 0]
        patch_idx = valid_indices[:, 1]
        valid_patches = self.T_pred[batch_idx, patch_idx]  # (N, h, w)

        return valid_patches

    def fidelity_loss(self, valid_patches):
        """
        计算保真项: E = mean_{patch i} || T̂_i - T_{θ,i} ||_F^2
        其中 T̂_i = max_c Lc_i[:,:,c]
        参数:
            valid_patches: shape (N, h, w)，等于 T_{θ,i}
        返回:
            scalar 张量，保真项损失（已经对所有 patch 平均）
        """
        # 预测光照
        T_theta = valid_patches  # (N, h, w)

        # 从原始 Lc 中提取相同索引的 patch
        batch_idx, patch_idx = self.valid_indices[:,0], self.valid_indices[:,1]
        Lc_patches = self.Lc[batch_idx, patch_idx]  # (N, h, w, 3)

        # 计算 T_hat = max_c Lc
        # Lc_patches[..., c] 对应三个通道
        T_hat = Lc_patches.max(dim=-1).values  # (N, h, w)

        # 逐 patch 计算 Frobenius 范数平方
        # 先 element-wise 差，再平方，再对 h,w 求和，最后对 patch 求平均
        per_patch_err = (T_hat - T_theta).pow(2).sum(dim=[1,2])  # (N,)

        # 平均到所有有效 patch
        loss = per_patch_err.mean()

        return loss

    def structure_loss(self, valid_patches, valid_idx, Lc):
        """
        计算结构平滑项（策略III）：
        α ∑_x ∑_d W_d(x) [∇_d T_theta(x)]^2 / (|∇_d T_hat(x)| + eps)
        """
        # 超参数
        alpha = 0.15
        sigma = 2.0
        eps = 1e-6

        # 提取对应 Lc patch 并计算 T_hat
        b_idx, p_idx = valid_idx.unbind(1)
        Lc_patches = Lc[b_idx, p_idx]  # (N, h, w, 3)
        T_hat = Lc_patches.max(dim=-1).values  # (N, h, w)
        T_theta = valid_patches  # (N, h, w)

        # 计算原始梯度 ∇_h 和 ∇_v (修正方向)
        # 水平梯度 (x方向)
        grad_h_hat = F.pad(T_hat[:, :, 1:] - T_hat[:, :, :-1], (0, 1, 0, 0), mode='constant', value=0)
        # 垂直梯度 (y方向)
        grad_v_hat = F.pad(T_hat[:, 1:, :] - T_hat[:, :-1, :], (0, 0, 0, 1), mode='constant', value=0)

        # 计算预测的梯度
        grad_h_theta = F.pad(T_theta[:, :, 1:] - T_theta[:, :, :-1], (0, 1, 0, 0), mode='constant', value=0)
        grad_v_theta = F.pad(T_theta[:, 1:, :] - T_theta[:, :-1, :], (0, 0, 0, 1), mode='constant', value=0)

        # 构造 Gaussian 核
        k = int(math.ceil(3 * sigma)) * 2 + 1  # 确保核大小是奇数
        r = k // 2
        coords = torch.arange(k, device=T_theta.device, dtype=T_theta.dtype) - r
        x = coords.view(1, -1).repeat(k, 1)
        y = coords.view(-1, 1).repeat(1, k)
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # 归一化
        kernel = kernel.view(1, 1, k, k)

        # 计算分母：高斯加权梯度绝对值
        abs_grad_h_hat = torch.abs(grad_h_hat).unsqueeze(1)  # (N, 1, h, w)
        abs_grad_v_hat = torch.abs(grad_v_hat).unsqueeze(1)  # (N, 1, h, w)

        # 使用卷积计算高斯加权和
        denom_h = F.conv2d(abs_grad_h_hat, kernel, padding=r).squeeze(1)  # (N, h, w)
        denom_v = F.conv2d(abs_grad_v_hat, kernel, padding=r).squeeze(1)  # (N, h, w)

        # 计算权重 W_h 和 W_v
        # 因为kernel已归一化，所以gauss_sum=1，但为了代码清晰度，显式计算
        gauss_sum = kernel.sum()
        W_h = gauss_sum / (denom_h + eps)
        W_v = gauss_sum / (denom_v + eps)

        # 计算最终正则化项 - 去掉多余的除法
        term_h = W_h * grad_h_theta.pow(2)
        term_v = W_v * grad_v_theta.pow(2)

        # 计算全部正则化损失
        reg = term_h + term_v

        # 按 patch 平均后，乘以 α
        loss = reg.sum(dim=[1, 2]).mean() * alpha

        return loss

    def forward(self, T_pred, Lc):
        """
        计算总损失：保真项 + 结构保持项

        参数:
            T_pred: 网络预测的光照图, 形状 (bsz, patches, h, w)
            Lc: 对应的原始低光图 patch，对应形状 (bsz, patches, h, w, 3)

        返回:
            scalar 张量，总损失
        """
        # 设置数据
        self.set_data(T_pred, Lc)

        # 过滤有效patchesmain
        valid_patches = self.filter_valid_patches()
        if valid_patches.shape[0] == 0:
            return torch.tensor(0.0, device=T_pred.device)

        # 计算保真项
        fidelity = self.fidelity_loss(valid_patches)

        # 计算结构保持项
        structure = self.structure_loss(valid_patches, self.valid_indices, self.Lc)

        # 总损失
        lime_loss = fidelity + structure

        return lime_loss


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

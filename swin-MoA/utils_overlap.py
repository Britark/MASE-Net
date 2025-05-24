import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np


class WindowOverlapProcessor(nn.Module):
    """处理重叠窗口的重建和混合"""

    def __init__(self, win_size, blend_mode='gaussian'):
        super().__init__()
        self.win_size = win_size
        self.blend_mode = blend_mode

        # 预计算混合权重
        if blend_mode == 'gaussian':
            self.blend_weights = self._create_gaussian_weights()
        elif blend_mode == 'linear':
            self.blend_weights = self._create_linear_weights()
        else:
            self.blend_weights = None

    def _create_gaussian_weights(self):
        """创建高斯权重矩阵用于窗口混合"""
        # 创建1D高斯核
        sigma = self.win_size / 4.0  # 标准差
        x = torch.arange(self.win_size, dtype=torch.float32)
        x = x - (self.win_size - 1) / 2.0
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        # 创建2D高斯权重
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        return gauss_2d

    def _create_linear_weights(self):
        """创建线性权重矩阵用于窗口混合"""
        # 从中心到边缘线性衰减
        weights = torch.ones(self.win_size, self.win_size, dtype=torch.float32)

        # 计算每个位置到边缘的最小距离
        for i in range(self.win_size):
            for j in range(self.win_size):
                dist_to_edge = min(i, j, self.win_size - 1 - i, self.win_size - 1 - j)
                weights[i, j] = (dist_to_edge + 1) / (self.win_size // 2 + 1)

        return weights

    def reconstruct_from_windows(self, windows, window_coords, h_windows, w_windows,
                                 stride, original_h, original_w, channels=3):
        """
        从重叠窗口重建完整图像

        Args:
            windows: 处理后的窗口, 形状 [B, num_windows, window_size, window_size, C]
            window_coords: 每个窗口的起始坐标
            h_windows, w_windows: 窗口网格大小
            stride: 窗口步长
            original_h, original_w: 原始图像的patch数
            channels: 通道数

        Returns:
            reconstructed: 重建的图像, 形状 [B, C, H, W]
        """
        with amp.autocast(enabled=torch.cuda.is_available()):
            B = windows.shape[0]
            device = windows.device

            # 确保blend_weights在正确的设备上
            if self.blend_weights is not None:
                self.blend_weights = self.blend_weights.to(device)

            # 创建输出张量和权重累积张量
            reconstructed = torch.zeros(B, channels, original_h, original_w, device=device)
            weight_sum = torch.zeros(B, 1, original_h, original_w, device=device)

            # 处理每个窗口
            for idx, (h_start, w_start) in enumerate(window_coords):
                # 获取当前窗口
                window = windows[:, idx]  # [B, window_size, window_size, C]

                # 转换为 [B, C, window_size, window_size]
                window = window.permute(0, 3, 1, 2)

                # 计算窗口在输出中的位置
                h_end = min(h_start + self.win_size, original_h)
                w_end = min(w_start + self.win_size, original_w)

                # 处理边界情况（窗口可能超出图像边界）
                h_size = h_end - h_start
                w_size = w_end - w_start

                # 获取混合权重
                if self.blend_weights is not None:
                    weights = self.blend_weights[:h_size, :w_size].unsqueeze(0).unsqueeze(0)
                else:
                    weights = torch.ones(1, 1, h_size, w_size, device=device)

                # 累加窗口内容和权重
                reconstructed[:, :, h_start:h_end, w_start:w_end] += window[:, :, :h_size, :w_size] * weights
                weight_sum[:, :, h_start:h_end, w_start:w_end] += weights

            # 归一化（避免除零）
            reconstructed = reconstructed / (weight_sum + 1e-8)

            return reconstructed

    def apply_random_shift(self, features, max_shift):
        """
        训练时对特征图应用随机偏移

        Args:
            features: 输入特征图 [B, C, H, W]
            max_shift: 最大偏移量（patch数）

        Returns:
            shifted_features: 偏移后的特征图
        """
        if max_shift == 0:
            return features

        B, C, H, W = features.shape

        # 生成随机偏移量
        shift_h = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        shift_w = torch.randint(-max_shift, max_shift + 1, (1,)).item()

        # 使用padding和cropping实现偏移
        if shift_h != 0 or shift_w != 0:
            # 计算padding
            pad_top = max(0, -shift_h)
            pad_bottom = max(0, shift_h)
            pad_left = max(0, -shift_w)
            pad_right = max(0, shift_w)

            # 应用padding
            features = F.pad(features, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

            # 裁剪到原始大小
            features = features[:, :, pad_top:pad_top + H, pad_left:pad_left + W]

        return features


def merge_overlapping_params(isp_params_list, window_coords, h_windows, w_windows,
                             win_size, stride, original_h, original_w, blend_mode='gaussian'):
    """
    合并重叠窗口的ISP参数

    Args:
        isp_params_list: ISP参数列表，每个元素形状 [B, num_windows, param_dim]
        window_coords: 窗口坐标
        其他参数同上

    Returns:
        merged_params_list: 合并后的ISP参数列表
    """
    processor = WindowOverlapProcessor(win_size, blend_mode)
    merged_params_list = []

    for isp_params in isp_params_list:
        B, num_windows, param_dim = isp_params.shape

        # 如果参数是矩阵形式（如色彩校正矩阵），需要特殊处理
        if param_dim == 9:  # 3x3矩阵
            # 重塑为 [B, num_windows, 3, 3]
            isp_params = isp_params.reshape(B, num_windows, 3, 3)
            # 展平为向量处理
            isp_params_flat = isp_params.reshape(B, num_windows, 9)

            # 为每个参数维度创建窗口
            merged_flat = []
            for i in range(9):
                param_windows = isp_params_flat[:, :, i].unsqueeze(-1)  # [B, num_windows, 1]
                param_windows = param_windows.unsqueeze(2).expand(-1, -1, win_size * win_size, -1)
                param_windows = param_windows.reshape(B, num_windows, win_size, win_size, 1)

                # 重建
                merged = processor.reconstruct_from_windows(
                    param_windows, window_coords, h_windows, w_windows,
                    stride, original_h, original_w, channels=1
                )
                merged_flat.append(merged)

            # 合并所有参数维度
            merged = torch.cat(merged_flat, dim=1)  # [B, 9, H, W]
            # 转换回原始形状
            merged = merged.permute(0, 2, 3, 1).reshape(B, original_h * original_w, 3, 3)

        else:
            # 对于其他参数，直接处理每个维度
            merged_list = []
            for i in range(param_dim):
                param_windows = isp_params[:, :, i].unsqueeze(-1)  # [B, num_windows, 1]
                param_windows = param_windows.unsqueeze(2).expand(-1, -1, win_size * win_size, -1)
                param_windows = param_windows.reshape(B, num_windows, win_size, win_size, 1)

                # 重建
                merged = processor.reconstruct_from_windows(
                    param_windows, window_coords, h_windows, w_windows,
                    stride, original_h, original_w, channels=1
                )
                merged_list.append(merged)

            # 合并所有参数维度
            merged = torch.cat(merged_list, dim=1)  # [B, param_dim, H, W]
            # 转换为 [B, H*W, param_dim]
            merged = merged.permute(0, 2, 3, 1).reshape(B, original_h * original_w, param_dim)

        merged_params_list.append(merged)

    return merged_params_list
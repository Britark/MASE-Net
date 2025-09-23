import torch
import torch.nn as nn
import torch.cuda.amp as amp  # 导入混合精度训练支持
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, patch_size=3, in_channels=3, embed_dim=48, win_size=4):
        super().__init__()
        # 直接使用一层卷积进行局部特征提取 (保持空间尺寸不变)
        self.local_feature = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, stride=1),
            nn.GELU()  # 使用GELU替代LeakyReLU
        )

        # Patch嵌入
        self.patch_embed = nn.Conv2d(
            in_channels=48,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 添加1×1卷积用于特征增强
        self.enhance = nn.Sequential(
            nn.GroupNorm(8, embed_dim),  # 归一化层
            nn.GELU(),  # 使用GELU替代LeakyReLU
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)  # 1×1卷积
        )

        # 窗口大小参数
        self.win_size = win_size

        # 使用CUDA融合优化器
        if torch.cuda.is_available():
            # 使这些模块使用CUDA优化的实现
            self.local_feature = torch.jit.script(self.local_feature)
            self.enhance = torch.jit.script(self.enhance)

    def forward(self, x, overlap_stride=None, return_coords=False):
        # 使用混合精度计算以提高速度和内存效率
        with amp.autocast(enabled=torch.cuda.is_available()):
            # 局部特征提取 (保持空间尺寸)
            x = self.local_feature(x)  # [B, 48, H, W]

            # Patch嵌入
            x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]

            # 特征增强
            x = self.enhance(x)  # [B, embed_dim, H/patch_size, W/patch_size]

            # 窗口分割
            if overlap_stride is not None:
                # 使用重叠窗口
                windows, h_windows, w_windows, window_coords = self.window_partition_overlap(x, overlap_stride)
                if return_coords:
                    return windows, h_windows, w_windows, window_coords
                else:
                    return windows, h_windows, w_windows
            else:
                # 使用原始的非重叠窗口
                windows, h_windows, w_windows = self.window_partition(x)
                return windows, h_windows, w_windows

    def window_partition(self, x):
        """
        将特征图分割成固定大小的窗口并重新排列
        Args:
            x: 输入特征图, 形状为 [B, C, H, W]
        Returns:
            windows: 窗口特征, 形状为 [B, num_windows, win_size*win_size, C]
        """
        B, C, H, W = x.shape
        assert H % self.win_size == 0 and W % self.win_size == 0
        embed_dim = C
        # 假设输入特征图的尺寸已经能够被win_size整除

        # 重排张量形状，便于窗口切分
        # [B, C, H, W] -> [B, C, H//win_size, win_size, W//win_size, win_size]
        # 这里使用reshape是安全的，因为我们没有改变内存中元素的顺序
        x = x.reshape(B, C, H // self.win_size, self.win_size, W // self.win_size, self.win_size)

        # 转置并重塑以获得所需的窗口表示
        # [B, C, H//win_size, win_size, W//win_size, win_size] ->
        # [B, H//win_size, W//win_size, win_size, win_size, C]
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # 添加contiguous确保内存连续

        # 计算窗口数量
        num_windows = (H // self.win_size) * (W // self.win_size)
        tokens = self.win_size * self.win_size

        # 在进行view操作前，确保张量内存是连续的
        # [B, H//win_size, W//win_size, win_size, win_size, C] ->
        # [B, num_windows, win_size*win_size, C]
        windows = x.view(B, num_windows, tokens, embed_dim)

        return windows, H // self.win_size, W // self.win_size  # 长宽都是多少个windows

    def window_partition_overlap(self, x, stride):
        """
        使用滑动窗口将特征图分割成重叠的窗口
        Args:
            x: 输入特征图, 形状为 [B, C, H, W]
            stride: 窗口滑动步长
        Returns:
            windows: 窗口特征, 形状为 [B, num_windows, win_size*win_size, C]
            h_windows: 高度方向的窗口数
            w_windows: 宽度方向的窗口数
            window_coords: 每个窗口的坐标信息 [(h_start, w_start), ...]
        """
        B, C, H, W = x.shape
        embed_dim = C

        # 使用unfold实现滑动窗口
        # unfold可以在指定维度上创建滑动窗口
        # 先在H维度上展开
        x_unfold = x.unfold(2, self.win_size, stride)  # [B, C, H_windows, W, win_size]
        # 再在W维度上展开
        x_unfold = x_unfold.unfold(3, self.win_size, stride)  # [B, C, H_windows, W_windows, win_size, win_size]

        # 计算窗口数量
        h_windows = x_unfold.shape[2]
        w_windows = x_unfold.shape[3]
        num_windows = h_windows * w_windows

        # 重新排列维度
        # [B, C, H_windows, W_windows, win_size, win_size] ->
        # [B, H_windows, W_windows, win_size, win_size, C]
        x_unfold = x_unfold.permute(0, 2, 3, 4, 5, 1).contiguous()

        # 合并窗口维度
        # [B, H_windows, W_windows, win_size, win_size, C] ->
        # [B, num_windows, win_size*win_size, C]
        windows = x_unfold.view(B, num_windows, self.win_size * self.win_size, embed_dim)

        # 记录每个窗口的起始坐标（用于后续重建）
        window_coords = []
        for h in range(h_windows):
            for w in range(w_windows):
                h_start = h * stride
                w_start = w * stride
                window_coords.append((h_start, w_start))

        return windows, h_windows, w_windows, window_coords
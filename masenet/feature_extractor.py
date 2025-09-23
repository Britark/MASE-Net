import torch
import torch.nn as nn
import torch.cuda.amp as amp  # 导入混合精度训练支持
import math


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

        self._initialize_weights()

        # 使用CUDA融合优化器
        if torch.cuda.is_available():
            # 使这些模块使用CUDA优化的实现
            self.local_feature = torch.jit.script(self.local_feature)
            self.enhance = torch.jit.script(self.enhance)

    def _initialize_weights(self):
        """LeCun Normal初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 使用混合精度计算以提高速度和内存效率
        with amp.autocast(enabled=torch.cuda.is_available()):
            # 局部特征提取 (保持空间尺寸)
            x = self.local_feature(x)  # [B, 48, H, W]

            # Patch嵌入
            x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]

            # 特征增强
            x = self.enhance(x)  # [B, embed_dim, H/patch_size, W/patch_size]

            # 窗口分割
            x = self.window_partition(x)  # [B, num_windows, win_size*win_size, embed_dim]

        return x

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
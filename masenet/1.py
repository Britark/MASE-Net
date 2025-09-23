import torch
import torch.nn as nn
#提取图像特征
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

    def forward(self, x):
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
        x = x.reshape(B, C, H // self.win_size, self.win_size, W // self.win_size, self.win_size)

        # 转置并重塑以获得所需的窗口表示
        # [B, C, H//win_size, win_size, W//win_size, win_size] ->
        # [B, H//win_size, W//win_size, win_size, win_size, C]
        x = x.permute(0, 2, 4, 3, 5, 1)

        # 计算窗口数量
        num_windows = (H // self.win_size) * (W // self.win_size)
        tokens = self.win_size * self.win_size
        # [B, H//win_size, W//win_size, win_size, win_size, C] ->
        # [B, num_windows, win_size*win_size, C]
        windows = x.reshape(B, num_windows, tokens, embed_dim)

        return windows, H // self.win_size, W // self.win_size #长宽都是多少个windows



import torch
import torch.nn as nn
import torch.nn.functional as F


class EispGeneratorFFN(nn.Module):
    """
    将特征图通过FFN层映射为E_isp张量

    输入特征图形状: [batches, windows, tokens, input_dim]
    输出E_isp形状: [batches, windows, output_dim]

    每个patch独立处理，将patch内所有tokens映射为一个output_dim维度的向量
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        """
        初始化EispGeneratorFFN

        参数:
            input_dim (int): 输入特征的维度
            hidden_dim (int): FFN隐藏层的维度
            output_dim (int): 输出E_isp的维度
            dropout_rate (float): Dropout比率，默认0.1
        """
        super(EispGeneratorFFN, self).__init__()

        # FFN结构定义
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()  # 使用GELU激活函数
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # 最终映射到output_dim
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # 初始化网络参数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        # 使用He初始化（适用于ReLU/GELU等激活函数）
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_out.weight, nonlinearity='relu')

        # 偏置初始化为0
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        """
        前向传播

        参数:
            x (Tensor): 形状为 [batches, windows, num_tokens, input_dim] 的特征图

        返回:
            E_isp (Tensor): 形状为 [batches, windows, output_dim] 的张量
        """
        # 获取输入形状
        batches, windows, num_tokens, input_dim = x.shape

        # 重塑为 [batches*windows, num_tokens, input_dim] 便于并行处理所有patch
        x_reshaped = x.reshape(batches * windows, num_tokens, input_dim)

        # 对输入进行归一化
        x_normalized = self.layer_norm(x_reshaped)

        # 应用FFN第一层
        x = self.fc1(x_normalized)
        x = self.dropout1(x)
        x = self.activation(x)

        # 应用FFN第二层
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.activation(x)

        # 对每个windows内的所有tokens进行池化操作，可以选择平均池化或注意力池化
        # 这里使用平均池化来聚合token信息
        x_pooled = torch.mean(x, dim=1)  # [batches*windows, hidden_dim]

        # 最终映射到输出维度
        E_isp = self.fc_out(x_pooled)  # [batches*windows, output_dim]

        # 重塑回原始batch和windows维度
        E_isp = E_isp.reshape(batches, windows, -1)  # [batches, windows, output_dim]

        return E_isp


class QKIspGenerator(nn.Module):
    """
    将E_isp转换为K_isp和Q_isp

    输入E_isp形状: [batches, windows, input_dim]
    输出K_isp形状: [batches, windows, N, input_dim]
    输出Q_isp形状: [batches, windows, 1, input_dim]
    """

    def __init__(self, dim, num_heads, dropout_rate):
        """
        初始化KQGenerator

        参数:
            dim (int): 输入和输出的维度
            num_heads (int): 生成的K_isp头数量N
            dropout_rate (float): Dropout比率，默认0.1
        """
        super(QKIspGenerator, self).__init__()

        self.dim = dim
        self.num_heads = num_heads

        # K路径：N个独立的映射
        self.k_projections = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_heads)
        ])

        # Q路径：单个映射
        self.q_projection = nn.Linear(dim, dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 层归一化
        self.norm = nn.LayerNorm(dim)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        # 初始化K投影矩阵
        for k_proj in self.k_projections:
            nn.init.xavier_uniform_(k_proj.weight)
            nn.init.zeros_(k_proj.bias)

        # 初始化Q投影矩阵
        nn.init.xavier_uniform_(self.q_projection.weight)
        nn.init.zeros_(self.q_projection.bias)

    def forward(self, E_isp):
        """
        前向传播

        参数:
            E_isp (Tensor): 形状为 [batches, windows, input_dim] 的张量

        返回:
            tuple: (K_isp, Q_isp)
                K_isp (Tensor): 形状为 [batches, windows, N, input_dim] 的张量
                Q_isp (Tensor): 形状为 [batches, windows, 1, input_dim] 的张量
        """
        # 应用层归一化
        E_isp_norm = self.norm(E_isp)

        # 获取输入形状
        batches, windows, input_dim = E_isp_norm.shape

        # K路径: 生成N个不同的K_isp向量
        K_isps = []
        for k_proj in self.k_projections:
            # 将E_isp映射为一个K_isp向量
            k_isp = k_proj(E_isp_norm)  # [batches, windows, input_dim]
            k_isp = self.dropout(k_isp)  # 应用dropout
            K_isps.append(k_isp)

        # 将N个K_isp向量堆叠在一起
        K_isp = torch.stack(K_isps, dim=2)  # [batches, windows, N, input_dim]

        # Q路径: 生成一个Q_isp向量
        Q_isp = self.q_projection(E_isp_norm)  # [batches, windows, input_dim]
        Q_isp = self.dropout(Q_isp)  # 应用dropout

        # 添加维度以形成[batches, windows, 1, input_dim]的形状
        Q_isp = Q_isp.unsqueeze(2)  # [batches, windows, 1, input_dim]

        return Q_isp, K_isp

class KVFeatureGenerator(nn.Module):
    """
            前向传播

            参数:
                feature (Tensor): 形状为 [batches, windows, tokens, embed_dim] 的张量

            返回:
                tuple: (k_feature, v_feature)
                    k_feature (Tensor): 与输入feature相同，形状为 [batches, windows, tokens, embed_dim]
                    v_feature (Tensor): 通过Wv映射后的feature，形状为 [batches, windows, tokens, embed_dim]
    """

    def __init__(self, embed_dim, dropout_rate):

        super(KVFeatureGenerator, self).__init__()

        # V路径的投影矩阵
        self.v_projection = nn.Linear(embed_dim, embed_dim)
        # K路径的投影矩阵
        self.k_projection = nn.Linear(embed_dim, embed_dim)

        # 分别为K和V路径创建Dropout层
        self.k_dropout = nn.Dropout(dropout_rate)
        self.v_dropout = nn.Dropout(dropout_rate)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        # 初始化V投影矩阵
        nn.init.xavier_uniform_(self.v_projection.weight)
        nn.init.zeros_(self.v_projection.bias)
        nn.init.xavier_uniform_(self.k_projection.weight)
        nn.init.zeros_(self.k_projection.bias)

    def forward(self, feature):

        # V路径: 通过投影矩阵映射
        v_feature = self.v_projection(feature)  # [batches, windows, tokens, embed_dim]
        v_feature = self.v_dropout(v_feature)  # 应用dropout

        # V路径: 通过投影矩阵映射
        k_feature = self.k_projection(feature)  # [batches, windows, tokens, embed_dim]
        k_feature = self.k_dropout(k_feature)  # 应用dropout

        return k_feature, v_feature

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from kornia.filters import gaussian_blur2d, bilateral_blur


def patch_gamma_correct(L, gamma_increment, epsilon=1e-5):
    """
    对每个 12×12 patch 做 γ 校正，输入的gamma_increment作为初始伽马值1的增量，
    并将最终伽马值限制在 [0.5, 1.6] 之间，输出结果限制在 [ε, 1]。

    参数:
        L: 原始低光图，shape = (bsz, patches, h, w, 3)，假设没有全零情况
        gamma_increment: 每个 patch 的 γ 增量值，shape = (bsz, patches, 1)
        epsilon: 最小值阈限
    返回:
        R: 增强结果，shape = (bsz, patches, h, w, 3)
    """
    bsz, patches, h, w, c = L.shape
    # 1) 计算最终的伽马值 = 初始值(1) + 增量
    final_gamma = 1.0 + gamma_increment

    # 2) 将伽马值限制在 [0.5, 1.6] 之间
    clamped_gamma = torch.clamp(final_gamma, min=0.5, max=1.4)

    # 然后将其重塑为[bsz, patches, 1, 1, 1]
    gamma_reshaped = clamped_gamma.view(bsz, patches, 1, 1, 1)

    # 最后扩展到与L相同的形状[bsz, patches, h, w, c]
    gamma_exp = gamma_reshaped.expand(bsz, patches, h, w, c)

    # 4) 幂运算
    R = L ** gamma_exp

    # 5) 限制最终结果在 [ε, 1]
    return torch.clamp(R, min=epsilon, max=1.0)


def white_balance(predicted_gains, I_source):
    """
    应用白平衡增益到低光图像上，其中predicted_gains是初始白平衡参数[1,1,1]的增量

    参数:
        predicted_gains: 预测的白平衡增益增量, 形状为 [B, P, 3], 其中P是patches数量
        I_source: 原始低光图像, 形状为 [B, P, H, W, 3]

    返回:
        I_whitebalance: 白平衡增强后的图像, 形状为 [B, P, H, W, 3]
    """
    # 获取维度信息
    _, _, h, w, _ = I_source.shape

    # 步骤1: 计算最终的白平衡参数 = 初始参数[1,1,1] + 预测的增量
    base_gains = torch.ones_like(predicted_gains)  # 初始白平衡参数[1,1,1]
    final_gains = base_gains + predicted_gains  # 加上增量得到最终参数

    # 步骤2: 对最终白平衡参数应用特定通道的约束
    processed_gains = torch.empty_like(final_gains)

    # R通道范围：0.5 ~ 2.0 (索引0)
    processed_gains[:, :, 0] = torch.clamp(final_gains[:, :, 0], 0.8, 1.5)

    # G通道范围：0.8 ~ 1.2 (索引1)
    processed_gains[:, :, 1] = torch.clamp(final_gains[:, :, 1], 0.9, 1.1)

    # B通道范围：0.5 ~ 2.0 (索引2)
    processed_gains[:, :, 2] = torch.clamp(final_gains[:, :, 2], 0.8, 1.5)

    # 步骤3: 应用白平衡增益到原始图像
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
    对原始图像应用色彩校正矩阵 (完全向量化实现，无循环)

    参数:
        I_source: 原始低光图像, 形状为 [B, P, H, W, 3]
        pred_matrices: 预测的色彩校正矩阵增量，形状为 [B, P, 3, 3]

    返回:
        I_color_corrected: 色彩校正后的图像, 形状为 [B, P, H, W, 3]
    """
    # 获取维度信息
    bsz, num_patches, h, w, _ = I_source.shape

    # 步骤1: 计算最终的色彩校正矩阵 = 初始矩阵(单位矩阵) + 预测的增量
    # 创建单位矩阵，形状匹配pred_matrices
    identity_matrix = torch.eye(3, device=pred_matrices.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_patches, -1,
                                                                                                 -1)

    # 单位矩阵加上增量得到最终的色彩校正矩阵
    final_matrices = identity_matrix + pred_matrices

    # 步骤2: 约束CCM矩阵在合理范围内
    # 创建对角元素掩码和非对角元素掩码
    diag_mask = torch.eye(3, device=pred_matrices.device).bool().unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    non_diag_mask = ~diag_mask

    # 约束对角元素范围：0.8 ~ 1.2
    constrained_matrices = torch.where(
        diag_mask.expand_as(final_matrices),
        torch.clamp(final_matrices, 0.9, 1.1),
        final_matrices
    )

    # 约束非对角元素范围：-0.2 ~ 0.2
    constrained_matrices = torch.where(
        non_diag_mask.expand_as(constrained_matrices),
        torch.clamp(constrained_matrices, -0.15, 0.15),
        constrained_matrices
    )

    # 步骤3: 应用色彩校正矩阵到原始图像
    # 将图像从[B, P, H, W, 3]重塑为[B, P, H*W, 3]以便进行批量矩阵乘法
    I_reshaped = I_source.reshape(bsz, num_patches, h * w, 3)

    # 使用爱因斯坦求和约定进行批量矩阵乘法
    # bpij,bpjk->bpik, 其中
    # b=batch, p=patch, i=pixel, j,k=matrix indices
    I_corrected = torch.einsum('bpij,bpkj->bpik', I_reshaped, constrained_matrices)

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
    对图像patches进行降噪和锐化，具有精细的噪声控制：
    - 扩大的双边滤波核以获得更好的平滑效果
    - 反转和指数化的噪声掩码以强调真实边缘
    - 软阈值处理通过sigmoid函数平滑地进行细节提取
    - 降低lambda范围以避免过度锐化
    - 提高tau下限以使噪声抑制曲线更陡峭
    - Skip Gate机制在噪声或细节过小时跳过锐化
    """

    def __init__(self, noise_thresh: float = 2e-3, kernel_size: int = 3, k_init: float = 1000.0,
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
        B, P, H, W, C = images.shape
        # 重塑为 [N, C, H, W]
        x = images.view(-1, H, W, C).permute(0, 3, 1, 2)
        p = params.view(-1, 7)

        # 将参数限制在安全范围内
        sigma_s = p[:, 0].clamp(0.5, 3.0)
        sigma_r = p[:, 1].clamp(0.1, 0.5)
        sigma_f = p[:, 2].clamp(0.5, 2.0)
        lam = p[:, 3].clamp(0.3, 1.0)
        tau = p[:, 4].clamp(1.0, 3.0)
        gain = p[:, 5].clamp(0.5, 1.0)
        offset = p[:, 6].clamp(0.1, 0.5)

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
        # 's手动处理任何仍然存在的无效值
        gf = torch.where(torch.isnan(gf) | torch.isinf(gf), x, gf)

        # 提取细节
        detail = x - gf

        # 估计噪声幅度
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
        should_skip = should_skip.float()

        # 只锐化真实结构，应用Skip Gate
        lam = lam.view(-1, 1, 1, 1)
        sharpening_term = lam * detail * noise_mask * detail_mask
        sharpened = torch.where(
            should_skip.expand_as(bf),
            x,  # 若应跳过，则直接使用原图
            bf + sharpening_term  # 否则进行锐化
        )

        # 确保输出在极小值到1之间
        min_value = 1e-5
        sharpened = torch.clamp(sharpened, min_value, 1.0)

        # 重新整形回原始形状
        out = sharpened.permute(0, 2, 3, 1).view(B, P, H, W, C)
        return out

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim, param_dim, hidden_dims=[96, 120, 150, 180]):
        super().__init__()

        layers = []
        input_dim = latent_dim

        # 构建MLP层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim

        # 最终输出层
        self.main = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], param_dim)

    def forward(self, x):
        x = self.main(x)
        params = self.output(x)
        return params


class ISPDecoder(nn.Module):
    """
    专家ISP处理器

    处理非零嵌入，并将结果恢复到原始位置
    """

    def __init__(self, latent_dim, param_dim, hidden_dims=[96, 120]):
        """
        初始化专家ISP处理器

        Args:
            latent_dim: 输入嵌入维度
            param_dim: 输出参数维度
            hidden_dims: 解码器隐藏层维度
        """
        super(ISPDecoder, self).__init__()

        # 创建解码器
        self.decoder = Decoder(latent_dim, param_dim, hidden_dims)
        self.param_dim = param_dim

    def forward(self, expert_embedding):
        """
        前向传播 - 处理专家嵌入并生成解码后的参数

        Args:
            expert_embedding: 形状为 [num_windows, embed_dim] 的张量
                              其中可能包含零嵌入表示未被选择的窗口

        Returns:
            output: 形状为 [num_windows, param_dim] 的张量
                              保持未选择位置为零
        """
        num_windows, embed_dim = expert_embedding.shape

        # 获取解码器的输出维度
        param_dim = self.decoder.output.out_features

        # 创建输出张量，初始化为1e-6，而不是0
        output = torch.zeros(num_windows, param_dim, device=expert_embedding.device)

        # 创建非零掩码：通过检查是否有非零元素
        non_zero_mask = torch.any(expert_embedding != 0, dim=1)  # [num_windows]

        # 如果没有非零嵌入，直接返回零张量
        if not torch.any(non_zero_mask):
            return output

        # 提取非零嵌入
        valid_embeddings = expert_embedding[non_zero_mask]  # [valid_count, embed_dim]

        # 使用解码器处理有效嵌入
        valid_outputs = self.decoder(valid_embeddings)  # [valid_count, param_dim]

        # 将解码结果放回原始位置
        output[non_zero_mask] = valid_outputs

        return output


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch.cuda.amp import custom_fwd, custom_bwd

from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,                          # 输入维度
        num_heads,                          # top-k个注意力头数
        dropout=0.0,                        # dropout概率
        bias=False,                         # 线性层偏置
        q_noise=0.0,                        # 量化噪声强度（用于模型参数 量化，默认0不启用）
        qn_block_size=8,                    # 量化噪声的分块大小（与q_noise配合使用）
        num_expert=4,                      # 专家数量（用于稀疏混合专家MoE结构）
        head_dim=24,                       # 每个注意力头的维度（若设置会覆盖num_heads计算逻辑）num_heads = embed_dim//head_dim
        use_attention_gate=False,           # 是否使用注意力门控，默认不走
        cvloss=0,                           # 专家负载均衡损失系数（MoE中平衡专家使用的损失项）
        aux_loss=0,                         # 辅助损失
        zloss=0,                            # 门控输出正则化损失系数（防止门控值过小）
        sample_topk=0,                      # 每个样本选择topk专家进行前向计算（0表示全选）
        noisy_gating=False,                 # 是否在门控函数中加入噪声（增强MoE鲁棒性）
        use_pos_bias=False,                 # 是否使用位置编码的标志位
    ):
        super().__init__()
        self.embed_dim = embed_dim                                            #模型定义维度
        self.kdim = embed_dim                   # Key的维度，等于embed_dim
        self.vdim = embed_dim                   # 同上

        self.num_heads = num_heads              #每个注意力头的维度embed_dim // num_heads
        # 使用fairseq定制的Dropout模块
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = head_dim #每个头的维度也就是Q⋅Wq_i后的维度也就是计算注意力机制的时候的维度
        self.scaling = self.head_dim**-0.5 #缩放因子1/√d_k

        #linearK，这个变换也就是KWk中的Wk，个专家共享相同Wk
        # q_noise量化噪声比例 (0.0~1.0)
        # qn_block_size噪声块大小 (如8表示8x8块)

        # 每个专家的linearQi，也就是Wq_i
        self.q_proj = quant_noise(
            MoELinearWrapper(
                input_size=embed_dim,
                head_size=self.head_dim,
                num_experts=num_expert,
                k=self.num_heads,
                cvloss=cvloss,
                aux_loss=aux_loss,
                zloss=zloss,
                noisy_gating=noisy_gating
            ),
            q_noise, qn_block_size
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, self.head_dim, bias=bias), q_noise, qn_block_size
        )
        #linearV，这个变换也就是VWv中的Wv，个专家共享相同Wk
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, self.head_dim, bias=bias), q_noise, qn_block_size
        )

        self.use_attention_gate = use_attention_gate

        self.sample_topk = sample_topk

        # 位置编码相关属性
        self.use_pos_bias = use_pos_bias
        self.pos_bias = None  # 位置编码嵌入表
        self.pos_indices = None  # 位置索引

        if use_pos_bias:
            self.init_pos_bias()

        self.reset_parameters() #初始化模型的权重

        self.skip_embed_dim_check = False #是否跳过嵌入维度检查

    def reset_parameters(self):
        # 当Q/K/V维度相同时，使用缩放版Xavier均匀初始化k_proj(Wk)，v_proj(Wv)权重（经验性优化）
        # 缩放因子 1/sqrt(2) 用于平衡多头注意力的合并结果
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))

        # nn.init.xavier_uniform_(self.out_proj.weight)
        # if self.out_proj.bias is not None:
        #     nn.init.constant_(self.out_proj.bias, 0.0)

    def init_pos_bias(self):
        """
        初始化二维相对位置编码，专为中心位置对周围8个位置的注意力设计

        每个位置学习一个向量，维度等于注意力头的数量
        """
        # 创建位置偏置嵌入层，8个固定的相对位置
        self.pos_bias = nn.Embedding(8, self.num_heads)

        # 直接使用从0到7的索引表示8个位置关系
        self.pos_indices = torch.arange(8)

        # 只有当缓冲区不存在时才注册
        if not hasattr(self, 'pos_indices'):
            self.register_buffer('pos_indices', self.pos_indices)

        # 为了清晰，可以添加位置到方向的映射注释
        # 0: 左上, 1: 上, 2: 右上, 3: 左, 4: 右, 5: 左下, 6: 下, 7: 右下

    def apply_pos_bias(self, attn_weights):
        """
        将位置偏置应用到注意力分数矩阵上

        参数:
            attn_weights: 注意力分数矩阵，形状为[bsz * patches, heads, tgt_len, src_len]

        返回:
            添加了位置偏置的注意力分数矩阵
        """
        # 确保pos_indices在正确的设备上
        device = attn_weights.device
        self.pos_indices = self.pos_indices.to(device)

        # 从嵌入层获取位置偏置
        bias = self.pos_bias(self.pos_indices)

        # 步骤1: 调整维度顺序，使heads维度在前
        bias = bias.permute(1, 0)

        # 步骤2: 将每个位置编码重复两次，确保每连续两个位置共享同一编码
        bias = bias.repeat_interleave(4, dim=1)

        # 步骤3: 添加维度，为批次和序列长度做准备
        bias = bias.unsqueeze(0).unsqueeze(2)

        # 步骤4: 将bias广播到与attn_weights相同的维度
        bias_broadcasted = bias.expand_as(attn_weights)

        # 步骤5: 添加到注意力分数上
        return attn_weights + (bias_broadcasted / self.scaling)

    def forward(
        self,
        query,
        key: Optional[Tensor],  #可选
        value: Optional[Tensor],#可选
        k_isp: Optional[Tensor] = None,
        need_weights: bool = True,# 是否返回多头平均注意力分数
        before_softmax: bool = False,# 是否返回softmax前的原始分数
        need_head_weights: bool = False,# 是否返回每个头的独立注意力分数
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """

        Args:
            need_weights (bool, 可选):
                是否返回注意力权重（经过多头平均后的结果，默认为False）。
            before_softmax (bool, 可选):
                是否返回经过softmax前的原始注意力权重和值。
            need_head_weights (bool, 可选):
                是否返回每个注意力头的权重（会强制启用need_weights）。
                默认返回所有注意力头的平均权重。
        """
        # 是否返回每个头的独立注意力分数
        if need_head_weights:
            need_weights = True

        bsz, patches, tgt_len, embed_dim = query.size()             # 返回query大小（目标序列长度（ISP超参数）、批次(图片数)、块数、嵌入维度）
        src_len = tgt_len                                           # 默认源序列长度等于目标序列长度（自注意力场景，MoA2）
        # 检查query的嵌入维度是否符合预期（仅在启用检查时）
        if not self.skip_embed_dim_check:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"     # query输入维度不等于模型定义维度
        assert list(query.size()) == [bsz, patches, tgt_len, embed_dim]  # 检查query的尺寸是否符合标准
        #处理key的存在情况（非自注意力时传入）
        if key is not None:
            key_bsz, key_p, src_len, _ = key.size()
            if not torch.jit.is_scripting():            # 脚本模式外进行维度验证（TorchScript会跳过这些检查）
                assert key_bsz == bsz                   # key和query的批大小必须相同
                assert key_p == patches
                assert value is not None                # 由key必须要有value
                assert (key_bsz, key_p, src_len) == value.shape[:3]  # key和value的前两个尺度必须一样

        # 如果是自注意力机制，计算q(qWq_i), k(KWk), v(VWv)，辅助损失
        # qkv都由query生成
        """
            q的结构
            [
                [样本1的专家1映射, 样本1的专家3映射],  # 样本1
                [样本2的专家2映射, 样本2的专家3映射],  # 样本2
                [样本3的专家1映射, 样本3的专家2映射],  # 样本3
            ]
        """

        assert key is not None and value is not None
        q, aux_loss = self.q_proj.map(
            query, k_isp=k_isp, sample_topk=self.sample_topk,
            attention_gate=self.use_attention_gate
            )#q 形状为[batch_size, p, tgt_len, k, head_size]
        k = self.k_proj(key)# k 的形状为 [bsz, p, src_len, head_dim]
        v = self.v_proj(value)# v 的形状为 [bsz, p, src_len, head_dim]
        q *= self.scaling   # q乘缩放因子1/√d_k，之后直接和k点积就行，softmax((q/√d_k)k)

        assert k is not None
        assert k.size(2) == src_len

        '''
        计算自注意力分数
        q的形状是(b,p,i,k,e)：批次(图片数)、块数、目标序列长度（ISP超参数）、注意力头数量、每个头的嵌入维度
        k的形状是(b,p,j,e)：批次(图片数)、块数、源序列长度（每块的token数）、嵌入维度
        'bpike,bpje->bpkij'输出张量的维度：bpkij，批次(图片数)、块数、注意力头数量、目标序列长度、源序列长度
        '''
        attn_weights = torch.einsum('bpike,bpje->bpkij', q, k)

        # 应用位置编码（如果启用）
        if self.use_pos_bias:
            # 获取当前形状
            bsz, patches, num_heads, tgt_len, src_len = attn_weights.shape  # [bsz, patches, self.num_heads, tgt_len, src_len]

            # 将attn_weights重新排列为形状[bsz*patches, heads, tgt_len, src_len]
            attn_weights = attn_weights.reshape(bsz * patches, self.num_heads, tgt_len, src_len)

            # 应用位置编码
            attn_weights = self.apply_pos_bias(attn_weights)

            # 恢复原始形状 [bsz, patches, self.num_heads, tgt_len, src_len]
            attn_weights = attn_weights.reshape(bsz, patches, self.num_heads, tgt_len, src_len)

        bsz, patches, num_heads, tgt_len, src_len = attn_weights.shape



        attn_weights = attn_weights.contiguous().reshape(bsz * patches * self.num_heads, tgt_len, src_len)# 将结果张量重塑为维度(批次大小×块数×注意力头数量，目标长度，源长度)

        assert list(attn_weights.size()) == [bsz * patches * self.num_heads, tgt_len, src_len]

        #如果需要返回softmax前的原始分数，返回qk/√d_k、和v
        if before_softmax:
            return attn_weights, v

        # 对注意力权重在最后一个维度上应用softmax函数，使权重归一化为概率分布
        attn_weights_float = utils.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights_float.type_as(attn_weights) # 将浮点型的注意力权重转换回原始注意力权重的数据类型
        #对注意力权重应用dropout，随机丢弃张量元素，增强模型的鲁棒性。在推理阶段，dropout会被关闭，所有权重都会被保留。
        attn_probs = self.dropout_module(attn_weights)

        # softmax(qk/√d_k)·v
        assert v is not None
        attn = torch.einsum('bpkij,bpje->bpike', attn_probs.reshape(bsz, patches, self.num_heads, tgt_len, src_len), v)#[bsz, patches, tgt_len, self.num_heads, self.head_dim]
        # 注意力计算后的形状
        assert list(attn.size()) == [bsz, patches, tgt_len, self.num_heads, self.head_dim]

        attn = self.q_proj.reduce(attn)# 加权求和得到MoA输出 原代码此处为attn = self.q_proj.reduce(attn).transpose(0, 1)，形状最终为[bsz, patches, tgt_len, self.head_dim]

        # 重塑注意力权重为[批次大小, 注意力头数量, 目标长度, 源长度]并转置
        if need_weights:
            attn_weights = attn_weights_float.reshape(
                bsz, patches, self.num_heads, tgt_len, src_len
            ).permute(2, 0, 1, 3, 4)#[self.num_heads, bsz, patches, tgt_len, src_len]
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)# 沿着头维度对注意力权重求平均
        else:
            attn_weights: Optional[Tensor] = None


        return attn, attn_weights, aux_loss#返回MoA注意力最终计算结果、注意力权重矩阵、辅助损失


class MoE(nn.Module):
    """ 使用一层FFN作为专家的稀疏门控。
        参数:
        input_size: 整数 - 输入的大小
        output_size: 整数 - 输出的大小
        num_experts: 整数 - 专家的数量
        hidden_size: 整数 - 专家的隐藏层大小
        noisy_gating: 布尔值 - 是否使用噪声门控
        k: 整数 - 对每个批次元素使用多少个专家
    """

    def __init__(self, input_size, head_size, num_experts, k, need_merge = False, cvloss=0, aux_loss=0, zloss=0, bias=False,
                 activation=None, noisy_gating=True):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating  # 是否使用噪声门控
        self.num_experts = num_experts  # 专家总数量
        self.input_size = input_size  # 输入维度
        self.head_size = head_size  # 注意力头大小
        self.need_merge = need_merge
        self.saved_top_k_indices = None
        self.token_expert_indices = None # 用于存储每个token选择的专家索引
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)  # 并行专家层，用于将输入转换为中间表示
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)  # 并行专家输出层，用于将中间表示转换回输入维度
        self.k = min(k, self.num_experts)
        self.cvloss = cvloss  # 变异系数损失
        self.aux_loss = aux_loss  # 切换损失
        self.zloss = zloss  # Z_loss
        self.activation = activation
        # 门控权重矩阵
        self.w_atten_gate = nn.Parameter(torch.randn(num_experts, num_experts) * 0.01, requires_grad=True)
        self.w_gate = nn.Parameter(torch.randn(input_size, num_experts) * 0.01,requires_grad=True)  # 初始化self.w_gate为小高斯噪声，可学习参数
        # 如果使用噪声门控
        if noisy_gating:
            self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

    def cv_squared(self, x):
        """ 样本的变异系数平方。
            作为损失函数时，鼓励正分布更加均匀。
            添加epsilon以保证数值稳定性。
            对于空张量返回0。
            Args:
                x: 一个张量
            Returns:
                一个标量
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean() ** 2 + eps)  # 计算变异系数的平方: 方差除以均值的平方

    # 计算变异系数
    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    # 1. 对输入的概率张量 probs 按照维度0求和，得到各专家的累积概率。
    # 2. 使用 F.normalize 进行归一化操作，将求和后的结果转换为和为1的分布，归一化时使用 L1 范数（p=1）沿 dim=0 进行归一化。
    # 3. 调用 cv_squared 函数计算归一化后概率分布的系数平方。

    def auxiliary_loss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * F.normalize(freqs.float(), p=1, dim=0)  # 归一化后的概率除归一化后的频率
        return loss.sum() * self.num_experts  # 求和乘专家数

    # 1. 对输入的logits 进行指数运算，转换为未归一化的概率量度。并沿着 dim=1 求和，得到每个样本对应的归一化因子
    # 2. 对求和的结果取对数，再平方平方。
    # 3. 计算上述值的均值，作为 z loss 的最终值。
    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def atten_gating(self, Q_isp, K_isp, sample_topk=0, noise_epsilon=1e-3):
        """
        前向传播

        参数:
            Q_isp (Tensor): 形状为 [batches, patches, 1, input_dim] 的张量
            K_isp (Tensor): 形状为 [batches, patches, N, input_dim] 的张量
            k (int): top-k中的k值，表示保留的权重数量

        返回:
            gate_weights (Tensor): 形状为 [-1, N] 的张量，
                                  表示门控权重，每个向量中只有k个非零元素
        """
        # 获取输入形状
        batches, patches, N, input_dim = K_isp.shape
        scale = 1.0 / math.sqrt(input_dim)
        # 计算注意力分数：Q_isp与K_isp的点积
        # Q_isp的形状是[batches, patches, 1, input_dim]
        # K_isp的形状是[batches, patches, N, input_dim]
        attention_scores = torch.matmul(Q_isp, K_isp.transpose(-1, -2))  # [batches, patches, 1, N]

        # 移除多余的维度
        attention_scores = attention_scores.squeeze(2)  # [batches, patches, N]

        attention_scores = attention_scores.reshape(-1, N)# [-1, N]

        attention_scores = attention_scores * scale

        attention_scores = torch.nan_to_num(
            attention_scores,
            nan=0.0,
            posinf=20.0,  # 上限可根据需要调到30、40
            neginf=-20.0
        )

        clean_logits = attention_scores
        # 如果启用了噪声门控并且当前处于训练模式，则添加噪声
        if self.noisy_gating and self.training:
            # 计算输入 x 与噪声权重矩阵 self.w_noise 的矩阵乘积，得到原始噪声标准差
            raw_noise_stddev = attention_scores @ self.w_atten_gate
            # 通过 softplus 激活函数保证噪声标准差为正值，并加上一个极小值 noise_epsilon
            noise_stddev = torch.nan_to_num(
                F.softplus(raw_noise_stddev) + noise_epsilon,
                nan=noise_epsilon,
                posinf=1e3, neginf=noise_epsilon
            )
            # 生成与 clean_logits 相同形状的随机噪声并按噪声标准差进行缩放，然后与 clean_logits 相加
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            # 使用带噪声的 logits
            logits = noisy_logits
        else:
            logits = clean_logits
        # 应用softmax获取注意力权重

        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0
        )


        attention_weights = F.softmax(logits, dim=1)  # [-1, N]

        # 展平attention_weights并保存为probs
        probs = attention_weights

        # 如果训练阶段，且sample_topk > 0启用混合选择策略
        if self.training and sample_topk > 0:
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k
            # 先在所有专家中选择topk中的topk - sample_topk个
            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6  # 添加极小值避免零概率
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0  # 创建掩码概率：已选位置置零防止重复选择
            k_indices = torch.multinomial(masked_probs, sample_topk)  # 从剩余概率中采样 sample_topk 个专家
            # 合并确定性和采样得到的专家索引
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)  # 常规top-k选择（确定性选择）

        # 门控值归一化
        top_k_gates = top_k_gates / \
                      (top_k_gates.sum(dim=1, keepdim=True) + 1e-6).detach()

        # 创建全零张量用于存储门控权重
        zeros = torch.zeros_like(probs, requires_grad=True)  # 构建稀疏门控矩阵
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 统计每个专家被选中的样本数（用于负载均衡计算）
        self.expert_size = (gates > 0).long().sum(0)  # 类似self.expert_size = [50, 49, 62, 51]
        # 门控值和对应索引展平为[batch_size * k]
        top_k_gates = top_k_gates.flatten()  # 门控值[0.5, 0.3, 0.7, 0.9, 0.6, 0.8] 3 个样本 × 2 个专家
        top_k_experts = top_k_indices.flatten()  # 门控索引[1, 3, 2, 4, 1, 2]3 个样本 × 2 个专家
        # 过滤零值门控（保留有效路由信息）
        nonzeros = top_k_gates.nonzero().squeeze(-1)
        top_k_experts_nonzero = top_k_experts[nonzeros]  # 门控索引[1, 3, 2, 4, 1, 2]
        # topk个专家按ID排序，优化后续计算效率
        _, _index_sorted_experts = top_k_experts_nonzero.sort(0)  # 排序后的门控索引
        self.index_sorted_experts = nonzeros[_index_sorted_experts]

        """
        top_k_indices = [[1, 3],  # 第一个样本选择了专家 1 和 3
                         [2, 4],  # 第二个样本选择了专家 2 和 4
                        [1, 2]]  # 第三个样本选择了专家 1 和 2
        top_k_gates = [[0.5, 0.3],  # 第一个样本选择专家 1 和 3 的门控值分别是 0.5 和 0.3
                       [0.7, 0.9],  # 第二个样本选择专家 2 和 4 的门控值分别是 0.7 和 0.9
                      [0.6, 0.8]]  # 第三个样本选择专家 1 和 2 的门控值分别是 0.6 和 0.8
        """
        self.batch_index = self.index_sorted_experts.div(self.k,
                                                         rounding_mode='trunc')  # 每个专家的样本池（每个专家处理哪些样本？样本池是样本原始索引组成的），再把样本池按照专家索引排序，[专家1样本池，专家2样本池，...]
        self.batch_gates = top_k_gates[
            self.index_sorted_experts]  # 每个专家的样本门控值池（每个专家处理的样本对应的门控值），再把样本门控值池按照专家索引排序，[专家1样本门控值池，专家2样本门控值池，...]
        # 计算损失
        loss = 0
        # 变异系数损失：鼓励均匀使用各专家
        loss += self.cvloss * self.compute_cvloss(gates)
        # 辅助损失
        loss += self.aux_loss * \
                self.auxiliary_loss(probs, self.expert_size)
        # zloss
        loss += self.zloss * self.compute_zloss(logits)
        return loss

    def top_k_gating(self, x, sample_topk=0, noise_epsilon=1e-3):
        """
            噪声 top-k 门控。
            参见论文：https://arxiv.org/abs/1701.06538

            参数:
              x: 输入张量，形状为 [-1, input_size]
              sample_topk: 训练阶段采样选取的专家数量（0 表示不采样）
              noise_epsilon: 防止数值不稳定的小浮点数
            返回:
              loss: 综合了 cv loss、switch loss 及 z loss 的最终损失值
        """
        d = x.size(1)
        scale = 1.0 / math.sqrt(d)
        # 计算输入 x 与门控权重矩阵 self.w_gate 的矩阵乘积，得到干净的 logits
        clean_logits = x @ self.w_gate
        clean_logits = clean_logits * scale
        clean_logits = torch.nan_to_num(
            clean_logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0
        )
        # 如果启用了噪声门控并且当前处于训练模式，则添加噪声
        if self.noisy_gating and self.training:
            # 计算输入 x 与噪声权重矩阵 self.w_noise 的矩阵乘积，得到原始噪声标准差
            raw_noise_stddev = x @ self.w_noise
            # 通过 softplus 激活函数保证噪声标准差为正值，并加上一个极小值 noise_epsilon
            noise_stddev = torch.nan_to_num(
                F.softplus(raw_noise_stddev) + noise_epsilon,
                nan=noise_epsilon,
                posinf=1e3, neginf=noise_epsilon
            )
            # 生成与 clean_logits 相同形状的随机噪声并按噪声标准差进行缩放，然后与 clean_logits 相加
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            # 使用带噪声的 logits
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0
        )

        probs = F.softmax(logits, dim=1) # [-1, N]

        # 如果训练阶段，且sample_topk > 0启用混合选择策略
        if self.training and sample_topk > 0:
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k
            # 先在所有专家中选择topk中的topk - sample_topk个
            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6  # 添加极小值避免零概率
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0  # 创建掩码概率：已选位置置零防止重复选择
            k_indices = torch.multinomial(masked_probs, sample_topk)  # 从剩余概率中采样 sample_topk 个专家
            # 合并确定性和采样得到的专家索引
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)  # 常规top-k选择（确定性选择）
        # 门控值归一化
        top_k_gates = top_k_gates / \
            (top_k_gates.sum(dim=1, keepdim=True) + 1e-6).detach()

        # 保存为实例变量，以便在forward和concat中使用
        self.saved_top_k_indices = top_k_indices

        zeros = torch.zeros_like(probs, requires_grad=True)  # 构建稀疏门控矩阵
        gates = zeros.scatter(1, top_k_indices, top_k_gates)  # 创建全零矩阵并通过scatter操作填充选中的门控值，仅保留top-k专家的激活值，其余位置为零

        # 统计每个专家被选中的样本数（用于负载均衡计算）
        self.expert_size = (gates > 0).long().sum(0)#类似self.expert_size = [50, 49, 62, 51]
        # 门控值和对应索引展平为[batch_size * k]
        top_k_gates = top_k_gates.flatten()# 门控值[0.5, 0.3, 0.7, 0.9, 0.6, 0.8] 3 个样本 × 2 个专家
        top_k_experts = top_k_indices.flatten()# 门控索引[1, 3, 2, 4, 1, 2]3 个样本 × 2 个专家
        # 过滤零值门控（保留有效路由信息）
        nonzeros = top_k_gates.nonzero().squeeze(-1)
        top_k_experts_nonzero = top_k_experts[nonzeros]# 门控索引[1, 3, 2, 4, 1, 2]
        # topk个专家按ID排序，优化后续计算效率
        _, _index_sorted_experts = top_k_experts_nonzero.sort(0)# 排序后的门控索引
        self.index_sorted_experts = nonzeros[_index_sorted_experts]

        """
        top_k_indices = [[1, 3],  # 第一个样本选择了专家 1 和 3
                         [2, 4],  # 第二个样本选择了专家 2 和 4
                        [1, 2]]  # 第三个样本选择了专家 1 和 2
        top_k_gates = [[0.5, 0.3],  # 第一个样本选择专家 1 和 3 的门控值分别是 0.5 和 0.3
                       [0.7, 0.9],  # 第二个样本选择专家 2 和 4 的门控值分别是 0.7 和 0.9
                      [0.6, 0.8]]  # 第三个样本选择专家 1 和 2 的门控值分别是 0.6 和 0.8
        """
        self.batch_index = self.index_sorted_experts.div(self.k, rounding_mode='trunc')#每个专家的样本池（每个专家处理哪些样本？样本池是样本原始索引组成的），再把样本池按照专家索引排序，[专家1样本池，专家2样本池，...]
        self.batch_gates = top_k_gates[self.index_sorted_experts]#每个专家的样本门控值池（每个专家处理的样本对应的门控值），再把样本门控值池按照专家索引排序，[专家1样本门控值池，专家2样本门控值池，...]
        # 计算损失
        loss = 0
        # 变异系数损失：鼓励均匀使用各专家
        loss += self.cvloss * self.compute_cvloss(gates)
        # 辅助损失
        loss += self.aux_loss * \
                self.auxiliary_loss(probs, self.expert_size)
        # zloss
        loss += self.zloss * self.compute_zloss(logits)
        # 在函数末尾添加
        return loss

    """
        MoE层前向传播过程，实现动态路由和专家计算

        参数:
          x: 输入张量，形状为 [bsz, patches, tgt_len, self.num_heads, self.head_dim]
          sample_topk: 训练时采样的专家数量（0表示禁用采样，正常topk选择）
          multiply_by_gates: 是否用门控值加权专家输出

        返回:
          y: 经过专家聚合的输出，形状同输入x
          loss: 门控机制计算的总损失
    """

    def forward(self, x, sample_topk=0, multiply_by_gates=True):
        """
        前向传播输入是[bsz, patches, tgt_len, self.input_size]
        """
        # 输入数据预处理
        bsz, patches, length, emb_size = x.size()  # 获取输入张量的维度：批次大小、块数量、序列长度、注意力头数量和每个头的嵌入维度
        x = x.contiguous()
        x = x.reshape(-1, emb_size)  # 将四维张量重塑为二维张量，新形状为[bsz*p*length*k, emb_size]

        # 计算门控权重更新到self属性，并返回loss
        loss = self.top_k_gating(x, sample_topk=sample_topk)
        # expert_inputs的每一项的索引对应专家索引，值是索引对应专家的输入数据
        expert_inputs = x[self.batch_index]
        # 并行计算所有专家的中间FFN的前向传播并应用激活函数
        h = self.experts(expert_inputs, self.expert_size)#继承自nn.Module，自动调用forward 方法
        h = self.activation(h)#继承自nn.Module，自动调用forward 方法
        # 并行计算所有专家输出层的前向传播
        expert_outputs = self.output_experts(h, self.expert_size)#继承自nn.Module，自动调用forward 方法
        # 门控加权
        # batch_gates: 每个路由项对应的门控权重，形状 [num_selected]
        # 通过[:, None]扩展维度实现逐元素相乘
        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]
        # 输出聚合
        # 创建全零基础张量用于聚合结果
        if self.need_merge:
            zeros = torch.zeros((bsz * patches * length, self.input_size),
                                dtype=expert_outputs.dtype, device=expert_outputs.device)
            # index_add操作：将专家输出累加到对应位置
            # 参数说明：
            #   dim=0 - 按行累加
            #   index=self.batch_index - 目标行索引
            #   source=expert_outputs - 需要添加的数据
            y = zeros.index_add(0, self.batch_index, expert_outputs)  # [batch_size*seq_len, output_dim]，包含了每个专家的输出向量相加（而不是合并）
            y = y.contiguous()
            y = y.reshape(bsz, patches, length, self.input_size)
        else:
            # need_merge=False 分支：使用 index_add 安全填充专家输出
            # 1) 计算总行数并创建全零扁平张量
            total_rows = bsz * patches * length * self.k
            zeros = torch.zeros(
                (total_rows, self.input_size),
                dtype=expert_outputs.dtype,
                device=expert_outputs.device
            )

            # 2) 重建 expert_positions（在原代码中已计算，用于指示在 k 维度的位置）
            expert_positions = torch.zeros(
                len(self.index_sorted_experts),
                dtype=torch.long,
                device=expert_outputs.device
            )
            expert_counts = torch.zeros(
                (bsz * patches * length),
                dtype=torch.long,
                device=expert_outputs.device
            )
            for i, sample_idx in enumerate(self.batch_index):
                expert_positions[i] = expert_counts[sample_idx]
                expert_counts[sample_idx] += 1

            # 3) 计算扁平后每行的索引：batch_index * k + expert_positions
            row_indices = self.batch_index * self.k + expert_positions  # shape [num_selected]

            # 4) 用 index_add 将每个 expert_outputs 累加到对应行
            y_flat = zeros.index_add(0, row_indices, expert_outputs)

            # 5) 最后 reshape 回 [bsz, patches, length, k, input_size]
            y = y_flat.view(bsz, patches, length, self.k, self.input_size)

            # 记录每个token选择的专家索引，用于concat方法
            self.token_expert_indices = self.saved_top_k_indices.reshape(bsz, patches, length, self.k)

        return y, loss

    def map(self, x, k_isp=None, sample_topk=0, attention_gate=False):
        """
            MoE映射函数（适配分块输入，保持独立计算）
            参数:
                x: 输入张量，形状 [batch_size, p=64, seq_len = 1, emb_size]
                k_isp: K张量(可选)，形状 [batch_size, p=64, N, emb_size]
                sample_topk: 训练时采样的专家数量（0表示禁用采样）
                attention_gate: 是否使用注意力门控，默认为False
            返回:
                y: 每个块中每个token对应的k个专家输出，形状 [batch_size, p=64, seq_len = 1, k = 2, head_size]
                loss: 门控机制的负载均衡损失
        """
        if attention_gate and k_isp is not None:
            # 使用注意力门控，保持原有维度
            bsz, p, seq_len, emb_size = x.size()  # 输入query的尺寸
            loss = self.atten_gating(x, k_isp, sample_topk=sample_topk)  # 计算辅助损失

            x = x.reshape(bsz * p * seq_len, emb_size)  # 直接合并前三个维度
        else:
            # 使用原始门控方式
            bsz, p, seq_len, emb_size = x.size()  # 输入query的尺寸
            x = x.reshape(bsz * p * seq_len, emb_size)  # 直接合并前三个维度
            loss = self.top_k_gating(x, sample_topk=sample_topk)  # 计算辅助损失

        # 门控后的索引信息
        # 此时x已经展平，直接使用
        # 此时对象已存储路由信息：
        #   self.batch_index: 每个专家处理的原始token索引 [num_selected]
        #   self.expert_size: 每个专家被选中的次数 [num_experts]
        #   self.index_sorted_experts: 按专家ID排序后的所有样本索引 [num_selected]
        expert_inputs = x[self.batch_index]  # 这个张量包含几部分，每一部分包含某个专家所处理的所有样本token
        expert_outputs = self.experts(expert_inputs, self.expert_size)  # 继承自nn.Module，自动调用forward方法

        # 输出聚合
        # 创建全零基础张量
        zeros = torch.zeros(
            (bsz * p * seq_len * self.k, self.head_size),
            dtype=expert_outputs.dtype,
            device=expert_outputs.device
        )
        # 按排序后的索引填充专家输出（确保同专家计算连续）
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.reshape(bsz, p, seq_len, self.k, self.head_size)  # 保持原始seq_len

        expert_outputs = self.experts(expert_inputs, self.expert_size)

        return y, loss  # 返回top-k个专家的qWq_i，下一步送入注意力点积运算以及辅助损失

    def reduce(self, x, multiply_by_gates=True):
        x = x.contiguous()
        bsz, patches, length, k, head_dim = x.size()

        # 重塑x为二维张量
        x = x.reshape(-1, head_dim)

        expert_inputs = x[self.index_sorted_experts]

        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            if self.batch_gates.shape[0] != expert_outputs.shape[0]:
                # 确保形状兼容
                min_size = min(self.batch_gates.shape[0], expert_outputs.shape[0])
                self.batch_gates = self.batch_gates[:min_size]
                expert_outputs = expert_outputs[:min_size]

            expert_outputs = expert_outputs * self.batch_gates[:, None]

        # 创建输出张量
        zeros = torch.zeros((bsz * patches * length, self.input_size),
                            dtype=expert_outputs.dtype, device=expert_outputs.device)

        # 确保batch_index所有值都在有效范围内
        max_index = bsz * patches * length - 1

        if self.batch_index.max() > max_index:
            # 裁剪越界索引
            valid_mask = self.batch_index <= max_index
            valid_indices = self.batch_index[valid_mask]
            valid_outputs = expert_outputs[valid_mask]
            y = zeros.index_add(0, valid_indices, valid_outputs)
        else:
            y = zeros.index_add(0, self.batch_index, expert_outputs)

        y = y.reshape(bsz, patches, length, self.input_size)
        return y

    def concat(self, y, e_isp):
        """
        将e_isp的最后一个维度拆分成4等份，然后按顺序加到y的对应k维度上

        参数:
            y: 前向传播输出的张量，形状为 [bsz, patches, length=1, k=4, self.input_size]
            e_isp: 输入张量，形状为 [bsz, patches, embed_dim]，embed_dim能被4整除

        返回:
            result: 求和后的结果，形状为 [bsz, patches, length=1, k=4, self.input_size]
        """
        bsz, patches, length, k, input_size = y.size()
        assert length == 1, "length维度应该为1"
        assert k == 4, "k维度应该为4"

        # 确认e_isp的最后一维可以被4整除
        embed_dim = e_isp.size(-1)
        assert embed_dim % 4 == 0, "embed_dim必须能被4整除"

        # 验证每个分块大小与input_size相匹配
        split_size = embed_dim // 4
        assert split_size == input_size, f"e_isp分块大小({split_size})必须与专家输出维度({input_size})匹配"

        # 将e_isp最后一个维度拆分成4等份
        # [bsz, patches, embed_dim] -> [bsz, patches, 4, embed_dim//4]
        e_isp_reshaped = e_isp.reshape(bsz, patches, 4, split_size)

        # 调整e_isp_reshaped的维度以匹配y
        # [bsz, patches, 4, split_size] -> [bsz, patches, 1, 4, split_size]
        e_isp_expanded = e_isp_reshaped.unsqueeze(2)

        # 直接将重塑后的e_isp与y相加
        combined = y + e_isp_expanded

        return combined

#ParallelLinear以及ParallelExperts实现了多专家并行计算的线性层
class ParallelLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, expert_size, weight, bias=None):
        output_list = []  # 初始化一个空列表，用于保存每个专家计算后的输出

        expert_size_list = expert_size.tolist()  # 将 expert_size 转换为 Python 列表
        # 将输入张量按照 expert_size_list 指定的尺寸进行分割
        input_list = input.split(expert_size_list, dim=0)

        # 记录权重形状用于调试
        for i in range(weight.size(0)):
            if expert_size_list[i] > 0:  # 只处理有样本的专家
                if bias is not None:
                    o_i = torch.mm(input_list[i], weight[i]) + bias[i]
                else:
                    o_i = torch.mm(input_list[i], weight[i])
                output_list.append(o_i)
            else:
                # 对于没有样本的专家，添加一个空张量
                if bias is not None:
                    empty_output = torch.empty((0, weight[i].size(1)),
                                               dtype=input.dtype,
                                               device=input.device)
                    output_list.append(empty_output)
                else:
                    empty_output = torch.empty((0, weight[i].size(1)),
                                               dtype=input.dtype,
                                               device=input.device)
                    output_list.append(empty_output)

        output = torch.cat(output_list, dim=0)  # 将所有专家的输出拼接成最终的输出张量

        # 将需要在反向传播中用到的变量保存起来
        variables = (input, expert_size, weight, bias)
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input, expert_size, weight, bias = ctx.saved_tensors  # 从之前保存的恢复前向传播保存的张量
        num_linears = weight.size(0)  # 专家数量赋值给num_linears

        expert_size_list = expert_size.tolist()
        input_list = input.split(expert_size_list, dim=0)  # 将输入按每个专家处理的样本数分割
        grad_list = grad_out.split(expert_size_list, dim=0)  # 将输出梯度按每个专家处理的样本数分割

        # 计算输入的梯度
        d_input_list = []
        for i in range(num_linears):
            if expert_size_list[i] > 0:  # 只处理有样本的专家

                # 使用转置矩阵乘法，确保维度匹配
                d_input_list.append(torch.mm(grad_list[i], weight[i].t()))
            else:
                # 对于没有样本的专家，添加一个空梯度
                empty_grad = torch.empty((0, weight[i].size(0)),
                                         dtype=grad_out.dtype,
                                         device=grad_out.device)
                d_input_list.append(empty_grad)

        d_input = torch.cat(d_input_list, dim=0)

        # 计算权重的梯度
        d_weight_list = []
        for i in range(num_linears):
            if expert_size_list[i] > 0:  # 只处理有样本的专家
                # 确保权重梯度的形状与权重相同
                # input_list[i]的形状为[batch_i, input_dim]
                # grad_list[i]的形状为[batch_i, output_dim]
                # 所需的权重梯度形状为[input_dim, output_dim]

                # 检查维度是否兼容
                if input_list[i].shape[1] != weight[i].shape[0] or grad_list[i].shape[1] != weight[i].shape[1]:
                    print(
                        f"警告：专家 {i} 的权重梯度形状不匹配：input_dim={input_list[i].shape[1]}, weight_in={weight[i].shape[0]}, weight_out={weight[i].shape[1]}, grad_dim={grad_list[i].shape[1]}")

                    # 如果梯度形状不匹配，进行适当的调整
                    if grad_list[i].shape[1] != weight[i].shape[1]:
                        # 将梯度调整为正确的形状
                        adjusted_grad = grad_list[i].view(grad_list[i].size(0), weight[i].size(1))
                        d_weight_list.append(torch.mm(input_list[i].t(), adjusted_grad))
                    else:
                        d_weight_list.append(torch.mm(input_list[i].t(), grad_list[i]))
                else:
                    d_weight_list.append(torch.mm(input_list[i].t(), grad_list[i]))
            else:
                # 对于没有样本的专家，添加一个形状正确的零梯度
                zero_grad = torch.zeros_like(weight[i])
                d_weight_list.append(zero_grad)

        d_weight = torch.stack(d_weight_list, dim=0)

        # 计算偏置的梯度
        if bias is not None:
            d_bias_list = []
            for i in range(num_linears):
                if expert_size_list[i] > 0:  # 只处理有样本的专家
                    if grad_list[i].shape[1] != bias[i].shape[0]:
                        print(
                            f"警告：专家 {i} 的偏置梯度形状不匹配：grad_dim={grad_list[i].shape[1]}, bias_dim={bias[i].shape[0]}")
                        # 将梯度调整为正确的形状
                        adjusted_grad = grad_list[i].view(grad_list[i].size(0), bias[i].size(0))
                        d_bias_list.append(adjusted_grad.sum(0))
                    else:
                        d_bias_list.append(grad_list[i].sum(0))
                else:
                    # 对于没有样本的专家，添加一个形状正确的零梯度
                    zero_bias_grad = torch.zeros_like(bias[i])
                    d_bias_list.append(zero_bias_grad)

            d_bias = torch.stack(d_bias_list, dim=0)
        else:
            d_bias = None

        return d_input, None, d_weight, d_bias

class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False) -> None:
        """
            初始化并行专家模块
            参数:
            num_experts: 专家的总数量
            input_size: 每个专家的输入特征维度
            output_size: 每个专家的输出特征维度
            bias: 是否使用偏置，默认为False
        """
        super().__init__()
        self.w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        if bias:
            self.b = nn.Parameter(torch.zeros(num_experts, output_size))
        else:
            self.b = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2.0 / float(self.w.size(1) + self.w.size(2)))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(self.w, -a, a)

    # 使用自定义的ParallelLinear进行前向计算
    def forward(self, inputs, expert_size):
        results = ParallelLinear.apply(inputs, expert_size, self.w, self.b)
        return results
    #self.activation


class MoELinearWrapper(nn.Linear):
    """
    包装 MoE 模块使其兼容 quant_noise 函数
    继承自 nn.Linear 以通过 quant_noise 的类型检查，
    同时不使用__getattr__转发所有 MoE 方法
    """

    def __init__(self, input_size, head_size, num_experts, k, need_merge=False,
                 cvloss=0, aux_loss=0, zloss=0, bias=False,
                 activation=None, noisy_gating=True):
        # 明确调用nn.Linear的初始化方法，避免使用super()
        nn.Linear.__init__(self, input_size, input_size, bias=False)

        # 创建MoE实例
        self.moe = MoE(
            input_size=input_size,
            head_size=head_size,
            num_experts=num_experts,
            k=k,
            need_merge=need_merge,
            cvloss=cvloss,
            aux_loss=aux_loss,
            zloss=zloss,
            bias=bias,
            activation=activation,
            noisy_gating=noisy_gating
        )

        # 禁用线性层的参数，因为我们不会使用它们
        self.weight.requires_grad = False

    # 明确转发所有需要的方法，不使用__getattr__
    def forward(self, x, sample_topk=0, multiply_by_gates=True):
        return self.moe(x, sample_topk, multiply_by_gates)

    def map(self, x, k_isp=None, sample_topk=0, attention_gate=False):
        return self.moe.map(x, k_isp, sample_topk, attention_gate)

    def reduce(self, x, multiply_by_gates=True):
        return self.moe.reduce(x, multiply_by_gates)

    def concat(self, y, e_isp):
        return self.moe.concat(y, e_isp)

    def cv_squared(self, x):
        return self.moe.cv_squared(x)

    def compute_cvloss(self, probs):
        return self.moe.compute_cvloss(probs)

    def auxiliary_loss(self, probs, freqs):
        return self.moe.auxiliary_loss(probs, freqs)

    def compute_zloss(self, logits):
        return self.moe.compute_zloss(logits)

    def atten_gating(self, Q_isp, K_isp, sample_topk=0, noise_epsilon=1e-2):
        return self.moe.atten_gating(Q_isp, K_isp, sample_topk, noise_epsilon)

    def top_k_gating(self, x, sample_topk=0, noise_epsilon=1e-2):
        return self.moe.top_k_gating(x, sample_topk, noise_epsilon)

    # 直接转发MoE的所有属性访问
    @property
    def batch_index(self):
        return self.moe.batch_index

    @property
    def batch_gates(self):
        return self.moe.batch_gates

    @property
    def token_expert_indices(self):
        return self.moe.token_expert_indices

    @property
    def saved_top_k_indices(self):
        return self.moe.saved_top_k_indices

    @property
    def index_sorted_experts(self):
        return self.moe.index_sorted_experts

    @property
    def expert_size(self):
        return self.moe.expert_size

# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import feature_extractor
from feature_extractor import FeatureExtractor
import config
from config import default_config
import emb_gen
from MoA import MultiheadAttention, MoE
from utils import *
from decoder import *
from ISP import *
from losses import *

class Model(nn.Module):
    """主模型类，组装各个预定义的模块"""

    def __init__(self, config=None):
        """
        使用配置字典初始化模型
        Args:
            config: 包含所有模型参数的配置字典，如果为None则使用默认配置
        """
        super(Model, self).__init__()
        self.expert_indices = None
        # 如果没有提供配置，使用默认配置
        if config is None:
            config = default_config()

        # 从config读取图像尺寸参数，设置H和W
        self.patch_size = config['image_size']['patch_size']
        self.win_size = config['image_size']['win_size']

        # 添加损失权重超参数
        loss_weights_config = config.get('loss_weights', {})
        self.reconstruction_weight = loss_weights_config.get('reconstruction_weight', 1.0)  # 重建损失权重
        self.auxiliary_weight = loss_weights_config.get('auxiliary_weight', 0.1)  # 辅助损失权重

        # 获取特征提取器配置
        fe_config = config.get('feature_extractor', {})

        # 实例化特征提取器
        self.feature_extractor = FeatureExtractor(
            patch_size=fe_config.get('patch_size', 2),
            in_channels=fe_config.get('in_channels', 3),
            embed_dim=fe_config.get('embed_dim', 128),
            win_size=fe_config.get('win_size', 2)
        )

        # 获取EispGeneratorFFN配置
        emb_gen_config = config.get('emb_gen', {})

        # 实例化EispGeneratorFFN
        self.emb_gen = emb_gen.EispGeneratorFFN(
            input_dim=emb_gen_config.get('input_dim', 128),
            hidden_dim=emb_gen_config.get('hidden_dim', 512),
            output_dim=emb_gen_config.get('output_dim', 128),
            dropout_rate=emb_gen_config.get('dropout_rate', 0.1)
        )

        # 添加线性层，将e_isp的最后一个维度映射为其4倍大
        self.e_isp_expand = nn.Linear(
            emb_gen_config.get('output_dim', 128),
            emb_gen_config.get('output_dim', 128) * 4
        )

        # 获取QKIspGenerator配置
        qk_gen_config = config.get('qk_gen', {})

        # 实例化QKIspGenerator
        self.qk_gen = emb_gen.QKIspGenerator(
            dim=qk_gen_config.get('dim', 128),
            num_heads=qk_gen_config.get('num_heads', 4),
            dropout_rate=qk_gen_config.get('dropout_rate', 0.1)
        )

        # 获取KVFeatureGenerator配置
        kv_gen_config = config.get('kv_gen', {})

        # 实例化KVFeatureGenerator
        self.kv_gen = emb_gen.KVFeatureGenerator(
            embed_dim=kv_gen_config.get('embed_dim', 128),
            dropout_rate=kv_gen_config.get('dropout_rate', 0.1)
        )

        # 获取MultiheadAttention配置
        multi_attn_1_config = config.get('multi_attention_1', {})

        # 实例化MultiheadAttention
        self.multi_attention_1 = MultiheadAttention(
            embed_dim=multi_attn_1_config.get('embed_dim', 128),
            num_heads=multi_attn_1_config.get('num_heads', 2),
            dropout=multi_attn_1_config.get('dropout', 0.1),
            bias=multi_attn_1_config.get('bias', True),
            q_noise=multi_attn_1_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_1_config.get('qn_block_size', 8),
            num_expert=multi_attn_1_config.get('num_expert', 4),
            head_dim=multi_attn_1_config.get('head_dim', 64),
            use_attention_gate=multi_attn_1_config.get('use_attention_gate', True),
            cvloss=multi_attn_1_config.get('cvloss', 0.05),
            aux_loss=multi_attn_1_config.get('aux_loss', 1),
            zloss=multi_attn_1_config.get('zloss', 0.001),
            sample_topk=multi_attn_1_config.get('sample_topk', 0),
            noisy_gating=multi_attn_1_config.get('noisy_gating', True),
            use_pos_bias=multi_attn_1_config.get('use_pos_bias', False)
        )

        # LayerNorm for attention residual
        self.attn_norm = nn.LayerNorm(multi_attn_1_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_1_config = config.get('moe_1', {})

        # 创建MoE实例
        self.moe_1 = MoE(
            input_size=moe_1_config.get('input_size', 128),
            head_size=moe_1_config.get('head_size', 512),
            num_experts=moe_1_config.get('num_experts', 8),
            k=moe_1_config.get('k', 4),
            need_merge=moe_1_config.get('need_merge', False),
            cvloss=moe_1_config.get('cvloss', 0.05),
            aux_loss=moe_1_config.get('aux_loss', 1),
            zloss=moe_1_config.get('zloss', 0.001),
            bias=moe_1_config.get('bias', False),
            activation=nn.GELU(),
            noisy_gating=moe_1_config.get('noisy_gating', True)
        )

        # 专门为moe_output_1定义的归一化层
        self.moe_1_norm = nn.LayerNorm(moe_1_config.get('input_size', 128))

        # 获取MultiheadAttention2配置
        multi_attn_2_config = config.get('multi_attention_2', {})

        # 实例化MultiheadAttention2
        self.multi_attention_2 = MultiheadAttention(
            embed_dim=multi_attn_2_config.get('embed_dim', 128),
            num_heads=multi_attn_2_config.get('num_heads', 2),
            dropout=multi_attn_2_config.get('dropout', 0.1),
            bias=multi_attn_2_config.get('bias', True),
            q_noise=multi_attn_2_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_2_config.get('qn_block_size', 8),
            num_expert=multi_attn_2_config.get('num_expert', 8),
            head_dim=multi_attn_2_config.get('head_dim', 64),
            use_attention_gate=multi_attn_2_config.get('use_attention_gate', False),
            cvloss=multi_attn_2_config.get('cvloss', 0.05),
            aux_loss=multi_attn_2_config.get('aux_loss', 1),
            zloss=multi_attn_2_config.get('zloss', 0.001),
            sample_topk=multi_attn_2_config.get('sample_topk', 0),
            noisy_gating=multi_attn_2_config.get('noisy_gating', True),
            use_pos_bias=multi_attn_2_config.get('use_pos_bias', False)
        )

        # LayerNorm for attention2 residual
        self.attn2_norm = nn.LayerNorm(multi_attn_2_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_2_config = config.get('moe_2', {})

        # 创建MoE实例
        self.moe_2 = MoE(
            input_size=moe_2_config.get('input_size', 128),
            head_size=moe_2_config.get('head_size', 512),
            num_experts=moe_2_config.get('num_experts', 8),
            k=moe_2_config.get('k', 2),
            need_merge=moe_2_config.get('need_merge', True),
            cvloss=moe_2_config.get('cvloss', 0.05),
            aux_loss=moe_2_config.get('aux_loss', 1),
            zloss=moe_2_config.get('zloss', 0.001),
            bias=moe_2_config.get('bias', False),
            activation=nn.GELU(),
            noisy_gating=moe_2_config.get('noisy_gating', True)
        )

        # 专门为moe_output_1定义的归一化层
        self.moe_2_norm = nn.LayerNorm(moe_2_config.get('input_size', 128))

        # 实例化PatchNeighborSearcher
        self.patch_neighbor_searcher = PatchNeighborSearcher()

        multi_attn_3_config = config.get('multi_attention_3', {})

        # 实例化MultiheadAttention3
        self.multi_attention_3 = MultiheadAttention(
            embed_dim=multi_attn_3_config.get('embed_dim', 128),
            num_heads=multi_attn_3_config.get('num_heads', 2),
            dropout=multi_attn_3_config.get('dropout', 0.1),
            bias=multi_attn_3_config.get('bias', True),
            q_noise=multi_attn_3_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_3_config.get('qn_block_size', 8),
            num_expert=multi_attn_3_config.get('num_expert', 8),
            head_dim=multi_attn_3_config.get('head_dim', 64),
            use_attention_gate=multi_attn_3_config.get('use_attention_gate', False),
            cvloss=multi_attn_3_config.get('cvloss', 0.05),
            aux_loss=multi_attn_3_config.get('aux_loss', 1),
            zloss=multi_attn_3_config.get('zloss', 0.001),
            sample_topk=multi_attn_3_config.get('sample_topk', 0),
            noisy_gating=multi_attn_3_config.get('noisy_gating', True),
            use_pos_bias=multi_attn_3_config.get('use_pos_bias', True)
        )

        # LayerNorm for attention2 residual
        self.attn3_norm = nn.LayerNorm(multi_attn_3_config.get('embed_dim', 128))

        # 获取MoE配置
        moe_3_config = config.get('moe_3', {})

        # 创建MoE实例
        self.moe_3 = MoE(
            input_size=moe_3_config.get('input_size', 128),
            head_size=moe_3_config.get('head_size', 512),
            num_experts=moe_3_config.get('num_experts', 8),
            k=moe_3_config.get('k', 2),
            need_merge=moe_3_config.get('need_merge', True),
            cvloss=moe_3_config.get('cvloss', 0.05),
            aux_loss=moe_3_config.get('aux_loss', 1),
            zloss=moe_3_config.get('zloss', 0.001),
            bias=moe_3_config.get('bias', False),
            activation=nn.GELU(),
            noisy_gating=moe_3_config.get('noisy_gating', True)
        )

        # 归一化层
        self.moe_3_norm = nn.LayerNorm(moe_3_config.get('input_size', 128))

        multi_attn_4_config = config.get('multi_attention_4', {})

        # 实例化MultiheadAttention4（自注意力）
        self.multi_attention_4 = MultiheadAttention(
            embed_dim=multi_attn_4_config.get('embed_dim', 128),
            num_heads=multi_attn_4_config.get('num_heads', 2),  # 可以适当增加头数
            dropout=multi_attn_4_config.get('dropout', 0.1),
            bias=multi_attn_4_config.get('bias', True),
            q_noise=multi_attn_4_config.get('q_noise', 0.005),
            qn_block_size=multi_attn_4_config.get('qn_block_size', 8),
            num_expert=multi_attn_4_config.get('num_expert', 8),
            head_dim=multi_attn_4_config.get('head_dim', 64),
            use_attention_gate=multi_attn_4_config.get('use_attention_gate', False),
            cvloss=multi_attn_4_config.get('cvloss', 0.05),
            aux_loss=multi_attn_4_config.get('aux_loss', 1),
            zloss=multi_attn_4_config.get('zloss', 0.001),
            sample_topk=multi_attn_4_config.get('sample_topk', 0),
            noisy_gating=multi_attn_4_config.get('noisy_gating', True),
            use_pos_bias=multi_attn_4_config.get('use_pos_bias', False)  # 自注意力中可以不使用位置偏置
        )

        # LayerNorm for attention2 residual
        self.attn4_norm = nn.LayerNorm(multi_attn_4_config.get('embed_dim', 128))

        # 获取ISP解码器配置
        isp_1_config = config.get('isp_1', {})
        isp_2_config = config.get('isp_2', {})
        isp_3_config = config.get('isp_3', {})
        isp_4_config = config.get('isp_4', {})

        # 实例化四个ISP解码器
        self.isp_1 = ISPDecoder(
            latent_dim=isp_1_config.get('latent_dim', 128),
            param_dim=isp_1_config.get('param_dim', 1),
            hidden_dims=isp_1_config.get('hidden_dims', [512, 256, 128, 64])
        )

        self.isp_2 = ISPDecoder(
            latent_dim=isp_2_config.get('latent_dim', 128),
            param_dim=isp_2_config.get('param_dim', 3),
            hidden_dims=isp_2_config.get('hidden_dims', [512, 256, 128, 64])
        )

        self.isp_3 = ISPDecoder(
            latent_dim=isp_3_config.get('latent_dim', 128),
            param_dim=isp_3_config.get('param_dim', 9),
            hidden_dims=isp_3_config.get('hidden_dims', [512, 256, 128, 64])
        )

        self.isp_4 = ISPDecoder(
            latent_dim=isp_4_config.get('latent_dim', 128),
            param_dim=isp_4_config.get('param_dim', 7),
            hidden_dims=isp_4_config.get('hidden_dims', [512, 256, 128, 64])
        )

    def forward(self, x, target=None, compute_loss=None):
        """
        定义模型的前向传播过程

        Args:
            x: 输入的低光照图像
            target: 目标正常光照图像。训练模式下必须提供，否则无法计算损失
            compute_loss: 是否计算损失。如果为None，则根据self.training判断

        Returns:
            如果compute_loss为True: (enhanced_image, loss) - 增强后的图像和计算的损失
            如果compute_loss为False: enhanced_image - 仅返回增强后的图像
        """
        # 确定是否计算损失
        if compute_loss is None:
            compute_loss = self.training

        # 如果需要计算损失但没有提供target，则抛出错误
        if compute_loss and target is None:
            raise ValueError("计算损失时必须提供target图像")

        batch_size, channels, height, width = x.shape
        # 特征提取
        features, h_windows, w_windows = self.feature_extractor(x)

        window_size = self.patch_size * self.win_size  # 计算窗口大小

        # 重塑输入图像为[batch_size, channels, h_windows, window_size, w_windows, window_size]格式
        x_reshaped = x.view(batch_size, channels, h_windows, window_size, w_windows, window_size)

        # 调整维度顺序为[batch_size, h_windows, w_windows, window_size, window_size, channels]
        x_reordered = x_reshaped.permute(0, 2, 4, 3, 5, 1)

        # 重塑为[batch_size, h_windows * w_windows, window_size, window_size, channels]
        x_win = x_reordered.reshape(batch_size, h_windows * w_windows, window_size, window_size, channels)

        # EispGeneratorFFN处理
        e_isp = self.emb_gen(features)

        e_isp_big = self.e_isp_expand(e_isp)

        # QKIspGenerator处理
        q_isp, k_isp = self.qk_gen(e_isp)

        # KVFeatureGenerator处理
        k_features, v_features = self.kv_gen(features)

        # MultiheadAttention处理attn_output_1[batch_size, patches, seq_len, input_size]
        attn_output_1, _, atten_aux_loss_1 = self.multi_attention_1(
            query=q_isp,
            key=k_features,
            value=v_features,
            k_isp=k_isp,
            need_weights=False,
            before_softmax=False,
            need_head_weights=False
        )

        # 归一化 & 残差连接
        attn_output_1 = self.attn_norm(attn_output_1) + q_isp

        # MoE处理
        moe_output_1, moe_aux_loss_1 = self.moe_1(
            x=attn_output_1,
            sample_topk=0,
            multiply_by_gates=True
        )

        # 归一化处理moe_output_1
        moe_output_1 = self.moe_1_norm(moe_output_1)  # 归一化操作
        moe_output_1 = self.moe_1.concat(moe_output_1, e_isp_big)
        moe_output_1 = moe_output_1.squeeze(2)

        # MultiheadAttention2处理
        attn_output_2, _, atten_aux_loss_2 = self.multi_attention_2(
            query=moe_output_1,
            key=moe_output_1,
            value=moe_output_1,
            k_isp=None,
            need_weights=False,
            before_softmax=False,
            need_head_weights=False
        )

        # 归一化处理attn_output_2
        attn_output_2 = self.attn2_norm(attn_output_2) + moe_output_1

        # MOE处理
        moe_output_2, moe_aux_loss_2 = self.moe_2(
            x=attn_output_2,
            sample_topk=0,
            multiply_by_gates=True
        )

        # 归一化处理moe_output_2+残差连接
        moe_output_2 = self.moe_2_norm(moe_output_2) + attn_output_2

        # PatchNeighborSearcher 模块
        neighbor_features = self.patch_neighbor_searcher(moe_output_2, h_windows, w_windows)

        # MultiheadAttention3处理
        attn_output_3, _, atten_aux_loss_3 = self.multi_attention_3(
            query=moe_output_2,
            key=neighbor_features,
            value=neighbor_features,
            k_isp=None,
            need_weights=False,
            before_softmax=False,
            need_head_weights=False
        )

        # 归一化处理attn_output_3
        attn_output_3 = self.attn3_norm(attn_output_3) + moe_output_2

        # MoE处理
        moe_output_3, moe_aux_loss_3 = self.moe_3(
            x=attn_output_3,
            sample_topk=0,
            multiply_by_gates=True
        )

        # 归一化处理moe_output_2
        moe_output_3 = self.moe_3_norm(moe_output_3) + attn_output_3

        # 自注意力处理
        attn_output_4, _, atten_aux_loss_4 = self.multi_attention_4(
            query=moe_output_3,
            key=moe_output_3,  # 自注意力：key和query相同
            value=moe_output_3,  # 自注意力：value和query相同
            k_isp=None,
            need_weights=False,
            before_softmax=False,
            need_head_weights=False
        )

        # 残差连接和归一化
        attn_output_4 = self.attn4_norm(attn_output_4) + moe_output_3

        # 1. 合并前两个维度
        bsz, patches, k, head_dim = attn_output_4.shape
        attn_output_4 = attn_output_4.reshape(-1, k, head_dim)  # [bsz*patches, tgt_len, head_dim]

        # 2. 将原始的第三个维度（tgt_len）放到第一个位置
        attn_output_4 = attn_output_4.transpose(0, 1)  # [tgt_len, bsz*patches, head_dim]

        # 直接从转置后的张量中获取四个专家嵌入
        isp_expert_1 = attn_output_4[0]
        isp_expert_2 = attn_output_4[1]
        isp_expert_3 = attn_output_4[2]
        isp_expert_4 = attn_output_4[3]

        # 分别输入到四个解码器中
        isp_final_1 = self.isp_1(isp_expert_1)  # [num_windows, param_dim_1]
        isp_final_2 = self.isp_2(isp_expert_2)
        isp_final_3 = self.isp_3(isp_expert_3)
        isp_final_4 = self.isp_4(isp_expert_4)

        # 获取配置中的param_dim参数
        param_dim_1 = self.isp_1.param_dim  # 通过解码器输出层获取param_dim
        param_dim_2 = self.isp_2.param_dim
        param_dim_3 = self.isp_3.param_dim
        param_dim_4 = self.isp_4.param_dim

        isp_final_1 = isp_final_1.reshape(-1, h_windows * w_windows, param_dim_1)
        isp_final_2 = isp_final_2.reshape(-1, h_windows * w_windows, param_dim_2)
        isp_final_3 = isp_final_3.reshape(-1, h_windows * w_windows, 3, 3)
        isp_final_4 = isp_final_4.reshape(-1, h_windows * w_windows, param_dim_4)

        enhanced_image = x_win

        # 按顺序应用四个增强方法
        # 4. 去噪与锐化
        denoising_sharpening = DenoisingSharpening()
        enhanced_image_1 = denoising_sharpening(images=enhanced_image, params=isp_final_4)

        # 1. LIME增强
        enhanced_image_2 = patch_gamma_correct(L=enhanced_image_1, gamma_increment = isp_final_1)
        # 2. 白平衡
        enhanced_image_3 = white_balance(predicted_gains=isp_final_2, I_source=enhanced_image_2)

        # 3. 色彩校正矩阵
        enhanced_image_4 = apply_color_correction_matrices(I_source=enhanced_image_3, pred_matrices=isp_final_3)


        # 将增强后的窗口图像重构回原始图像形状
        enhanced_image_4 = enhanced_image_4.permute(0, 1, 4, 2, 3)  # [B, P, C, H, W]
        enhanced_image_4 = enhanced_image_4.reshape(
            batch_size, h_windows, w_windows, channels, window_size, window_size
        )
        enhanced_image_4 = enhanced_image_4.permute(0, 3, 1, 4, 2, 5)  # [B, C, h_win, win_size, w_win, win_size]
        enhanced_image_4 = enhanced_image_4.reshape(batch_size, channels,
                                                h_windows * window_size, w_windows * window_size)

        # 训练模式下才计算损失并返回
        if self.training:
            # 计算总的辅助损失
            auxiliary_loss = (atten_aux_loss_1 + moe_aux_loss_1 + atten_aux_loss_2 + \
                             moe_aux_loss_2 + atten_aux_loss_3 + moe_aux_loss_3 + atten_aux_loss_4) / 7

            # 使用目标图像计算重建损失
            l1_loss = L1ReconstructionLoss()

            reconstruction_loss = l1_loss(enhanced_image_4, target)

            # 使用权重系数组合不同的损失
            loss = (self.reconstruction_weight * reconstruction_loss +
                    self.auxiliary_weight * auxiliary_loss)

            return enhanced_image_4, loss, reconstruction_loss
        else:
            # 测试模式，只返回增强后的图像
            return enhanced_image_4


import torch
import torch.nn as nn

def initialize_2d_relative_position_bias(self, heads=1):
    """
    初始化二维相对位置编码，专为3×3网格中心位置对周围8个位置的注意力设计

    参数:
        heads: 注意力头的数量
    """
    # 直接创建位置偏置嵌入层，8个固定的相对位置
    self.pos_bias = nn.Embedding(8, heads)

    # 直接使用从0到7的索引表示8个位置关系
    pos_indices = torch.arange(8)

    # 注册为模型缓冲区
    self.register_buffer('pos_indices', pos_indices)

    # 为了清晰，可以保存相对位置的语义标记（可选）
    # 例如：0=左上, 1=上, 2=右上, 3=左, 4=右, 5=左下, 6=下, 7=右下


def apply_center_to_surrounding_pos_bias(self, fmap, scale):
    """
    将位置偏置应用到中心对周围的注意力分数矩阵上

    参数:
        fmap: 注意力分数矩阵，形状为[batch, heads, 1, 8]
        scale: 缩放因子

    返回:
        添加了位置偏置的注意力分数矩阵
    """
    # 从嵌入层获取位置偏置
    bias = self.pos_bias(self.pos_indices)  # 形状为[8, heads]

    # 重排为[1, heads, 1, 8]
    bias = bias.permute(1, 0)  # 将 [8, heads] 变为 [heads, 8]
    bias = bias.unsqueeze(0).unsqueeze(2)  # 添加批次和查询维度，得到 [1, heads, 1, 8]

    # 应用偏置
    return fmap + (bias / scale)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
import numpy as np
from datetime import datetime
import argparse

# 导入自定义模块
from data_loader import get_data_loaders
from models import Model


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    训练一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备（GPU/CPU）
        epoch: 当前epoch

    Returns:
        平均总损失和平均重建损失
    """
    model.train()  # 设置为训练模式
    total_loss = 0
    total_reconstruction_loss = 0

    # 使用tqdm创建进度条
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch_idx, batch_data in progress_bar:
        # 获取低光照输入和正常光照目标
        inputs = batch_data['input'].to(device)
        targets = batch_data['target'].to(device)  # 从数据加载器中获取目标图像

        # 清除梯度
        optimizer.zero_grad()

        # 前向传播 - 现在模型返回三个值：增强图像、总损失和重建损失
        enhanced_images, loss, reconstruction_loss = model(inputs, targets)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累计损失
        total_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()

        # 更新进度条，同时显示总损失和重建损失
        progress_bar.set_postfix(loss=loss.item(), recon_loss=reconstruction_loss.item())

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_reconstruction_loss / len(train_loader)

    return avg_loss, avg_recon_loss


def validate(model, val_loader, device):
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备（GPU/CPU）

    Returns:
        平均验证总损失和平均验证重建损失
    """
    model.eval()  # 设置为评估模式
    total_val_loss = 0
    total_val_recon_loss = 0

    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data['input'].to(device)
            targets = batch_data['target'].to(device)  # 获取目标图像

            # 临时将模型设为训练模式以计算损失，但保持梯度禁用
            model.train()
            enhanced_images, loss, reconstruction_loss = model(inputs, targets)
            model.eval()  # 恢复为评估模式

            total_val_loss += loss.item()
            total_val_recon_loss += reconstruction_loss.item()

    # 计算平均验证损失
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_recon_loss = total_val_recon_loss / len(val_loader)

    return avg_val_loss, avg_val_recon_loss


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建结果和检查点目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 初始化数据加载器
    root_dir = os.path.expanduser(args.data_dir)
    train_loader, val_loader_1, _, _ = get_data_loaders(
        root_dir=root_dir,
        batch_size=args.batch_size,  # 使用命令行参数中的batch size
        eval_batch_size=32,  # 评估时使用32的batch size
        num_workers=args.num_workers  # 使用命令行参数中的线程数
    )

    # 初始化模型
    model = Model().to(device)

    # 检查是否从检查点恢复
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")

    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 如果恢复训练，也加载优化器状态
    if args.resume and os.path.isfile(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state")

    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 训练日志
    log_file = os.path.join(save_dir, 'training_log.txt')
    if start_epoch == 1:  # 如果是新训练，创建新日志文件
        with open(log_file, 'w') as f:
            f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Args: {args}\n\n")
            f.write("Epoch,Train Loss,Train Recon Loss,Val Loss,Val Recon Loss,Learning Rate,Time(s)\n")

    best_val_loss = float('inf')
    # 如果恢复训练，更新最佳验证损失
    if args.resume and os.path.isfile(args.resume) and 'val_loss' in checkpoint:
        best_val_loss = checkpoint['val_loss']
        print(f"Restored best validation loss: {best_val_loss:.4f}")

    # 早停相关变量初始化
    early_stopping_counter = 0
    early_stopping_min_val_loss = float('inf')

    # 开始训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        # 训练一个epoch
        train_loss, train_recon_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # 验证 (仅使用val_loader_1)
        val_loss, val_recon_loss = validate(model, val_loader_1, device)

        # 学习率调度器更新
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 计算用时
        epoch_time = time.time() - start_time

        # 打印结果，现在包括重建损失
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Recon Loss: {train_recon_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Recon Loss: {val_recon_loss:.4f}, "
              f"LR: {current_lr:.6f}, "
              f"Time: {epoch_time:.2f}s")

        # 记录日志，包括重建损失
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{train_recon_loss:.6f},{val_loss:.6f},{val_recon_loss:.6f},{current_lr:.6f},{epoch_time:.2f}\n")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存完整检查点
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_recon_loss': val_recon_loss,
                'train_loss': train_loss,
                'train_recon_loss': train_recon_loss
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")

            # 额外保存一个只包含模型权重的文件，便于部署
            weights_path = os.path.join(save_dir, 'best_model_weights.pth')
            torch.save(model.state_dict(), weights_path)
            print(f"Saved best model weights to {weights_path}")

        # 每隔特定epochs保存一次模型
        if epoch % args.save_interval == 0:
            # 保存完整检查点
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_recon_loss': val_recon_loss,
                'train_loss': train_loss,
                'train_recon_loss': train_recon_loss
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            # 保存权重文件
            weights_path = os.path.join(save_dir, f'model_epoch_{epoch}_weights.pth')
            torch.save(model.state_dict(), weights_path)
            print(f"Saved model weights to {weights_path}")

        # 早停逻辑
        if val_loss < early_stopping_min_val_loss - args.min_delta:
            # 验证损失显著下降，重置计数器
            early_stopping_min_val_loss = val_loss
            early_stopping_counter = 0
            print(f"Validation loss decreased significantly. Early stopping counter reset.")
        else:
            # 验证损失没有显著下降，增加计数器
            early_stopping_counter += 1
            print(
                f"Validation loss did not decrease significantly. Early stopping counter: {early_stopping_counter}/{args.patience}")

            # 如果连续多个epoch没有改善，触发早停
            if early_stopping_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs!")
                break

    # 保存最终模型
    final_checkpoint_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_recon_loss': val_recon_loss,
        'train_loss': train_loss,
        'train_recon_loss': train_recon_loss
    }, final_checkpoint_path)
    print(f"Training completed. Final model saved to {final_checkpoint_path}")

    # 保存最终模型权重
    final_weights_path = os.path.join(save_dir, 'final_model_weights.pth')
    torch.save(model.state_dict(), final_weights_path)
    print(f"Final model weights saved to {final_weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train low-light image enhancement model")
    parser.add_argument('--data_dir', type=str, default="/root/autodl-tmp/swin-MOA/LOL-v2-Dataset/archive/LOL-v2/Real_captured",
                        help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='directory to save models')
    parser.add_argument('--save_interval', type=int, default=10, help='save model every N epochs')
    parser.add_argument('--num_workers', type=int, default=20, help='number of worker threads for dataloader')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    # 添加早停相关参数
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值：验证损失多少个epoch没有显著下降后停止训练')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='定义验证损失"显著"下降的最小幅度')

    args = parser.parse_args()
    main(args)

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchNeighborSearcher(nn.Module):
    """
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
        batches, patches, length, input_size = E_flat.shape
        assert patches == H * W, (
            f"patches ({patches}) 必须等于 H*W ({H}*{W}={H*W})"
        )

        # ----- 1. 恢复为网格形式 [B, H, W, L, D] -----
        E_grid = E_flat.view(batches, H, W, length, input_size)

        # ----- 2. 合并 length 和 input_size 作为通道维度，准备用 Unfold -----
        # [B, H, W, L, D] -> [B, L, D, H, W]
        X = E_grid.permute(0, 3, 4, 1, 2)
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
        neighbor_indices = [0, 1, 2, 3, 5, 6, 7, 8]
        neigh = patches_unfold[:, :, neighbor_indices, :]  # [B, L*D, 8, P]

        # ----- 6. 恢复维度并 permute -----
        neigh = neigh.view(batches, length, input_size, 8, patches)
        neigh = neigh.permute(0, 4, 1, 3, 2)  # [B, P, L, 8, D]

        # ----- 7. 合并 length 和 8 维度 -> length*8 -----
        # [B, P, L, 8, D] -> [B, P, L*8, D]
        neigh_features = neigh.reshape(batches, patches, length * 8, input_size)

        return neigh_features


class ISPParameterGenerator(nn.Module):
    """
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
        # 获取输入张量的形状
        batches, windows, k, embed_dim = isp_per_win.shape
        num_windows = batches * windows

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

        # 使用index_put_操作将嵌入分配到输出张量中
        # 注意：如果有多个相同(expert_idx, win_idx)对，后面的会覆盖前面的
        expert_embeddings.index_put_(
            (flat_expert_indices, flat_win_indices),
            flat_embeddings
        )

        return expert_embeddings


# 使用示例
# 使用示例
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)

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















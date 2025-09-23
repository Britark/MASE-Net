#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试窗口重叠实现的脚本
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models import Model
from config import default_config


def test_window_overlap():
    """测试窗口重叠功能"""
    print("=" * 80)
    print("测试窗口重叠实现")
    print("=" * 80)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建配置
    config = default_config()

    # 测试两种模式
    for overlap_mode in [False, True]:
        print(f"\n{'=' * 60}")
        print(f"测试模式: {'重叠窗口' if overlap_mode else '非重叠窗口'}")
        print(f"{'=' * 60}")

        # 更新配置
        config['image_size']['window_overlap'] = overlap_mode
        config['image_size']['overlap_ratio'] = 0.5
        config['image_size']['blend_mode'] = 'gaussian'

        # 创建模型
        model = Model(config).to(device)
        model.eval()  # 设置为评估模式

        # 创建测试输入 - 棋盘格模式的图像，便于观察窗口边界效果
        batch_size = 1
        height, width = 400, 600

        # 创建棋盘格测试图像
        checkerboard = create_checkerboard_image(height, width, square_size=40)
        test_input = torch.tensor(checkerboard, dtype=torch.float32).unsqueeze(0).to(device)

        print(f"输入形状: {test_input.shape}")

        # 前向传播
        with torch.no_grad():
            try:
                output = model(test_input)
                print(f"输出形状: {output.shape}")
                print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

                # 计算输出的平滑度（相邻像素差异）
                diff_h = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]).mean()
                diff_w = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1]).mean()
                smoothness = (diff_h + diff_w) / 2
                print(f"输出平滑度指标: {smoothness.item():.6f} (越小越平滑)")

            except Exception as e:
                print(f"错误: {str(e)}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


def create_checkerboard_image(height, width, square_size=40):
    """创建棋盘格测试图像"""
    image = np.zeros((3, height, width), dtype=np.float32)

    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                # 亮方块 - 不同通道设置不同值以测试色彩处理
                image[0, i:i + square_size, j:j + square_size] = 0.8  # R
                image[1, i:i + square_size, j:j + square_size] = 0.7  # G
                image[2, i:i + square_size, j:j + square_size] = 0.6  # B
            else:
                # 暗方块
                image[0, i:i + square_size, j:j + square_size] = 0.2  # R
                image[1, i:i + square_size, j:j + square_size] = 0.3  # G
                image[2, i:i + square_size, j:j + square_size] = 0.4  # B

    return image


def test_blend_weights():
    """测试混合权重的生成"""
    from utils_overlap import WindowOverlapProcessor

    print("\n" + "=" * 80)
    print("测试混合权重生成")
    print("=" * 80)

    for blend_mode in ['gaussian', 'linear']:
        print(f"\n混合模式: {blend_mode}")
        processor = WindowOverlapProcessor(win_size=4, blend_mode=blend_mode)

        if processor.blend_weights is not None:
            weights = processor.blend_weights.numpy()
            print(f"权重形状: {weights.shape}")
            print(f"权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
            print(f"权重和: {weights.sum():.3f}")

            # 可视化权重
            plt.figure(figsize=(6, 6))
            plt.imshow(weights, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(f'{blend_mode.capitalize()} Blend Weights')
            plt.tight_layout()
            plt.savefig(f'blend_weights_{blend_mode}.png', dpi=150)
            plt.close()
            print(f"权重可视化已保存为: blend_weights_{blend_mode}.png")


def test_isp_parameter_merging():
    """测试ISP参数合并功能"""
    from utils_overlap import merge_overlapping_params

    print("\n" + "=" * 80)
    print("测试ISP参数合并")
    print("=" * 80)

    # 模拟参数
    batch_size = 2
    num_windows = 9  # 3x3窗口网格
    win_size = 4
    stride = 2  # 50%重叠

    # 创建测试ISP参数
    isp_params_1 = torch.randn(batch_size, num_windows, 1)  # gamma
    isp_params_2 = torch.randn(batch_size, num_windows, 3)  # white balance
    isp_params_3 = torch.randn(batch_size, num_windows, 9)  # color correction matrix

    # 创建窗口坐标
    window_coords = [(i * stride, j * stride) for i in range(3) for j in range(3)]

    # 测试合并
    params_list = [isp_params_1, isp_params_2, isp_params_3]

    try:
        merged_params = merge_overlapping_params(
            params_list, window_coords, 3, 3,
            win_size, stride, 12, 12, 'gaussian'
        )

        print("参数合并成功！")
        for i, params in enumerate(merged_params):
            print(f"参数{i + 1}形状: {params.shape}")

    except Exception as e:
        print(f"参数合并失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行所有测试
    test_window_overlap()
    test_blend_weights()
    test_isp_parameter_merging()

    print("\n所有测试完成！")
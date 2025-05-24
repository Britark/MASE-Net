import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# 导入自定义模块
from data_loader import get_data_loaders
from models import Model
from config import default_config


def test_model(args):
    """
    在验证集第一部分上测试模型并保存增强后的图像与原始图像的对比图

    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存结果的目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取验证集第一部分数据加载器
    root_dir = os.path.expanduser(args.data_dir)
    train_loader, val_loader_1, val_loader_2, test_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 创建配置并启用窗口重叠（如果指定）
    config = default_config()
    if args.use_window_overlap:
        print("启用窗口重叠推理")
        config['image_size']['window_overlap'] = True
        config['image_size']['overlap_ratio'] = args.overlap_ratio
        config['image_size']['blend_mode'] = args.blend_mode
        # 推理时不需要随机偏移
        config['image_size']['train_random_shift'] = False

    # 初始化模型
    model = Model(config).to(device)

    # 加载模型权重
    print(f"正在从 {args.weights_path} 加载模型权重")
    if args.weights_path.endswith('_weights.pth'):
        # 加载只包含权重的文件
        model.load_state_dict(torch.load(args.weights_path, map_location=device))
    else:
        # 加载包含完整训练状态的检查点
        checkpoint = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 设置为评估模式
    model.eval()

    # 进度条
    progress_bar = tqdm(enumerate(val_loader_1), total=len(val_loader_1), desc="在验证集1上测试")

    # 用于后续可视化的图像计数
    image_count = 0

    # 用于统计棋盘格伪影的减少情况
    smoothness_original = []
    smoothness_enhanced = []

    with torch.no_grad():
        for batch_idx, batch_data in progress_bar:
            # 获取低光照输入
            inputs = batch_data['input'].to(device)

            # 可以用于比较的参考图像
            targets = batch_data['target']

            # 推理 - 不计算损失
            enhanced_images = model(inputs)

            # 逐个保存增强后的图像
            for i in range(inputs.shape[0]):
                # 原始低光照图像
                low_light_img = inputs[i].cpu().permute(1, 2, 0).numpy()
                low_light_img = np.clip(low_light_img, 0, 1)

                # 增强后图像
                enhanced_img = enhanced_images[i].cpu().permute(1, 2, 0).numpy()
                enhanced_img = np.clip(enhanced_img, 0, 1)

                # 目标参考图像
                normal_light_img = targets[i].cpu().permute(1, 2, 0).numpy()
                normal_light_img = np.clip(normal_light_img, 0, 1)

                # 计算平滑度指标（用于评估棋盘格伪影）
                if args.compute_smoothness:
                    # 计算原始图像的边缘强度
                    orig_diff_h = np.abs(low_light_img[1:, :, :] - low_light_img[:-1, :, :]).mean()
                    orig_diff_w = np.abs(low_light_img[:, 1:, :] - low_light_img[:, :-1, :]).mean()
                    orig_smoothness = (orig_diff_h + orig_diff_w) / 2
                    smoothness_original.append(orig_smoothness)

                    # 计算增强图像的边缘强度
                    enh_diff_h = np.abs(enhanced_img[1:, :, :] - enhanced_img[:-1, :, :]).mean()
                    enh_diff_w = np.abs(enhanced_img[:, 1:, :] - enhanced_img[:, :-1, :]).mean()
                    enh_smoothness = (enh_diff_h + enh_diff_w) / 2
                    smoothness_enhanced.append(enh_smoothness)

                # 创建对比图 (原始低光照图像和增强后图像的对比)
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.imshow(low_light_img)
                plt.title('原始低光照图像')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(enhanced_img)
                plt.title(f'增强后图像{"(重叠窗口)" if args.use_window_overlap else "(非重叠窗口)"}')
                plt.axis('off')

                plt.tight_layout()
                comparison_path = os.path.join(args.output_dir, f"对比图_{image_count:04d}.png")
                plt.savefig(comparison_path, dpi=200)
                plt.close()

                # 如果需要，保存三图对比
                if args.save_comparison:
                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(low_light_img)
                    plt.title('原始低光照图像')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(enhanced_img)
                    plt.title(f'增强后图像{"(重叠窗口)" if args.use_window_overlap else "(非重叠窗口)"}')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(normal_light_img)
                    plt.title('正常光照参考图像')
                    plt.axis('off')

                    plt.tight_layout()
                    three_comparison_path = os.path.join(args.output_dir, f"三图对比_{image_count:04d}.png")
                    plt.savefig(three_comparison_path, dpi=200)
                    plt.close()

                # 保存单独的增强图像
                if args.save_enhanced:
                    enhanced_img_pil = Image.fromarray((enhanced_img * 255).astype(np.uint8))
                    enhanced_path = os.path.join(args.output_dir, f"增强图_{image_count:04d}.png")
                    enhanced_img_pil.save(enhanced_path)

                # 如果需要，保存原始低光照图像
                if args.save_originals:
                    low_light_img_pil = Image.fromarray((low_light_img * 255).astype(np.uint8))
                    low_light_path = os.path.join(args.output_dir, f"低光照原图_{image_count:04d}.png")
                    low_light_img_pil.save(low_light_path)

                image_count += 1

    print(f"验证集1测试完成。共生成 {image_count} 张对比图，保存在 {args.output_dir}")

    # 打印平滑度统计信息
    if args.compute_smoothness and smoothness_original:
        avg_orig_smooth = np.mean(smoothness_original)
        avg_enh_smooth = np.mean(smoothness_enhanced)
        improvement = (avg_orig_smooth - avg_enh_smooth) / avg_orig_smooth * 100

        print(f"\n平滑度分析:")
        print(f"原始图像平均边缘强度: {avg_orig_smooth:.6f}")
        print(f"增强图像平均边缘强度: {avg_enh_smooth:.6f}")
        print(f"边缘强度降低: {improvement:.2f}% (正值表示更平滑)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在验证集1上测试低光照图像增强模型")
    parser.add_argument('--data_dir', type=str,
                        default="/root/autodl-tmp/swin-MOA/LOL-v2-Dataset/archive/LOL-v2/Real_captured",
                        help='数据集路径')
    parser.add_argument('--weights_path', type=str,
                        default="/root/autodl-tmp/swin-MOA/checkpoints/final_model_weights.pth",
                        help='模型权重路径')
    parser.add_argument('--output_dir', type=str, default="./validation_results",
                        help='保存增强图像的目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='测试批次大小（通常val_loader_1为32）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作线程数')
    parser.add_argument('--save_comparison', action='store_true',
                        help='是否保存三图对比可视化')
    parser.add_argument('--save_originals', action='store_true',
                        help='是否保存原始低光照图像')
    parser.add_argument('--save_enhanced', action='store_true',
                        help='是否单独保存增强后图像')

    # 窗口重叠相关参数
    parser.add_argument('--use_window_overlap', action='store_true',
                        help='是否使用窗口重叠推理')
    parser.add_argument('--overlap_ratio', type=float, default=0.5,
                        help='窗口重叠比例（0-1之间）')
    parser.add_argument('--blend_mode', type=str, default='gaussian',
                        choices=['gaussian', 'linear'],
                        help='重叠区域的混合模式')
    parser.add_argument('--compute_smoothness', action='store_true',
                        help='是否计算平滑度指标来评估棋盘格伪影')

    args = parser.parse_args()
    test_model(args)
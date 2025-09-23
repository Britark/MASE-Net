import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json
import sys

# 添加上级目录到路径，以便导入模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from data_loader import get_data_loaders
from models import Model


def calculate_single_image_metrics(img1, img2):
    """
    计算两张图像之间的PSNR和SSIM

    Args:
        img1: 第一张图像 numpy array (H, W, C)
        img2: 第二张图像 numpy array (H, W, C)

    Returns:
        psnr_value: PSNR值
        ssim_value: SSIM值
    """
    # 确保数值在[0, 1]范围内
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)

    # 计算PSNR
    psnr_val = psnr(img1, img2, data_range=1.0)

    # 计算SSIM
    if img1.shape[-1] == 3:  # RGB图像
        ssim_val = ssim(img1, img2, data_range=1.0, channel_axis=-1)
    else:  # 灰度图像
        ssim_val = ssim(img1.squeeze(), img2.squeeze(), data_range=1.0)

    return psnr_val, ssim_val


def demo_model(args):
    """
    Demo演示模型效果

    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置输出目录为demo_results
    args.output_dir = "./demo_results"

    # 创建保存结果的目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取数据加载器
    root_dir = os.path.expanduser(args.data_dir)
    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Demo只使用测试集
    data_loader = test_loader
    print("Demo on test set")

    # 初始化模型
    model = Model().to(device)

    # 加载模型权重
    print(f"Loading model weights from {args.weights_path}")
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
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Demo processing")

    # 图像计数
    image_count = 0

    # 用于记录总体指标
    total_enhanced_psnr = 0
    total_enhanced_ssim = 0
    total_original_psnr = 0
    total_original_ssim = 0
    total_images = 0

    with torch.no_grad():
        for batch_idx, batch_data in progress_bar:
            # 获取低光照输入
            inputs = batch_data['input'].to(device)

            # 可以用于比较的参考图像（groundtruth）
            targets = batch_data['target']

            # 推理
            enhanced_images = model(inputs)

            # 逐个保存增强后的图像
            for i in range(inputs.shape[0]):
                # 原始低光照图像
                low_light_img = inputs[i].cpu().permute(1, 2, 0).numpy()
                low_light_img = np.clip(low_light_img, 0, 1)

                # 增强后图像
                enhanced_img = enhanced_images[i].cpu().permute(1, 2, 0).numpy()
                enhanced_img = np.clip(enhanced_img, 0, 1)

                # 目标参考图像（groundtruth）
                normal_light_img = targets[i].cpu().permute(1, 2, 0).numpy()
                normal_light_img = np.clip(normal_light_img, 0, 1)

                # 计算指标
                original_psnr, original_ssim = calculate_single_image_metrics(low_light_img, normal_light_img)
                enhanced_psnr, enhanced_ssim = calculate_single_image_metrics(enhanced_img, normal_light_img)

                # 累积总体指标
                total_enhanced_psnr += enhanced_psnr
                total_enhanced_ssim += enhanced_ssim
                total_original_psnr += original_psnr
                total_original_ssim += original_ssim
                total_images += 1

                # 创建三图对比图
                plt.figure(figsize=(18, 6))

                plt.subplot(1, 3, 1)
                plt.imshow(low_light_img)
                plt.title(f'Original Low-light Image\nPSNR: {original_psnr:.2f} dB, SSIM: {original_ssim:.4f}',
                          fontsize=12)
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(enhanced_img)
                plt.title(f'Enhanced Image\nPSNR: {enhanced_psnr:.2f} dB, SSIM: {enhanced_ssim:.4f}',
                          fontsize=12)
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(normal_light_img)
                plt.title('Ground Truth\n(Reference)', fontsize=12)
                plt.axis('off')

                plt.tight_layout()
                comparison_path = os.path.join(args.output_dir, f"demo_comparison_{image_count:04d}.png")
                plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
                plt.close()

                # 保存单独的增强图像
                if args.save_enhanced:
                    enhanced_img_pil = Image.fromarray((enhanced_img * 255).astype(np.uint8))
                    enhanced_path = os.path.join(args.output_dir, f"demo_enhanced_{image_count:04d}.png")
                    enhanced_img_pil.save(enhanced_path)

                # 如果需要，保存原始低光照图像
                if args.save_originals:
                    low_light_img_pil = Image.fromarray((low_light_img * 255).astype(np.uint8))
                    low_light_path = os.path.join(args.output_dir, f"demo_low_light_{image_count:04d}.png")
                    low_light_img_pil.save(low_light_path)

                image_count += 1

                # 在进度条中显示当前图像的指标
                progress_bar.set_postfix(
                    enh_psnr=f"{enhanced_psnr:.2f}",
                    enh_ssim=f"{enhanced_ssim:.4f}",
                    orig_psnr=f"{original_psnr:.2f}",
                    orig_ssim=f"{original_ssim:.4f}"
                )


    # 计算平均指标
    avg_enhanced_psnr = total_enhanced_psnr / total_images
    avg_enhanced_ssim = total_enhanced_ssim / total_images
    avg_original_psnr = total_original_psnr / total_images
    avg_original_ssim = total_original_ssim / total_images

    # 保存Demo结果
    save_demo_results(avg_enhanced_psnr, avg_enhanced_ssim, avg_original_psnr, avg_original_ssim, total_images, args.output_dir)

    print(f"\n=== Demo结果 ===")
    print(f"处理图像总数: {total_images}")
    print(f"原始低光照图像平均 PSNR: {avg_original_psnr:.2f} dB")
    print(f"原始低光照图像平均 SSIM: {avg_original_ssim:.4f}")
    print(f"增强后图像平均 PSNR: {avg_enhanced_psnr:.2f} dB")
    print(f"增强后图像平均 SSIM: {avg_enhanced_ssim:.4f}")
    print(f"PSNR 提升: {avg_enhanced_psnr - avg_original_psnr:.2f} dB")
    print(f"SSIM 提升: {avg_enhanced_ssim - avg_original_ssim:.4f}")

    print(f"\nDemo completed. Results saved in {args.output_dir}")


def save_demo_results(avg_enhanced_psnr, avg_enhanced_ssim, avg_original_psnr, avg_original_ssim, total_images, output_dir):
    """保存Demo结果到文件"""
    results_file = os.path.join(output_dir, "demo_results.txt")
    json_file = os.path.join(output_dir, "demo_results.json")

    # 保存文本格式
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== 图像增强Demo结果 ===\n\n")
        f.write(f"处理图像总数: {total_images}\n")
        f.write(f"原始低光照图像平均 PSNR: {avg_original_psnr:.2f} dB\n")
        f.write(f"原始低光照图像平均 SSIM: {avg_original_ssim:.4f}\n")
        f.write(f"增强后图像平均 PSNR: {avg_enhanced_psnr:.2f} dB\n")
        f.write(f"增强后图像平均 SSIM: {avg_enhanced_ssim:.4f}\n")
        f.write(f"PSNR 提升: {avg_enhanced_psnr - avg_original_psnr:.2f} dB\n")
        f.write(f"SSIM 提升: {avg_enhanced_ssim - avg_original_ssim:.4f}\n")

    # 保存JSON格式
    results_data = {
        'total_images': total_images,
        'avg_original_psnr': avg_original_psnr,
        'avg_original_ssim': avg_original_ssim,
        'avg_enhanced_psnr': avg_enhanced_psnr,
        'avg_enhanced_ssim': avg_enhanced_ssim,
        'psnr_improvement': avg_enhanced_psnr - avg_original_psnr,
        'ssim_improvement': avg_enhanced_ssim - avg_original_ssim
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"详细结果已保存到: {results_file}")
    print(f"JSON格式结果已保存到: {json_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo low-light image enhancement model")

    # 基本参数
    # Demo专用数据集路径
    parser.add_argument('--data_dir', type=str,
                        default="./lol_dataset",
                        help='Demo dataset path (relative to demo directory)')

    # 使用LOLv1权重
    parser.add_argument('--weights_path', type=str,
                        default="../checkpoints/LOLv1_checkpoints.pth",
                        help='Model weights path (relative to demo directory)')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Demo batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')

    # 保存选项
    parser.add_argument('--save_enhanced', action='store_true',
                        default=True,
                        help='Whether to save enhanced images separately')
    parser.add_argument('--save_originals', action='store_true',
                        help='Whether to save original low-light images')

    args = parser.parse_args()

    print(f"=== Demo配置 ===")
    print(f"数据集: 测试集(test)")
    print(f"模型权重: {args.weights_path}")
    print(f"数据路径: {args.data_dir}")

    demo_model(args)
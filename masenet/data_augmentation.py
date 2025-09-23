import os
import cv2
import numpy as np
import argparse
from PIL import Image, ImageFilter
import random
from tqdm import tqdm
import glob


def random_scale_rotate_crop(image, target_height=400, target_width=600):
    """
    随机放大旋转并裁剪到指定尺寸

    Args:
        image: 输入图像 (PIL Image)
        target_height: 目标高度
        target_width: 目标宽度

    Returns:
        处理后的图像 (PIL Image)
    """
    # 随机缩放比例 (1.2-2.0倍)
    scale_factor = random.uniform(1.2, 2.0)

    # 随机旋转角度 (-15到15度)
    rotation_angle = random.uniform(-15, 15)

    # 获取原始尺寸
    orig_width, orig_height = image.size

    # 计算缩放后的尺寸
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    # 缩放图像
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # 旋转图像
    image = image.rotate(rotation_angle, expand=True, fillcolor=(0, 0, 0))

    # 获取旋转后的尺寸
    rotated_width, rotated_height = image.size

    # 确保图像足够大可以裁剪
    if rotated_width < target_width or rotated_height < target_height:
        # 如果图像太小，再次放大
        scale_x = target_width / rotated_width if rotated_width < target_width else 1
        scale_y = target_height / rotated_height if rotated_height < target_height else 1
        scale = max(scale_x, scale_y) * 1.1  # 多留一点边距

        new_width = int(rotated_width * scale)
        new_height = int(rotated_height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        rotated_width, rotated_height = new_width, new_height

    # 随机裁剪位置
    max_x = rotated_width - target_width
    max_y = rotated_height - target_height

    crop_x = random.randint(0, max_x) if max_x > 0 else 0
    crop_y = random.randint(0, max_y) if max_y > 0 else 0

    # 裁剪到目标尺寸
    image = image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))

    return image


def apply_bilateral_filter(image, probability=0.5):
    """
    随机应用双边滤波

    Args:
        image: 输入图像 (PIL Image)
        probability: 应用滤波的概率

    Returns:
        处理后的图像 (PIL Image)
    """
    if random.random() < probability:
        # 转换为numpy数组
        img_array = np.array(image)

        # 随机双边滤波参数
        d = random.choice([5, 7, 9])  # 滤波直径
        sigma_color = random.uniform(50, 150)  # 颜色空间的标准差
        sigma_space = random.uniform(50, 150)  # 坐标空间的标准差

        # 应用双边滤波
        filtered = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)

        # 转换回PIL图像
        image = Image.fromarray(filtered)

    return image


def process_dataset(input_dir, output_dir, target_height=400, target_width=600,
                   bilateral_prob=0.5, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    处理整个数据集

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        target_height: 目标高度
        target_width: 目标宽度
        bilateral_prob: 双边滤波应用概率
        extensions: 支持的图像格式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, '**', f'*{ext}')
        image_files.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(input_dir, '**', f'*{ext.upper()}')
        image_files.extend(glob.glob(pattern, recursive=True))

    if not image_files:
        print(f"在 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件")
    print(f"目标尺寸: {target_width}x{target_height}")
    print(f"双边滤波概率: {bilateral_prob}")

    # 处理每个图像
    successful = 0
    failed = 0

    for img_path in tqdm(image_files, desc="处理图像"):
        try:
            # 读取图像
            image = Image.open(img_path)

            # 转换为RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # 随机缩放旋转裁剪
            processed_image = random_scale_rotate_crop(image, target_height, target_width)

            # 随机应用双边滤波
            processed_image = apply_bilateral_filter(processed_image, bilateral_prob)

            # 生成输出文件名
            relative_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)

            # 创建输出目录
            output_subdir = os.path.dirname(output_path)
            os.makedirs(output_subdir, exist_ok=True)

            # 保存处理后的图像
            processed_image.save(output_path, quality=95)
            successful += 1

        except Exception as e:
            print(f"处理 {img_path} 时出错: {str(e)}")
            failed += 1
            continue

    print(f"\n处理完成!")
    print(f"成功处理: {successful} 个文件")
    print(f"处理失败: {failed} 个文件")
    print(f"结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="数据集图像增强处理工具")

    # 必需参数
    parser.add_argument('--input_dir', type=str, required=True,
                        help='输入数据集路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出数据集路径')

    # 可选参数
    parser.add_argument('--height', type=int, default=400,
                        help='目标图像高度 (默认: 400)')
    parser.add_argument('--width', type=int, default=600,
                        help='目标图像宽度 (默认: 600)')
    parser.add_argument('--bilateral_prob', type=float, default=0.5,
                        help='双边滤波应用概率 (0.0-1.0, 默认: 0.5)')
    parser.add_argument('--extensions', nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.bmp'],
                        help='支持的图像格式 (默认: .jpg .jpeg .png .bmp)')

    args = parser.parse_args()

    # 验证参数
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 {args.input_dir} 不存在")
        return

    if args.bilateral_prob < 0 or args.bilateral_prob > 1:
        print("错误: bilateral_prob 必须在 0.0-1.0 之间")
        return

    if args.height <= 0 or args.width <= 0:
        print("错误: 高度和宽度必须大于0")
        return

    print(f"=== 数据增强配置 ===")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标尺寸: {args.width}x{args.height}")
    print(f"双边滤波概率: {args.bilateral_prob}")
    print(f"支持格式: {', '.join(args.extensions)}")
    print(f"==================")

    # 处理数据集
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_height=args.height,
        target_width=args.width,
        bilateral_prob=args.bilateral_prob,
        extensions=args.extensions
    )


if __name__ == "__main__":
    main()
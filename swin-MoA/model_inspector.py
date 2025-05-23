#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型结构检查工具 - 用于分析和打印PyTorch模型的结构与参数信息
专门用于分析models.py中定义的Model
"""

import argparse
import sys
import os
import torch
from torchinfo import summary

# 添加当前目录到系统路径，确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入models.py中的Model及其依赖
try:
    # 尝试导入所有可能的依赖项
    import config
    from models import Model
except ImportError as e:
    print(f"错误：无法导入必要的模块: {e}")
    print("请确保以下文件在当前目录或Python路径中:")
    print("  - models.py")
    print("  - config.py")
    print("  - feature_extractor.py")
    print("  - emb_gen.py")
    print("  - MoA.py")
    print("  - utils.py")
    print("  - decoder.py")
    print("  - ISP.py")
    print("  - losses.py")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch模型结构检查工具')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='模拟的批次大小（默认：32）')
    parser.add_argument('--channels', type=int, default=3,
                        help='输入通道数（默认：3）')
    parser.add_argument('--height', type=int, default=400,
                        help='输入高度（默认：400）')
    parser.add_argument('--width', type=int, default=600,
                        help='输入宽度（默认：600）')
    parser.add_argument('--device', type=str, default='cpu',
                        help='使用的设备，"cpu"或"cuda"（默认：cpu）')
    parser.add_argument('--verbose', type=int, default=1,
                        help='输出详细程度，0-2（默认：1）')
    parser.add_argument('--col_names', type=str,
                        default='input_size,output_size,num_params,trainable,mult_adds',
                        help='要显示的列名，逗号分隔')
    parser.add_argument('--model_args', type=str, default='',
                        help='传递给Model的参数，格式为"param1=value1,param2=value2"')
    parser.add_argument('--depth', type=int, default=3,
                        help='模型层次结构的显示深度（默认：3）')
    return parser.parse_args()


def load_model(args):
    """
    加载models.py中的Model

    参数:
        args: 命令行参数，包含model_args

    返回:
        实例化的模型对象
    """
    try:
        # 处理model_args参数
        model_kwargs = {}
        if args.model_args:
            for arg_pair in args.model_args.split(','):
                if '=' in arg_pair:
                    key, value = arg_pair.split('=')
                    # 尝试将value转换为合适的类型
                    try:
                        # 尝试作为数字
                        if '.' in value:
                            model_kwargs[key] = float(value)
                        else:
                            model_kwargs[key] = int(value)
                    except ValueError:
                        # 如果不是数字，保持为字符串
                        model_kwargs[key] = value

        # 创建模型实例
        print(f"正在创建Model实例...")
        model = Model(**model_kwargs)

        # 保存原始forward方法
        original_forward = model.forward

        # 创建检查专用的forward方法，忽略compute_loss和target参数
        def inspector_forward(x, *args, **kwargs):
            # 强制禁用损失计算，只返回增强图像
            return original_forward(x, compute_loss=False)

        # 替换模型的forward方法
        model.forward = inspector_forward

        print(f"Model实例创建成功")
        return model
    except TypeError as e:
        print(f"创建模型时参数错误: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def inspect_model(model, args):
    """
    检查模型结构并打印详细信息

    参数:
        model: PyTorch模型实例
        args: 命令行参数
    """
    print(f"\n{'=' * 80}")
    print(f"模型类型: {type(model).__name__}")
    print(f"{'=' * 80}\n")

    # 确保设备可用
    device_str = args.device
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        device_str = 'cpu'
    device = torch.device(device_str)

    # 将模型移至指定设备
    try:
        model = model.to(device)
        print(f"模型已移至设备: {device}")
    except Exception as e:
        print(f"将模型移至设备{device}时出错: {str(e)}")
        print("继续使用CPU...")
        device = torch.device('cpu')
        model = model.to(device)

    # 准备显示的列
    col_names = [col.strip() for col in args.col_names.split(',')]

    # 使用torchinfo.summary分析模型
    try:
        print(f"正在分析模型结构...")
        print(
            f"输入形状: [batch_size={args.batch_size}, channels={args.channels}, height={args.height}, width={args.width}]")

        # 创建一个相同形状的示例输入，用于跟踪计算流程
        dummy_input = torch.randn(args.batch_size, args.channels, args.height, args.width, device=device)

        # 使用torchinfo.summary分析模型
        model_stats = summary(
            model,
            input_data=dummy_input,
            verbose=args.verbose,
            col_names=col_names,
            depth=args.depth,
            device=device,
            mode='eval',
            # 添加以下行，显式传递compute_loss=False给模型的forward方法
            forward_pass_kwargs={'compute_loss': False}
        )

        # 打印额外的摘要信息
        print(f"\n{'=' * 80}")
        print("模型参数摘要:")
        print(f"{'=' * 80}")
        print(f"总参数量: {model_stats.total_params:,}")
        print(f"可训练参数量: {model_stats.trainable_params:,}")
        print(f"不可训练参数量: {model_stats.total_params - model_stats.trainable_params:,}")

        # 检查是否有FLOPS计算
        if hasattr(model_stats, 'total_mult_adds'):
            print(f"模型计算量 (MAdd): {model_stats.total_mult_adds / 1e6:.2f}M")

        # 计算模型大小（MB）
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        print(f"模型大小: {model_size:.2f} MB")

        # 显示模型输出的形状
        try:
            with torch.no_grad():
                model.eval()  # 临时设为评估模式
                outputs = model(dummy_input)
                if isinstance(outputs, tuple):
                    print(f"\n输出信息:")
                    for i, output in enumerate(outputs):
                        if isinstance(output, torch.Tensor):
                            print(f"  输出 {i}: 形状 = {list(output.shape)}")
                        else:
                            print(f"  输出 {i}: 类型 = {type(output)}")
                else:
                    if isinstance(outputs, torch.Tensor):
                        print(f"\n输出形状: {list(outputs.shape)}")
                    else:
                        print(f"\n输出类型: {type(outputs)}")
        except Exception as e:
            print(f"\n获取模型输出形状时出错: {str(e)}")

    except Exception as e:
        print(f"分析模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 配置信息和欢迎信息
    print("\n" + "=" * 80)
    print(f"PyTorch模型结构检查工具 - 适用于自定义ISP图像增强模型")
    print("=" * 80)

    # 解析命令行参数
    args = parse_args()

    # 打印配置信息
    print(f"模型: models.py中的Model类")
    if args.model_args:
        print(f"模型参数: {args.model_args}")
    print(f"输入形状: [{args.batch_size}, {args.channels}, {args.height}, {args.width}]")
    print(f"运行设备: {args.device}")
    print(f"显示深度: {args.depth}")

    try:
        # 导入必要的依赖
        print("\n正在检查依赖项...")
        import torch
        print(f"PyTorch版本: {torch.__version__}")

        if args.device == 'cuda':
            if torch.cuda.is_available():
                print(f"CUDA是否可用: 是")
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"可用GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print(f"CUDA是否可用: 否，将使用CPU")

        import torchinfo
        print(f"torchinfo版本: {torchinfo.__version__ if hasattr(torchinfo, '__version__') else '未知'}")

        # 加载模型
        print("\n正在加载模型...")
        model = load_model(args)

        # 打印模型信息概要
        print("\n模型概要:")
        print(f"  模型类型: {type(model).__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数总数: {total_params:,}")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

        # 检查模型
        inspect_model(model, args)

        # 完成
        print("\n" + "=" * 80)
        print("检查完成!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n运行期间发生错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("检查未完成!")
        print("=" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        print(f"\n{'=' * 80}")
        print(f"ISP图像增强模型结构分析工具 v1.0")
        print(f"{'=' * 80}")
        print("该工具用于分析models.py中的Model类结构与参数")
        print("使用torchinfo查看模型的详细信息，包括总参数量、可训练参数量和不可训练参数量")
        print("\n参数示例:")
        print("  --batch_size 32           # 设置批次大小为2")
        print("  --height 400 --width 600 # 设置输入图像尺寸为396×600")
        print("  --device cuda            # 使用GPU进行分析")
        print("  --depth 5                # 显示5层深度的模型结构")
        print("  --verbose 2              # 显示最详细的信息")
        print("\n运行示例:")
        print("  python model_inspector.py --batch_size 2 --height 400 --width 600 --device cuda")
        print(f"{'=' * 80}\n")

        main()
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
        sys.exit(0)
#model_stats = summary
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
import math
from datetime import datetime
import argparse
from torch.cuda.amp import autocast, GradScaler  # 导入混合精度工具
from lion_pytorch import Lion

# 导入自定义模块
from data_loader import get_data_loaders
from models import Model


def warmup_cosine_schedule(optimizer, epoch, warmup_epochs, total_epochs, min_lr_factor=0.01):
    """
    实现warmup + 余弦退火学习率调度的函数

    Args:
        optimizer: 优化器
        epoch: 当前epoch
        warmup_epochs: 预热阶段的轮数
        total_epochs: 总训练轮数
        min_lr_factor: 最小学习率是初始学习率的倍数
    """
    # 保存基础学习率
    if not hasattr(warmup_cosine_schedule, 'base_lrs'):
        warmup_cosine_schedule.base_lrs = [group['lr'] for group in optimizer.param_groups]

    # 预热阶段
    if epoch < warmup_epochs:
        # 线性预热
        factor = float(epoch) / float(max(1, warmup_epochs))
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = warmup_cosine_schedule.base_lrs[i] * factor
    # 余弦退火阶段
    else:
        # 计算余弦退火的进度
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        factor = max(min_lr_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = warmup_cosine_schedule.base_lrs[i] * factor


def train_one_epoch(model, train_loader, optimizer, device, epoch, scaler=None, use_amp=False):
    """
    训练一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备（GPU/CPU）
        epoch: 当前epoch
        scaler: 混合精度缩放器，用于防止梯度下溢
        use_amp: 是否使用混合精度训练

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
        inputs = batch_data['input'].to(device, non_blocking=True)  # 添加non_blocking=True提高GPU传输效率
        targets = batch_data['target'].to(device, non_blocking=True)  # 从数据加载器中获取目标图像

        # 清除梯度
        optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True代替置零，提高效率

        # 使用混合精度训练
        if use_amp:
            with autocast():
                # 前向传播 - 现在模型返回三个值：增强图像、总损失和重建损失
                enhanced_images, loss, reconstruction_loss = model(inputs, targets)

            # 使用缩放器进行反向传播
            scaler.scale(loss).backward()

            # 在更新参数前进行梯度展开和剪裁
            scaler.unscale_(optimizer)
            # 添加梯度剪裁，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向传播 - 使用原始精度
            enhanced_images, loss, reconstruction_loss = model(inputs, targets)

            # 反向传播
            loss.backward()

            # 添加梯度剪裁，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

        # 累计损失
        total_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()

        # 更新进度条，同时显示总损失和重建损失
        progress_bar.set_postfix(loss=loss.item(), recon_loss=reconstruction_loss.item())

        # 显式释放不需要的张量，减少内存占用
        del inputs, targets, enhanced_images, loss, reconstruction_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清理缓存，减少内存碎片

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_reconstruction_loss / len(train_loader)

    return avg_loss, avg_recon_loss


def validate(model, val_loader, device, use_amp=False):
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        device: 设备（GPU/CPU）
        use_amp: 是否使用混合精度

    Returns:
        平均验证总损失和平均验证重建损失
    """
    model.eval()  # 设置为评估模式
    total_val_loss = 0
    total_val_recon_loss = 0

    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data['input'].to(device, non_blocking=True)
            targets = batch_data['target'].to(device, non_blocking=True)  # 获取目标图像

            # 使用混合精度计算
            if use_amp:
                with autocast():
                    # 临时将模型设为训练模式以计算损失，但保持梯度禁用
                    model.train()
                    enhanced_images, loss, reconstruction_loss = model(inputs, targets)
                    model.eval()  # 恢复为评估模式
            else:
                # 临时将模型设为训练模式以计算损失，但保持梯度禁用
                model.train()
                enhanced_images, loss, reconstruction_loss = model(inputs, targets)
                model.eval()  # 恢复为评估模式

            total_val_loss += loss.item()
            total_val_recon_loss += reconstruction_loss.item()

            # 释放内存
            del inputs, targets, enhanced_images, loss, reconstruction_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 计算平均验证损失
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_recon_loss = total_val_recon_loss / len(val_loader)

    return avg_val_loss, avg_val_recon_loss


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 启用混合精度训练
    use_amp = args.use_amp and torch.cuda.is_available()
    if use_amp:
        print("启用混合精度训练 (AMP)")
        scaler = GradScaler()
    else:
        scaler = None

    # 设置CUDA性能优化
    if torch.cuda.is_available():
        # 启用CUDNN自动优化
        torch.backends.cudnn.benchmark = True
        # 确保算法确定性
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)

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

    # 使用torch.compile优化模型（如果PyTorch版本>=2.0）
    if args.compile and hasattr(torch, 'compile'):
        print("使用torch.compile优化模型")
        model = torch.compile(model)

    # 检查是否从检查点恢复
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"从 {args.resume} 加载检查点")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"从第 {start_epoch} 轮恢复训练")
        else:
            print(f"在 {args.resume} 未找到检查点")

    # 初始化Lion优化器，注意参数调整
    # Lion学习率一般比AdamW小3-10倍，权重衰减大3-10倍
    optimizer = Lion(
        model.parameters(),
        lr=args.lr,  # 使用命令行参数中的学习率
        weight_decay=args.weight_decay * 3.0  # 权重衰减增大3倍
    )

    print(f"使用Lion优化器: 学习率={args.lr}, 权重衰减={args.weight_decay * 3.0}")

    # 打印学习率调度器信息
    print(f"使用Warmup({args.warmup_epochs}轮) + 余弦退火学习率调度")
    print(f"初始学习率: {args.lr}, 最小学习率: {args.lr * args.min_lr_factor}")

    # 如果恢复训练，也加载优化器状态
    if args.resume and os.path.isfile(args.resume):
        # 注意：如果之前使用的是AdamW，这里状态可能不兼容
        # 建议在切换优化器时重新训练，而不是从之前的检查点恢复
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("加载优化器状态")
        except:
            print("优化器状态加载失败，可能是由于优化器类型变更，将使用新的优化器状态")

        # 如果使用混合精度，恢复scaler状态
        if use_amp and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("加载混合精度缩放器状态")

    # 训练日志
    log_file = os.path.join(save_dir, 'training_log.txt')
    if start_epoch == 1:  # 如果是新训练，创建新日志文件
        with open(log_file, 'w') as f:
            f.write(f"训练开始于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"参数: {args}\n")
            f.write(f"优化器: Lion, 学习率={args.lr}, 权重衰减={args.weight_decay * 3.0}\n\n")
            f.write("轮次,训练损失,训练重建损失,验证损失,验证重建损失,学习率,用时(秒)\n")

    best_val_loss = float('inf')
    # 如果恢复训练，更新最佳验证损失
    if args.resume and os.path.isfile(args.resume) and 'val_loss' in checkpoint:
        best_val_loss = checkpoint['val_loss']
        print(f"恢复最佳验证损失: {best_val_loss:.4f}")

    # 早停相关变量初始化
    early_stopping_counter = 0
    early_stopping_min_val_loss = float('inf')

    # 开始训练循环
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        # 更新学习率 - 使用自定义函数实现warmup + 余弦退火
        warmup_cosine_schedule(
            optimizer,
            epoch - 1,  # 因为epoch从1开始，我们需要减1使其从0开始
            args.warmup_epochs,
            args.epochs,
            args.min_lr_factor
        )
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

        # 训练一个epoch
        train_loss, train_recon_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler, use_amp)

        # 验证 (仅使用val_loader_1)
        val_loss, val_recon_loss = validate(model, val_loader_1, device, use_amp)

        # 计算用时
        epoch_time = time.time() - start_time

        # 打印结果，现在包括重建损失
        print(f"轮次 {epoch}/{args.epochs} - "
              f"训练损失: {train_loss:.4f}, "
              f"训练重建损失: {train_recon_loss:.4f}, "
              f"验证损失: {val_loss:.4f}, "
              f"验证重建损失: {val_recon_loss:.4f}, "
              f"学习率: {current_lr:.6f}, "
              f"用时: {epoch_time:.2f}秒")

        # 记录日志，包括重建损失
        with open(log_file, 'a') as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_recon_loss:.6f},{val_loss:.6f},{val_recon_loss:.6f},{current_lr:.6f},{epoch_time:.2f}\n")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存完整检查点
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')

            # 构建保存字典
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_recon_loss': val_recon_loss,
                'train_loss': train_loss,
                'train_recon_loss': train_recon_loss,
                'lr': current_lr  # 保存当前学习率
            }

            # 如果使用混合精度，保存scaler状态
            if use_amp:
                save_dict['scaler_state_dict'] = scaler.state_dict()

            torch.save(save_dict, checkpoint_path)
            print(f"保存最佳模型检查点到 {checkpoint_path}")

            # 额外保存一个只包含模型权重的文件，便于部署
            weights_path = os.path.join(save_dir, 'best_model_weights.pth')
            torch.save(model.state_dict(), weights_path)
            print(f"保存最佳模型权重到 {weights_path}")

        # 每隔特定epochs保存一次模型
        if epoch % args.save_interval == 0:
            # 保存完整检查点
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')

            # 构建保存字典
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_recon_loss': val_recon_loss,
                'train_loss': train_loss,
                'train_recon_loss': train_recon_loss,
                'lr': current_lr  # 保存当前学习率
            }

            # 如果使用混合精度，保存scaler状态
            if use_amp:
                save_dict['scaler_state_dict'] = scaler.state_dict()

            torch.save(save_dict, checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")

            # 保存权重文件
            weights_path = os.path.join(save_dir, f'model_epoch_{epoch}_weights.pth')
            torch.save(model.state_dict(), weights_path)
            print(f"保存模型权重到 {weights_path}")

        # 早停逻辑
        if val_loss < early_stopping_min_val_loss - args.min_delta:
            # 验证损失显著下降，重置计数器
            early_stopping_min_val_loss = val_loss
            early_stopping_counter = 0
            print(f"验证损失显著下降。早停计数器重置。")
        else:
            # 验证损失没有显著下降，增加计数器
            early_stopping_counter += 1
            print(
                f"验证损失未显著下降。早停计数器: {early_stopping_counter}/{args.patience}")

            # 如果连续多个epoch没有改善，触发早停
            if early_stopping_counter >= args.patience:
                print(f"早停在第{epoch}轮训练后触发！")
                break

    # 保存最终模型
    final_checkpoint_path = os.path.join(save_dir, 'final_model.pth')

    # 构建保存字典
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_recon_loss': val_recon_loss,
        'train_loss': train_loss,
        'train_recon_loss': train_recon_loss,
        'lr': current_lr  # 保存当前学习率
    }

    # 如果使用混合精度，保存scaler状态
    if use_amp:
        save_dict['scaler_state_dict'] = scaler.state_dict()

    torch.save(save_dict, final_checkpoint_path)
    print(f"训练完成。最终模型保存到 {final_checkpoint_path}")

    # 保存最终模型权重
    final_weights_path = os.path.join(save_dir, 'final_model_weights.pth')
    torch.save(model.state_dict(), final_weights_path)
    print(f"最终模型权重保存到 {final_weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练低光照图像增强模型")
    parser.add_argument('--data_dir', type=str,
                        default="/root/autodl-tmp/swin-MOA/LOL-v2-Dataset/archive/LOL-v2/Real_captured",
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')  # 调整为Lion推荐的更小学习率
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')  # Lion会在内部将这个值乘以3
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='保存模型的目录')
    parser.add_argument('--save_interval', type=int, default=10, help='每N个epoch保存一次模型')
    parser.add_argument('--num_workers', type=int, default=20, help='数据加载器的工作线程数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    # 添加早停相关参数
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值：验证损失多少个epoch没有显著下降后停止训练')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='定义验证损失"显著"下降的最小幅度')
    # 添加GPU优化相关参数
    parser.add_argument('--use_amp', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--deterministic', action='store_true', help='是否使用确定性算法（可能降低性能）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--compile', action='store_true', help='使用torch.compile优化模型(需要PyTorch 2.0+)')
    # 添加Warmup和余弦退火相关参数
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Warmup的轮数')
    parser.add_argument('--min_lr_factor', type=float, default=0.01, help='余弦退火最小学习率是初始学习率的多少倍')

    args = parser.parse_args()
    main(args)
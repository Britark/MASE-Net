import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import *
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
import math
from datetime import datetime
import argparse
import kornia.metrics as K
import shutil

# 兼容不同PyTorch版本的混合精度导入
try:
    from torch.amp import autocast, GradScaler

    AMP_AVAILABLE = True
    USE_NEW_AMP = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler

        AMP_AVAILABLE = True
        USE_NEW_AMP = False
    except ImportError:
        AMP_AVAILABLE = False
        USE_NEW_AMP = False
        print("警告：无法导入混合精度训练模块，将禁用AMP")

# 导入自定义模块
from data_loader import get_data_loaders
from models import Model
from config import default_config

# Optuna集成
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告：Optuna未安装，将跳过超参数优化功能")


def check_gradients(model, batch_idx, epoch):
    """
    检查模型梯度状态，专注显示零梯度参数
    Args:
        model: 训练模型
        batch_idx: 当前batch索引
        epoch: 当前epoch
    """

    total_params = 0
    zero_grad_params = 0
    nan_grad_params = 0
    normal_grad_params = 0
    
    # 记录零梯度参数
    zero_grad_details = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_norm = torch.norm(grad).item()

            # 检查梯度状态
            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()
            is_zero = (grad_norm < 1e-10)

            if has_nan:
                nan_grad_params += 1
            elif is_zero:
                zero_grad_params += 1
                # 记录零梯度参数详情
                module_name = name.split('.')[0]
                if module_name not in zero_grad_details:
                    zero_grad_details[module_name] = []
                zero_grad_details[module_name].append(name)
            else:
                normal_grad_params += 1

            total_params += 1

    print(f"\n📊 梯度统计 [Epoch {epoch}, Batch {batch_idx}]:")
    print(f"  总参数: {total_params}, 正常: {normal_grad_params} ({normal_grad_params / total_params * 100:.1f}%), 零梯度: {zero_grad_params} ({zero_grad_params / total_params * 100:.1f}%), NaN: {nan_grad_params} ({nan_grad_params / total_params * 100:.1f}%)")
    
    # 只显示有零梯度的模块和具体参数
    if zero_grad_details:
        print(f"\n🔍 零梯度参数详情:")
        for module_name, param_names in zero_grad_details.items():
            print(f"  📦 {module_name} ({len(param_names)}个零梯度参数):")
            for param_name in param_names:
                print(f"    - {param_name}")
        print("=" * 80)
    else:
        print("✅ 所有参数都有正常梯度流！")
        print("=" * 80)


def calculate_psnr(img1, img2):
    """使用Kornia计算PSNR"""
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    return K.psnr(img1, img2, max_val=1.0).mean().item()


def calculate_ssim(img1, img2):
    """使用Kornia计算SSIM"""
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    return K.ssim(img1, img2, window_size=11, max_val=1.0).mean().item()


def suggest_hyperparameters(trial):
    """建议超参数（基于Optuna重要性分析结果优化）"""
    return {
        # === Tier 1: 最高重要性参数（精细调整）===
        'learning_rate': trial.suggest_float('learning_rate', 6.5e-5, 7.8e-5),  # 重要性0.25-0.26
        'psnr_weight': trial.suggest_float('psnr_weight', 0.08, 0.12),          # 重要性0.26-0.28
        
        # === Tier 2: 高重要性参数（适度调整）===
        'lab_color_weight': trial.suggest_float('lab_color_weight', 0.006, 0.015), # 重要性0.15-0.28
        'reconstruction_weight': trial.suggest_float('reconstruction_weight', 1.48, 1.58), # 重要性0.07-0.33 (PED ANOVA最高)
        
        # === 固定参数（基于最佳试验#23的值，低重要性）===
        'perceptual_weight': 0.232,    # 重要性0.02-0.07，固定为最佳值
        'ssim_weight': 0.126,          # 重要性0.01-0.05，固定为最佳值  
        'auxiliary_weight': 0.107,     # 重要性0.02-0.11，固定为最佳值
    }


def create_config_with_weights(loss_weights):
    """创建包含指定损失权重的配置，并增加验证"""
    config = default_config()
    config['loss_weights'] = loss_weights
    
    # === 新增：详细日志 ===
    print(f"📋 创建配置，损失权重:")
    total_weight = sum(loss_weights.values())
    for key, value in loss_weights.items():
        percentage = (value / total_weight) * 100
        print(f"  {key}: {value:.6f} ({percentage:.1f}%)")
    print(f"  总权重: {total_weight:.6f}")
    
    return config


def train_and_evaluate_for_optuna(config, data_loaders, device, learning_rate, max_epochs=8):
    """为Optuna优化设计的训练和评估函数"""
    train_loader, val_loader = data_loaders
    
    # === 添加随机种子控制确保可重现性 ===
    import random
    import numpy as np
    
    seed = 3407  # 固定种子确保可重现性
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 设置随机种子: {seed}")
    
    # === 验证传入的损失权重配置 ===
    print(f"\n🔍 验证传入的损失权重配置:")
    loss_weights = config.get('loss_weights', {})
    total_weight = sum(loss_weights.values())
    
    for key, value in loss_weights.items():
        percentage = (value / total_weight) * 100 if total_weight > 0 else 0
        print(f"  {key}: {value:.6f} ({percentage:.1f}%)")
    print(f"  总权重: {total_weight:.6f}")
    
    # 验证必要参数是否存在
    required_params = ['reconstruction_weight', 'auxiliary_weight', 'psnr_weight', 
                       'ssim_weight', 'lab_color_weight', 'perceptual_weight']
    missing_params = [param for param in required_params if param not in loss_weights]
    if missing_params:
        print(f"❌ 警告: 缺失参数 {missing_params}, 将使用默认值")
        return float('inf')  # 如果参数传递失败，直接返回最差分数
    
    # CUDA状态检查和清理（只在开始时进行）
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        print("⚠️ CUDA清理失败，继续执行...")
        pass
    
    # 创建模型
    try:
        model = Model(config).to(device)
    except Exception as e:
        print(f"模型创建失败: {e}")
        return float('inf')
    
    # 创建优化器（使用Optuna建议的学习率）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,  # 使用Optuna建议的学习率
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 混合精度设置
    use_amp = torch.cuda.is_available() and AMP_AVAILABLE
    scaler = GradScaler() if use_amp else None
    
    best_val_psnr = 0.0
    patience = 0
    max_patience = 999  # 基本不会触发早停
    
    for epoch in range(1, max_epochs + 1):
        # 训练一个epoch
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 限制每个epoch的batch数量以加速（增加到100以获得更准确的评估）
        max_train_batches = min(100, len(train_loader))
        
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_train_batches:
                break
                
            inputs = batch_data['input'].to(device, non_blocking=True)
            targets = batch_data['target'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            try:
                if use_amp:
                    if USE_NEW_AMP:
                        with autocast('cuda'):
                            enhanced_images, loss, reconstruction_loss = model(inputs, targets)
                    else:
                        with autocast():
                            enhanced_images, loss, reconstruction_loss = model(inputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    enhanced_images, loss, reconstruction_loss = model(inputs, targets)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "index out of bounds" in str(e) or "device-side assert" in str(e):
                    print(f"🚫 CUDA错误，跳过此批次: {e}")
                    # 不调用torch.cuda.empty_cache()，避免进一步CUDA错误
                    print(f"🚫 发现问题图片！跳过batch {batch_idx} (这是正常的，约1%概率)")
                    # 记录问题batch信息到文件
                    with open('problematic_batches.log', 'a') as f:
                        f.write(f"Epoch {epoch}, Batch {batch_idx}: CUDA error\n")
                    
                    # 重置CUDA上下文
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except:
                        pass
                    continue
                else:
                    print(f"其他运行时错误: {e}")
                    continue
            except Exception as e:
                print(f"训练批次失败: {e}")
                continue
        
        if num_batches == 0:
            return float('inf')  # 训练失败
            
        avg_train_loss = total_loss / num_batches
        
        # 验证
        val_psnr = validate_for_optuna(model, val_loader, device, use_amp)
        
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val PSNR: {val_psnr:.2f}")
        
        # 早停检查
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"早停于epoch {epoch}")
                break
    
    return -best_val_psnr  # 返回负值因为Optuna要最小化


def validate_for_optuna(model, val_loader, device, use_amp=False):
    """为Optuna优化设计的验证函数"""
    model.eval()
    total_psnr = 0
    num_batches = 0
    
    # 限制验证batch数量
    max_val_batches = min(30, len(val_loader))
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            if batch_idx >= max_val_batches:
                break
                
            inputs = batch_data['input'].to(device, non_blocking=True)
            targets = batch_data['target'].to(device, non_blocking=True)
            
            try:
                # 验证时需要将模型设为训练模式以获取损失计算，但不更新梯度
                model.train()  # 临时设为训练模式以获取损失
                if use_amp:
                    if USE_NEW_AMP:
                        with autocast('cuda'):
                            enhanced_images, total_loss, reconstruction_loss = model(inputs, targets)
                    else:
                        with autocast():
                            enhanced_images, total_loss, reconstruction_loss = model(inputs, targets)
                else:
                    enhanced_images, total_loss, reconstruction_loss = model(inputs, targets)
                model.eval()  # 恢复评估模式
                
                # 计算PSNR用于优化目标
                psnr_value = calculate_psnr(enhanced_images, targets)
                total_psnr += psnr_value
                num_batches += 1
                
            except RuntimeError as e:
                if "CUDA" in str(e) or "index out of bounds" in str(e) or "device-side assert" in str(e):
                    print(f"🚫 CUDA错误，跳过验证批次: {e}")
                    # 不调用torch.cuda.empty_cache()，避免进一步CUDA错误
                    continue
                else:
                    print(f"其他验证错误: {e}")
                    continue
            except Exception as e:
                print(f"验证批次失败: {e}")
                continue
    
    return total_psnr / max(num_batches, 1)


def optuna_objective(trial, data_loaders, device, args):
    """Optuna目标函数"""
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna未安装，无法进行超参数优化")
    
    # 每个trial开始前尝试重置CUDA状态
    reset_cuda_context()
    
    # 建议超参数（包括损失权重和学习率）
    hyperparams = suggest_hyperparameters(trial)
    
    # 分离损失权重和学习率
    learning_rate = hyperparams.pop('learning_rate')
    loss_weights = hyperparams
    
    # 创建配置
    config = create_config_with_weights(loss_weights)
    
    # 打印当前试验的超参数
    print(f"\nTrial {trial.number} 超参数:")
    print(f"  learning_rate: {learning_rate:.6f}")
    for key, value in loss_weights.items():
        print(f"  {key}: {value:.4f}")
    
    # 训练和评估
    try:
        score = train_and_evaluate_for_optuna(config, data_loaders, device, learning_rate, max_epochs=args.optuna_epochs)
        
        # 检查score是否有效
        if score == float('inf') or score != score:  # score != score 检查NaN
            print(f"Trial {trial.number} 失败: 无效得分 {score}")
            return float('inf')
            
        print(f"Trial {trial.number} 完成，得分: {score:.4f}")
        
        # 向trial报告中间结果
        trial.report(score, args.optuna_epochs)
        
        return score
    except RuntimeError as e:
        if "CUDA" in str(e) or "index out of bounds" in str(e) or "device-side assert" in str(e):
            print(f"🚫 Trial {trial.number} CUDA错误，跳过: {e}")
            # 不调用torch.cuda.empty_cache()，避免进一步CUDA错误
            return float('inf')  # 返回最差分数，让Optuna跳过
        else:
            print(f"Trial {trial.number} 运行时错误: {e}")
            return float('inf')
    except Exception as e:
        print(f"Trial {trial.number} 其他错误: {e}")
        return float('inf')


def reset_cuda_context():
    """重置CUDA上下文，用于从严重CUDA错误中恢复"""
    try:
        if torch.cuda.is_available():
            # 尝试重置CUDA上下文
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except:
        try:
            # 如果正常重置失败，尝试强制重置
            import subprocess
            result = subprocess.run(['nvidia-smi', '--gpu-reset'], 
                                   capture_output=True, text=True)
            print("🔄 执行了GPU重置")
        except:
            print("⚠️ 无法重置CUDA上下文，继续执行...")
            pass


def run_diagnostic_test():
    """运行诊断测试，验证参数传递和模型配置"""
    print("🔧 开始诊断测试...")
    
    # 创建已知的权重配置
    test_weights = {
        'reconstruction_weight': 1.5,
        'auxiliary_weight': 0.4,
        'psnr_weight': 0.3,
        'ssim_weight': 0.2,
        'lab_color_weight': 0.05,
        'perceptual_weight': 0.15
    }
    
    print("📋 测试权重配置:")
    for key, value in test_weights.items():
        print(f"  {key}: {value}")
    
    # 创建配置并模型
    config = create_config_with_weights(test_weights)
    try:
        model = Model(config)
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 验证模型中的权重是否正确设置
    print("\n🔍 模型权重验证:")
    print(f"  reconstruction_weight: {model.reconstruction_weight}")
    print(f"  auxiliary_weight: {model.auxiliary_weight}")
    print(f"  perceptual_weight: {model.perceptual_weight}")
    print(f"  psnr_weight: {model.psnr_weight}")
    print(f"  ssim_weight: {model.ssim_weight}")
    print(f"  lab_color_weight: {model.lab_color_weight}")
    
    # 检查属性是否存在
    required_attrs = ['reconstruction_weight', 'auxiliary_weight', 'perceptual_weight', 
                      'psnr_weight', 'ssim_weight', 'lab_color_weight']
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(model, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        print(f"❌ 缺失属性: {missing_attrs}")
        return False
    else:
        print("✅ 所有权重属性正确设置")
        return True


def run_optuna_optimization(args):
    """运行Optuna超参数优化"""
    if not OPTUNA_AVAILABLE:
        print("错误：Optuna未安装，请先安装: pip install optuna")
        return
    
    print("🚀 开始Optuna超参数优化...")
    print(f"试验次数: {args.optuna_trials}")
    print(f"每个试验训练轮数: {args.optuna_epochs}")
    print(f"研究名称: {args.optuna_study_name}")
    print(f"存储位置: {args.optuna_storage}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    root_dir = os.path.expanduser(args.data_dir)
    train_loader, val_loader, _ = get_data_loaders(
        root_dir=root_dir,
        batch_size=args.batch_size,
        eval_batch_size=4,  # 更小的验证batch
        num_workers=args.num_workers
    )
    
    data_loaders = (train_loader, val_loader)
    
    # 创建存储目录
    storage_dir = os.path.dirname(args.optuna_storage.replace('sqlite:///', ''))
    if storage_dir:
        os.makedirs(storage_dir, exist_ok=True)
    
    # 创建或加载研究
    try:
        study = optuna.create_study(
            study_name=args.optuna_study_name,
            storage=args.optuna_storage,
            load_if_exists=True,
            direction="minimize",  # 最小化损失(负PSNR)
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=3,          # 最少训练3个epoch
                max_resource=args.optuna_epochs,  # 最多训练指定epochs
                reduction_factor=3       # 官方推荐的减少因子
            )
        )
        print(f"✅ 研究 '{args.optuna_study_name}' 创建/加载成功")
    except Exception as e:
        print(f"❌ 创建研究失败: {e}")
        return
    
    # 定义优化目标的包装函数
    def objective_wrapper(trial):
        return optuna_objective(trial, data_loaders, device, args)
    
    # 开始优化
    try:
        print(f"\n🔍 开始优化，目标: 最大化验证PSNR...")
        study.optimize(objective_wrapper, n_trials=args.optuna_trials)
        
        # 显示结果
        print("\n🎉 优化完成！")
        print(f"最佳验证PSNR: {-study.best_value:.2f}")
        print("最佳损失权重:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value:.4f}")
        
        # 保存最佳参数到文件
        best_config = create_config_with_weights(study.best_params)
        import json
        with open(f'best_loss_weights_{args.optuna_study_name}.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        print(f"\n💾 最佳参数已保存到: best_loss_weights_{args.optuna_study_name}.json")
        
        # 提示如何查看dashboard
        print(f"\n📊 查看优化结果:")
        print(f"1. 启动dashboard: optuna-dashboard {args.optuna_storage}")
        print(f"2. 通过SSH隧道访问: ssh -L 8080:localhost:8080 用户名@服务器IP")
        print(f"3. 浏览器打开: http://localhost:8080")
        
    except KeyboardInterrupt:
        print("\n⏹️ 优化被用户中断")
        if study.trials:
            print(f"已完成 {len(study.trials)} 个试验")
            if study.best_trial:
                print(f"当前最佳PSNR: {-study.best_value:.2f}")
    except Exception as e:
        print(f"❌ 优化过程出错: {e}")
        import traceback
        traceback.print_exc()


def warmup_cosine_schedule(optimizer, epoch, warmup_epochs, total_epochs, min_lr_factor=0.01, steepness=2.0,acceleration_factor=1.0):
    """
    实现warmup + 余弦退火学习率调度的函数（增加陡峭度参数）

    Args:
        optimizer: 优化器
        epoch: 当前epoch
        warmup_epochs: 预热阶段的轮数
        total_epochs: 总训练轮数
        min_lr_factor: 最小学习率是初始学习率的倍数
        steepness: 陡峭度参数，越大衰减越快
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
        progress = float(epoch - warmup_epochs) * acceleration_factor / float(max(1, total_epochs - warmup_epochs))
        # 添加steepness参数让衰减更陡峭
        cosine_factor = math.cos(math.pi * progress)
        factor = max(min_lr_factor,
                     0.5 * (1.0 + math.pow(abs(cosine_factor), steepness) * (1 if cosine_factor >= 0 else -1)))
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = warmup_cosine_schedule.base_lrs[i] * factor


def train_one_epoch(model, train_loader, optimizer, device, epoch, scaler=None, use_amp=False, check_grad_every=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_reconstruction_loss = 0
    total_psnr = 0
    total_ssim = 0


    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", ncols=0, leave=True)

    for batch_idx, batch_data in progress_bar:
        inputs = batch_data['input'].to(device, non_blocking=True)
        targets = batch_data['target'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            if USE_NEW_AMP:
                with autocast('cuda'):
                    enhanced_images, loss, reconstruction_loss = model(inputs, targets)
            else:
                with autocast():
                    enhanced_images, loss, reconstruction_loss = model(inputs, targets)

            scaler.scale(loss).backward()

            # ========== 每个batch都检查梯度 ==========
            # 需要先unscale来检查真实梯度
            scaler.unscale_(optimizer)
            check_gradients(model, batch_idx, epoch)

            # 梯度裁剪前检查是否需要裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            if grad_norm > 1.0:
                print(f"🔧 [Epoch {epoch}, Batch {batch_idx}] 梯度范数过大: {grad_norm:.6f} > 1.0, 执行梯度裁剪!")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            enhanced_images, loss, reconstruction_loss = model(inputs, targets)
            loss.backward()

            # ========== 每个batch都检查梯度 ==========
            check_gradients(model, batch_idx, epoch)

            # 梯度裁剪前检查是否需要裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            if grad_norm > 1.0:
                print(f"🔧 [Epoch {epoch}, Batch {batch_idx}] 梯度范数过大: {grad_norm:.6f} > 1.0, 执行梯度裁剪!")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                if batch_idx % check_grad_every == 0:  # 只在检查梯度的batch显示正常信息
                    print(f"✅ [Epoch {epoch}, Batch {batch_idx}] 梯度范数正常: {grad_norm:.6f}, 无需裁剪")

            optimizer.step()

        # 计算PSNR和SSIM
        with torch.no_grad():
            psnr_value = calculate_psnr(enhanced_images, targets)
            ssim_value = calculate_ssim(enhanced_images, targets)

        total_loss += loss.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_psnr += psnr_value
        total_ssim += ssim_value

        progress_bar.set_postfix(
            loss=loss.item(),
            recon_loss=reconstruction_loss.item(),
            psnr=psnr_value,
            ssim=ssim_value
        )

        del inputs, targets, enhanced_images, loss, reconstruction_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_reconstruction_loss / len(train_loader)
    avg_psnr = total_psnr / len(train_loader)
    avg_ssim = total_ssim / len(train_loader)

    return avg_loss, avg_recon_loss, avg_psnr, avg_ssim


def validate(model, val_loader, device, use_amp=False):
    """
    在验证集上评估模型
    """
    model.eval()
    total_val_loss = 0
    total_val_recon_loss = 0
    total_val_psnr = 0
    total_val_ssim = 0

    batch_count = 0
    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data['input'].to(device, non_blocking=True)
            targets = batch_data['target'].to(device, non_blocking=True)

            # 使用混合精度计算
            if use_amp:
                if USE_NEW_AMP:
                    with autocast('cuda'):
                        enhanced_images = model(inputs)  # 验证模式只返回enhanced_images
                else:
                    with autocast():
                        enhanced_images = model(inputs)  # 验证模式只返回enhanced_images
            else:
                enhanced_images = model(inputs)  # 验证模式只返回enhanced_images

            # 手动计算损失（与训练时保持一致）
            l1_loss = L1ReconstructionLoss()
            reconstruction_loss = l1_loss(enhanced_images, targets)
            loss = reconstruction_loss  # 验证时不需要辅助损失

            # 计算PSNR和SSIM
            psnr_value = calculate_psnr(enhanced_images, targets)
            ssim_value = calculate_ssim(enhanced_images, targets)

            total_val_loss += loss.item()
            total_val_recon_loss += reconstruction_loss.item()
            total_val_psnr += psnr_value
            total_val_ssim += ssim_value
            batch_count += 1

            del inputs, targets, enhanced_images, loss, reconstruction_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_val_loss = total_val_loss / batch_count
    avg_val_recon_loss = total_val_recon_loss / batch_count
    avg_val_psnr = total_val_psnr / batch_count
    avg_val_ssim = total_val_ssim / batch_count

    return avg_val_loss, avg_val_recon_loss, avg_val_psnr, avg_val_ssim


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 启用混合精度训练
    use_amp = args.use_amp and torch.cuda.is_available() and AMP_AVAILABLE
    if use_amp:
        print(f"启用混合精度训练 (AMP) - 使用{'新' if USE_NEW_AMP else '旧'}版本API")
        scaler = GradScaler()
    else:
        scaler = None
        if args.use_amp and not AMP_AVAILABLE:
            print("警告：请求启用AMP但不可用，将使用普通精度训练")

    # 设置CUDA性能优化 - 默认使用确定性设置以与Optuna一致
    if torch.cuda.is_available():
        # 使用确定性CUDNN设置（与Optuna优化一致）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        print("🔒 使用确定性CUDNN设置（与Optuna优化一致）")

    # 创建结果和检查点目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 初始化数据加载器
    root_dir = os.path.expanduser(args.data_dir)
    train_loader, val_loader, _ = get_data_loaders(
        root_dir=root_dir,
        batch_size=args.batch_size,
        eval_batch_size=16,
        num_workers=args.num_workers
    )

    print(f"训练集batch数量: {len(train_loader)}")
    print(f"验证集batch数量: {len(val_loader)}")

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

    # 初始化AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    print(f"使用AdamW优化器: 学习率={args.lr}, 权重衰减={args.weight_decay}")

    # 打印学习率信息
    print(f"使用固定学习率: {args.lr}")

    # 如果恢复训练，也加载优化器状态
    if args.resume and os.path.isfile(args.resume):
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("加载优化器状态")
        except:
            print("优化器状态加载失败，可能是由于优化器类型变更，将使用新的优化器状态")

        # 如果使用混合精度，恢复scaler状态
        if use_amp and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("加载混合精度缩放器状态")

    # 训练日志设置
    log_file = os.path.join(save_dir, 'training_log.txt')
    if start_epoch == 1:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"训练开始于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"参数: {args}\n")
            f.write(f"优化器: AdamW, 学习率={args.lr}, 权重衰减={args.weight_decay}\n")
            f.write(f"学习率调度: 固定学习率 (与Optuna一致)\n")
            f.write(f"模型保存策略: 仅保存验证PSNR最高的模型\n")
            f.write(f"早停策略: 基于验证PSNR停止改善进行早停\n")
            f.write(f"梯度检查频率: 每{args.check_grad_every}个batch检查一次\n\n")
            f.write(
                "轮次,训练损失,训练重建损失,训练PSNR,训练SSIM,验证损失,验证重建损失,验证PSNR,验证SSIM,学习率,用时(秒)\n")

    # 修改：使用验证PSNR作为最佳模型判断标准
    best_val_psnr = 0.0  # 初始化为0，因为PSNR越高越好
    if args.resume and os.path.isfile(args.resume) and 'val_psnr' in checkpoint:
        best_val_psnr = checkpoint['val_psnr']
        print(f"恢复最佳验证PSNR: {best_val_psnr:.4f}")

    # 早停相关变量初始化 - 修改为基于PSNR
    early_stopping_counter = 0
    early_stopping_max_val_psnr = 0.0  # 记录最高的验证PSNR

    # 开始训练循环
    print(f"\n开始训练，共{args.epochs}轮...")
    print(f"模型保存策略: 仅保存验证PSNR最高的模型")
    print(f"早停策略: 基于验证PSNR停止改善进行早停 (耐心值: {args.patience}, 最小改善: {args.min_delta})")
    print(f"🔍 梯度检查: 每{args.check_grad_every}个batch检查一次梯度状态")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        # model.color_transform.capture_epoch(epoch)  # 新版本不需要epoch信息

        # 使用固定学习率，不进行调度
        current_lr = optimizer.param_groups[0]['lr']

        # 训练一个epoch（添加梯度检查）
        train_loss, train_recon_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scaler, use_amp, args.check_grad_every
        )

        # 验证
        val_loss, val_recon_loss, val_psnr, val_ssim = validate(model, val_loader, device, use_amp)

        # 计算用时
        epoch_time = time.time() - start_time

        # 打印结果
        print(f"轮次 {epoch}/{args.epochs} - "
              f"训练损失: {train_loss:.4f}, "
              f"训练重建损失: {train_recon_loss:.4f}, "
              f"训练PSNR: {train_psnr:.2f}, "
              f"训练SSIM: {train_ssim:.4f}, "
              f"验证损失: {val_loss:.4f}, "
              f"验证重建损失: {val_recon_loss:.4f}, "
              f"验证PSNR: {val_psnr:.2f}, "
              f"验证SSIM: {val_ssim:.4f}, "
              f"学习率: {current_lr:.6f}, "
              f"用时: {epoch_time:.2f}秒")

        # 记录日志
        with open(log_file, 'a') as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_recon_loss:.6f},{train_psnr:.4f},{train_ssim:.6f},"
                f"{val_loss:.6f},{val_recon_loss:.6f},{val_psnr:.4f},{val_ssim:.6f},"
                f"{current_lr:.6f},{epoch_time:.2f}\n")

        # 修改：基于验证PSNR保存最佳模型
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')

            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_recon_loss': val_recon_loss,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'train_loss': train_loss,
                'train_recon_loss': train_recon_loss,
                'train_psnr': train_psnr,
                'train_ssim': train_ssim,
                'lr': current_lr
            }

            if use_amp:
                save_dict['scaler_state_dict'] = scaler.state_dict()

            torch.save(save_dict, checkpoint_path)
            print(f"★ 验证PSNR创新高！保存最佳模型检查点到 {checkpoint_path} (PSNR: {val_psnr:.4f})")

            weights_path = os.path.join(save_dir, 'best_model_weights.pth')
            torch.save(model.state_dict(), weights_path)
            print(f"★ 保存最佳模型权重到 {weights_path}")


        # 修改：基于验证PSNR的早停逻辑
        if val_psnr > early_stopping_max_val_psnr + args.min_delta:
            early_stopping_max_val_psnr = val_psnr
            early_stopping_counter = 0
            print(f"验证PSNR显著提升。早停计数器重置。")
        else:
            early_stopping_counter += 1
            print(f"验证PSNR未显著提升。早停计数器: {early_stopping_counter}/{args.patience}")

            if early_stopping_counter >= args.patience:
                print(f"早停在第{epoch}轮训练后触发！最佳验证PSNR: {best_val_psnr:.4f}")
                break

    # 训练完成
    print(f"训练完成。最佳验证PSNR: {best_val_psnr:.4f}")
    print(f"最佳模型已保存到: {save_dir}/best_model.pth")
    print(f"最佳权重已保存到: {save_dir}/best_model_weights.pth")

    # 数据集路径选择:
    # LOLv1: ./datasets/LOL_V1/lol_dataset
    # LOLv2: ./datasets/LOL-v2
    # LSRW:  ./datasets/OpenDataLab___LSRW/raw/LSRW
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练低光照图像增强模型")
    parser.add_argument('--data_dir', type=str,
                        default="./datasets/LOL_V1/lol_dataset",
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=1200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=7.11e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='权重衰减')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='保存模型的目录')
    parser.add_argument('--num_workers', type=int, default=20, help='数据加载器的工作线程数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    # 早停相关参数
    parser.add_argument('--patience', type=int, default=2000, help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='早停最小改善阈值')
    # GPU优化相关参数
    parser.add_argument('--use_amp', action='store_true', help='是否使用混合精度训练')
    # 移除deterministic参数，现在默认使用确定性设置以与Optuna一致
    # parser.add_argument('--deterministic', action='store_true', help='是否使用确定性算法')
    parser.add_argument('--seed', type=int, default=3407, help='随机种子')
    parser.add_argument('--compile', action='store_true', help='使用torch.compile优化模型')
    # 学习率调度相关参数
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Warmup的轮数')
    parser.add_argument('--min_lr_factor', type=float, default=0.001, help='最小学习率倍数')
    parser.add_argument('--steepness', type=float, default=10, help='余弦退火陡峭度参数')
    parser.add_argument('--acceleration_factor', type=float, default=3.0, help='学习率衰减加速因子')
    # 梯度检查相关参数
    parser.add_argument('--check_grad_every', type=int, default=1, help='每N个batch检查一次梯度')
    
    # Optuna超参数优化相关参数
    parser.add_argument('--optuna', action='store_true', help='启用Optuna超参数优化')
    parser.add_argument('--optuna_trials', type=int, default=30, help='Optuna试验次数 (官方建议20-50)')
    parser.add_argument('--optuna_epochs', type=int, default=8, help='每个试验的训练轮数 (官方建议8-10)')
    parser.add_argument('--optuna_study_name', type=str, default='mase_net_optimization', help='Optuna研究名称')
    parser.add_argument('--optuna_storage', type=str, default='sqlite:///optuna_studies/mase_net_full_2_study.db', help='Optuna存储数据库路径')
    parser.add_argument('--diagnostic', action='store_true', help='运行诊断测试')

    args = parser.parse_args()
    
    if args.diagnostic:
        success = run_diagnostic_test()
        if success:
            print("🎉 诊断测试通过，可以开始正式优化")
        else:
            print("❌ 诊断测试失败，请检查代码修改")
    elif args.optuna:
        run_optuna_optimization(args)
    else:
        main(args)

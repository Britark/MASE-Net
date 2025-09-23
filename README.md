# MaSE-Net: Mixture attention & Selective Enhancement Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

MaSE-Net是一个用于低光图像增强的深度学习网络，结合了混合注意力机制(MOA)和混合专家模型(MoE)，实现高效的自适应图像增强。


## 🌟 主要特性

- **双层耦合机制**: 创新的"指引-内容"双层耦合，实现选择与处理的深度整合
- **分离式MoE**: 为不同ISP方法创建专用信息通道，避免信息混淆
- **高效架构**: 仅3M参数即可达到最高PSNR=26, SSIM=0.86的优秀性能
- **端到端训练**: 支持完整的训练和推理流程
- **多数据集支持**: 支持LOL-v1、LOL-v2、LSRW等主流低光增强数据集
<img width="2584" height="882" alt="MASE-Net" src="https://github.com/user-attachments/assets/585d9858-e89a-40c5-a0bd-bcc6c915a9ff" />

## 📊 性能表现

| 数据集 | PSNR | SSIM | 参数量 |
|--------|------|------|---------|
| LOL-v1      | 26.8 | 0.82 | 3M |
| LOL-v2-real | 23.2 | 0.86 | 3M |
| LSRW        | 22.8 | 0.69 | 3M |



## 🚀 快速开始

### 环境要求

**Python版本**：Python 3.9+

**安装依赖**：
# 方法1：使用requirements.txt（推荐）
pip install -r requirements.txt

# 方法2：手动安装核心依赖
pip install torch torchvision numpy matplotlib scikit-image tqdm pillow kornia optuna

### 数据集准备

数据集可从链接获取 https://pan.baidu.com/s/1a9V1A6iIcr0IHUbL-9-a_A?pwd=vn7c

项目支持以下数据集结构：

```
datasets/
├── LOL_V1/
│   └── lol_dataset/
│       ├── Train/
│       ├── Test/
│       └── Val/
├── LOL_v2/
│   ├── Train/
│   ├── Test/
│   └── Val/
└── OpenDataLab___LSRW/
    └── raw/LSRW/
        ├── Train/
        ├── Test/
        └── Val/
```

将数据集下载后放置在对应目录即可。

### ⚙️ 重要配置说明

**针对不同数据集的关键配置参数：**

| 数据集 | patch_size | win_size | 说明 |
|--------|------------|----------|------|
| LOL-v1 | 4          | 2        | 适用于LOL-v1数据集的配置 |
| LSRW   | 4          | 2        | 适用于LSRW数据集的配置   |
| LOL-v2 | 2          | 4        | 适用于LOL-v2数据集的配置 |

**配置修改方法：**
1. 在 `config.py` 文件中修改对应参数：
   ```python
   # LOL-v1和LSRW数据集
   'patch_size': 4,
   'win_size': 2,

   # LOL-v2数据集
   'patch_size': 2,
   'win_size': 4,
   ```

2. **如果需要更改输入输出图像尺寸**，需要同时修改：
   - `config.py` 中的 `input_size` 和 `output_size` 参数
   - `data_loader.py` 中对应的图像预处理尺寸设置

⚠️ **注意**: 不同的patch_size和win_size组合会影响模型的窗口分割和特征提取策略，务必根据所使用的数据集选择正确的配置。

### 预训练模型

将预训练权重文件放置在 `checkpoints/` 目录下：
- `LOLv1_checkpoints.pth` - LOL-v1数据集训练的模型
- `LOLv2_real_checkpoints.pth` - LOL-v2-real数据集训练的模型
- `LSRW_checkpoints.pth` - LSRW数据集训练的模型

## 📖 使用说明

### 训练模型

#### 基础训练
```bash
# 在LOL-v2数据集上训练
python train.py --data_dir ./datasets/LOL_v2 --epochs 1200 --batch_size 4

# 自定义参数训练
python train.py \
    --data_dir ./datasets/LOL_V1/lol_dataset \
    --epochs 800 \
    --batch_size 8 \
    --lr 7.11e-5 \
    --save_dir ./checkpoints
```

#### Optuna超参数优化
```bash
# 启用Optuna自动调参（推荐）
python train.py \
    --optuna \
    --optuna_trials 30 \
    --optuna_epochs 8 \
    --data_dir ./datasets/LOL_v2
```

#### 主要训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 数据集路径 | `./datasets/LOL_v2` |
| `--epochs` | 训练轮数 | 1200 |
| `--batch_size` | 批大小 | 4 |
| `--lr` | 学习率 | 7.11e-5 |
| `--use_amp` | 混合精度训练 | False |
| `--resume` | 恢复训练检查点 | None |
| `--optuna` | 启用超参数优化 | False |

### 测试模型

#### 在测试集上评估
```bash
# 使用LOL-v1模型测试
python test.py \
    --data_dir ./datasets/LOL_V1/lol_dataset \
    --weights_path ./checkpoints/LOLv1_checkpoints.pth \
    --dataset_split test

# 使用LOL-v2模型测试
python test.py \
    --data_dir ./datasets/LOL_v2 \
    --weights_path ./checkpoints/LOLv2_real_checkpoints.pth \
    --dataset_split test
```

#### 测试参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | 数据集路径 | `./datasets/LOL-v2` |
| `--weights_path` | 模型权重路径 | `./checkpoints/LOLv1_checkpoints.pth` |
| `--dataset_split` | 测试集选择 | `test` |
| `--batch_size` | 测试批大小 | 4 |
| `--save_enhanced` | 保存增强图像 | True |
| `--save_originals` | 保存原始图像 | False |

测试结果会自动保存在 `./result/{dataset_type}/` 目录下，包括：
- 三图对比效果图 (`comparison_XXXX.png`)
- 增强后的单独图像 (`enhanced_XXXX.png`)
- 测试指标报告 (`test_results.txt`, `test_results.json`)

### Demo演示

在demo目录下运行演示：

```bash
cd demo

# 使用默认配置运行demo
python demo.py

# 自定义demo配置
python demo.py \
    --data_dir ./lol_dataset \
    --weights_path ../checkpoints/LOLv1_checkpoints.pth \
    --batch_size 2
```

Demo会在 `demo_results/` 目录下生成：
- 演示对比图像 (`demo_comparison_XXXX.png`)
- 演示结果报告 (`demo_results.txt`, `demo_results.json`)

## 🖼️ 效果展示

项目根目录下的 `comparison_1.png` ~ `comparison_8.png` 展示了MaSE-Net在不同场景下的增强效果对比：

- **comparison_1.png**: 室内场景低光增强效果
- **comparison_2.png**: 户外夜景增强对比
- **comparison_3.png**: 复杂光照条件处理
- **comparison_4.png**: 细节保持和噪声抑制
- **comparison_5.png**: 颜色还原准确性
- **comparison_6.png**: 高对比度场景处理
- **comparison_7.png**: 极低光条件增强
- **comparison_8.png**: 混合光源场景优化

每张对比图包含三部分：原始低光图像、MaSE-Net增强结果、参考真值图像，并显示PSNR和SSIM指标。

## 📁 项目结构

```
masenet/
├── README.md              # 项目说明文档
├── config.py              # 模型配置文件
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── models.py              # 主模型定义
├── MoA.py                 # MOA和MOE模块实现
├── feature_extractor.py   # 特征提取器
├── ISP.py                 # ISP操作模块
├── decoder.py             # ISP参数解码器
├── data_loader.py         # 数据加载器
├── data_augmentation.py   # 数据增强工具
├── losses.py              # 损失函数定义
├── utils.py               # 工具函数
├── emb_gen.py             # 嵌入生成器
├── checkpoints/           # 预训练模型目录
├── datasets/              # 数据集目录
├── demo/                  # 演示脚本目录
│   ├── demo.py            # 演示脚本
│   └── lol_dataset/       # 演示数据集
└── comparison_*.png       # 效果展示图片
```

## 🔧 技术细节

### 核心创新

1. **双层耦合机制**:
   - 指引层: Q_isp + K_isp → 智能选择处理方向
   - 内容层: 基于指引的Q_new + K_features/V_features → 针对性内容处理

2. **分离式MoE设计**:
   - MoE1的need_merge=False创建分离通道
   - 为每个ISP方法创建独立信息传递通道
   - 避免不同ISP方法信息相互干扰

3. **渐进式精细化**:
   - 从window级决策到pixel级执行
   - 4个ISP专家分别处理gamma、颜色、去噪、饱和度

### 损失函数

组合损失包含：
- L1重建损失
- 感知损失(VGG特征)
- SSIM结构相似性损失
- PSNR优化损失
- LAB色彩空间损失
- 辅助正则化损失(MoE负载均衡)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📚 引用

如果您在研究中使用了MaSE-Net，请考虑引用：

```bibtex
@misc{masenet2024,
  title={MaSE-Net: Mixture attention \& Selective Enhancement Network for Low-light Image Enhancement},
  author={Xutong Lin},
  year={2025},
  howpublished={\url{https://github.com/Britark/MASE-Net}}
}
```

## 🙏 致谢

感谢以下开源项目和数据集：
- [LOL Dataset](https://daooshee.github.io/BMVC2018website/)
- [Optuna](https://optuna.org/)

---

**联系方式**: 如有问题请提交Issue或发送邮件至 britarklxt@gmail.com

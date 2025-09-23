import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms


class LOLDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        读取低光照增强数据集，加载成对的低光照和正常光照图像
        Args:
            root_dir: 数据集根目录
            mode: 'train', 'val', 或 'test'
            transform: 图像变换
        """
        self.mode = mode
        self.transform = transform

        # 根据mode设置数据路径
        if mode == 'train':
            self.data_path = os.path.join(root_dir, 'Train')
        elif mode == 'val':
            self.data_path = os.path.join(root_dir, 'Val')
        else:  # 'test'
            self.data_path = os.path.join(root_dir, 'Test')

        # 获取所有低光照图像的路径
        self.low_light_dir = os.path.join(self.data_path, 'Low')
        self.normal_light_dir = os.path.join(self.data_path, 'Normal')

        # 获取所有低光照图像文件名
        self.low_light_images = sorted([f for f in os.listdir(self.low_light_dir)
                                        if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 验证每个低光照图像都有对应的正常光照图像
        self.image_pairs = []
        for low_img_name in self.low_light_images:
            normal_img_name = low_img_name.replace('Low', 'Normal')
            normal_img_path = os.path.join(self.normal_light_dir, normal_img_name)

            # 如果文件名替换规则不正确，尝试找匹配的正常光照图像
            if not os.path.exists(normal_img_path):
                # 假设文件编号是相同的
                img_num = ''.join(filter(str.isdigit, low_img_name))
                for normal_file in os.listdir(self.normal_light_dir):
                    if img_num in normal_file:
                        normal_img_name = normal_file
                        normal_img_path = os.path.join(self.normal_light_dir, normal_img_name)
                        break

            if os.path.exists(normal_img_path):
                self.image_pairs.append((
                    os.path.join(self.low_light_dir, low_img_name),
                    normal_img_path
                ))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        low_img_path, normal_img_path = self.image_pairs[idx]

        # 读取低光照图像和正常光照图像
        low_img = Image.open(low_img_path).convert('RGB')
        normal_img = Image.open(normal_img_path).convert('RGB')

        # 应用变换
        if self.transform:
            low_img = self.transform(low_img)
            normal_img = self.transform(normal_img)

        # 返回一个字典，包含输入图像和目标图像
        return {'input': low_img, 'target': normal_img}


def get_transform():
    return transforms.Compose([
        transforms.Resize((400, 600)),
        transforms.ToTensor(),  # 保持 [0,1] 范围
        transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))  # 安全保护
    ])


def get_data_loaders(root_dir, batch_size=16, eval_batch_size=None, num_workers=20):
    """
    获取训练、验证和测试数据加载器

    Args:
        root_dir: 数据集根目录
        batch_size: 训练的批次大小
        eval_batch_size: 评估和测试的批次大小，如果为None则使用batch_size
        num_workers: 数据加载的工作线程数

    Returns:
        train_loader, val_loader, test_loader
    """
    # 如果没有指定评估批量大小，使用训练批量大小
    if eval_batch_size is None:
        eval_batch_size = batch_size

    transform = get_transform()

    # 创建数据集
    train_dataset = LOLDataset(root_dir=root_dir, mode='train', transform=transform)
    val_dataset = LOLDataset(root_dir=root_dir, mode='val', transform=transform)
    test_dataset = LOLDataset(root_dir=root_dir, mode='test', transform=transform)

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,  # 使用评估批量大小
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"训练集大小: {len(train_dataset)}张图像对，{len(train_loader)}个批次")

    return train_loader, val_loader, test_loader


# 用法示例
if __name__ == "__main__":
    # 根目录
    root_dir = "./LOL_V1/lol_dataset"
    # 将相对路径转为绝对路径
    root_dir = os.path.expanduser(root_dir)

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        root_dir=root_dir,
        batch_size=16,
        num_workers=4
    )

    # 验证训练数据加载是否成功
    for batch_idx, batch_data in enumerate(train_loader):
        inputs = batch_data['input']
        targets = batch_data['target']
        print(f"训练批次 {batch_idx + 1}:")
        print(f"输入(低光照)图像形状: {inputs.shape}")
        print(f"目标(正常光照)图像形状: {targets.shape}")
        break  # 只打印第一个批次

    # 验证验证数据加载器
    for batch_data in val_loader:
        inputs = batch_data['input']
        targets = batch_data['target']
        print(f"验证集:")
        print(f"输入(低光照)图像形状: {inputs.shape}")
        print(f"目标(正常光照)图像形状: {targets.shape}")
        break

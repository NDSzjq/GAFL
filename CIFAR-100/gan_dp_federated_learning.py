"""
GAN增强差分隐私联邦学习 - 完整实现
基于对抗学习方案，包含两个阶段：
- 阶段0（第1-70轮）：标准FedAvg训练
- 阶段1（第71-500轮）：GAN增强训练
数据集: CIFAR-100 (32x32x3 RGB图像)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from collections import defaultdict
import copy
import math
import os
import gc
import time
import sys
import signal
import json
import atexit  # 用于程序退出时自动保存
import traceback  # 用于打印异常信息

from secure_aggregation import SecureAggregator, SecureAggregationClient, SecureAggregationServer

PERF_DEBUG = os.environ.get("PERF_DEBUG", "0") == "1"
GATE_TEACHER_MIN_PROB = float(os.environ.get("GATE_TEACHER_MIN_PROB", "0.7"))
FEDPROX_MU = 0.0  # FedProx近端项系数（设为0表示取消FedProx）
TEMPERATURE_T = 1.0  # 温度系数，控制生成采样的集中度（方案3）

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
NUM_CLASSES = 100


class Logger:
    """
    日志类：同时输出到终端和文件
    将所有print输出保存到指定的txt文件中
    """
    def __init__(self, log_file='training_log.txt'):
        """
        初始化日志记录器
        
        参数:
            log_file: 日志文件路径
        """
        self.log_file = log_file
        self.terminal = sys.stdout
        # 创建日志文件，写入头部信息
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("GAN增强差分隐私联邦学习 - 训练日志\n")
            f.write("=" * 60 + "\n")
            f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
    
    def write(self, message):
        """
        写入消息到终端和文件
        """
        # 输出到终端
        self.terminal.write(message)
        # 同时写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        """
        刷新缓冲区
        """
        self.terminal.flush()
    
    def close(self):
        """
        关闭日志记录器，写入结束信息
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
        # 恢复标准输出
        sys.stdout = self.terminal

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _cpu_state_dict(state_dict):
    return {k: v.detach().cpu() for k, v in state_dict.items()}

def _cpu_state_dict_fp16(state_dict):
    out = {}
    for k, v in state_dict.items():
        t = v.detach().cpu()
        if torch.is_floating_point(t) and t.dtype == torch.float32:
            t = t.half()
        out[k] = t
    return out


def set_seed(seed=42):
    """设置随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def worker_init_fn(worker_id):
    """DataLoader worker的随机种子初始化函数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==================== 模型定义 ====================

class Generator(nn.Module):
    """
    条件生成器网络
    输入: 噪声z (100维) + 条件标签y (one-hot, 100维)
    输出: 生成图像 (32x32x3 RGB图像)
    """
    def __init__(self, noise_dim=100, num_classes=NUM_CLASSES, output_dim=3072):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # 定义网络结构
        # 输入: 噪声(100) + 标签(100) = 200维
        self.fc1 = nn.Linear(noise_dim + num_classes, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, output_dim)  # 32*32*3 = 3072
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(2048)
        
        self.register_buffer('cifar_mean', torch.tensor([0.5071, 0.4867, 0.4408], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('cifar_std', torch.tensor([0.2675, 0.2565, 0.2761], dtype=torch.float32).view(1, 3, 1, 1))
    
    def forward(self, z, y):
        """
        前向传播
        参数:
            z: 噪声向量 (batch_size, noise_dim)
            y: 条件标签 one-hot (batch_size, num_classes)
        返回:
            x: 生成图像 (batch_size, 3, 32, 32)
        """
        # 拼接噪声和标签
        x = torch.cat([z, y], dim=1)
        
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        # 第三层
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        
        # 第四层
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
        
        # 输出层
        x = self.fc5(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 3, 32, 32)
        x = (x - self.cifar_mean) / self.cifar_std
        return x


class ResBlockGenerator(nn.Module):
    """
    生成器残差块
    用于Deep-Aligned ResGen-9M
    """
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else None
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.learnable_shortcut = upsample or (in_channels != out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False) if self.learnable_shortcut else nn.Identity()
    
    def forward(self, x):
        h = self.upsample(x) if self.upsample is not None else x
        residual = self.shortcut(h if self.upsample is not None else x)
        out = F.relu(self.bn1(self.conv1(h)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class DeepAlignedResGen9M(nn.Module):
    """
    深度对齐残差生成器
    与ResNet18深度对齐
    参数量: ~9.0M
    
    架构说明:
    - Projector: z(128) + y(128) → 512×4×4
    - Group 1: 与ResNet18 Stage4对齐 (512→256, 4×4→8×8)
    - Group 2: 与ResNet18 Stage3对齐 (256→128, 8×8→16×16)
    - Group 3: 与ResNet18 Stage2/1对齐 (128→64, 16×16→32×32)
    - Head: 64→3, 32×32
    """
    def __init__(self, noise_dim=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # 标签嵌入 (128维)
        self.label_embed = nn.Embedding(num_classes, 128)
        
        # Projector: z(128) + y(128) → 512×4×4
        self.projector = nn.Sequential(
            nn.Linear(noise_dim + 128, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU()
        )
        
        # Group 1: 与ResNet18 Stage4对齐 (512→256, 4×4→8×8)
        self.group1 = nn.Sequential(
            ResBlockGenerator(512, 256, upsample=True),
            ResBlockGenerator(256, 256, upsample=False)
        )
        
        # Group 2: 与ResNet18 Stage3对齐 (256→128, 8×8→16×16)
        self.group2 = nn.Sequential(
            ResBlockGenerator(256, 128, upsample=True),
            ResBlockGenerator(128, 128, upsample=False)
        )
        
        # Group 3: 与ResNet18 Stage2/1对齐 (128→64, 16×16→32×32)
        self.group3 = nn.Sequential(
            ResBlockGenerator(128, 64, upsample=True),
            ResBlockGenerator(64, 64, upsample=False)
        )
        
        # Head: 64→3, 32×32
        self.head = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
        self.register_buffer('cifar_mean', torch.tensor([0.5071, 0.4867, 0.4408], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('cifar_std', torch.tensor([0.2675, 0.2565, 0.2761], dtype=torch.float32).view(1, 3, 1, 1))
    
    def forward(self, z, y):
        """
        前向传播
        
        参数:
            z: 噪声向量 (batch_size, noise_dim)
            y: 标签，可以是one-hot编码或标签索引
        
        返回:
            生成图像 (batch_size, 3, 32, 32)
        """
        # y可以是one-hot或标签索引
        if y.dim() == 2:  # one-hot
            y_idx = y.argmax(dim=1)
        else:
            y_idx = y
        
        # 嵌入标签
        y_embed = self.label_embed(y_idx)
        
        # 拼接噪声和标签
        x = torch.cat([z, y_embed], dim=1)
        
        # Projector
        x = self.projector(x)
        x = x.view(-1, 512, 4, 4)
        
        # Groups (逐步上采样)
        x = self.group1(x)  # 4×4 → 8×8
        x = self.group2(x)  # 8×8 → 16×16
        x = self.group3(x)  # 16×16 → 32×32
        
        # Head
        x = self.head(x)  # Tanh输出 [-1, 1]
        
        # 从Tanh输出[-1,1]转换到标准化后的图像
        x = (x + 1) / 2  # [-1,1] → [0,1]
        x = (x - self.cifar_mean) / self.cifar_std
        
        return x


class BasicBlock(nn.Module):
    """
    ResNet基础块
    用于构建ResNet20网络
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        """初始化BasicBlock"""
        super(BasicBlock, self).__init__()
        # 第一个卷积层: 3x3卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层: 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 残差连接（shortcut）
        self.shortcut = nn.Sequential()
        # 如果stride!=1或通道数不同，需要进行下采样
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """前向传播: F(x) + x"""
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        # 残差连接
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """
    ResNet20 for CIFAR-100
    专门为CIFAR-100设计的小型残差网络
    参数量: 约0.27M
    结构: 1个初始卷积 + 9个BasicBlock(18层) + 1个FC = 20层
    """
    def __init__(self, num_classes=NUM_CLASSES):
        """初始化ResNet20"""
        super(ResNet20, self).__init__()
        self.in_channels = 16  # 初始通道数
        
        # 初始卷积层: 3x3卷积, 16通道
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 三个stage，每个stage 3个BasicBlock
        # Stage1: 16通道, stride=1 (不降采样)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        # Stage2: 32通道, stride=2 (降采样)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        # Stage3: 64通道, stride=2 (降采样)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        
        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """构建一个stage，包含多个BasicBlock"""
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个block用stride，其余为1
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        # 初始卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # 三个stage
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # 全局平均池化
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        # 全连接层
        out = self.fc(out)
        return out
    
    def get_bn_stats(self, x):
        """
        获取各BN层的统计量（均值和方差）
        用于BN正则化损失
        ResNet20共有19个BN层: 1个初始 + 9个BasicBlock × 2个
        """
        bn_stats = []
        
        # 初始卷积层的BN
        out = self.conv1(x)
        out = self.bn1(out)
        bn_stats.append((self.bn1.running_mean, self.bn1.running_var))
        out = F.relu(out)
        
        # 遍历所有layer中的BN层
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                # 第一个BN层
                out = block.conv1(out)
                out = block.bn1(out)
                bn_stats.append((block.bn1.running_mean, block.bn1.running_var))
                out = F.relu(out)
                # 第二个BN层
                out = block.conv2(out)
                out = block.bn2(out)
                bn_stats.append((block.bn2.running_mean, block.bn2.running_var))
                out = F.relu(out)
        
        return bn_stats


class ResNet18(nn.Module):
    """
    ResNet18 for CIFAR-100
    标准ResNet18结构，针对CIFAR-100调整
    参数量: 约11M
    结构: 1个初始卷积 + 8个BasicBlock(16层) + 1个FC = 18层
    """
    def __init__(self, num_classes=NUM_CLASSES):
        """初始化ResNet18"""
        super(ResNet18, self).__init__()
        self.in_channels = 64  # 初始通道数
        
        # 初始卷积层: 3x3卷积, 64通道 (针对CIFAR-100调整，使用较小的kernel)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 四个stage，每个stage 2个BasicBlock
        # Stage1: 64通道, stride=1 (不降采样)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        # Stage2: 128通道, stride=2 (降采样)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        # Stage3: 256通道, stride=2 (降采样)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        # Stage4: 512通道, stride=2 (降采样)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """构建一个stage，包含多个BasicBlock"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        # 初始卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # 四个stage
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 全局平均池化
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        # 全连接层
        out = self.fc(out)
        return out
    
    def get_bn_stats(self, x):
        """
        获取各BN层的统计量（均值和方差）
        用于BN正则化损失
        ResNet18共有17个BN层: 1个初始 + 8个BasicBlock × 2个
        """
        bn_stats = []
        
        # 初始卷积层的BN
        out = self.conv1(x)
        out = self.bn1(out)
        bn_stats.append((self.bn1.running_mean, self.bn1.running_var))
        out = F.relu(out)
        
        # 遍历所有layer中的BN层
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                # 第一个BN层
                out = block.conv1(out)
                out = block.bn1(out)
                bn_stats.append((block.bn1.running_mean, block.bn1.running_var))
                out = F.relu(out)
                # 第二个BN层
                out = block.conv2(out)
                out = block.bn2(out)
                bn_stats.append((block.bn2.running_mean, block.bn2.running_var))
                out = F.relu(out)
        
        return bn_stats


# ==================== 辅助函数 ====================

def save_training_history_json(history, filename='training_history.json'):
    """
    保存训练历史到JSON文件
    
    参数:
        history: 训练历史字典，包含round, train_loss, test_accuracy, test_loss
        filename: 保存的文件名
    """
    # 创建保存数据的副本，确保数据类型可JSON序列化
    save_data = {
        'round': [int(r) for r in history['round']],
        'train_loss': [float(l) for l in history['train_loss']],
        'test_accuracy': [float(a) for a in history['test_accuracy']],
        'test_loss': [float(l) for l in history['test_loss']]
    }
    
    # 使用临时文件+原子写入，防止写入过程中断导致文件损坏
    temp_filename = filename + '.tmp'
    with open(temp_filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    # 原子性重命名（Windows下需要先删除目标文件）
    if os.path.exists(filename):
        os.remove(filename)
    os.rename(temp_filename, filename)


def create_signal_handler(history_ref, filename='training_history.json'):
    """
    创建信号处理函数，用于捕获中断信号时保存数据
    
    参数:
        history_ref: 训练历史字典的引用
        filename: 保存的文件名
    
    返回:
        signal_handler: 信号处理函数
    """
    def signal_handler(signum, frame):
        # 获取信号名称
        signal_names = {
            signal.SIGINT: 'SIGINT (Ctrl+C)',
            signal.SIGTERM: 'SIGTERM (终止信号)',
        }
        signal_name = signal_names.get(signum, f'信号 {signum}')
        
        print(f"\n\n收到中断信号 ({signal_name})，正在保存训练历史...")
        save_training_history_json(history_ref, filename)
        print(f"训练历史已保存到: {filename}")
        print(f"已完成轮次: {len(history_ref['round'])}")
        sys.exit(0)
    return signal_handler


def setup_safe_exit(history_ref, filename='training_history.json'):
    """
    设置安全退出机制，包括：
    1. 信号处理 (SIGINT, SIGTERM)
    2. atexit 注册（程序正常退出时调用）
    3. 全局异常钩子（未捕获异常时调用）
    
    参数:
        history_ref: 训练历史字典的引用
        filename: 保存的文件名
    """
    # 保存函数，供多个退出机制调用
    def save_and_log(reason="退出"):
        try:
            if len(history_ref['round']) > 0:
                print(f"\n[{reason}] 正在保存训练历史...")
                save_training_history_json(history_ref, filename)
                print(f"训练历史已保存到: {filename}")
                print(f"已完成轮次: {len(history_ref['round'])}")
        except Exception as e:
            print(f"保存训练历史时出错: {e}")
    
    # 1. 注册 atexit（程序正常退出时调用）
    def atexit_handler():
        save_and_log("程序退出")
    atexit.register(atexit_handler)
    
    # 2. 设置信号处理
    signal_handler = create_signal_handler(history_ref, filename)
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill 命令
    
    # 3. 设置全局异常钩子（未捕获异常时调用）
    original_excepthook = sys.excepthook
    
    def custom_excepthook(exc_type, exc_value, exc_tb):
        # 打印完整的异常信息
        print("\n" + "=" * 60)
        print("程序发生未捕获的异常！")
        print("=" * 60)
        traceback.print_exception(exc_type, exc_value, exc_tb)
        print("=" * 60)
        
        # 保存训练历史
        save_and_log("异常退出")
        
        # 调用原始的异常钩子
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        
        # 退出程序
        sys.exit(1)
    
    sys.excepthook = custom_excepthook
    
    print("安全退出机制已启用:")
    print("  - 信号处理: SIGINT (Ctrl+C), SIGTERM")
    print("  - atexit 注册: 程序正常退出时自动保存")
    print("  - 异常钩子: 未捕获异常时自动保存")


def load_training_history(history_path='training_history.json'):
    """
    加载训练历史JSON文件
    
    参数:
        history_path: 训练历史文件路径
    
    返回:
        history: 训练历史字典，如果文件不存在则返回空字典
    """
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        print(f"  已加载训练历史: {history_path}")
        print(f"  已完成轮次: {len(history['round'])}")
        return history
    else:
        return {
            'round': [],
            'train_loss': [],
            'test_accuracy': [],
            'test_loss': []
        }


def restore_random_states(random_states):
    """
    恢复随机状态以实现可复现的继续训练
    
    参数:
        random_states: 随机状态字典
    """
    if random_states is None:
        return
    
    # 恢复PyTorch CPU随机状态
    if 'torch_cpu' in random_states:
        torch.set_rng_state(random_states['torch_cpu'])
    
    # 恢复NumPy随机状态
    if 'numpy' in random_states:
        np.random.set_state(random_states['numpy'])
    
    # 恢复Python随机状态
    if 'python' in random_states:
        random.setstate(random_states['python'])
    
    # 恢复GPU随机状态
    if 'torch_cuda' in random_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(random_states['torch_cuda'])
    
    print("  已恢复随机状态")


def compute_cosine_similarity_matrix(client_params_list):
    """
    计算客户端模型参数之间的余弦相似度矩阵
    
    参数:
        client_params_list: 客户端模型参数列表 [params1, params2, ...]
        每个params是一个字典 {key: tensor}
    
    返回:
        similarity_matrix: 余弦相似度矩阵 (n_clients x n_clients) 的列表形式
    """
    # 获取客户端数量
    n_clients = len(client_params_list)
    
    # 如果只有一个客户端，返回1x1矩阵
    if n_clients == 1:
        return [[1.0]]
    
    # Step 1: 将每个客户端的参数展平为向量
    vectors = []
    for params in client_params_list:
        # 将所有参数拼接成一个向量
        vec = torch.cat([p.flatten() for p in params.values()])
        vectors.append(vec)
    
    # Step 2: 堆叠为矩阵 (n_clients x param_dim)
    matrix = torch.stack(vectors)
    
    # Step 3: L2归一化（使向量长度为1）
    norms = matrix.norm(dim=1, keepdim=True)
    normalized = matrix / (norms + 1e-8)  # 加小常数避免除零
    
    # Step 4: 计算余弦相似度矩阵
    # 归一化后，余弦相似度 = 归一化向量的点积
    similarity_matrix = torch.mm(normalized, normalized.t())
    
    # Step 5: 转换为列表格式
    similarity_list = similarity_matrix.tolist()
    
    return similarity_list


def apply_logit_masking(logits, local_classes, num_classes=NUM_CLASSES, class_counts=None, num_samples=None):
    """
    对logits进行动态掩码操作（方案3：基于样本数量阈值）
    
    参数:
        logits: 模型输出 (batch_size, num_classes)
        local_classes: 客户端拥有的类别列表（保留参数兼容性）
        num_classes: 总类别数
        class_counts: 每个类别的样本数量（新增，方案3）
        num_samples: 客户端总样本数（新增，方案3）
    
    返回:
        masked_logits: 掩码后的logits
        minority_classes: 少数类集合（需要生成的类别）
    """
    # 方案3：计算阈值 tau = N_total / K
    # 如果提供了num_samples，使用动态阈值；否则使用默认值
    if num_samples is not None and class_counts is not None:
        # 动态阈值：总样本数/类别数
        tau = num_samples / num_classes
        
        # 确定多数类（样本充足，无需生成）和少数类（样本稀缺，需要生成）
        majority_classes = [c for c in range(num_classes) if class_counts[c] >= tau]
        minority_classes = [c for c in range(num_classes) if class_counts[c] < tau]
        
        # 漏洞修复：如果少数类为空（所有类别样本都充足），则使用缺失类
        if len(minority_classes) == 0:
            minority_classes = [i for i in range(num_classes) if i not in local_classes]
            majority_classes = list(local_classes)
    else:
        # 兼容原有逻辑：客户端拥有的类别为多数类，缺失的类别为少数类
        minority_classes = [i for i in range(num_classes) if i not in local_classes]
        majority_classes = list(local_classes)
    
    # 创建掩码：多数类为0，少数类为1
    mask = torch.ones(logits.shape[0], num_classes, device=logits.device)
    for c in majority_classes:
        mask[:, c] = 0
    
    # 应用掩码：将多数类对应的logit设为-100（避免数值溢出）
    masked_logits = logits.clone()
    masked_logits[mask == 0] = -100
    
    return masked_logits, minority_classes


def diversity_loss(generator, z1, z2, y):
    """
    多样性损失：防止模式崩溃
    
    参数:
        generator: 生成器
        z1, z2: 两个不同的噪声向量
        y: 条件标签
    
    返回:
        loss: 多样性损失（负值，因为要最小化）
    """
    # 生成两张图像
    img1 = generator(z1, y)
    img2 = generator(z2, y)
    
    # 计算图像差异与噪声差异的比例
    img_diff = torch.norm(img1 - img2, p=2)
    z_diff = torch.norm(z1 - z2, p=2)
    
    # 使用tanh归一化，将值映射到[-1, 0]范围
    # tanh输出[0,1]，取负后变为[-1,0]，避免极端值
    normalized_loss = -torch.tanh(img_diff / (z_diff + 1e-8))
    
    return normalized_loss


def bn_regularization_loss(global_model, fake_images):
    """
    BN层正则化损失
    
    参数:
        global_model: 全局模型（带BN层）
        fake_images: 生成的假图像
    
    返回:
        loss: BN正则化损失
    """
    # 存储各BN层的输入
    bn_inputs = {}
    
    # 定义钩子函数
    def hook_fn(name):
        def hook(module, input, output):
            bn_inputs[name] = input[0].detach()
        return hook
    
    # 注册钩子到所有BN层
    hooks = []
    bn_layers = {}
    
    for name, module in global_model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers[name] = module
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    was_training = global_model.training
    global_model.eval()
    _ = global_model(fake_images)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 计算损失
    loss = 0.0
    for name, module in bn_layers.items():
        if name not in bn_inputs:
            continue
        
        bn_input = bn_inputs[name]
        
        # 计算生成图像的统计量
        if isinstance(module, nn.BatchNorm2d):
            # Conv层: [N, C, H, W] -> [C]
            fake_mean = bn_input.mean(dim=[0, 2, 3])
            fake_var = bn_input.var(dim=[0, 2, 3], unbiased=False)
        else:
            # FC层: [N, C] -> [C]
            fake_mean = bn_input.mean(dim=0)
            fake_var = bn_input.var(dim=0, unbiased=False)
        
        # 获取真实数据的统计量
        running_mean = module.running_mean
        running_var = module.running_var
        
        # 计算MSE损失
        loss += F.mse_loss(fake_mean, running_mean)
        loss += F.mse_loss(fake_var, running_var)
    
    if was_training:
        global_model.train()
    else:
        global_model.eval()
    return loss


def get_lambda(*_args, **_kwargs):
    """
    固定混合权重λ为0.3
    """
    return 0.3


# ==================== 训练函数 ====================

def train_generator(generator, global_model, local_classes, 
                    num_epochs=10, batch_size=64, 
                    alpha=0.1, beta=0.05, lr=0.0002,
                    device='cpu', class_counts=None, num_samples=None,
                    temperature=1.0):
    """
    训练生成器（方案3：支持频率加权采样）
    
    参数:
        generator: 生成器网络
        global_model: 冻结的全局模型（判别器）
        local_classes: 客户端拥有的类别（保留参数兼容性）
        num_epochs: 训练轮次
        batch_size: 批量大小
        alpha: 多样性损失权重
        beta: BN正则化权重
        lr: 学习率
        class_counts: 每个类别的样本数量（新增，方案3）
        num_samples: 客户端总样本数（新增，方案3）
        temperature: 温度系数T，控制采样集中度（新增，方案3）
    
    返回:
        generator: 训练后的生成器
        avg_loss: 平均总损失
        avg_cls_loss: 平均分类损失
        avg_div_loss: 平均多样性损失
        avg_bn_loss: 平均BN正则化损失
    """
    # 冻结全局模型参数
    for param in global_model.parameters():
        param.requires_grad = False
    
    # 关键：将全局模型设为评估模式，防止BatchNorm更新统计量
    global_model.eval()
    
    # 设置生成器为训练模式
    generator.train()
    
    # 定义优化器
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 方案3：计算动态掩码和采样权重
    if num_samples is not None and class_counts is not None:
        # 动态阈值：总样本数/类别数
        tau = num_samples / NUM_CLASSES
        # 确定少数类（样本稀缺，需要生成）
        minority_classes = [c for c in range(NUM_CLASSES) if class_counts[c] < tau]
        
        # 漏洞修复：如果少数类为空（所有类别样本都充足），则使用缺失类
        if len(minority_classes) == 0:
            minority_classes = [i for i in range(NUM_CLASSES) if i not in local_classes]
        
        # 漏洞修复：如果minority_classes仍然为空（所有100个类别都有），则跳过训练
        if len(minority_classes) == 0:
            # 解冻全局模型参数
            for param in global_model.parameters():
                param.requires_grad = True
            # 恢复全局模型为训练模式
            global_model.train()
            # 返回未训练的生成器和0损失
            return generator, 0.0, 0.0, 0.0, 0.0
        
        # 方案3：计算频率加权采样权重 P(y=c) = exp(-n_c / T) / sum(exp(-n_k / T))
        sampling_weights = torch.zeros(NUM_CLASSES, device=device)
        for c in minority_classes:
            # 样本越少，权重越大（完全缺失的类权重最大）
            if class_counts[c] > 0:
                sampling_weights[c] = math.exp(-class_counts[c] / temperature)
            else:
                sampling_weights[c] = math.exp(0)  # 完全缺失的类权重为1
        
        # 漏洞修复：如果权重全为0，则均匀分布
        if sampling_weights.sum() == 0:
            for c in minority_classes:
                sampling_weights[c] = 1.0 / len(minority_classes) if len(minority_classes) > 0 else 0.0
        else:
            # 归一化
            sampling_weights = sampling_weights / sampling_weights.sum()
    else:
        # 兼容原有逻辑：缺失的类别为少数类
        minority_classes = [i for i in range(NUM_CLASSES) if i not in local_classes]
        sampling_weights = None
        
        # 漏洞修复：如果minority_classes为空（客户端拥有所有类别），则跳过训练
        if len(minority_classes) == 0:
            # 解冻全局模型参数
            for param in global_model.parameters():
                param.requires_grad = True
            # 恢复全局模型为训练模式
            global_model.train()
            # 返回未训练的生成器和0损失
            return generator, 0.0, 0.0, 0.0, 0.0
    
    # 训练循环
    total_loss = 0.0
    total_cls_loss = 0.0  # 记录分类损失
    total_div_loss = 0.0  # 记录多样性损失
    total_bn_loss = 0.0   # 记录BN正则化损失
    num_batches = 0
    
    # 减小batch_size以节省显存
    effective_batch_size = min(batch_size, 64)  # 限制最大64
    
    # 注册BN层钩子，在前向传播时收集统计量
    bn_stats = {}
    def hook_fn(name):
        def hook(module, input, output):
            x = input[0]
            if isinstance(module, nn.BatchNorm2d):
                fake_mean = x.mean(dim=(0, 2, 3))
                fake_var = x.var(dim=(0, 2, 3), unbiased=False)
            else:
                fake_mean = x.mean(dim=0)
                fake_var = x.var(dim=0, unbiased=False)
            bn_stats[name] = (fake_mean, fake_var)
        return hook
    
    hooks = []
    bn_layers = {}
    for name, module in global_model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers[name] = module
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    for epoch in range(num_epochs):
        epoch_cls_loss = 0.0
        epoch_div_loss = 0.0
        epoch_bn_loss = 0.0
        epoch_batches = 0
        
        for _ in range(10):  # 每个epoch 10个batch
            # 采样噪声
            z = torch.randn(effective_batch_size, generator.noise_dim, device=device)
            
            # 方案3：频率加权采样类别（越稀疏，越优先）
            if sampling_weights is not None:
                # 使用多项式分布进行加权采样
                y_indices = torch.multinomial(sampling_weights, effective_batch_size, replacement=True)
            else:
                # 兼容原有逻辑：均匀采样
                y_indices = torch.tensor(
                    [minority_classes[i] for i in torch.randint(0, len(minority_classes), (effective_batch_size,))],
                    device=device
                )
            
            # 创建one-hot标签
            y = torch.zeros(effective_batch_size, generator.num_classes, device=device)
            for i, class_label in enumerate(y_indices):
                y[i, class_label] = 1
            
            # 生成假图像
            fake_images = generator(z, y)
            
            # 清空BN统计量
            bn_stats.clear()
            
            # 计算分类损失（带动态掩码，方案3）
            # 同时通过钩子收集BN层统计量
            logits = global_model(fake_images)
            masked_logits, _ = apply_logit_masking(logits, local_classes, 
                                                    class_counts=class_counts, 
                                                    num_samples=num_samples)
            loss_cls = F.cross_entropy(masked_logits, y_indices)
            
            # 计算BN正则化损失（使用收集到的统计量）
            loss_bn = torch.tensor(0.0, device=device)
            for name, module in bn_layers.items():
                if name not in bn_stats:
                    continue
                fake_mean, fake_var = bn_stats[name]
                loss_bn = loss_bn + F.mse_loss(fake_mean, module.running_mean)
                loss_bn = loss_bn + F.mse_loss(fake_var, module.running_var)
            
            # 计算多样性损失（使用更小的batch）
            z1 = torch.randn(effective_batch_size, generator.noise_dim, device=device)
            z2 = torch.randn(effective_batch_size, generator.noise_dim, device=device)
            loss_div = diversity_loss(generator, z1, z2, y)
            
            # 总损失
            loss = loss_cls + alpha * loss_div + beta * loss_bn
            
            # 记录损失（在删除变量之前）
            epoch_cls_loss += loss_cls.item()
            epoch_div_loss += loss_div.item()
            epoch_bn_loss += loss_bn.item()
            epoch_batches += 1
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 清理显存
            del fake_images, logits, masked_logits, z, y, z1, z2, loss_cls, loss_div, loss_bn, loss
            bn_stats.clear()
        
        # 不再输出每个epoch的损失，只记录
        total_cls_loss += epoch_cls_loss
        total_div_loss += epoch_div_loss
        total_bn_loss += epoch_bn_loss
        num_batches += epoch_batches
    
    # 计算总体平均损失
    avg_cls_loss = total_cls_loss / num_batches
    avg_div_loss = total_div_loss / num_batches
    avg_bn_loss = total_bn_loss / num_batches
    avg_loss = avg_cls_loss + alpha * avg_div_loss + beta * avg_bn_loss
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 解冻全局模型参数
    for param in global_model.parameters():
        param.requires_grad = True
    
    # 恢复全局模型为训练模式
    global_model.train()
    
    return generator, avg_loss, avg_cls_loss, avg_div_loss, avg_bn_loss


def generate_synthetic_data(generator, missing_classes, 
                           num_samples_per_class=100, device='cpu',
                           sampling_weights=None, total_samples=None):
    """
    生成缺失类别的数据（方案3：支持频率加权采样）
    
    参数:
        generator: 训练好的生成器
        missing_classes: 缺失的类别列表（少数类）
        num_samples_per_class: 每个类别生成的样本数
        device: 计算设备
        sampling_weights: 采样权重向量（方案3）
        total_samples: 总生成样本数（方案3，如果使用加权采样）
    
    返回:
        synthetic_images: 生成的图像张量
        synthetic_labels: 生成的标签张量
    """
    generator.eval()
    
    # 漏洞修复：如果missing_classes为空，直接返回空张量
    if len(missing_classes) == 0:
        return torch.zeros(0, 3, 32, 32), torch.tensor([], dtype=torch.long)
    
    synthetic_data = {c: [] for c in missing_classes}
    synthetic_images = []
    synthetic_labels = []
    
    with torch.no_grad():
        # 方案3：如果提供了采样权重，使用频率加权采样
        if sampling_weights is not None and total_samples is not None and total_samples > 0:
            # 漏洞修复：确保total_samples不小于missing_classes的数量
            # 每个类别至少分配1个样本
            effective_total = max(total_samples, len(missing_classes))
            
            # 根据采样权重确定每个类别的生成数量
            class_samples = {}
            remaining = effective_total
            for i, c in enumerate(missing_classes):
                if i < len(missing_classes) - 1:
                    # 按权重分配样本数（至少分配1个样本）
                    # 漏洞修复：sampling_weights[c]可能为0（如果该类不在minority_classes中）
                    weight_val = sampling_weights[c].item() if sampling_weights[c] > 0 else 1.0 / len(missing_classes)
                    samples = max(1, int(effective_total * weight_val))
                    samples = min(samples, remaining - (len(missing_classes) - i - 1))  # 确保剩余类别至少有1个样本
                    class_samples[c] = max(1, samples)
                    remaining -= class_samples[c]
                else:
                    # 最后一个类别获得剩余样本（至少1个）
                    class_samples[c] = max(1, remaining)
            
            # 按加权数量生成数据（分批生成以节省显存）
            gen_batch_size = 64  # 每批生成的数量
            for class_label in missing_classes:
                num_samples = class_samples[class_label]
                if num_samples <= 0:
                    continue
                
                class_images = []
                # 分批生成
                for start in range(0, num_samples, gen_batch_size):
                    end = min(start + gen_batch_size, num_samples)
                    batch_num = end - start
                    
                    # 采样噪声
                    z = torch.randn(batch_num, generator.noise_dim, device=device)
                    
                    # 创建one-hot标签
                    y = torch.zeros(batch_num, generator.num_classes, device=device)
                    y[:, class_label] = 1
                    
                    # 生成图像
                    fake_images = generator(z, y)
                    class_images.append(fake_images.cpu())
                    
                    # 清理显存
                    del z, y, fake_images
                
                # 合并该类别的所有图像
                if class_images:
                    all_class_images = torch.cat(class_images, dim=0)
                    synthetic_data[class_label] = all_class_images
                    synthetic_images.append(all_class_images)
                    synthetic_labels.extend([class_label] * num_samples)
        else:
            # 原有逻辑：每个类别生成固定数量的样本（分批生成以节省显存）
            gen_batch_size = 64
            for class_label in missing_classes:
                class_images = []
                remaining = num_samples_per_class
                
                for start in range(0, num_samples_per_class, gen_batch_size):
                    end = min(start + gen_batch_size, num_samples_per_class)
                    batch_num = end - start
                    
                    # 采样噪声
                    z = torch.randn(batch_num, generator.noise_dim, device=device)
                    
                    # 创建one-hot标签
                    y = torch.zeros(batch_num, generator.num_classes, device=device)
                    y[:, class_label] = 1
                    
                    # 生成图像
                    fake_images = generator(z, y)
                    class_images.append(fake_images.cpu())
                    
                    # 清理显存
                    del z, y, fake_images
                
                # 合并该类别的所有图像
                if class_images:
                    all_class_images = torch.cat(class_images, dim=0)
                    synthetic_data[class_label] = all_class_images
                    synthetic_images.append(all_class_images)
                    synthetic_labels.extend([class_label] * num_samples_per_class)
    
    # 合并所有生成数据
    if synthetic_images:
        synthetic_images = torch.cat(synthetic_images, dim=0)
        synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.long)
    else:
        # 如果没有生成任何数据，返回空张量
        synthetic_images = torch.zeros(0, 3, 32, 32)
        synthetic_labels = torch.tensor([], dtype=torch.long)
    
    return synthetic_images, synthetic_labels


def train_with_mixed_data(model, real_data_loader, synthetic_images, synthetic_labels,
                         epochs=6, batch_size=64, 
                         real_weight=0.7, synthetic_weight=0.3,
                         device='cpu', clipping_threshold=1.2,
                         global_model_params=None,
                         minority_classes=None, tau=None, lambda_val=0.2):
    """
    使用真实数据和生成数据训练模型（方案3：γ调节因子 + 混合损失公式）
    
    核心改进：
    1. 每类目标配额 b = Batch_Size / K
    2. 真实样本每轮只使用一次（去冗余）
    3. 生成样本补齐剩余配额
    4. 强制Batch类别均衡
    5. γ调节因子：γ_c = 1 - n_c/τ
    6. 混合损失公式：
       L = ∑_{c∈C_maj} L_CE(Real) + ∑_{c∈C_min} [(1-γ)L_CE(Real) + γλL_CE(Syn)]
    
    参数:
        model: 本地模型
        real_data_loader: 真实数据加载器
        synthetic_images: 生成图像
        synthetic_labels: 生成标签
        epochs: 训练轮次
        batch_size: 批量大小
        real_weight: 真实数据权重（已弃用，使用γ调节）
        synthetic_weight: 生成数据权重（已弃用，使用γλ）
        device: 计算设备
        clipping_threshold: 梯度裁剪阈值
        global_model_params: 全局模型参数（用于FedProx近端项）
        minority_classes: 少数类列表（方案3）
        tau: 动态阈值（方案3）
        lambda_val: 生成数据权重λ（方案3）
    
    返回:
        model: 训练后的模型
        avg_loss: 平均损失
        client_grad_norm_median: 客户端内部梯度范数中位数
    """
    model.train()
    
    # 保存全局模型参数用于FedProx近端项
    global_params = None
    if global_model_params is not None and FEDPROX_MU > 0:
        global_params = {}
        for name, param in model.named_parameters():
            if name in global_model_params:
                global_params[name] = global_model_params[name].clone().detach().to(device)
    
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    
    # 方案3：去冗余均衡Batch构建
    num_classes = 100
    # 修改：每类目标配额至少为1，确保batch_size能覆盖所有类别
    quota_per_class = max(1, batch_size // num_classes)  # 每类目标配额
    
    dataset = real_data_loader.dataset
    real_indices_by_class = {c: [] for c in range(num_classes)}
    if isinstance(dataset, Subset):
        base = dataset.dataset
        if hasattr(base, 'targets'):
            labels = [int(base.targets[i]) for i in dataset.indices]
        else:
            labels = [int(dataset[i][1]) for i in range(len(dataset))]
    elif hasattr(dataset, 'targets'):
        labels = [int(x) for x in dataset.targets]
    elif isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 2:
        labels = [int(x) for x in dataset.tensors[1].detach().cpu().tolist()]
    else:
        labels = [int(dataset[i][1]) for i in range(len(dataset))]

    for i, label in enumerate(labels):
        if 0 <= label < num_classes:
            real_indices_by_class[label].append(i)

    real_counts = {c: len(real_indices_by_class[c]) for c in range(num_classes)}
    total_real = sum(real_counts.values())
    
    # 计算γ调节因子（方案3）
    # γ_c = 1 - n_c/τ
    gamma = {}
    if minority_classes is not None and tau is not None:
        for c in range(num_classes):
            if c in minority_classes:
                # 少数类：γ_c = 1 - n_c/τ
                gamma[c] = max(0.0, min(1.0, 1.0 - real_counts[c] / tau))
            else:
                # 多数类：γ = 0（不使用生成数据）
                gamma[c] = 0.0
    else:
        # 兼容旧逻辑：所有类γ=0.2
        for c in range(num_classes):
            gamma[c] = 0.2
    
    # 计算总批次数（基于真实样本数）
    num_batches_per_epoch = max(1, total_real // batch_size)
    
    syn_indices_by_class = {c: [] for c in range(num_classes)}
    if int(synthetic_images.shape[0]) > 0:
        syn_labels_list = synthetic_labels.detach().cpu().tolist()
        for i, label in enumerate(syn_labels_list):
            if 0 <= int(label) < num_classes:
                syn_indices_by_class[int(label)].append(i)
    
    # 训练循环
    total_loss = 0.0
    num_batches = 0
    all_batch_global_grad_norms = []
    
    # 统计实际使用的样本数量
    total_real_used = 0  # 实际使用的真实样本总数
    total_syn_used = 0   # 实际使用的生成样本总数
    minority_real_used = 0  # 少数类真实样本使用数
    minority_syn_used = 0   # 少数类生成样本使用数
    
    # 按类别统计生成样本使用情况
    syn_used_by_class = {c: 0 for c in range(num_classes)}
    real_used_by_class = {c: 0 for c in range(num_classes)}
    
    for epoch in range(epochs):
        # 打乱每个类别的真实数据索引
        real_indices = {c: real_indices_by_class[c].copy() for c in range(num_classes)}
        for c in range(num_classes):
            random.shuffle(real_indices[c])
        
        # 记录每个类别已使用的真实样本数
        used_real_count = {c: 0 for c in range(num_classes)}
        
        for batch_idx in range(num_batches_per_epoch):
            batch_data = []
            batch_labels = []
            batch_is_synthetic = []
            batch_class_labels = []
            
            for c in range(num_classes):
                # 计算该类需要的真实样本数
                remaining_real = real_counts[c] - used_real_count[c]
                remaining_batches = num_batches_per_epoch - batch_idx
                
                if remaining_batches > 0 and remaining_real > 0:
                    # 按比例分配真实样本
                    real_quota = min(quota_per_class, remaining_real // remaining_batches)
                    real_quota = max(0, real_quota)
                else:
                    real_quota = 0
                
                # 生成样本补齐（仅对少数类）
                if minority_classes is not None and c in minority_classes:
                    syn_quota = quota_per_class - real_quota
                else:
                    syn_quota = 0  # 多数类不使用生成数据
                
                # 添加真实样本
                if real_quota > 0 and used_real_count[c] < len(real_indices[c]):
                    real_added = 0  # 记录实际添加的真实样本数
                    for _ in range(real_quota):
                        if used_real_count[c] >= len(real_indices[c]):
                            break
                        sample_idx = real_indices[c][used_real_count[c]]
                        x, _ = dataset[sample_idx]
                        batch_data.append(x)
                        batch_labels.append(c)
                        batch_is_synthetic.append(False)
                        batch_class_labels.append(c)
                        used_real_count[c] += 1
                        real_added += 1
                    # 统计真实样本使用
                    total_real_used += real_added
                    real_used_by_class[c] += real_added  # 按类别统计
                    if minority_classes is not None and c in minority_classes:
                        minority_real_used += real_added
                
                # 添加生成样本（仅对少数类）
                if syn_quota > 0 and len(syn_indices_by_class[c]) > 0:
                    syn_added = 0  # 记录实际添加的生成样本数
                    for _ in range(syn_quota):
                        syn_idx = random.choice(syn_indices_by_class[c])
                        batch_data.append(synthetic_images[syn_idx])
                        batch_labels.append(c)
                        batch_is_synthetic.append(True)
                        batch_class_labels.append(c)
                        syn_added += 1
                    # 统计生成样本使用
                    total_syn_used += syn_added
                    syn_used_by_class[c] += syn_added  # 按类别统计
                    if minority_classes is not None and c in minority_classes:
                        minority_syn_used += syn_added
            
            # 如果batch为空，跳过
            if len(batch_data) == 0:
                continue
            
            # 合并batch数据
            batch_data = torch.stack(batch_data, dim=0).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
            batch_is_synthetic = torch.tensor(batch_is_synthetic, dtype=torch.bool, device=device)
            batch_class_labels = torch.tensor(batch_class_labels, dtype=torch.long, device=device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(batch_data)
            
            # 方案3：使用γ调节因子的混合损失公式
            # L = ∑_{c∈C_maj} L_CE(Real) + ∑_{c∈C_min} [(1-γ)L_CE(Real) + γλL_CE(Syn)]
            loss = 0.0
            
            for c in range(num_classes):
                # 获取该类别的样本mask
                class_mask = (batch_class_labels == c)
                if not class_mask.any():
                    continue
                
                class_output = output[class_mask]
                class_labels = batch_labels[class_mask]
                class_is_syn = batch_is_synthetic[class_mask]
                
                if minority_classes is not None and c in minority_classes:
                    # 少数类：使用γ调节
                    # 真实数据权重：(1-γ_c)
                    # 生成数据权重：γ_c * λ
                    real_mask_c = ~class_is_syn
                    syn_mask_c = class_is_syn
                    
                    if real_mask_c.any():
                        real_output = class_output[real_mask_c]
                        real_labels = class_labels[real_mask_c]
                        loss += (1.0 - gamma[c]) * F.cross_entropy(real_output, real_labels)
                    
                    if syn_mask_c.any():
                        syn_output = class_output[syn_mask_c]
                        syn_labels = class_labels[syn_mask_c]
                        loss += gamma[c] * lambda_val * F.cross_entropy(syn_output, syn_labels)
                else:
                    # 多数类：只使用真实数据，权重为1
                    real_mask_c = ~class_is_syn
                    if real_mask_c.any():
                        real_output = class_output[real_mask_c]
                        real_labels = class_labels[real_mask_c]
                        loss += F.cross_entropy(real_output, real_labels)
            
            # 添加FedProx近端项
            if global_params is not None:
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    if name in global_params and param.requires_grad:
                        proximal_term += (param - global_params[name]).norm(2) ** 2
                loss = loss + (FEDPROX_MU / 2) * proximal_term
            
            loss.backward()
            
            # 计算梯度范数
            all_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    all_grads.append(param.grad.view(-1))
            if all_grads:
                global_grad_norm = torch.norm(torch.cat(all_grads), p=2).item()
                all_batch_global_grad_norms.append(global_grad_norm)
            
            # 梯度裁剪
            clip_model_gradients(model, clipping_threshold)
            
            optimizer.step()
            total_loss += float(loss.detach().item())
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # 计算梯度范数中位数
    client_global_grad_norm_median = np.median(all_batch_global_grad_norms) if all_batch_global_grad_norms else 0.0
    
    # 返回实际使用的样本统计
    sample_usage_stats = {
        'total_real_used': total_real_used,
        'total_syn_used': total_syn_used,
        'minority_real_used': minority_real_used,
        'minority_syn_used': minority_syn_used,
        'real_used_by_class': real_used_by_class,
        'syn_used_by_class': syn_used_by_class,
        'minority_classes': minority_classes  # 添加少数类列表
    }
    
    return model, avg_loss, client_global_grad_norm_median, sample_usage_stats


# ==================== 数据加载和划分 ====================

def load_cifar100(data_path=None, use_augmentation=True):
    """
    加载CIFAR-100数据集（支持数据增强，方案3）
    
    参数:
        data_path: 数据存储路径
        use_augmentation: 是否使用数据增强（随机裁剪、翻转）
    
    返回:
        train_dataset: 训练数据集 (50000个样本)
        test_dataset: 测试数据集 (10000个样本)
    """
    if data_path is None:
        data_path = DATA_DIR

    # 测试集不使用数据增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 训练集使用数据增强（方案3：随机裁剪、翻转）
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),      # 随机裁剪，填充4像素
            transforms.RandomHorizontalFlip(),          # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    else:
        # 不使用数据增强时，使用与测试集相同的变换
        train_transform = test_transform
    
    # 下载并加载训练集
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 下载并加载测试集
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform
    )
    
    return train_dataset, test_dataset


def partition_data_noniid(train_dataset, num_clients=100, num_classes_per_client=2):
    """
    Non-IID数据划分
    每个客户端只包含指定数量的不同类别
    CIFAR-100: 每个类别500个样本，每个客户端每类25个样本
    """
    targets = np.array(train_dataset.targets)
    
    # 按类别分组数据索引
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    
    num_classes = len(set(targets.tolist()))
    print("\n数据集各类别样本数量:")
    for class_label in range(num_classes):
        print(f"  类别 {class_label}: {len(class_indices[class_label])} 个样本")
    
    # 打乱每个类别中的索引
    for label in class_indices:
        np.random.shuffle(class_indices[label])
    
    # 计算每个类别应该被分配给多少个客户端
    # 100个客户端 × 2个类别 = 200个类别分配
    # 100个类别，平均每个类别被分配2次
    num_assignments_per_class = (num_clients * num_classes_per_client) // max(num_classes, 1)
    
    # 计算每个类别每次分配的样本数
    # CIFAR-100: 500 / 2 = 250
    samples_per_assignment = {}
    print("\n每个类别每次分配的样本数:")
    for class_label in range(num_classes):
        samples_per_assignment[class_label] = len(class_indices[class_label]) // num_assignments_per_class
        print(f"  类别 {class_label}: 每次分配 {samples_per_assignment[class_label]} 个样本")
    
    # 生成所有可能的类别组合 C(100,2) = 4950种
    from itertools import combinations
    all_combinations = list(combinations(range(num_classes), num_classes_per_client))
    
    # 为每个客户端分配不同的类别组合
    client_data_indices = defaultdict(list)
    client_classes = {}
    class_assigned = defaultdict(int)
    
    # 记录每个类别被分配的次数
    class_assignment_count = defaultdict(int)
    
    for client_id in range(num_clients):
        # 选择一个类别组合，优先选择被分配次数较少的类别
        # 计算每个组合的"优先级"（组合中类别被分配次数之和）
        combo_scores = []
        for combo in all_combinations:
            score = sum(class_assignment_count[c] for c in combo)
            combo_scores.append((combo, score))
        
        # 按分数排序，选择分数最低的组合（平衡分配）
        combo_scores.sort(key=lambda x: x[1])
        
        # 从分数最低的几个组合中随机选择一个
        min_score = combo_scores[0][1]
        low_score_combos = [c for c, s in combo_scores if s == min_score]
        selected_combo = list(random.choice(low_score_combos))
        
        client_classes[client_id] = selected_combo
        
        # 更新类别分配计数
        for class_label in selected_combo:
            class_assignment_count[class_label] += 1
        
        # 分配数据
        for class_label in selected_combo:
            num_samples = samples_per_assignment[class_label]
            start_idx = class_assigned[class_label]
            end_idx = start_idx + num_samples
            
            client_data_indices[client_id].extend(
                class_indices[class_label][start_idx:end_idx]
            )
            class_assigned[class_label] = end_idx
    
    # 打印数据分配统计信息
    print("\n数据分配统计:")
    total_samples = 0
    for client_id in range(num_clients):
        num_samples = len(client_data_indices[client_id])
        total_samples += num_samples
        if client_id < 5:
            print(f"客户端 {client_id}: {num_samples} 个样本, 类别: {client_classes[client_id]}")
    print(f"总样本数: {total_samples}")
    
    # 统计类别组合分布
    class_combinations = {}
    for client_id in range(num_clients):
        combo = tuple(sorted(client_classes[client_id]))
        if combo not in class_combinations:
            class_combinations[combo] = []
        class_combinations[combo].append(client_id)
    
    print(f"\n类别组合分布:")
    print(f"  理论组合数: C({num_classes},{num_classes_per_client})")
    print(f"  实际组合数: {len(class_combinations)} 种")
    print(f"  组合详情 (前10种):")
    sorted_combos = sorted(class_combinations.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (combo, clients) in enumerate(sorted_combos[:10]):
        print(f"    组合 {combo}: {len(clients)} 个客户端")
    
    # 验证每个类别被分配的次数
    print(f"\n各类别被分配次数:")
    for class_label in range(num_classes):
        print(f"  类别 {class_label}: {class_assignment_count[class_label]} 次")
    
    return client_data_indices, client_classes


def partition_data_dirichlet(train_dataset, num_clients=100, alpha=0.1, min_samples=1, seed=42):
    """
    使用狄利克雷分布进行Non-IID数据划分
    
    参数:
        train_dataset: 训练数据集
        num_clients: 客户端数量
        alpha: 狄利克雷分布浓度参数（越小越不均衡）
        min_samples: 每个客户端每个类别的最小样本数
        seed: 随机种子（固定后每次运行结果相同，方便调参对比）
    
    返回:
        client_data_indices: 每个客户端的数据索引
        client_classes: 每个客户端拥有的类别列表
    """
    # 设置随机种子，确保每次运行结果相同
    np.random.seed(seed)
    random.seed(seed)
    
    targets = np.array(train_dataset.targets)
    if hasattr(train_dataset, 'classes') and train_dataset.classes is not None:
        num_classes = len(train_dataset.classes)
    else:
        num_classes = int(targets.max()) + 1 if targets.size > 0 else NUM_CLASSES
    
    # 按类别分组数据索引
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    
    # 打乱每个类别中的索引
    for label in class_indices:
        np.random.shuffle(class_indices[label])
    
    # 为每个类别生成狄利克雷分布的分配比例
    # 每个类别生成一个狄利克雷分布向量，表示该类别分配给各客户端的比例
    client_data_indices = defaultdict(list)
    client_classes = {i: [] for i in range(num_clients)}
    
    # 为每个类别生成分配比例
    for class_label in range(num_classes):
        # 从狄利克雷分布采样，得到该类别分配给各客户端的比例
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # 计算该类别分配给每个客户端的样本数
        total_samples = len(class_indices[class_label])
        samples_per_client = (proportions * total_samples).astype(int)
        
        # 确保至少有样本的客户端能分到数据
        # 找到非零分配的客户端，确保他们至少有 min_samples 个样本
        nonzero_mask = samples_per_client > 0
        if nonzero_mask.sum() > 0:
            # 计算需要调整的样本数
            adjustment_needed = min_samples * nonzero_mask.sum() - samples_per_client[nonzero_mask].sum()
            if adjustment_needed > 0:
                # 从零分配的客户端那里借样本（如果有的话）
                zero_mask = samples_per_client == 0
                if zero_mask.sum() > 0:
                    # 不从零分配的客户端借，而是调整比例
                    pass
        
        # 简化处理：直接按比例分配，不强制最小样本数
        # 调整样本数，确保总和等于实际样本数
        diff = total_samples - samples_per_client.sum()
        if diff > 0:
            # 需要增加样本，随机选择客户端增加
            for _ in range(diff):
                client_id = np.random.randint(0, num_clients)
                samples_per_client[client_id] += 1
        elif diff < 0:
            # 需要减少样本，从样本数最多的客户端减少
            while samples_per_client.sum() > total_samples:
                max_idx = samples_per_client.argmax()
                if samples_per_client[max_idx] > 0:
                    samples_per_client[max_idx] -= 1
                else:
                    break
        
        # 分配样本
        start_idx = 0
        for client_id in range(num_clients):
            num_samples = samples_per_client[client_id]
            if num_samples > 0:
                end_idx = min(start_idx + num_samples, len(class_indices[class_label]))
                client_data_indices[client_id].extend(
                    class_indices[class_label][start_idx:end_idx]
                )
                # 记录该客户端拥有该类别
                if class_label not in client_classes[client_id]:
                    client_classes[client_id].append(class_label)
                start_idx = end_idx
    
    # 检查并移除没有数据的客户端
    empty_clients = [i for i in range(num_clients) if len(client_data_indices[i]) == 0]
    if empty_clients:
        print(f"\n警告: 发现 {len(empty_clients)} 个客户端没有数据，正在重新分配...")
        # 从有数据的客户端那里借一些数据给空客户端
        for empty_client in empty_clients:
            # 找一个有足够数据的客户端
            for donor_client in range(num_clients):
                if len(client_data_indices[donor_client]) > 100:
                    # 转移一部分数据
                    transfer_count = min(50, len(client_data_indices[donor_client]) // 2)
                    transfer_indices = client_data_indices[donor_client][:transfer_count]
                    client_data_indices[empty_client].extend(transfer_indices)
                    client_data_indices[donor_client] = client_data_indices[donor_client][transfer_count:]
                    # 更新类别信息
                    for idx in transfer_indices:
                        label = targets[idx]
                        if label not in client_classes[empty_client]:
                            client_classes[empty_client].append(label)
                    break
    
    return client_data_indices, client_classes


# ==================== 差分隐私函数 ====================

def calculate_noise_std(epsilon, delta, sensitivity, q, T):
    """基于cDP到(ε,δ)-DP反解计算噪声标准差"""
    if epsilon <= 0:
        raise ValueError("epsilon 必须大于 0")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta 必须在 (0, 1) 区间")
    if sensitivity <= 0:
        return 0.0
    if q <= 0 or T <= 0:
        return 0.0

    log_term = math.log(1.0 / delta)
    sqrt_rho = math.sqrt(log_term + epsilon) - math.sqrt(log_term)
    rho_total = max(sqrt_rho * sqrt_rho, 1e-18)
    sigma = sensitivity * math.sqrt((q * T) / (2.0 * rho_total))
    return sigma


def calculate_sensitivity(learning_rate, clipping_threshold, num_samples):
    """计算本地训练过程的敏感度 (公式14)"""
    sensitivity = (2 * learning_rate * clipping_threshold) / num_samples
    return sensitivity


def clip_gradient(gradient, clipping_threshold):
    """单个梯度张量裁剪（兼容保留）"""
    grad_norm = torch.norm(gradient, p=2)
    if grad_norm > clipping_threshold:
        clipped_gradient = gradient * (clipping_threshold / grad_norm)
    else:
        clipped_gradient = gradient
    return clipped_gradient


def clip_model_gradients(model, clipping_threshold):
    """全模型统一L2范数裁剪（UDP基础版修正）"""
    grad_tensors = []
    for param in model.parameters():
        if param.grad is not None:
            grad_tensors.append(param.grad.view(-1))

    if not grad_tensors:
        return 0.0

    global_grad_norm = torch.norm(torch.cat(grad_tensors), p=2)
    clip_coef = clipping_threshold / (global_grad_norm + 1e-12)
    clip_coef = min(1.0, float(clip_coef))

    if clip_coef < 1.0:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(clip_coef)

    return float(global_grad_norm.item())


def _stochastic_round_to_int64(x):
    x_floor = torch.floor(x)
    frac = x - x_floor
    r = torch.rand_like(frac)
    return (x_floor + (r < frac).to(x_floor.dtype)).to(torch.int64)


def add_discrete_noise_to_model_update(global_params_cpu, local_params_cpu, sigma, device, num_bits=16):
    qmax = (1 << (num_bits - 1)) - 1
    noisy_params = {}

    for key, local_param in local_params_cpu.items():
        if local_param.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            global_param = global_params_cpu[key]
            if global_param.dtype != local_param.dtype:
                global_param = global_param.to(dtype=local_param.dtype)

            delta = local_param - global_param
            max_abs = float(delta.detach().abs().max().item()) if delta.numel() > 0 else 0.0
            if not np.isfinite(max_abs) or max_abs <= 0.0:
                scale = 1.0
            else:
                scale = max_abs / float(qmax)
                if scale <= 0.0 or not np.isfinite(scale):
                    scale = 1.0

            scale_t = torch.tensor(scale, device=device, dtype=torch.float64)
            delta_fp = delta.to(device=device, dtype=torch.float64)
            delta_scaled = delta_fp / scale_t
            delta_int = _stochastic_round_to_int64(delta_scaled).clamp(min=-qmax, max=qmax)

            noise_std = float(sigma) / float(scale)
            noise_fp = torch.randn(delta_fp.shape, device=device, dtype=torch.float64) * noise_std
            noise_int = torch.round(noise_fp).to(torch.int64)

            delta_noisy_fp = (delta_int + noise_int).to(torch.float64) * scale_t
            delta_noisy = delta_noisy_fp.to(dtype=local_param.dtype)
            noisy_params[key] = global_param + delta_noisy
        else:
            noisy_params[key] = local_param

    return noisy_params


# ==================== 客户端类 ====================

class ClientWithGAN:
    """带GAN数据增强的客户端类（方案3：支持动态掩码和频率加权采样）"""
    
    def __init__(self, client_id, data_indices, local_classes, dataset,
                 generator_init_params, device, local_epochs=6, batch_size=64, 
                 learning_rate=0.005, clipping_threshold=0.32,
                 epsilon=0.5, delta=1e-3, sampling_ratio=0.32, 
                 total_rounds=50, gen_epochs=10, gen_lr=0.0002,
                 num_synthetic_per_class=None, alpha=0.1, beta=0.05,
                 temperature=1.0, generator_cache_dir='generator_cache'):
        """初始化客户端（方案3：添加温度系数参数）"""
        self.client_id = client_id
        self.data_indices = data_indices
        self.local_classes = local_classes
        self.missing_classes = [i for i in range(NUM_CLASSES) if i not in local_classes]
        self.dataset = dataset
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.clipping_threshold = clipping_threshold
        self.epsilon = epsilon
        self.delta = delta
        self.sampling_ratio = sampling_ratio
        self.total_rounds = total_rounds
        self.gen_epochs = gen_epochs
        self.gen_lr = gen_lr
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature  # 方案3：温度系数
        
        # 生成器热启动相关
        self.generator_init_params_cpu = generator_init_params
        self.generator_cache_dir = generator_cache_dir
        self.generator_cache_path = os.path.join(generator_cache_dir, f'client_{client_id}.pth')
        self.generator_trained = False
        
        # 创建本地数据加载器
        self.data_loader = self._create_data_loader()
        
        # 计算敏感度和噪声标准差
        self.num_samples = len(data_indices)
        
        # 动态计算每类生成样本数：0.6 × 客户端总数据量 / NUM_CLASSES
        if num_synthetic_per_class is None:
            self.num_synthetic_per_class = int(0.6 * self.num_samples / NUM_CLASSES)
        else:
            self.num_synthetic_per_class = num_synthetic_per_class
        
        self.sensitivity = calculate_sensitivity(
            learning_rate, clipping_threshold, self.num_samples
        )
        self.noise_sigma = calculate_noise_std(
            epsilon, delta, self.sensitivity, sampling_ratio, total_rounds
        )
        
        # 方案3：统计每个类别的样本数量
        self.class_counts = self._compute_class_counts()
        
        # 方案3：计算动态掩码和采样权重
        self.majority_classes, self.minority_classes = self._compute_dynamic_mask()
        self.sampling_weights = self._compute_sampling_weights()
    
    def _compute_class_counts(self):
        """方案3：统计每个类别的样本数量"""
        class_counts = torch.zeros(NUM_CLASSES)
        # 遍历数据索引，统计每个类别的样本数
        for idx in self.data_indices:
            # 获取该样本的标签
            label = self.dataset.targets[idx]
            class_counts[label] += 1
        return class_counts
    
    def _compute_dynamic_mask(self):
        """
        方案3：计算动态统计掩码
        
        返回:
            majority_classes: 多数类集合（样本充足，无需生成）
            minority_classes: 少数类集合（样本稀缺，需要生成）
        """
        # 计算阈值 tau = N_total / K
        tau = self.num_samples / NUM_CLASSES
        
        # 划分多数类和少数类
        majority_classes = [c for c in range(NUM_CLASSES) if self.class_counts[c] >= tau]
        minority_classes = [c for c in range(NUM_CLASSES) if self.class_counts[c] < tau]
        
        # 漏洞修复：如果少数类为空（所有类别样本都充足），则使用缺失类作为少数类
        # 这种情况在每客户端恰好有2类各250样本时会发生
        if len(minority_classes) == 0:
            minority_classes = self.missing_classes.copy()
            # 多数类为客户端拥有的类别
            majority_classes = list(self.local_classes)
        
        return majority_classes, minority_classes
    
    def _compute_sampling_weights(self):
        """
        方案3：计算频率加权采样权重
        
        公式: P(y=c) = exp(-n_c / T) / sum(exp(-n_k / T))
        样本越少，权重越大（完全缺失的类权重最大）
        """
        weights = torch.zeros(NUM_CLASSES)
        
        # 只对少数类计算权重
        for c in self.minority_classes:
            if self.class_counts[c] > 0:
                # 样本数越多，权重越小
                weights[c] = math.exp(-self.class_counts[c] / self.temperature)
            else:
                # 完全缺失的类权重最大
                weights[c] = math.exp(0)  # = 1.0
        
        # 漏洞修复：如果权重全为0（少数类为空），则均匀分布
        if weights.sum() == 0:
            for c in self.minority_classes:
                weights[c] = 1.0 / len(self.minority_classes) if len(self.minority_classes) > 0 else 0.0
        else:
            # 归一化
            weights = weights / weights.sum()
        
        return weights
    
    def _create_data_loader(self):
        """创建本地数据加载器"""
        subset = Subset(self.dataset, self.data_indices)
        generator = torch.Generator()
        generator.manual_seed(42)
        loader = DataLoader(
            subset, batch_size=self.batch_size, shuffle=True,
            generator=generator, worker_init_fn=worker_init_fn, num_workers=0,
            drop_last=True  # 丢弃最后一个不完整的batch，避免BatchNorm错误
        )
        return loader
    
    def train_phase0(self, global_model_params, model, local_epochs=None):
        """阶段0：标准FedAvg训练（添加FedProx近端项）
        
        参数:
            global_model_params: 全局模型参数
            model: 本地模型
        """
        model.load_state_dict(global_model_params)
        model.train()
        
        # 保存全局模型参数用于FedProx近端项
        global_params = None
        if FEDPROX_MU > 0:
            global_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        # 定义优化器和损失函数
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # 本地训练
        total_loss = 0.0
        total_batches = 0
        
        # 收集所有batch的全局梯度范数（客户端内部）
        all_batch_global_grad_norms = []
        
        effective_local_epochs = self.local_epochs if local_epochs is None else int(local_epochs)
        for epoch in range(effective_local_epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # 添加FedProx近端项：约束本地模型不要偏离全局模型太远
                if FEDPROX_MU > 0 and global_params is not None:
                    proximal_term = 0.0
                    for name, param in model.named_parameters():
                        if name in global_params and param.requires_grad:
                            proximal_term += (param - global_params[name]).norm(2) ** 2
                    loss = loss + (FEDPROX_MU / 2) * proximal_term
                
                loss.backward()
                
                # 计算裁剪前的全局梯度范数（所有参数拼接后的L2范数）
                all_grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        all_grads.append(param.grad.view(-1))
                if all_grads:
                    global_grad_norm = torch.norm(torch.cat(all_grads), p=2).item()
                    all_batch_global_grad_norms.append(global_grad_norm)
                
                # 使用全模型统一范数裁剪，避免逐参数裁剪与UDP假设不一致
                clip_model_gradients(model, self.clipping_threshold)
                
                optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / max(total_batches, 1)
        
        # 计算客户端内部所有batch的全局梯度范数的中位数
        client_global_grad_norm_median = np.median(all_batch_global_grad_norms) if all_batch_global_grad_norms else 0.0
        
        # 添加差分隐私噪声
        params_cpu = _cpu_state_dict(model.state_dict())
        noisy_params = add_discrete_noise_to_model_update(
            global_model_params, params_cpu, self.noise_sigma, device='cpu', num_bits=16
        )
        
        return noisy_params, self.num_samples, avg_loss, client_global_grad_norm_median
    
    def train_phase1(self, global_model_params, test_accuracy_history, model, generator, local_epochs=None):
        """阶段1：GAN增强训练
        
        参数:
            global_model_params: 全局模型参数
            test_accuracy_history: 测试准确率历史列表（用于计算λ权重）
            model: 本地模型
            generator: 生成器模型
        """
        model.load_state_dict(global_model_params)
        
        # 记录训练前的状态（用于正确显示是否热启动）
        was_trained = self.generator_trained
        
        # 步骤1: 训练生成器（热启动）
        # 检查生成器参数是否匹配当前架构
        os.makedirs(self.generator_cache_dir, exist_ok=True)
        loaded_state = None
        loaded_from_cache = False
        if self.generator_trained and os.path.exists(self.generator_cache_path):
            loaded_state = torch.load(self.generator_cache_path, map_location='cpu')
            loaded_from_cache = True
        elif self.generator_init_params_cpu is not None and len(self.generator_init_params_cpu) > 0:
            loaded_state = self.generator_init_params_cpu

        if loaded_state is not None and len(loaded_state) > 0:
            current_keys = set(generator.state_dict().keys())
            saved_keys = set(loaded_state.keys())
            if current_keys == saved_keys:
                device_state = {}
                for k, v in loaded_state.items():
                    t = v.to(self.device)
                    if torch.is_floating_point(t):
                        t = t.float()
                    device_state[k] = t
                generator.load_state_dict(device_state)
            else:
                print(f"  警告: 生成器参数架构不匹配，使用随机初始化")
                print(f"    当前架构参数: {len(current_keys)} 个")
                print(f"    保存的参数: {len(saved_keys)} 个")
                if loaded_from_cache:
                    try:
                        os.remove(self.generator_cache_path)
                    except OSError:
                        pass
                self.generator_trained = False
        
        # 方案3：传入class_counts、num_samples和temperature参数
        generator, gen_loss, cls_loss, div_loss, bn_loss = train_generator(
            generator, model, self.local_classes,
            num_epochs=10, batch_size=64,
            alpha=self.alpha, beta=self.beta, lr=self.gen_lr,
            device=self.device,
            class_counts=self.class_counts,      # 方案3：类别样本数量
            num_samples=self.num_samples,         # 方案3：客户端总样本数
            temperature=self.temperature          # 方案3：温度系数
        )
        torch.save(_cpu_state_dict_fp16(generator.state_dict()), self.generator_cache_path)
        self.generator_trained = True
        
        # 返回生成器损失信息（包含是否热启动的标志）
        gen_losses = (gen_loss, cls_loss, div_loss, bn_loss, was_trained)
        
        # 步骤2: 生成缺失类别数据（方案3：使用频率加权采样）
        synthetic_images, synthetic_labels = generate_synthetic_data(
            generator, self.minority_classes,  # 方案3：使用少数类而非缺失类
            num_samples_per_class=self.num_synthetic_per_class,
            device=self.device,
            sampling_weights=self.sampling_weights,  # 方案3：采样权重
            total_samples=self.num_synthetic_per_class * len(self.minority_classes)  # 方案3：总样本数
        )

        gate_total = int(synthetic_labels.shape[0])
        gate_kept = 0
        gate_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        if gate_total > 0:
            model_was_training = model.training
            model.eval()
            teacher_probs = []
            with torch.no_grad():
                bs = 256
                # 门控机制：只使用教师模型的置信度筛选生成数据
                # 教师模型置信度 >= 0.6 的样本被保留
                for start in range(0, int(synthetic_images.shape[0]), bs):
                    end = min(start + bs, int(synthetic_images.shape[0]))
                    x = synthetic_images[start:end].to(self.device)
                    y = synthetic_labels[start:end].to(self.device)
                    logits = model(x)
                    p_teacher = torch.softmax(logits, dim=1)
                    teacher_probs.append(p_teacher.gather(1, y.view(-1, 1)).squeeze(1).detach().cpu())
            if model_was_training:
                model.train()

            teacher_probs = torch.cat(teacher_probs, dim=0)
            # 只使用教师模型的门控条件：置信度 >= 0.6
            keep_mask = teacher_probs >= GATE_TEACHER_MIN_PROB
            keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
            synthetic_images = synthetic_images.index_select(0, keep_idx) if int(keep_idx.numel()) > 0 else synthetic_images[:0]
            synthetic_labels = synthetic_labels.index_select(0, keep_idx) if int(keep_idx.numel()) > 0 else synthetic_labels[:0]
            gate_kept = int(synthetic_labels.shape[0])
            if gate_kept > 0:
                gate_counts = torch.bincount(synthetic_labels, minlength=NUM_CLASSES).to(torch.long)
        
        # 步骤3: 混合数据训练
        # 获取动态权重（基于近5轮测试准确率的加权平均）
        lambda_val = get_lambda(test_accuracy_history)
        real_weight = 1.0 - lambda_val
        synthetic_weight = lambda_val
        
        effective_local_epochs = self.local_epochs if local_epochs is None else int(local_epochs)
        model, avg_loss, client_grad_norm_median, sample_usage_stats = train_with_mixed_data(
            model, self.data_loader, synthetic_images, synthetic_labels,
            epochs=effective_local_epochs, batch_size=self.batch_size,
            real_weight=real_weight, synthetic_weight=synthetic_weight,
            device=self.device, clipping_threshold=self.clipping_threshold,
            global_model_params=global_model_params,
            minority_classes=self.minority_classes,  # 方案3：少数类列表
            tau=self.num_samples / NUM_CLASSES,  # 方案3：动态阈值
            lambda_val=lambda_val  # 方案3：生成数据权重λ
        )
        
        # 步骤4: 添加差分隐私噪声
        params_cpu = _cpu_state_dict(model.state_dict())
        noisy_params = add_discrete_noise_to_model_update(
            global_model_params, params_cpu, self.noise_sigma, device='cpu', num_bits=16
        )
        
        # 清理不需要的变量，释放内存
        del synthetic_images, synthetic_labels
        
        return noisy_params, self.num_samples, avg_loss, gen_losses, client_grad_norm_median, (gate_total, gate_kept, gate_counts), sample_usage_stats


# ==================== 服务器类 ====================

class ServerWithGAN:
    """带GAN增强的服务器类"""
    
    def __init__(self, model, device, test_dataset, num_clients=100, sample_ratio=0.32):
        """初始化服务器"""
        self.model = model
        self.device = device
        self.test_dataset = test_dataset
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        
        # 创建测试数据加载器
        self.test_loader = DataLoader(
            test_dataset, batch_size=100, shuffle=False
        )
        
        # 初始化全局模型参数
        self.global_model_params = _cpu_state_dict(model.state_dict())
    
    def select_clients(self):
        """随机选择一部分客户端参与训练"""
        num_selected = int(self.num_clients * self.sample_ratio)
        selected_indices = np.random.choice(
            self.num_clients, size=num_selected, replace=False
        )
        return selected_indices
    
    def aggregate_models(self, client_updates, participating_clients=None, secure_server=None):
        """聚合客户端模型更新"""
        if not client_updates:
            return 0.0

        if secure_server is None or participating_clients is None:
            raise ValueError("安全聚合必须提供 secure_server 与 participating_clients")

        masked_updates = []
        for update in client_updates:
            if len(update) >= 4:
                params, num_samples, train_loss, client_id = update[0], update[1], update[2], update[3]
            else:
                params, num_samples, train_loss = update[0], update[1], update[2]
                client_id = -1
            masked_updates.append((params, int(num_samples), int(client_id)))

        aggregated_params = secure_server.aggregate_with_secure_masking(
            masked_updates=masked_updates,
            participating_clients=[int(x) for x in participating_clients],
            device='cpu'
        )
        self.global_model_params = aggregated_params

        avg_loss = sum(update[2] for update in client_updates) / len(client_updates)
        return avg_loss
    
    def evaluate(self):
        """在测试集上评估全局模型"""
        self.model.load_state_dict(self.global_model_params)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        return accuracy, avg_loss


# ==================== GPU选择函数 ====================

def select_gpu():
    """
    让用户通过数字选择使用哪一个GPU
    
    返回:
        device: 选择的设备 (torch.device)
    """
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，将使用CPU")
        return torch.device('cpu')
    
    # 获取可用GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"\n检测到 {num_gpus} 个可用的GPU:")
    
    # 列出所有GPU信息
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        # 获取GPU显存信息
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # 转换为GB
        print(f"  [{i}] {gpu_name} (显存: {total_memory:.1f} GB)")
    
    # 让用户选择GPU
    while True:
        try:
            choice = input(f"\n请输入要使用的GPU编号 (0-{num_gpus-1})，或按Enter使用默认GPU[0]: ").strip()
            
            # 如果用户直接按Enter，使用默认GPU 0
            if choice == '':
                selected_gpu = 0
            else:
                selected_gpu = int(choice)
            
            # 验证选择是否有效
            if 0 <= selected_gpu < num_gpus:
                # 设置CUDA设备
                torch.cuda.set_device(selected_gpu)
                device = torch.device(f'cuda:{selected_gpu}')
                print(f"\n已选择GPU [{selected_gpu}]: {torch.cuda.get_device_name(selected_gpu)}")
                return device
            else:
                print(f"错误: 请输入 0 到 {num_gpus-1} 之间的数字")
        except ValueError:
            print("错误: 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n用户取消，使用默认GPU[0]")
            torch.cuda.set_device(0)
            return torch.device('cuda:0')


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 60)
    print("GAN增强差分隐私联邦学习实验 (CIFAR-100)")
    print("=" * 60)
    
    # 设置随机种子
    SEED = 42
    set_seed(SEED)
    
    # 设置计算设备 - 通过数字选择GPU
    device = select_gpu()
    print(f"使用设备: {device}")
    
    # 实验参数
    NUM_CLIENTS = 100  # 客户端总数
    SAMPLE_RATIO = 0.1  # 采样率，每轮10个客户端参与训练
    PHASE0_LOCAL_EPOCHS = 10
    PHASE1_LOCAL_EPOCHS = 10
    BATCH_SIZE = 64
    PHASE0_LEARNING_RATE = 0.005  # 阶段0学习率
    PHASE1_LEARNING_RATE = 0.005  # 阶段1学习率
    GLOBAL_ROUNDS = 500  # 总轮次
    PHASE0_MAX_ROUNDS = 70  # 阶段0最大轮次（实际根据准确率提前结束）
    
    # 阶段0特有参数
    PHASE0_ACCURACY_THRESHOLD = 59.0  # 阶段0停止阈值：最高准确率超过59%时停止
    
    # 差分隐私参数
    EPSILON = 0.5
    DELTA = 1e-5
    CLIPPING_THRESHOLD = 0.32  # 梯度裁剪阈值
    
    # GAN参数
    GEN_EPOCHS = 10
    GEN_LR = 0.0001  # 生成器学习率
    # NUM_SYNTHETIC_PER_CLASS: 动态计算 = 0.6 × 客户端总数据量 / NUM_CLASSES
    ALPHA = 0.05  # 多样性损失权重 L_div
    BETA = 0.15   # BN正则化权重 L_bn
    TEMPERATURE = TEMPERATURE_T  # 方案3：温度系数，控制采样集中度
    
    # 加载数据集
    train_dataset, test_dataset = load_cifar100(data_path=DATA_DIR)
    
    # 狄利克雷分布Non-IID数据划分
    # alpha=0.1: 极度不均衡的数据分布
    # seed=42: 固定随机种子，确保每次运行结果相同，方便调参对比
    DIRICHLET_ALPHA = 0.1
    DIRICHLET_SEED = 42
    client_data_indices, client_classes = partition_data_dirichlet(
        train_dataset, 
        num_clients=NUM_CLIENTS, 
        alpha=DIRICHLET_ALPHA,
        min_samples=10,
        seed=DIRICHLET_SEED
    )
    
    # 初始化全局模型
    global_model = ResNet18(num_classes=NUM_CLASSES).to(device)
    
    # 创建服务器
    server = ServerWithGAN(
        model=global_model, device=device, test_dataset=test_dataset,
        num_clients=NUM_CLIENTS, sample_ratio=SAMPLE_RATIO
    )
    
    client_model = ResNet18(num_classes=NUM_CLASSES).to(device)
    # 使用当前生成器（约9M参数）
    shared_generator = DeepAlignedResGen9M(noise_dim=128, num_classes=NUM_CLASSES).to(device)

    # 创建所有客户端
    clients = []
    generator_init_params_cpu = _cpu_state_dict(shared_generator.state_dict())
    for client_id in range(NUM_CLIENTS):
        client = ClientWithGAN(
            client_id=client_id,
            data_indices=client_data_indices[client_id],
            local_classes=client_classes[client_id],
            dataset=train_dataset,
            generator_init_params=generator_init_params_cpu,
            device=device,
            local_epochs=PHASE1_LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=PHASE0_LEARNING_RATE,  # 初始使用阶段0学习率
            clipping_threshold=CLIPPING_THRESHOLD,
            epsilon=EPSILON,
            delta=DELTA,
            sampling_ratio=SAMPLE_RATIO,
            total_rounds=GLOBAL_ROUNDS,
            gen_epochs=GEN_EPOCHS,
            gen_lr=GEN_LR,
            # num_synthetic_per_class: 使用默认值，动态计算 = 0.6 × 客户端数据量 / NUM_CLASSES
            alpha=ALPHA,
            beta=BETA,
            temperature=TEMPERATURE  # 方案3：温度系数
        )
        clients.append(client)

    secure_aggregator = SecureAggregator(num_clients=NUM_CLIENTS, seed=SEED)
    secure_server = SecureAggregationServer(secure_aggregator)
    secure_clients = []
    for client_id in range(NUM_CLIENTS):
        keys = secure_aggregator.generate_client_keys(client_id)
        secure_clients.append(SecureAggregationClient(client_id, keys, NUM_CLIENTS))

    # 阶段1本地训练轮次固定为10
    def get_phase1_local_epochs(round_idx):
        return 10  # 固定为10

    history = {
        'round': [], 'train_loss': [], 'test_accuracy': [], 'test_loss': []
    }
    best_accuracy = 0.0
    best_accuracy_round = 0
    
    # 阶段0: 标准FedAvg (第1-70轮)
    phase0_start = 1
    if phase0_start <= PHASE0_MAX_ROUNDS:
        print("\n" + "=" * 60)
        print("阶段0: 联邦学习训练 (FedAvg)")
        print("=" * 60)
        print(f"  学习率: {PHASE0_LEARNING_RATE}")
        print(f"  停止条件: 最高准确率 > {PHASE0_ACCURACY_THRESHOLD}%")
        print(f"  使用分布式差分隐私")
        
        phase0_completed_rounds = 0
        
        for round_idx in range(phase0_start, PHASE0_MAX_ROUNDS + 1):
                print(f"\n全局轮次 {round_idx}/{GLOBAL_ROUNDS} (阶段0)")
                selected_client_indices = server.select_clients()
                participating_clients = [int(x) for x in selected_client_indices]

                for client_id in participating_clients:
                    other_ids = [x for x in participating_clients if x != client_id]
                    seeds = secure_aggregator.get_all_pairwise_seeds_for_client(client_id, other_ids)
                    secure_clients[client_id].set_pairwise_seeds(seeds)

                client_updates = []
                client_grad_norm_medians = []
                num_selected = len(selected_client_indices)
                for i, idx in enumerate(selected_client_indices):
                    if i % 8 == 0:
                        print(f"  训练进度: {i+1}/{num_selected} 客户端")

                    noisy_params, num_samples, train_loss, client_global_grad_norm_median = clients[idx].train_phase0(
                        server.global_model_params, client_model, PHASE0_LOCAL_EPOCHS
                    )

                    sa_client = secure_clients[int(idx)]
                    pairwise_mask, self_mask = sa_client.generate_mask(
                        noisy_params, participating_clients, device='cpu'
                    )
                    masked_params = sa_client.apply_mask(noisy_params, pairwise_mask, self_mask)
                    secure_server.collect_self_mask_seed(int(idx), sa_client.get_self_mask_seed())
                    client_updates.append((masked_params, num_samples, train_loss, int(idx)))

                    client_grad_norm_medians.append(client_global_grad_norm_median)

                print(f"  训练进度: {num_selected}/{num_selected} 客户端 - 完成")
                avg_train_loss = server.aggregate_models(
                    client_updates,
                    participating_clients=participating_clients,
                    secure_server=secure_server
                )
                round_global_grad_norm_median = np.median(client_grad_norm_medians) if client_grad_norm_medians else 0.0
                del client_updates
                if PERF_DEBUG:
                    gc.collect()
                
                # 评估模型
                test_accuracy, test_loss = server.evaluate()
                
                # 更新最高准确率
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_accuracy_round = round_idx
                
                # 保存训练历史
                history['round'].append(round_idx)
                history['train_loss'].append(avg_train_loss)
                history['test_accuracy'].append(test_accuracy)
                history['test_loss'].append(test_loss)
                
                pass
                
                print(f"轮次 {round_idx} 结果:")
                print(f"  平均训练损失: {avg_train_loss:.4f}")
                print(f"  测试准确率: {test_accuracy:.2f}%")
                print(f"  最高准确率: {best_accuracy:.2f}%")
                print(f"  测试损失: {test_loss:.4f}")
                print(f"  全局梯度范数中位数: {round_global_grad_norm_median:.4f}")
                
                # 检查是否达到停止条件（最高准确率超过59%）
                if best_accuracy > PHASE0_ACCURACY_THRESHOLD:
                    print(f"\n*** 最高准确率已超过{PHASE0_ACCURACY_THRESHOLD}%，停止阶段0 ***")
                    phase0_completed_rounds = round_idx
                    break
                
                phase0_completed_rounds = round_idx
                
                print(f"\n  生成器预热:")
                print(f"    本轮预热客户端数: {num_selected}")
                print(f"    生成器参数: lr={GEN_LR}, epochs={GEN_EPOCHS}, batch_size=64")
                
                # 加载全局模型参数用于生成器训练
                global_model.load_state_dict(server.global_model_params)
                
                warmup_completed = 0
                for client_idx in selected_client_indices:
                    client = clients[client_idx]
                    # 使用少数类（minority_classes）训练生成器
                    if len(client.minority_classes) > 0:
                        # 每个客户端训练自己的生成器
                        client_generator = DeepAlignedResGen9M(noise_dim=128, num_classes=NUM_CLASSES).to(device)
                        # 加载客户端缓存的生成器参数（热启动）
                        if client.generator_trained and os.path.exists(client.generator_cache_path):
                            try:
                                saved_gen_params = torch.load(client.generator_cache_path, map_location='cpu')
                                client_generator.load_state_dict({k: v.to(device) for k, v in saved_gen_params.items()})
                            except Exception:
                                pass
                        elif client.generator_init_params_cpu is not None:
                            client_generator.load_state_dict(client.generator_init_params_cpu)
                        
                        # 训练生成器（基于少数类）
                        client_generator, gen_loss, cls_loss, div_loss, bn_loss = train_generator(
                            client_generator, global_model, client.minority_classes,
                            num_epochs=GEN_EPOCHS, batch_size=64,
                            alpha=ALPHA, beta=BETA, lr=GEN_LR,
                            device=device
                        )
                        
                        # 保存客户端生成器参数
                        os.makedirs(client.generator_cache_dir, exist_ok=True)
                        torch.save(_cpu_state_dict_fp16(client_generator.state_dict()), client.generator_cache_path)
                        client.generator_trained = True
                        
                        # 清理显存
                        del client_generator, gen_loss, cls_loss, div_loss, bn_loss
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        warmup_completed += 1
                
                print(f"    已完成 {warmup_completed} 个客户端的生成器预热")
        
        actual_phase0_rounds = phase0_completed_rounds if phase0_completed_rounds > 0 else PHASE0_MAX_ROUNDS
        print(f"阶段0实际完成轮次: {actual_phase0_rounds}")
    
    # 阶段1: GAN增强FedAvg (分布式差分隐私)
    # 计算阶段1的起始轮次（使用实际完成的阶段0轮次）
    # 如果阶段0提前结束，阶段1从下一轮开始
    if 'actual_phase0_rounds' in dir():
        phase1_start = actual_phase0_rounds + 1
    else:
        phase1_start = PHASE0_MAX_ROUNDS + 1
    
    # 生成器启用阈值：最高准确率超过50%后才启用生成器数据
    GENERATOR_ENABLE_THRESHOLD = 50.0  # 准确率阈值
    
    if phase1_start <= GLOBAL_ROUNDS:
        print("\n" + "=" * 60)
        print(f"阶段1: GAN增强FedAvg训练 (分布式差分隐私)")
        print("=" * 60)
        print(f"生成器启用阈值: 最高准确率超过 {GENERATOR_ENABLE_THRESHOLD}% 后启用生成数据")
        
        # 更新所有客户端的学习率为阶段1学习率
        # 注意：阶段0和阶段1学习率都是0.005，这里保持一致
        print(f"\n阶段1客户端学习率: {PHASE1_LEARNING_RATE}")
        for client in clients:
            client.learning_rate = PHASE1_LEARNING_RATE
            # 重新计算敏感度和噪声（因为学习率改变了）
            client.sensitivity = calculate_sensitivity(
                PHASE1_LEARNING_RATE, client.clipping_threshold, client.num_samples
            )
            client.noise_sigma = calculate_noise_std(
                client.epsilon, client.delta, client.sensitivity, 
                client.sampling_ratio, client.total_rounds
            )
        print(f"所有客户端学习率已更新为 {PHASE1_LEARNING_RATE}")
        
        for round_idx in range(phase1_start, GLOBAL_ROUNDS + 1):
            print(f"\n全局轮次 {round_idx}/{GLOBAL_ROUNDS}")
            
            # 检查是否启用生成器数据（最高准确率超过阈值）
            current_best_accuracy = max(history['test_accuracy']) if history['test_accuracy'] else 0.0
            generator_enabled = current_best_accuracy > GENERATOR_ENABLE_THRESHOLD
            
            if generator_enabled:
                print(f"  [生成器已启用] 最高准确率: {current_best_accuracy:.2f}% > {GENERATOR_ENABLE_THRESHOLD}%")
            else:
                print(f"  [生成器未启用] 最高准确率: {current_best_accuracy:.2f}% <= {GENERATOR_ENABLE_THRESHOLD}% (仅预热)")
            
            # 计算基于近5轮测试准确率的加权平均λ
            lambda_current = get_lambda(history['test_accuracy'])
            
            # 如果生成器未启用，强制λ=0（不使用生成数据）
            if not generator_enabled:
                lambda_current = 0.0
            
            # 计算并显示加权平均准确率
            recent_accs = history['test_accuracy'][-5:] if history['test_accuracy'] else []
            if recent_accs:
                weights = list(range(1, len(recent_accs) + 1))
                weighted_avg_acc = sum(w * a for w, a in zip(weights, recent_accs)) / sum(weights)
                print(f"  近{len(recent_accs)}轮加权平均准确率: {weighted_avg_acc:.2f}%")
            print(f"  本轮混合权重 λ: {lambda_current:.4f} (真实: {1.0-lambda_current:.4f}, 生成: {lambda_current:.4f})")
            current_phase1_local_epochs = get_phase1_local_epochs(round_idx)
            print(f"  阶段1本轮local_epochs: {current_phase1_local_epochs}")
            
            selected_client_indices = server.select_clients()
            participating_clients = [int(x) for x in selected_client_indices]

            for client_id in participating_clients:
                other_ids = [x for x in participating_clients if x != client_id]
                seeds = secure_aggregator.get_all_pairwise_seeds_for_client(client_id, other_ids)
                secure_clients[client_id].set_pairwise_seeds(seeds)

            client_updates = []
            client_grad_norm_medians = []  # 收集各客户端的梯度范数中位数
            round_gate_total = 0
            round_gate_kept = 0
            round_gate_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
            
            # 统计本轮客户端的样本使用情况
            round_total_real_samples = 0  # 所有客户端真实样本总数
            round_total_syn_samples = 0   # 所有客户端生成样本总数
            round_minority_real_samples = 0  # 所有客户端少数类真实样本总数
            round_minority_syn_samples = 0   # 所有客户端少数类生成样本总数
            
            # 按类别统计生成样本使用情况
            round_syn_used_by_class = {c: 0 for c in range(NUM_CLASSES)}
            round_real_used_by_class = {c: 0 for c in range(NUM_CLASSES)}
            
            round_class_as_minority_count = {c: 0 for c in range(NUM_CLASSES)}
            
            # 客户端训练进度
            num_selected = len(selected_client_indices)
            for i, idx in enumerate(selected_client_indices):
                # 每8个客户端输出一条进度信息
                if i % 8 == 0:
                    print(f"  训练进度: {i+1}/{num_selected} 客户端")
                
                client = clients[idx]  # 获取当前客户端对象
                
                noisy_params, num_samples, train_loss, gen_losses, client_global_grad_norm_median, gate_info, sample_usage_stats = clients[idx].train_phase1(
                    server.global_model_params, history['test_accuracy'], client_model, shared_generator, current_phase1_local_epochs
                )

                sa_client = secure_clients[int(idx)]
                pairwise_mask, self_mask = sa_client.generate_mask(
                    noisy_params, participating_clients, device='cpu'
                )
                masked_params = sa_client.apply_mask(noisy_params, pairwise_mask, self_mask)
                secure_server.collect_self_mask_seed(int(idx), sa_client.get_self_mask_seed())
                client_updates.append((masked_params, num_samples, train_loss, int(idx)))

                client_grad_norm_medians.append(client_global_grad_norm_median)
                gate_total, gate_kept, gate_counts = gate_info
                round_gate_total += int(gate_total)
                round_gate_kept += int(gate_kept)
                round_gate_counts += gate_counts
                
                # 统计实际使用的样本数量
                round_total_real_samples += sample_usage_stats['total_real_used']
                round_total_syn_samples += sample_usage_stats['total_syn_used']
                round_minority_real_samples += sample_usage_stats['minority_real_used']
                round_minority_syn_samples += sample_usage_stats['minority_syn_used']
                
                # 按类别统计
                for c in range(NUM_CLASSES):
                    round_syn_used_by_class[c] += sample_usage_stats['syn_used_by_class'].get(c, 0)
                    round_real_used_by_class[c] += sample_usage_stats['real_used_by_class'].get(c, 0)
                
                # 统计少数类
                client_minority_classes = sample_usage_stats.get('minority_classes', [])
                if client_minority_classes is not None:
                    for c in client_minority_classes:
                        round_class_as_minority_count[c] += 1
                
                # 解包生成器损失信息
                gen_loss, cls_loss, div_loss, bn_loss, is_warm_start = gen_losses
                
                # 只输出第一个客户端的生成器训练情况
            # 计算并输出实际使用的样本统计
            avg_real_samples = round_total_real_samples / num_selected if num_selected > 0 else 0
            avg_syn_samples = round_total_syn_samples / num_selected if num_selected > 0 else 0
            avg_minority_real = round_minority_real_samples / num_selected if num_selected > 0 else 0
            avg_minority_syn = round_minority_syn_samples / num_selected if num_selected > 0 else 0
            
            # 计算按类别的平均值
            avg_real_by_class = {c: round_real_used_by_class[c] / num_selected if num_selected > 0 else 0 for c in range(NUM_CLASSES)}
            avg_syn_by_class = {c: round_syn_used_by_class[c] / num_selected if num_selected > 0 else 0 for c in range(NUM_CLASSES)}
            
            # 计算少数类的正确统计（只统计作为少数类的类别的生成样本）
            # 每个类别作为少数类时的平均生成样本使用量
            avg_minority_syn_by_class = {}
            for c in range(NUM_CLASSES):
                if round_class_as_minority_count[c] > 0:
                    avg_minority_syn_by_class[c] = round_syn_used_by_class[c] / round_class_as_minority_count[c]
                else:
                    avg_minority_syn_by_class[c] = 0
            
            minority_classes_list = [c for c in range(NUM_CLASSES) if round_class_as_minority_count[c] > 0]
            total_minority_syn = sum(avg_minority_syn_by_class[c] for c in minority_classes_list)
            
            # 统计本轮客户端的平均类别分布（真实数据）
            round_class_counts_real = np.zeros(NUM_CLASSES)      # 真实数据各类别数量
            round_client_count = 0
            
            for idx in selected_client_indices:
                client = clients[idx]
                round_client_count += 1
                # 真实数据各类别数量
                for c in range(NUM_CLASSES):
                    round_class_counts_real[c] += client.class_counts[c].item()
            
            # 计算平均值
            avg_class_counts_real = round_class_counts_real / round_client_count if round_client_count > 0 else round_class_counts_real
            # 生成数据平均值：round_gate_counts已经是所有客户端的总和，直接除以客户端数
            avg_class_counts_syn = np.array([int(round_gate_counts[c].item()) for c in range(NUM_CLASSES)]) / round_client_count if round_client_count > 0 else np.zeros(NUM_CLASSES)
            avg_class_counts_total = avg_class_counts_real + avg_class_counts_syn
            
            nonzero = [(i, int(round_gate_counts[i].item())) for i in range(NUM_CLASSES) if int(round_gate_counts[i].item()) > 0]
            counts_str = ", ".join([f"{i}:{c}" for i, c in nonzero[:10]]) + "..." if len(nonzero) > 10 else ", ".join([f"{i}:{c}" for i, c in nonzero]) if nonzero else "无"
            avg_train_loss = server.aggregate_models(
                client_updates,
                participating_clients=participating_clients,
                secure_server=secure_server
            )
            
            # 计算客户端间的全局梯度范数中位数
            round_global_grad_norm_median = np.median(client_grad_norm_medians) if client_grad_norm_medians else 0.0
            
            # 清理客户端更新数据，释放内存
            del client_updates
            if PERF_DEBUG:
                gc.collect()
            
            test_accuracy, test_loss = server.evaluate()
            
            # 更新最高准确率
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_accuracy_round = round_idx
            
            history['round'].append(round_idx)
            history['train_loss'].append(avg_train_loss)
            history['test_accuracy'].append(test_accuracy)
            history['test_loss'].append(test_loss)
            
            pass
            
            print(f"轮次 {round_idx} 结果:")
            print(f"  平均训练损失: {avg_train_loss:.4f}")
            print(f"  测试准确率: {test_accuracy:.2f}%")
            print(f"  最高准确率: {best_accuracy:.2f}%")
            print(f"  测试损失: {test_loss:.4f}")
            print(f"  全局梯度范数中位数: {round_global_grad_norm_median:.4f}")
            
    
    # 训练完成
    print("\n" + "=" * 60)
    print("GAN增强差分隐私联邦学习训练完成!")
    print("=" * 60)
    
    print(f"\n最终测试准确率: {history['test_accuracy'][-1]:.2f}%")
    print(f"最终测试损失: {history['test_loss'][-1]:.4f}")
    print(f"最高测试准确率: {best_accuracy:.2f}% (第 {best_accuracy_round} 轮)")
    print(f"隐私保证: ({EPSILON}, {DELTA})-DP")
    
    pass
    
    # 保存最终模型
    print("保存最终模型...")
    torch.save(server.global_model_params, 'gan_dp_global_model.pth')
    
    print("\n实验完成!")
    
    pass


if __name__ == '__main__':
    main()

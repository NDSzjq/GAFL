"""
GANNOTE - NOTE
NOTE,NOTE:
- NOTE0(NOTE1-10NOTE):NOTEFedAvgNOTE
- NOTE1(NOTE11-300NOTE):GANNOTE
NOTE: MNIST (28x28x1 NOTE)
NOTE: LeNet5
NOTE: LeGen28
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

from secure_aggregation import SecureAggregator, SecureAggregationClient, SecureAggregationServer

PERF_DEBUG = os.environ.get("PERF_DEBUG", "0") == "1"
GATE_TEACHER_MIN_PROB = float(os.environ.get("GATE_TEACHER_MIN_PROB", "0.6"))
FEDPROX_MU = 0.0  # FedProxNOTE(NOTE0NOTEFedProx)
TEMPERATURE_T = 1.0  # NOTE,NOTE(NOTE3)


class Logger:
    """
    NOTE:NOTE
    NOTEprintNOTEtxtNOTE
    """
    def __init__(self, log_file='training_log.txt'):
        """
        NOTE
        
        NOTE:
            log_file: NOTE
        """
        self.log_file = log_file
        self.terminal = sys.stdout
        # NOTE,NOTE
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("GANNOTE - NOTE\n")
            f.write("=" * 60 + "\n")
            f.write(f"NOTE: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
    
    def write(self, message):
        """
        NOTE
        """
        # NOTE
        self.terminal.write(message)
        # NOTE
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        """
        NOTE
        """
        self.terminal.flush()
    
    def close(self):
        """
        NOTE,NOTE
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"NOTE: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
        # NOTE
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
    """NOTE,NOTE"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def worker_init_fn(worker_id):
    """DataLoader workerNOTE"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==================== NOTE ====================

class Generator(nn.Module):
    """
    NOTE
    NOTE: NOTEz (100NOTE) + NOTEy (one-hot, 10NOTE)
    NOTE: NOTE (32x32x3 RGBNOTE)
    """
    def __init__(self, noise_dim=100, num_classes=10, output_dim=3072):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # NOTE
        # NOTE: NOTE(100) + NOTE(10) = 110NOTE
        self.fc1 = nn.Linear(noise_dim + num_classes, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, output_dim)  # 32*32*3 = 3072
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(2048)
        
        self.register_buffer('cifar_mean', torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('cifar_std', torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(1, 3, 1, 1))
    
    def forward(self, z, y):
        """
        NOTE
        NOTE:
            z: NOTE (batch_size, noise_dim)
            y: NOTE one-hot (batch_size, num_classes)
        NOTE:
            x: NOTE (batch_size, 3, 32, 32)
        """
        # NOTE
        x = torch.cat([z, y], dim=1)
        
        # NOTE
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        # NOTE
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        # NOTE
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        
        # NOTE
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
        
        # NOTE
        x = self.fc5(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 3, 32, 32)
        x = (x - self.cifar_mean) / self.cifar_std
        return x


class ResBlockGenerator(nn.Module):
    """
    NOTE
    NOTEDeep-Aligned ResGen-9M
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
    Deep-Aligned ResGen-9M NOTE
    NOTEResNet18NOTE
    NOTE: ~9.0M
    
    NOTE:
    - Projector: z(128) + y(128) → 512×4×4
    - Group 1: NOTEResNet18 Stage4NOTE (512→256, 4×4→8×8)
    - Group 2: NOTEResNet18 Stage3NOTE (256→128, 8×8→16×16)
    - Group 3: NOTEResNet18 Stage2/1NOTE (128→64, 16×16→32×32)
    - Head: 64→3, 32×32
    """
    def __init__(self, noise_dim=128, num_classes=10):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # NOTE (128NOTE)
        self.label_embed = nn.Embedding(num_classes, 128)
        
        # Projector: z(128) + y(128) → 512×4×4
        self.projector = nn.Sequential(
            nn.Linear(noise_dim + 128, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU()
        )
        
        # Group 1: NOTEResNet18 Stage4NOTE (512→256, 4×4→8×8)
        self.group1 = nn.Sequential(
            ResBlockGenerator(512, 256, upsample=True),
            ResBlockGenerator(256, 256, upsample=False)
        )
        
        # Group 2: NOTEResNet18 Stage3NOTE (256→128, 8×8→16×16)
        self.group2 = nn.Sequential(
            ResBlockGenerator(256, 128, upsample=True),
            ResBlockGenerator(128, 128, upsample=False)
        )
        
        # Group 3: NOTEResNet18 Stage2/1NOTE (128→64, 16×16→32×32)
        self.group3 = nn.Sequential(
            ResBlockGenerator(128, 64, upsample=True),
            ResBlockGenerator(64, 64, upsample=False)
        )
        
        # Head: 64→3, 32×32
        self.head = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # CIFAR-10NOTE
        self.register_buffer('cifar_mean', torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('cifar_std', torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(1, 3, 1, 1))
    
    def forward(self, z, y):
        """
        NOTE
        
        NOTE:
            z: NOTE (batch_size, noise_dim)
            y: NOTE,NOTEone-hotNOTE
        
        NOTE:
            NOTE (batch_size, 3, 32, 32)
        """
        # yNOTEone-hotNOTE
        if y.dim() == 2:  # one-hot
            y_idx = y.argmax(dim=1)
        else:
            y_idx = y
        
        # NOTE
        y_embed = self.label_embed(y_idx)
        
        # NOTE
        x = torch.cat([z, y_embed], dim=1)
        
        # Projector
        x = self.projector(x)
        x = x.view(-1, 512, 4, 4)
        
        # Groups (NOTE)
        x = self.group1(x)  # 4×4 → 8×8
        x = self.group2(x)  # 8×8 → 16×16
        x = self.group3(x)  # 16×16 → 32×32
        
        # Head
        x = self.head(x)  # TanhNOTE [-1, 1]
        
        # NOTETanhNOTE[-1,1]NOTE
        x = (x + 1) / 2  # [-1,1] → [0,1]
        x = (x - self.cifar_mean) / self.cifar_std  # NOTECIFAR-10NOTE
        
        return x


class ResGen32(nn.Module):
    def __init__(self, noise_dim=128, num_classes=10, base_channels=32):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.base_channels = base_channels

        self.label_embed = nn.Embedding(num_classes, 128)

        ch3 = base_channels * 8
        ch2 = base_channels * 4
        ch1 = base_channels * 2
        ch0 = base_channels

        self.projector = nn.Sequential(
            nn.Linear(noise_dim + 128, ch3 * 4 * 4, bias=False),
            nn.BatchNorm1d(ch3 * 4 * 4),
            nn.ReLU(),
        )

        self.block1 = nn.Sequential(
            ResBlockGenerator(ch3, ch2, upsample=True),
            ResBlockGenerator(ch2, ch2, upsample=False),
        )
        self.block2 = nn.Sequential(
            ResBlockGenerator(ch2, ch1, upsample=True),
            ResBlockGenerator(ch1, ch1, upsample=False),
        )
        self.block3 = nn.Sequential(
            ResBlockGenerator(ch1, ch0, upsample=True),
            ResBlockGenerator(ch0, ch0, upsample=False),
        )

        self.head = nn.Sequential(
            nn.Conv2d(ch0, 3, 3, padding=1),
            nn.Tanh(),
        )

        self.register_buffer('cifar_mean', torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('cifar_std', torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, z, y):
        if y.dim() == 2:
            y_idx = y.argmax(dim=1)
        else:
            y_idx = y

        y_embed = self.label_embed(y_idx)
        x = torch.cat([z, y_embed], dim=1)
        x = self.projector(x)
        x = x.view(-1, self.base_channels * 8, 4, 4)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)

        x = (x + 1) / 2
        x = (x - self.cifar_mean) / self.cifar_std
        return x


class LeGen28(nn.Module):
    """
    LeGen28: NOTEMNIST 28x28NOTE
    NOTELeNet-5NOTE
    
    NOTE:
    - NOTE: NOTEz(100NOTE) + NOTEy(10NOTEone-hot)
    - NOTE: 1x28x28 NOTE
    - NOTE: NOTE + NOTE
    """
    def __init__(self, noise_dim=100, num_classes=10):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # NOTE
        self.label_embed = nn.Embedding(num_classes, noise_dim)
        
        # NOTE: NOTE7x7NOTE
        # NOTE: noise_dim(100) + noise_dim(100) = 200
        # NOTE: 256 * 7 * 7 = 12544
        self.projector = nn.Sequential(
            nn.Linear(noise_dim * 2, 256 * 7 * 7, bias=False),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(),
        )
        
        # NOTE1: 7x7 -> 14x14
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # NOTE2: 14x14 -> 28x28
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # NOTE: NOTE
        self.head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
        # MNISTNOTE (NOTE=0.1307, NOTE=0.3081)
        self.register_buffer('mnist_mean', torch.tensor([0.1307], dtype=torch.float32).view(1, 1, 1, 1))
        self.register_buffer('mnist_std', torch.tensor([0.3081], dtype=torch.float32).view(1, 1, 1, 1))
    
    def forward(self, z, y):
        """
        NOTE
        
        NOTE:
            z: NOTE (batch_size, noise_dim)
            y: NOTE,NOTEone-hotNOTE
        
        NOTE:
            NOTE (batch_size, 1, 28, 28)
        """
        # NOTE
        if y.dim() == 2:  # one-hotNOTE
            y_idx = y.argmax(dim=1)
        else:
            y_idx = y
        
        # NOTE
        y_embed = self.label_embed(y_idx)
        
        # NOTE
        x = torch.cat([z, y_embed], dim=1)
        
        # NOTE
        x = self.projector(x)
        x = x.view(-1, 256, 7, 7)
        
        # NOTE
        x = self.deconv1(x)  # 7x7 -> 14x14
        x = self.deconv2(x)  # 14x14 -> 28x28
        
        # NOTE
        x = self.head(x)  # TanhNOTE [-1, 1]
        
        # NOTETanhNOTE[-1,1]NOTE
        x = (x + 1) / 2  # [-1,1] → [0,1]
        x = (x - self.mnist_mean) / self.mnist_std  # NOTEMNISTNOTE
        
        return x


class BasicBlock(nn.Module):
    """
    ResNetNOTE
    NOTEResNet20NOTE
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        """NOTEBasicBlock"""
        super(BasicBlock, self).__init__()
        # NOTE: 3x3NOTE
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # NOTE: 3x3NOTE
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # NOTE(shortcut)
        self.shortcut = nn.Sequential()
        # NOTEstride!=1NOTE,NOTE
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """NOTE: F(x) + x"""
        # NOTE
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # NOTE
        out = self.conv2(out)
        out = self.bn2(out)
        # NOTE
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """
    ResNet20 for CIFAR-10
    NOTECIFAR-10NOTE
    NOTE: NOTE0.27M
    NOTE: 1NOTE + 9NOTEBasicBlock(18NOTE) + 1NOTEFC = 20NOTE
    """
    def __init__(self, num_classes=10):
        """NOTEResNet20"""
        super(ResNet20, self).__init__()
        self.in_channels = 16  # NOTE
        
        # NOTE: 3x3NOTE, 16NOTE
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # NOTEstage,NOTEstage 3NOTEBasicBlock
        # Stage1: 16NOTE, stride=1 (NOTE)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        # Stage2: 32NOTE, stride=2 (NOTE)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        # Stage3: 64NOTE, stride=2 (NOTE)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        
        # NOTE
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """NOTEstage,NOTEBasicBlock"""
        strides = [stride] + [1] * (num_blocks - 1)  # NOTEblockNOTEstride,NOTE1
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """NOTE"""
        # NOTE
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # NOTEstage
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # NOTE
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        # NOTE
        out = self.fc(out)
        return out
    
    def get_bn_stats(self, x):
        """
        NOTEBNNOTE(NOTE)
        NOTEBNNOTE
        ResNet20NOTE19NOTEBNNOTE: 1NOTE + 9NOTEBasicBlock × 2NOTE
        """
        bn_stats = []
        
        # NOTEBN
        out = self.conv1(x)
        out = self.bn1(out)
        bn_stats.append((self.bn1.running_mean, self.bn1.running_var))
        out = F.relu(out)
        
        # NOTElayerNOTEBNNOTE
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                # NOTEBNNOTE
                out = block.conv1(out)
                out = block.bn1(out)
                bn_stats.append((block.bn1.running_mean, block.bn1.running_var))
                out = F.relu(out)
                # NOTEBNNOTE
                out = block.conv2(out)
                out = block.bn2(out)
                bn_stats.append((block.bn2.running_mean, block.bn2.running_var))
                out = F.relu(out)
        
        return bn_stats


class ResNet18(nn.Module):
    """
    ResNet18 for CIFAR-10
    NOTEResNet18NOTE,NOTECIFAR-10NOTE
    NOTE: NOTE11M
    NOTE: 1NOTE + 8NOTEBasicBlock(16NOTE) + 1NOTEFC = 18NOTE
    """
    def __init__(self, num_classes=10):
        """NOTEResNet18"""
        super(ResNet18, self).__init__()
        self.in_channels = 64  # NOTE
        
        # NOTE: 3x3NOTE, 64NOTE (NOTECIFAR-10NOTE,NOTEkernel)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # NOTEstage,NOTEstage 2NOTEBasicBlock
        # Stage1: 64NOTE, stride=1 (NOTE)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        # Stage2: 128NOTE, stride=2 (NOTE)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        # Stage3: 256NOTE, stride=2 (NOTE)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        # Stage4: 512NOTE, stride=2 (NOTE)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # NOTE
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """NOTEstage,NOTEBasicBlock"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """NOTE"""
        # NOTE
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # NOTEstage
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # NOTE
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        # NOTE
        out = self.fc(out)
        return out
    
    def get_bn_stats(self, x):
        """
        NOTEBNNOTE(NOTE)
        NOTEBNNOTE
        ResNet18NOTE17NOTEBNNOTE: 1NOTE + 8NOTEBasicBlock × 2NOTE
        """
        bn_stats = []
        
        # NOTEBN
        out = self.conv1(x)
        out = self.bn1(out)
        bn_stats.append((self.bn1.running_mean, self.bn1.running_var))
        out = F.relu(out)
        
        # NOTElayerNOTEBNNOTE
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                # NOTEBNNOTE
                out = block.conv1(out)
                out = block.bn1(out)
                bn_stats.append((block.bn1.running_mean, block.bn1.running_var))
                out = F.relu(out)
                # NOTEBNNOTE
                out = block.conv2(out)
                out = block.bn2(out)
                bn_stats.append((block.bn2.running_mean, block.bn2.running_var))
                out = F.relu(out)
        
        return bn_stats


class LeNet5(nn.Module):
    """
    LeNet-5 for MNIST
    NOTELeNet-5NOTE,NOTEMNIST 28x28NOTE
    NOTE: NOTE60K
    NOTE: 2NOTE + 3NOTE
    NOTE: 1x28x28 NOTE
    NOTE: 10NOTE
    """
    def __init__(self, num_classes=10):
        """NOTELeNet-5"""
        super(LeNet5, self).__init__()
        # NOTE: NOTE1NOTE, NOTE6NOTE, 5x5NOTE
        # NOTE: 28x28 -> NOTE: 24x24 (NOTEpadding)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        # NOTE: 2x2NOTE
        # NOTE: 24x24 -> NOTE: 12x12
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # NOTE: NOTE6NOTE, NOTE16NOTE, 5x5NOTE
        # NOTE: 12x12 -> NOTE: 8x8 (NOTEpadding)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        # NOTE: 2x2NOTE
        # NOTE: 8x8 -> NOTE: 4x4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # NOTE
        # NOTE: 16 * 4 * 4 = 256
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        """NOTE"""
        # NOTE: conv -> bn -> relu -> pool
        out = self.conv1(x)          # 28x28 -> 24x24
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)        # 24x24 -> 12x12
        
        # NOTE: conv -> bn -> relu -> pool
        out = self.conv2(out)        # 12x12 -> 8x8
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool2(out)        # 8x8 -> 4x4
        
        # NOTE
        out = out.view(out.size(0), -1)  # 16 * 4 * 4 = 256
        
        # NOTE
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out
    
    def get_bn_stats(self, x):
        """
        NOTEBNNOTE(NOTE)
        NOTEBNNOTE
        LeNet5NOTE2NOTEBNNOTE
        """
        bn_stats = []
        
        # NOTEBNNOTE
        out = self.conv1(x)
        out = self.bn1(out)
        bn_stats.append((self.bn1.running_mean, self.bn1.running_var))
        out = F.relu(out)
        out = self.pool1(out)
        
        # NOTEBNNOTE
        out = self.conv2(out)
        out = self.bn2(out)
        bn_stats.append((self.bn2.running_mean, self.bn2.running_var))
        
        return bn_stats


# ==================== NOTE ====================

def flatten_params(params_dict):
    """
    NOTE
    
    NOTE:
        params_dict: NOTE {name: tensor}
    
    NOTE:
        flat_vector: NOTE
    """
    # NOTE
    tensors = []
    for name, param in params_dict.items():
        if param.dtype in [torch.float16, torch.float32, torch.float64]:
            tensors.append(param.view(-1))
    
    # NOTE
    if tensors:
        return torch.cat(tensors)
    else:
        return torch.tensor([])


def normalize_vector(vec):
    """
    NOTEL2NOTE
    
    NOTE:
    - NOTEL2NOTE(NOTE)
    - L2NOTE = sqrt(sum(x_i^2))
    - NOTE,NOTEL2NOTE1
    - NOTE:NOTE,NOTE
    
    NOTE:
        vec: NOTE
    
    NOTE:
        normalized_vec: NOTE
    """
    norm = torch.norm(vec, p=2)  # NOTEL2NOTE
    if norm > 1e-8:
        return vec / norm
    else:
        return vec


def compute_cosine_similarity(vec1, vec2):
    """
    NOTE
    
    NOTE:
    - NOTE
    - NOTE:[-1, 1]
    - 1 NOTE
    - 0 NOTE(NOTE)
    - -1 NOTE
    
    NOTE:cos(θ) = (A · B) / (||A|| * ||B||)
    
    NOTE,NOTE = NOTE
    
    NOTE:
        vec1: NOTE(NOTE)
        vec2: NOTE(NOTE)
    
    NOTE:
        similarity: NOTE
    """
    # NOTE,NOTE
    similarity = torch.dot(vec1, vec2).item()
    return similarity


def compute_client_similarity_matrix(client_params_list, client_ids=None):
    """
    NOTE
    
    NOTE:
        client_params_list: NOTE,NOTE
        client_ids: NOTEIDNOTE(NOTE,NOTE)
    
    NOTE:
        similarity_matrix: NOTE (n_clients x n_clients)
        similarity_data: NOTEJSONNOTE
    """
    n_clients = len(client_params_list)
    
    # NOTE1:NOTE
    flat_vectors = []
    for params in client_params_list:
        vec = flatten_params(params)
        flat_vectors.append(vec)
    
    # NOTE2:NOTEL2NOTE
    normalized_vectors = []
    for vec in flat_vectors:
        normalized_vec = normalize_vector(vec)
        normalized_vectors.append(normalized_vec)
    
    # NOTE3:NOTE
    similarity_matrix = torch.zeros(n_clients, n_clients)
    
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                # NOTE1
                similarity_matrix[i, j] = 1.0
            elif j > i:
                # NOTE,NOTE
                sim = compute_cosine_similarity(normalized_vectors[i], normalized_vectors[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # NOTE
    
    # NOTEJSONNOTE
    similarity_data = {
        'matrix': similarity_matrix.tolist(),  # NOTE
        'client_ids': client_ids if client_ids else list(range(n_clients)),  # NOTEID
        'n_clients': n_clients,  # NOTE
        'description': 'Cosine similarity matrix of client model parameters after local training'
    }
    
    return similarity_matrix, similarity_data


def save_similarity_to_json(similarity_data, round_idx, save_dir='similarity_results'):
    """
    NOTEJSONNOTE(NOTE)
    
    NOTE:
        similarity_data: NOTE
        round_idx: NOTE
        save_dir: NOTE
    
    NOTE:
        json_path: JSONNOTE
    """
    return None


# ==================== NOTE ====================

def apply_logit_masking(logits, local_classes, num_classes=10, class_counts=None, num_samples=None):
    """
    NOTElogitsNOTE(NOTE3:NOTE)
    
    NOTE:
        logits: NOTE (batch_size, num_classes)
        local_classes: NOTE(NOTE)
        num_classes: NOTE
        class_counts: NOTE(NOTE,NOTE3)
        num_samples: NOTE(NOTE,NOTE3)
    
    NOTE:
        masked_logits: NOTElogits
        minority_classes: NOTE(NOTE)
    """
    # NOTE3:NOTE tau = N_total / K
    # NOTEnum_samples,NOTE；NOTE
    if num_samples is not None and class_counts is not None:
        # NOTE:NOTE/NOTE
        tau = num_samples / num_classes
        
        # NOTE(NOTE,NOTE)NOTE(NOTE,NOTE)
        majority_classes = [c for c in range(num_classes) if class_counts[c] >= tau]
        minority_classes = [c for c in range(num_classes) if class_counts[c] < tau]
        
        # NOTE:NOTE(NOTE),NOTE
        if len(minority_classes) == 0:
            minority_classes = [i for i in range(num_classes) if i not in local_classes]
            majority_classes = list(local_classes)
    else:
        # NOTE:NOTE,NOTE
        minority_classes = [i for i in range(num_classes) if i not in local_classes]
        majority_classes = list(local_classes)
    
    # NOTE:NOTE0,NOTE1
    mask = torch.ones(logits.shape[0], num_classes, device=logits.device)
    for c in majority_classes:
        mask[:, c] = 0
    
    # NOTE:NOTElogitNOTE-100(NOTE)
    masked_logits = logits.clone()
    masked_logits[mask == 0] = -100
    
    return masked_logits, minority_classes


def diversity_loss(generator, z1, z2, y):
    """
    NOTE:NOTE
    
    NOTE:
        generator: NOTE
        z1, z2: NOTE
        y: NOTE
    
    NOTE:
        loss: NOTE(NOTE,NOTE)
    """
    # NOTE
    img1 = generator(z1, y)
    img2 = generator(z2, y)
    
    # NOTE
    img_diff = torch.norm(img1 - img2, p=2)
    z_diff = torch.norm(z1 - z2, p=2)
    
    # NOTEtanhNOTE,NOTE[-1, 0]NOTE
    # tanhNOTE[0,1],NOTE[-1,0],NOTE
    normalized_loss = -torch.tanh(img_diff / (z_diff + 1e-8))
    
    return normalized_loss


def bn_regularization_loss(global_model, fake_images):
    """
    BNNOTE
    
    NOTE:
        global_model: NOTE(NOTEBNNOTE)
        fake_images: NOTE
    
    NOTE:
        loss: BNNOTE
    """
    # NOTEBNNOTE
    bn_inputs = {}
    
    # NOTE
    def hook_fn(name):
        def hook(module, input, output):
            bn_inputs[name] = input[0].detach()
        return hook
    
    # NOTEBNNOTE
    hooks = []
    bn_layers = {}
    
    for name, module in global_model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers[name] = module
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    was_training = global_model.training
    global_model.eval()
    _ = global_model(fake_images)
    
    # NOTE
    for hook in hooks:
        hook.remove()
    
    # NOTE
    loss = 0.0
    for name, module in bn_layers.items():
        if name not in bn_inputs:
            continue
        
        bn_input = bn_inputs[name]
        
        # NOTE
        if isinstance(module, nn.BatchNorm2d):
            # ConvNOTE: [N, C, H, W] -> [C]
            fake_mean = bn_input.mean(dim=[0, 2, 3])
            fake_var = bn_input.var(dim=[0, 2, 3], unbiased=False)
        else:
            # FCNOTE: [N, C] -> [C]
            fake_mean = bn_input.mean(dim=0)
            fake_var = bn_input.var(dim=0, unbiased=False)
        
        # NOTE
        running_mean = module.running_mean
        running_var = module.running_var
        
        # NOTEMSENOTE
        loss += F.mse_loss(fake_mean, running_mean)
        loss += F.mse_loss(fake_var, running_var)
    
    if was_training:
        global_model.train()
    else:
        global_model.eval()
    return loss


def get_lambda(*_args, **_kwargs):
    """
    NOTEλNOTE0.2
    """
    return 0.2


# ==================== NOTE ====================

def train_generator(generator, global_model, local_classes, 
                    num_epochs=10, batch_size=64, 
                    alpha=0.1, beta=0.05, lr=0.0002,
                    device='cpu', class_counts=None, num_samples=None,
                    temperature=1.0):
    """
    NOTE(NOTE3:NOTE)
    
    NOTE:
        generator: NOTE
        global_model: NOTE(NOTE)
        local_classes: NOTE(NOTE)
        num_epochs: NOTE
        batch_size: NOTE
        alpha: NOTE
        beta: BNNOTE
        lr: NOTE
        class_counts: NOTE(NOTE,NOTE3)
        num_samples: NOTE(NOTE,NOTE3)
        temperature: NOTET,NOTE(NOTE,NOTE3)
    
    NOTE:
        generator: NOTE
        avg_loss: NOTE
        avg_cls_loss: NOTE
        avg_div_loss: NOTE
        avg_bn_loss: NOTEBNNOTE
    """
    # NOTE
    for param in global_model.parameters():
        param.requires_grad = False
    
    # NOTE:NOTE,NOTEBatchNormNOTE
    global_model.eval()
    
    # NOTE
    generator.train()
    
    # NOTE
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # NOTE3:NOTE
    if num_samples is not None and class_counts is not None:
        # NOTE:NOTE/NOTE
        tau = num_samples / 10
        # NOTE(NOTE,NOTE)
        minority_classes = [c for c in range(10) if class_counts[c] < tau]
        
        # NOTE:NOTE(NOTE),NOTE
        if len(minority_classes) == 0:
            minority_classes = [i for i in range(10) if i not in local_classes]
        
        # NOTE:NOTEminority_classesNOTE(NOTE10NOTE),NOTE
        if len(minority_classes) == 0:
            # NOTE
            for param in global_model.parameters():
                param.requires_grad = True
            # NOTE
            global_model.train()
            # NOTE0NOTE
            return generator, 0.0, 0.0, 0.0, 0.0
        
        # NOTE3:NOTE P(y=c) = exp(-n_c / T) / sum(exp(-n_k / T))
        sampling_weights = torch.zeros(10, device=device)
        for c in minority_classes:
            # NOTE,NOTE(NOTE)
            if class_counts[c] > 0:
                sampling_weights[c] = math.exp(-class_counts[c] / temperature)
            else:
                sampling_weights[c] = math.exp(0)  # NOTE1
        
        # NOTE:NOTE0,NOTE
        if sampling_weights.sum() == 0:
            for c in minority_classes:
                sampling_weights[c] = 1.0 / len(minority_classes) if len(minority_classes) > 0 else 0.0
        else:
            # NOTE
            sampling_weights = sampling_weights / sampling_weights.sum()
    else:
        # NOTE:NOTE
        minority_classes = [i for i in range(10) if i not in local_classes]
        sampling_weights = None
        
        # NOTE:NOTEminority_classesNOTE(NOTE),NOTE
        if len(minority_classes) == 0:
            # NOTE
            for param in global_model.parameters():
                param.requires_grad = True
            # NOTE
            global_model.train()
            # NOTE0NOTE
            return generator, 0.0, 0.0, 0.0, 0.0
    
    # NOTE
    total_loss = 0.0
    total_cls_loss = 0.0  # NOTE
    total_div_loss = 0.0  # NOTE
    total_bn_loss = 0.0   # NOTEBNNOTE
    num_batches = 0
    
    # NOTEbatch_sizeNOTE
    effective_batch_size = min(batch_size, 64)  # NOTE64
    
    # NOTEBNNOTE,NOTE
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
        
        for _ in range(10):  # NOTEepoch 10NOTEbatch
            # NOTE
            z = torch.randn(effective_batch_size, generator.noise_dim, device=device)
            
            # NOTE3:NOTE(NOTE,NOTE)
            if sampling_weights is not None:
                # NOTE
                y_indices = torch.multinomial(sampling_weights, effective_batch_size, replacement=True)
            else:
                # NOTE:NOTE
                y_indices = torch.tensor(
                    [minority_classes[i] for i in torch.randint(0, len(minority_classes), (effective_batch_size,))],
                    device=device
                )
            
            # NOTEone-hotNOTE
            y = torch.zeros(effective_batch_size, generator.num_classes, device=device)
            for i, class_label in enumerate(y_indices):
                y[i, class_label] = 1
            
            # NOTE
            fake_images = generator(z, y)
            
            # NOTEBNNOTE
            bn_stats.clear()
            
            # NOTE(NOTE,NOTE3)
            # NOTEBNNOTE
            logits = global_model(fake_images)
            masked_logits, _ = apply_logit_masking(logits, local_classes, 
                                                    class_counts=class_counts, 
                                                    num_samples=num_samples)
            loss_cls = F.cross_entropy(masked_logits, y_indices)
            
            # NOTEBNNOTE(NOTE)
            loss_bn = torch.tensor(0.0, device=device)
            for name, module in bn_layers.items():
                if name not in bn_stats:
                    continue
                fake_mean, fake_var = bn_stats[name]
                loss_bn = loss_bn + F.mse_loss(fake_mean, module.running_mean)
                loss_bn = loss_bn + F.mse_loss(fake_var, module.running_var)
            
            # NOTE(NOTEbatch)
            z1 = torch.randn(effective_batch_size, generator.noise_dim, device=device)
            z2 = torch.randn(effective_batch_size, generator.noise_dim, device=device)
            loss_div = diversity_loss(generator, z1, z2, y)
            
            # NOTE
            loss = loss_cls + alpha * loss_div + beta * loss_bn
            
            # NOTE(NOTE)
            epoch_cls_loss += loss_cls.item()
            epoch_div_loss += loss_div.item()
            epoch_bn_loss += loss_bn.item()
            epoch_batches += 1
            
            # NOTE
            optimizer.zero_grad()
            loss.backward()
            
            # NOTE,NOTE
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # NOTE
            del fake_images, logits, masked_logits, z, y, z1, z2, loss_cls, loss_div, loss_bn, loss
            bn_stats.clear()
        
        # NOTEepochNOTE,NOTE
        total_cls_loss += epoch_cls_loss
        total_div_loss += epoch_div_loss
        total_bn_loss += epoch_bn_loss
        num_batches += epoch_batches
    
    # NOTE
    avg_cls_loss = total_cls_loss / num_batches
    avg_div_loss = total_div_loss / num_batches
    avg_bn_loss = total_bn_loss / num_batches
    avg_loss = avg_cls_loss + alpha * avg_div_loss + beta * avg_bn_loss
    
    # NOTE
    for hook in hooks:
        hook.remove()
    
    # NOTE
    for param in global_model.parameters():
        param.requires_grad = True
    
    # NOTE
    global_model.train()
    
    return generator, avg_loss, avg_cls_loss, avg_div_loss, avg_bn_loss


def generate_synthetic_data(generator, missing_classes, 
                           num_samples_per_class=100, device='cpu',
                           sampling_weights=None, total_samples=None):
    """
    NOTE(NOTE3:NOTE)
    
    NOTE:
        generator: NOTE
        missing_classes: NOTE(NOTE)
        num_samples_per_class: NOTE
        device: NOTE
        sampling_weights: NOTE(NOTE3)
        total_samples: NOTE(NOTE3,NOTE)
    
    NOTE:
        synthetic_images: NOTE
        synthetic_labels: NOTE
    """
    generator.eval()
    
    # NOTE:NOTEmissing_classesNOTE,NOTE
    # MNISTNOTE: 1x28x28
    if len(missing_classes) == 0:
        return torch.zeros(0, 1, 28, 28), torch.tensor([], dtype=torch.long)
    
    synthetic_data = {c: [] for c in missing_classes}
    synthetic_images = []
    synthetic_labels = []
    
    with torch.no_grad():
        # NOTE3:NOTE,NOTE
        if sampling_weights is not None and total_samples is not None and total_samples > 0:
            # NOTE:NOTEtotal_samplesNOTEmissing_classesNOTE
            # NOTE1NOTE
            effective_total = max(total_samples, len(missing_classes))
            
            # NOTE
            class_samples = {}
            remaining = effective_total
            for i, c in enumerate(missing_classes):
                if i < len(missing_classes) - 1:
                    # NOTE(NOTE1NOTE)
                    # NOTE:sampling_weights[c]NOTE0(NOTEminority_classesNOTE)
                    weight_val = sampling_weights[c].item() if sampling_weights[c] > 0 else 1.0 / len(missing_classes)
                    samples = max(1, int(effective_total * weight_val))
                    samples = min(samples, remaining - (len(missing_classes) - i - 1))  # NOTE1NOTE
                    class_samples[c] = max(1, samples)
                    remaining -= class_samples[c]
                else:
                    # NOTE(NOTE1NOTE)
                    class_samples[c] = max(1, remaining)
            
            # NOTE(NOTE)
            gen_batch_size = 64  # NOTE
            for class_label in missing_classes:
                num_samples = class_samples[class_label]
                if num_samples <= 0:
                    continue
                
                class_images = []
                # NOTE
                for start in range(0, num_samples, gen_batch_size):
                    end = min(start + gen_batch_size, num_samples)
                    batch_num = end - start
                    
                    # NOTE
                    z = torch.randn(batch_num, generator.noise_dim, device=device)
                    
                    # NOTEone-hotNOTE
                    y = torch.zeros(batch_num, generator.num_classes, device=device)
                    y[:, class_label] = 1
                    
                    # NOTE
                    fake_images = generator(z, y)
                    class_images.append(fake_images.cpu())
                    
                    # NOTE
                    del z, y, fake_images
                
                # NOTE
                if class_images:
                    all_class_images = torch.cat(class_images, dim=0)
                    synthetic_data[class_label] = all_class_images
                    synthetic_images.append(all_class_images)
                    synthetic_labels.extend([class_label] * num_samples)
        else:
            # NOTE:NOTE(NOTE)
            gen_batch_size = 64
            for class_label in missing_classes:
                class_images = []
                remaining = num_samples_per_class
                
                for start in range(0, num_samples_per_class, gen_batch_size):
                    end = min(start + gen_batch_size, num_samples_per_class)
                    batch_num = end - start
                    
                    # NOTE
                    z = torch.randn(batch_num, generator.noise_dim, device=device)
                    
                    # NOTEone-hotNOTE
                    y = torch.zeros(batch_num, generator.num_classes, device=device)
                    y[:, class_label] = 1
                    
                    # NOTE
                    fake_images = generator(z, y)
                    class_images.append(fake_images.cpu())
                    
                    # NOTE
                    del z, y, fake_images
                
                # NOTE
                if class_images:
                    all_class_images = torch.cat(class_images, dim=0)
                    synthetic_data[class_label] = all_class_images
                    synthetic_images.append(all_class_images)
                    synthetic_labels.extend([class_label] * num_samples_per_class)
    
    # NOTE
    if synthetic_images:
        synthetic_images = torch.cat(synthetic_images, dim=0)
        synthetic_labels = torch.tensor(synthetic_labels, dtype=torch.long)
    else:
        # NOTE,NOTE
        # MNISTNOTE: 1x28x28
        synthetic_images = torch.zeros(0, 1, 28, 28)
        synthetic_labels = torch.tensor([], dtype=torch.long)
    
    return synthetic_images, synthetic_labels


def train_with_mixed_data(model, real_data_loader, synthetic_images, synthetic_labels,
                         epochs=6, batch_size=64, 
                         real_weight=0.7, synthetic_weight=0.3,
                         device='cpu', clipping_threshold=1.2,
                         global_model_params=None,
                         minority_classes=None, tau=None, lambda_val=0.2):
    """
    NOTE(NOTE3:γNOTE + NOTE)
    
    NOTE:
    1. NOTE b = Batch_Size / K
    2. NOTE(NOTE)
    3. NOTE
    4. NOTEBatchNOTE
    5. γNOTE:γ_c = 1 - n_c/τ
    6. NOTE:
       L = ∑_{c∈C_maj} L_CE(Real) + ∑_{c∈C_min} [(1-γ)L_CE(Real) + γλL_CE(Syn)]
    
    NOTE:
        model: NOTE
        real_data_loader: NOTE
        synthetic_images: NOTE
        synthetic_labels: NOTE
        epochs: NOTE
        batch_size: NOTE
        real_weight: NOTE(NOTE,NOTEγNOTE)
        synthetic_weight: NOTE(NOTE,NOTEγλ)
        device: NOTE
        clipping_threshold: NOTE
        global_model_params: NOTE(NOTEFedProxNOTE)
        minority_classes: NOTE(NOTE3)
        tau: NOTE(NOTE3)
        lambda_val: NOTEλ(NOTE3)
    
    NOTE:
        model: NOTE
        avg_loss: NOTE
        client_grad_norm_median: NOTE
    """
    model.train()
    
    # NOTEFedProxNOTE
    global_params = None
    if global_model_params is not None and FEDPROX_MU > 0:
        global_params = {}
        for name, param in model.named_parameters():
            if name in global_model_params:
                global_params[name] = global_model_params[name].clone().detach().to(device)
    
    # NOTE
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    
    # NOTE3:NOTEBatchNOTE
    num_classes = 10
    quota_per_class = batch_size // num_classes  # NOTE
    
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
    
    # NOTEγNOTE(NOTE3)
    # γ_c = 1 - n_c/τ
    gamma = {}
    if minority_classes is not None and tau is not None:
        for c in range(num_classes):
            if c in minority_classes:
                # NOTE:γ_c = 1 - n_c/τ
                gamma[c] = max(0.0, min(1.0, 1.0 - real_counts[c] / tau))
            else:
                # NOTE:γ = 0(NOTE)
                gamma[c] = 0.0
    else:
        # NOTE:NOTEγ=0.2
        for c in range(num_classes):
            gamma[c] = 0.2
    
    # NOTE(NOTE)
    num_batches_per_epoch = max(1, total_real // batch_size)
    
    syn_indices_by_class = {c: [] for c in range(num_classes)}
    if int(synthetic_images.shape[0]) > 0:
        syn_labels_list = synthetic_labels.detach().cpu().tolist()
        for i, label in enumerate(syn_labels_list):
            if 0 <= int(label) < num_classes:
                syn_indices_by_class[int(label)].append(i)
    
    # NOTE
    total_loss = 0.0
    num_batches = 0
    all_batch_global_grad_norms = []
    
    # NOTE
    total_real_used = 0  # NOTE
    total_syn_used = 0   # NOTE
    minority_real_used = 0  # NOTE
    minority_syn_used = 0   # NOTE
    
    for epoch in range(epochs):
        # NOTE
        real_indices = {c: real_indices_by_class[c].copy() for c in range(num_classes)}
        for c in range(num_classes):
            random.shuffle(real_indices[c])
        
        # NOTE
        used_real_count = {c: 0 for c in range(num_classes)}
        
        for batch_idx in range(num_batches_per_epoch):
            batch_data = []
            batch_labels = []
            batch_is_synthetic = []
            batch_class_labels = []
            
            for c in range(num_classes):
                # NOTE
                remaining_real = real_counts[c] - used_real_count[c]
                remaining_batches = num_batches_per_epoch - batch_idx
                
                if remaining_batches > 0 and remaining_real > 0:
                    # NOTE
                    real_quota = min(quota_per_class, remaining_real // remaining_batches)
                    real_quota = max(0, real_quota)
                else:
                    real_quota = 0
                
                # NOTE(NOTE)
                if minority_classes is not None and c in minority_classes:
                    syn_quota = quota_per_class - real_quota
                else:
                    syn_quota = 0  # NOTE
                
                # NOTE
                if real_quota > 0 and used_real_count[c] < len(real_indices[c]):
                    real_added = 0  # NOTE
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
                    # NOTE
                    total_real_used += real_added
                    if minority_classes is not None and c in minority_classes:
                        minority_real_used += real_added
                
                # NOTE(NOTE)
                if syn_quota > 0 and len(syn_indices_by_class[c]) > 0:
                    syn_added = 0  # NOTE
                    for _ in range(syn_quota):
                        syn_idx = random.choice(syn_indices_by_class[c])
                        batch_data.append(synthetic_images[syn_idx])
                        batch_labels.append(c)
                        batch_is_synthetic.append(True)
                        batch_class_labels.append(c)
                        syn_added += 1
                    # NOTE
                    total_syn_used += syn_added
                    if minority_classes is not None and c in minority_classes:
                        minority_syn_used += syn_added
            
            # NOTEbatchNOTE,NOTE
            if len(batch_data) == 0:
                continue
            
            # NOTEbatchNOTE
            batch_data = torch.stack(batch_data, dim=0).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
            batch_is_synthetic = torch.tensor(batch_is_synthetic, dtype=torch.bool, device=device)
            batch_class_labels = torch.tensor(batch_class_labels, dtype=torch.long, device=device)
            
            # NOTE
            optimizer.zero_grad()
            output = model(batch_data)
            
            # NOTE3:NOTEγNOTE
            # L = ∑_{c∈C_maj} L_CE(Real) + ∑_{c∈C_min} [(1-γ)L_CE(Real) + γλL_CE(Syn)]
            loss = 0.0
            
            for c in range(num_classes):
                # NOTEmask
                class_mask = (batch_class_labels == c)
                if not class_mask.any():
                    continue
                
                class_output = output[class_mask]
                class_labels = batch_labels[class_mask]
                class_is_syn = batch_is_synthetic[class_mask]
                
                if minority_classes is not None and c in minority_classes:
                    # NOTE:NOTEγNOTE
                    # NOTE:(1-γ_c)
                    # NOTE:γ_c * λ
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
                    # NOTE:NOTE,NOTE1
                    real_mask_c = ~class_is_syn
                    if real_mask_c.any():
                        real_output = class_output[real_mask_c]
                        real_labels = class_labels[real_mask_c]
                        loss += F.cross_entropy(real_output, real_labels)
            
            # NOTEFedProxNOTE
            if global_params is not None:
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    if name in global_params and param.requires_grad:
                        proximal_term += (param - global_params[name]).norm(2) ** 2
                loss = loss + (FEDPROX_MU / 2) * proximal_term
            
            loss.backward()
            
            # NOTE
            all_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    all_grads.append(param.grad.view(-1))
            if all_grads:
                global_grad_norm = torch.norm(torch.cat(all_grads), p=2).item()
                all_batch_global_grad_norms.append(global_grad_norm)
            
            # NOTE
            clip_model_gradients(model, clipping_threshold)
            
            optimizer.step()
            total_loss += float(loss.detach().item())
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # NOTE
    client_global_grad_norm_median = np.median(all_batch_global_grad_norms) if all_batch_global_grad_norms else 0.0
    
    # NOTE
    sample_usage_stats = {
        'total_real_used': total_real_used,
        'total_syn_used': total_syn_used,
        'minority_real_used': minority_real_used,
        'minority_syn_used': minority_syn_used
    }
    
    return model, avg_loss, client_global_grad_norm_median, sample_usage_stats


# ==================== NOTE ====================

def load_mnist(data_path=None, use_augmentation=True):
    """
    NOTEMNISTNOTE(NOTE)
    
    NOTE:
        data_path: NOTE
        use_augmentation: NOTE(NOTE)
    
    NOTE:
        train_dataset: NOTE (60000NOTE)
        test_dataset: NOTE (10000NOTE)
    """
    # MNISTNOTE: NOTE=0.1307, NOTE=0.3081
    mnist_mean = 0.1307
    mnist_std = 0.3081
    
    # NOTE
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,))
    ])
    
    # NOTE
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),  # NOTE,NOTE4NOTE
            transforms.ToTensor(),
            transforms.Normalize((mnist_mean,), (mnist_std,))
        ])
    else:
        # NOTE,NOTE
        train_transform = test_transform
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if data_path is None:
        data_path = os.path.join(base_dir, 'data')
    elif not os.path.isabs(data_path):
        data_path = os.path.join(base_dir, data_path)
    os.makedirs(data_path, exist_ok=True)
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_path,
        train=True,
        download=False,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=False,
        transform=test_transform
    )
    
    return train_dataset, test_dataset


def load_cifar10(data_path='./data', use_augmentation=True):
    """
    NOTECIFAR-10NOTE(NOTE,NOTE3)
    
    NOTE:
        data_path: NOTE
        use_augmentation: NOTE(NOTE,NOTE)
    
    NOTE:
        train_dataset: NOTE (50000NOTE)
        test_dataset: NOTE (10000NOTE)
    """
    # NOTE
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # NOTE(NOTE3:NOTE,NOTE)
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),      # NOTE,NOTE4NOTE
            transforms.RandomHorizontalFlip(),          # NOTE
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    else:
        # NOTE,NOTE
        train_transform = test_transform
    
    # NOTE
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # NOTE
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform
    )
    
    print(f"NOTE: {len(train_dataset)}")
    print(f"NOTE: {len(test_dataset)}")
    print(f"NOTE: {'NOTE' if use_augmentation else 'NOTE'}")
    
    return train_dataset, test_dataset


def partition_data_noniid(train_dataset, num_clients=100, num_classes_per_client=2):
    """
    Non-IIDNOTE
    NOTE
    MNIST: NOTE6000NOTE
    """
    targets = np.array(train_dataset.targets)
    
    # NOTE
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    
    # NOTE
    print("\nNOTE:")
    for class_label in range(10):
        print(f"  NOTE {class_label}: {len(class_indices[class_label])} NOTE")
    
    # NOTE
    for label in class_indices:
        np.random.shuffle(class_indices[label])
    
    # NOTE
    # 100NOTE × 2NOTE = 200NOTE
    # 10NOTE,NOTE20NOTE
    num_assignments_per_class = (num_clients * num_classes_per_client) // 10
    
    # NOTE
    samples_per_assignment = {}
    print("\nNOTE:")
    for class_label in range(10):
        samples_per_assignment[class_label] = len(class_indices[class_label]) // num_assignments_per_class
        print(f"  NOTE {class_label}: NOTE {samples_per_assignment[class_label]} NOTE")
    
    # NOTE C(10,2) = 45NOTE
    from itertools import combinations
    all_combinations = list(combinations(range(10), num_classes_per_client))
    
    # NOTE
    client_data_indices = defaultdict(list)
    client_classes = {}
    class_assigned = defaultdict(int)
    
    # NOTE
    class_assignment_count = defaultdict(int)
    
    for client_id in range(num_clients):
        # NOTE,NOTE
        # NOTE"NOTE"(NOTE)
        combo_scores = []
        for combo in all_combinations:
            score = sum(class_assignment_count[c] for c in combo)
            combo_scores.append((combo, score))
        
        # NOTE,NOTE(NOTE)
        combo_scores.sort(key=lambda x: x[1])
        
        # NOTE
        min_score = combo_scores[0][1]
        low_score_combos = [c for c, s in combo_scores if s == min_score]
        selected_combo = list(random.choice(low_score_combos))
        
        client_classes[client_id] = selected_combo
        
        # NOTE
        for class_label in selected_combo:
            class_assignment_count[class_label] += 1
        
        # NOTE
        for class_label in selected_combo:
            num_samples = samples_per_assignment[class_label]
            start_idx = class_assigned[class_label]
            end_idx = start_idx + num_samples
            
            client_data_indices[client_id].extend(
                class_indices[class_label][start_idx:end_idx]
            )
            class_assigned[class_label] = end_idx
    
    # NOTE
    print("\nNOTE:")
    total_samples = 0
    for client_id in range(num_clients):
        num_samples = len(client_data_indices[client_id])
        total_samples += num_samples
        if client_id < 5:
            print(f"NOTE {client_id}: {num_samples} NOTE, NOTE: {client_classes[client_id]}")
    print(f"NOTE: {total_samples}")
    
    # NOTE
    class_combinations = {}
    for client_id in range(num_clients):
        combo = tuple(sorted(client_classes[client_id]))
        if combo not in class_combinations:
            class_combinations[combo] = []
        class_combinations[combo].append(client_id)
    
    print(f"\nNOTE:")
    print(f"  NOTE: C(10,2) = 45 NOTE")
    print(f"  NOTE: {len(class_combinations)} NOTE")
    print(f"  NOTE (NOTE10NOTE):")
    sorted_combos = sorted(class_combinations.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (combo, clients) in enumerate(sorted_combos[:10]):
        print(f"    NOTE {combo}: {len(clients)} NOTE")
    
    # NOTE
    print(f"\nNOTE:")
    for class_label in range(10):
        print(f"  NOTE {class_label}: {class_assignment_count[class_label]} NOTE")
    
    return client_data_indices, client_classes


def partition_data_dirichlet(train_dataset, num_clients=10, alpha=0.1, min_samples=10, seed=42):
    """
    NOTENon-IIDNOTE
    
    NOTE:
        train_dataset: NOTE
        num_clients: NOTE
        alpha: NOTE(NOTE)
        min_samples: NOTE
        seed: NOTE(NOTE,NOTE)
    
    NOTE:
        client_data_indices: NOTE
        client_classes: NOTE
    """
    # NOTE,NOTE
    np.random.seed(seed)
    random.seed(seed)
    
    targets = np.array(train_dataset.targets)
    num_classes = 10
    
    # NOTE
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    
    # NOTE
    for label in class_indices:
        np.random.shuffle(class_indices[label])
    
    # NOTE
    # NOTE,NOTE
    client_data_indices = defaultdict(list)
    client_classes = {i: [] for i in range(num_clients)}
    
    # NOTE
    for class_label in range(num_classes):
        # NOTE,NOTE
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # NOTE
        total_samples = len(class_indices[class_label])
        samples_per_client = (proportions * total_samples).astype(int)
        
        # NOTEmin_samplesNOTE(NOTE)
        # NOTE
        samples_per_client = np.maximum(samples_per_client, min_samples)
        
        # NOTE,NOTE
        while samples_per_client.sum() > total_samples:
            # NOTE,NOTE1NOTE
            max_idx = samples_per_client.argmax()
            if samples_per_client[max_idx] > min_samples:
                samples_per_client[max_idx] -= 1
            else:
                break
        
        # NOTE
        start_idx = 0
        for client_id in range(num_clients):
            num_samples = samples_per_client[client_id]
            if num_samples > 0:
                end_idx = min(start_idx + num_samples, len(class_indices[class_label]))
                client_data_indices[client_id].extend(
                    class_indices[class_label][start_idx:end_idx]
                )
                # NOTE
                if class_label not in client_classes[client_id]:
                    client_classes[client_id].append(class_label)
                start_idx = end_idx
    
    return client_data_indices, client_classes


# ==================== NOTE ====================

def calculate_noise_std(epsilon, delta, sensitivity, q, T):
    """NOTEcDPNOTE(ε,δ)-DPNOTE"""
    if epsilon <= 0:
        raise ValueError("epsilon NOTE 0")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta NOTE (0, 1) NOTE")
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
    """NOTE (NOTE14)"""
    sensitivity = (2 * learning_rate * clipping_threshold) / num_samples
    return sensitivity


def clip_gradient(gradient, clipping_threshold):
    """NOTE(NOTE)"""
    grad_norm = torch.norm(gradient, p=2)
    if grad_norm > clipping_threshold:
        clipped_gradient = gradient * (clipping_threshold / grad_norm)
    else:
        clipped_gradient = gradient
    return clipped_gradient


def clip_model_gradients(model, clipping_threshold):
    """NOTEL2NOTE(UDPNOTE)"""
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


# ==================== NOTE ====================

class ClientWithGAN:
    """NOTEGANNOTE(NOTE3:NOTE)"""
    
    def __init__(self, client_id, data_indices, local_classes, dataset,
                 generator_init_params, device, local_epochs=6, batch_size=64, 
                 learning_rate=0.005, clipping_threshold=0.32,
                 epsilon=0.5, delta=1e-3, sampling_ratio=0.32, 
                 total_rounds=50, gen_epochs=10, gen_lr=0.0002,
                 num_synthetic_per_class=50, alpha=0.1, beta=0.05,
                 temperature=1.0, generator_cache_dir='generator_cache'):
        """NOTE(NOTE3:NOTE)"""
        self.client_id = client_id
        self.data_indices = data_indices
        self.local_classes = local_classes
        self.missing_classes = [i for i in range(10) if i not in local_classes]
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
        self.num_synthetic_per_class = num_synthetic_per_class
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature  # NOTE3:NOTE
        
        # NOTE
        self.generator_init_params_cpu = generator_init_params
        self.generator_cache_dir = generator_cache_dir
        self.generator_cache_path = os.path.join(generator_cache_dir, f'client_{client_id}.pth')
        self.generator_trained = False
        self.generator_state_cpu = None
        
        # NOTE
        self.data_loader = self._create_data_loader()
        
        # NOTE
        self.num_samples = len(data_indices)
        self.sensitivity = calculate_sensitivity(
            learning_rate, clipping_threshold, self.num_samples
        )
        self.noise_sigma = calculate_noise_std(
            epsilon, delta, self.sensitivity, sampling_ratio, total_rounds
        )
        
        # NOTE3:NOTE
        self.class_counts = self._compute_class_counts()
        
        # NOTE3:NOTE
        self.majority_classes, self.minority_classes = self._compute_dynamic_mask()
        self.sampling_weights = self._compute_sampling_weights()
    
    def _compute_class_counts(self):
        """NOTE3:NOTE"""
        class_counts = torch.zeros(10)
        # NOTE,NOTE
        for idx in self.data_indices:
            # NOTE
            label = self.dataset.targets[idx]
            class_counts[label] += 1
        return class_counts
    
    def _compute_dynamic_mask(self):
        """
        NOTE3:NOTE
        
        NOTE:
            majority_classes: NOTE(NOTE,NOTE)
            minority_classes: NOTE(NOTE,NOTE)
        """
        # NOTE tau = N_total / K
        tau = self.num_samples / 10
        
        # NOTE
        majority_classes = [c for c in range(10) if self.class_counts[c] >= tau]
        minority_classes = [c for c in range(10) if self.class_counts[c] < tau]
        
        # NOTE:NOTE(NOTE),NOTE
        # NOTE2NOTE250NOTE
        if len(minority_classes) == 0:
            minority_classes = self.missing_classes.copy()
            # NOTE
            majority_classes = list(self.local_classes)
        
        return majority_classes, minority_classes
    
    def _compute_sampling_weights(self):
        """
        NOTE3:NOTE
        
        NOTE: P(y=c) = exp(-n_c / T) / sum(exp(-n_k / T))
        NOTE,NOTE(NOTE)
        """
        weights = torch.zeros(10)
        
        # NOTE
        for c in self.minority_classes:
            if self.class_counts[c] > 0:
                # NOTE,NOTE
                weights[c] = math.exp(-self.class_counts[c] / self.temperature)
            else:
                # NOTE
                weights[c] = math.exp(0)  # = 1.0
        
        # NOTE:NOTE0(NOTE),NOTE
        if weights.sum() == 0:
            for c in self.minority_classes:
                weights[c] = 1.0 / len(self.minority_classes) if len(self.minority_classes) > 0 else 0.0
        else:
            # NOTE
            weights = weights / weights.sum()
        
        return weights
    
    def _create_data_loader(self):
        """NOTE"""
        subset = Subset(self.dataset, self.data_indices)
        generator = torch.Generator()
        generator.manual_seed(42)
        loader = DataLoader(
            subset, batch_size=self.batch_size, shuffle=True,
            generator=generator, worker_init_fn=worker_init_fn, num_workers=0,
            drop_last=True  # NOTEbatch,NOTEBatchNormNOTE
        )
        return loader
    
    def train_phase0(self, global_model_params, model, local_epochs=None):
        """NOTE0:NOTEFedAvgNOTE(NOTEFedProxNOTE)
        
        NOTE:
            global_model_params: NOTE
            model: NOTE
        """
        model.load_state_dict(global_model_params)
        model.train()
        
        # NOTEFedProxNOTE
        global_params = None
        if FEDPROX_MU > 0:
            global_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        # NOTE
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # NOTE
        total_loss = 0.0
        total_batches = 0
        
        # NOTEbatchNOTE(NOTE)
        all_batch_global_grad_norms = []
        
        effective_local_epochs = self.local_epochs if local_epochs is None else int(local_epochs)
        for epoch in range(effective_local_epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # NOTEFedProxNOTE:NOTE
                if FEDPROX_MU > 0 and global_params is not None:
                    proximal_term = 0.0
                    for name, param in model.named_parameters():
                        if name in global_params and param.requires_grad:
                            proximal_term += (param - global_params[name]).norm(2) ** 2
                    loss = loss + (FEDPROX_MU / 2) * proximal_term
                
                loss.backward()
                
                # NOTE(NOTEL2NOTE)
                all_grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        all_grads.append(param.grad.view(-1))
                if all_grads:
                    global_grad_norm = torch.norm(torch.cat(all_grads), p=2).item()
                    all_batch_global_grad_norms.append(global_grad_norm)
                
                # NOTE,NOTEUDPNOTE
                clip_model_gradients(model, self.clipping_threshold)
                
                optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / max(total_batches, 1)
        
        # NOTEbatchNOTE
        client_global_grad_norm_median = np.median(all_batch_global_grad_norms) if all_batch_global_grad_norms else 0.0
        
        # NOTE
        params_cpu = _cpu_state_dict(model.state_dict())
        noisy_params = add_discrete_noise_to_model_update(
            global_model_params, params_cpu, self.noise_sigma, device='cpu', num_bits=16
        )
        
        return noisy_params, self.num_samples, avg_loss, client_global_grad_norm_median
    
    def train_phase1(self, global_model_params, test_accuracy_history, model, generator, local_epochs=None):
        """NOTE1:GANNOTE
        
        NOTE:
            global_model_params: NOTE
            test_accuracy_history: NOTE(NOTEλNOTE)
            model: NOTE
            generator: NOTE
        """
        model.load_state_dict(global_model_params)
        
        # NOTE(NOTE)
        was_trained = self.generator_trained
        
        # NOTE1: NOTE(NOTE)
        # NOTE
        loaded_state = None
        if self.generator_trained and self.generator_state_cpu is not None and len(self.generator_state_cpu) > 0:
            loaded_state = self.generator_state_cpu
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
                print(f"  NOTE: NOTE,NOTE")
                print(f"    NOTE: {len(current_keys)} NOTE")
                print(f"    NOTE: {len(saved_keys)} NOTE")
                self.generator_trained = False
                self.generator_state_cpu = None
        
        # NOTE3:NOTEclass_counts,num_samplesNOTEtemperatureNOTE
        generator, gen_loss, cls_loss, div_loss, bn_loss = train_generator(
            generator, model, self.local_classes,
            num_epochs=10, batch_size=64,
            alpha=self.alpha, beta=self.beta, lr=self.gen_lr,
            device=self.device,
            class_counts=self.class_counts,      # NOTE3:NOTE
            num_samples=self.num_samples,         # NOTE3:NOTE
            temperature=self.temperature          # NOTE3:NOTE
        )
        self.generator_state_cpu = _cpu_state_dict_fp16(generator.state_dict())
        self.generator_trained = True
        
        # NOTE(NOTE)
        gen_losses = (gen_loss, cls_loss, div_loss, bn_loss, was_trained)
        
        # NOTE2: NOTE(NOTE3:NOTE)
        synthetic_images, synthetic_labels = generate_synthetic_data(
            generator, self.minority_classes,  # NOTE3:NOTE
            num_samples_per_class=self.num_synthetic_per_class,
            device=self.device,
            sampling_weights=self.sampling_weights,  # NOTE3:NOTE
            total_samples=self.num_synthetic_per_class * len(self.minority_classes)  # NOTE3:NOTE
        )

        gate_total = int(synthetic_labels.shape[0])
        gate_kept = 0
        gate_counts = torch.zeros(10, dtype=torch.long)
        if gate_total > 0:
            model_was_training = model.training
            model.eval()
            teacher_probs = []
            with torch.no_grad():
                bs = 256
                # NOTE:NOTE
                # NOTE >= 0.6 NOTE
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
            # NOTE:NOTE >= 0.6
            keep_mask = teacher_probs >= GATE_TEACHER_MIN_PROB
            keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
            synthetic_images = synthetic_images.index_select(0, keep_idx) if int(keep_idx.numel()) > 0 else synthetic_images[:0]
            synthetic_labels = synthetic_labels.index_select(0, keep_idx) if int(keep_idx.numel()) > 0 else synthetic_labels[:0]
            gate_kept = int(synthetic_labels.shape[0])
            if gate_kept > 0:
                gate_counts = torch.bincount(synthetic_labels, minlength=10).to(torch.long)
        
        # NOTE3: NOTE
        # NOTE(NOTE5NOTE)
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
            minority_classes=self.minority_classes,  # NOTE3:NOTE
            tau=self.num_samples / 10,  # NOTE3:NOTE
            lambda_val=lambda_val  # NOTE3:NOTEλ
        )
        
        # NOTE4: NOTE
        params_cpu = _cpu_state_dict(model.state_dict())
        noisy_params = add_discrete_noise_to_model_update(
            global_model_params, params_cpu, self.noise_sigma, device='cpu', num_bits=16
        )
        
        # NOTE,NOTE
        del synthetic_images, synthetic_labels
        
        return noisy_params, self.num_samples, avg_loss, gen_losses, client_grad_norm_median, (gate_total, gate_kept, gate_counts), sample_usage_stats


# ==================== NOTE ====================

class ServerWithGAN:
    """NOTEGANNOTE"""
    
    def __init__(self, model, device, test_dataset, num_clients=100, sample_ratio=0.32, secure_agg_server=None):
        """NOTE"""
        self.model = model
        self.device = device
        self.test_dataset = test_dataset
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.secure_agg_server = secure_agg_server
        
        # NOTE
        self.test_loader = DataLoader(
            test_dataset, batch_size=100, shuffle=False
        )
        
        # NOTE
        self.global_model_params = _cpu_state_dict(model.state_dict())
    
    def select_clients(self):
        """NOTE"""
        num_selected = int(self.num_clients * self.sample_ratio)
        selected_indices = np.random.choice(
            self.num_clients, size=num_selected, replace=False
        )
        return selected_indices
    
    def aggregate_models(self, client_updates, participating_clients=None):
        """NOTE"""
        if self.secure_agg_server is None:
            raise RuntimeError("secure_agg_server NOTE,NOTE")
        if participating_clients is None:
            raise RuntimeError("participating_clients NOTE,NOTE")
        if not client_updates:
            raise RuntimeError("client_updates NOTE,NOTE")

        masked_updates = [(params, num_samples, client_id) for params, num_samples, _, client_id in client_updates]
        self.global_model_params = self.secure_agg_server.aggregate_with_secure_masking(
            masked_updates, participating_clients, device='cpu'
        )
        avg_loss = sum([update[2] for update in client_updates]) / len(client_updates)
        return avg_loss
    
    def evaluate(self):
        """NOTE"""
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


# ==================== NOTE ====================

def main():
    """NOTE"""
    print("=" * 60)
    print("GANNOTE (MNIST)")
    print("=" * 60)
    
    SEED = 42
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # NOTE
    NUM_CLIENTS = 10  # NOTE
    SAMPLE_RATIO = 0.5  # NOTE,NOTE5NOTE
    PHASE0_LOCAL_EPOCHS = 10
    PHASE1_LOCAL_EPOCHS = 3
    BATCH_SIZE = 64
    LEARNING_RATE = 0.005
    GLOBAL_ROUNDS = 300
    PHASE0_ROUNDS = 10  # NOTEFedAvgNOTE
    PHASE1_ROUNDS = GLOBAL_ROUNDS - PHASE0_ROUNDS  # GANNOTE

    # NOTE
    EPSILON = 0.5
    DELTA = 1e-3
    CLIPPING_THRESHOLD = 0.32  # NOTE
    
    # GANNOTE
    GEN_EPOCHS = 10
    GEN_LR = 0.0001  # NOTE
    NUM_SYNTHETIC_PER_CLASS = 50  # NOTE
    ALPHA = 0.05  # NOTE L_div
    BETA = 0.15   # BNNOTE L_bn
    TEMPERATURE = TEMPERATURE_T  # NOTE3:NOTE,NOTE
    
    train_dataset, test_dataset = load_mnist()
    
    # NOTENon-IIDNOTE
    # alpha=0.1: NOTE
    # seed=42: NOTE,NOTE,NOTE
    DIRICHLET_ALPHA = 0.1
    DIRICHLET_SEED = 42
    client_data_indices, client_classes = partition_data_dirichlet(
        train_dataset, 
        num_clients=NUM_CLIENTS, 
        alpha=DIRICHLET_ALPHA,
        min_samples=10,
        seed=DIRICHLET_SEED
    )
    
    global_model = LeNet5().to(device)
    
    secagg_aggregator = SecureAggregator(num_clients=NUM_CLIENTS, seed=SEED)
    secagg_server = SecureAggregationServer(secagg_aggregator)
    secagg_clients = {}
    for client_id in range(NUM_CLIENTS):
        keys = secagg_aggregator.generate_client_keys(client_id)
        sec_client = SecureAggregationClient(client_id, keys, NUM_CLIENTS)
        secagg_clients[client_id] = sec_client
        secagg_server.collect_self_mask_seed(client_id, sec_client.get_self_mask_seed())

    server = ServerWithGAN(
        model=global_model, device=device, test_dataset=test_dataset,
        num_clients=NUM_CLIENTS, sample_ratio=SAMPLE_RATIO, secure_agg_server=secagg_server
    )
    
    client_model = LeNet5().to(device)
    shared_generator = LeGen28(noise_dim=100, num_classes=10).to(device)

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
            learning_rate=LEARNING_RATE,
            clipping_threshold=CLIPPING_THRESHOLD,
            epsilon=EPSILON,
            delta=DELTA,
            sampling_ratio=SAMPLE_RATIO,
            total_rounds=GLOBAL_ROUNDS,
            gen_epochs=GEN_EPOCHS,
            gen_lr=GEN_LR,
            num_synthetic_per_class=NUM_SYNTHETIC_PER_CLASS,
            alpha=ALPHA,
            beta=BETA,
            temperature=TEMPERATURE  # NOTE3:NOTE
        )
        client.secure_agg_client = secagg_clients[client_id]
        clients.append(client)

    # NOTE1NOTE3
    def get_phase1_local_epochs(round_idx):
        return 3  # NOTE3
    
    # NOTE
    history = {
        'round': [], 'train_loss': [], 'test_accuracy': [], 'test_loss': []
    }
    
    # NOTE
    best_accuracy = 0.0
    
    # NOTE0: NOTEFedAvg (NOTE1-10NOTE)
    print("\n" + "=" * 60)
    print(f"NOTE0: NOTEFedAvgNOTE (NOTE1-{PHASE0_ROUNDS}NOTE)")
    print("=" * 60)
    
    for round_idx in range(1, PHASE0_ROUNDS + 1):
        _cuda_sync()
        
        print(f"\nNOTE {round_idx}/{GLOBAL_ROUNDS}")
        
        selected_client_indices = server.select_clients()
        participating_client_ids = [int(clients[idx].client_id) for idx in selected_client_indices]
        for cid in participating_client_ids:
            other_ids = [x for x in participating_client_ids if x != cid]
            seeds = secagg_aggregator.get_all_pairwise_seeds_for_client(cid, other_ids)
            secagg_clients[cid].set_pairwise_seeds(seeds)
        client_updates = []
        client_grad_norm_medians = []
        
        for idx in selected_client_indices:
            noisy_params, num_samples, train_loss, client_global_grad_norm_median = clients[idx].train_phase0(
                server.global_model_params, client_model, PHASE0_LOCAL_EPOCHS
            )
            cid = int(clients[idx].client_id)
            sec_client = clients[idx].secure_agg_client
            pairwise_mask, self_mask = sec_client.generate_mask(noisy_params, participating_client_ids, device='cpu')
            masked_params = sec_client.apply_mask(noisy_params, pairwise_mask, self_mask)
            client_updates.append((masked_params, num_samples, train_loss, cid))
            del pairwise_mask, self_mask, masked_params
            client_grad_norm_medians.append(client_global_grad_norm_median)
        
        round_global_grad_norm_median = np.median(client_grad_norm_medians) if client_grad_norm_medians else 0.0
        
        avg_train_loss = server.aggregate_models(client_updates, participating_clients=participating_client_ids)
        
        del client_updates
        if PERF_DEBUG:
            gc.collect()
        
        test_accuracy, test_loss = server.evaluate()
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
        
        history['round'].append(round_idx)
        history['train_loss'].append(avg_train_loss)
        history['test_accuracy'].append(test_accuracy)
        history['test_loss'].append(test_loss)
        
        print(f"NOTE {round_idx} NOTE:")
        print(f"  NOTE: {avg_train_loss:.4f}")
        print(f"  NOTE: {test_accuracy:.2f}%")
        print(f"  NOTE: {best_accuracy:.2f}%")
        print(f"  NOTE: {test_loss:.4f}")
        print(f"  NOTE: {round_global_grad_norm_median:.4f}")
        
        all_missing_classes = []
        for idx in selected_client_indices:
            client = clients[idx]
            all_missing_classes.extend(client.missing_classes)
        all_missing_classes = list(set(all_missing_classes))
        
        if len(all_missing_classes) > 0:
            global_model.load_state_dict(server.global_model_params)
            shared_generator, gen_loss, cls_loss, div_loss, bn_loss = train_generator(
                shared_generator, global_model, all_missing_classes,
                num_epochs=GEN_EPOCHS, batch_size=64,
                alpha=ALPHA, beta=BETA, lr=GEN_LR,
                device=device
            )
            del gen_loss, cls_loss, div_loss, bn_loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    phase0_shared_generator_params_cpu = _cpu_state_dict(shared_generator.state_dict())
    for client in clients:
        client.generator_init_params_cpu = phase0_shared_generator_params_cpu
        client.generator_trained = False
        client.generator_state_cpu = None
    
    # NOTE1: GANNOTEFedAvg (NOTE11-150NOTE)
    print("\n" + "=" * 60)
    print(f"NOTE1: GANNOTEFedAvgNOTE (NOTE{PHASE0_ROUNDS + 1}-{GLOBAL_ROUNDS}NOTE)")
    print("=" * 60)
    
    print("NOTE0NOTE1(NOTE)")
    
    for round_idx in range(PHASE0_ROUNDS + 1, GLOBAL_ROUNDS + 1):
        _cuda_sync()
        
        print(f"\nNOTE {round_idx}/{GLOBAL_ROUNDS}")
        
        # NOTE5NOTEλ
        lambda_current = get_lambda(history['test_accuracy'])
        
        # NOTE
        recent_accs = history['test_accuracy'][-5:] if history['test_accuracy'] else []
        if recent_accs:
            weights = list(range(1, len(recent_accs) + 1))
            weighted_avg_acc = sum(w * a for w, a in zip(weights, recent_accs)) / sum(weights)
            print(f"  NOTE{len(recent_accs)}NOTE: {weighted_avg_acc:.2f}%")
        print(f"  NOTE λ: {lambda_current:.4f} (NOTE: {1.0-lambda_current:.4f}, NOTE: {lambda_current:.4f})")
        current_phase1_local_epochs = get_phase1_local_epochs(round_idx)
        print(f"  NOTE1NOTElocal_epochs: {current_phase1_local_epochs}")
        
        selected_client_indices = server.select_clients()
        participating_client_ids = [int(clients[idx].client_id) for idx in selected_client_indices]
        for cid in participating_client_ids:
            other_ids = [x for x in participating_client_ids if x != cid]
            seeds = secagg_aggregator.get_all_pairwise_seeds_for_client(cid, other_ids)
            secagg_clients[cid].set_pairwise_seeds(seeds)
        client_updates = []
        client_grad_norm_medians = []
        for idx in selected_client_indices:
            noisy_params, num_samples, train_loss, gen_losses, client_global_grad_norm_median, gate_info, sample_usage_stats = clients[idx].train_phase1(
                server.global_model_params, history['test_accuracy'], client_model, shared_generator, current_phase1_local_epochs
            )
            cid = int(clients[idx].client_id)
            sec_client = clients[idx].secure_agg_client
            pairwise_mask, self_mask = sec_client.generate_mask(noisy_params, participating_client_ids, device='cpu')
            masked_params = sec_client.apply_mask(noisy_params, pairwise_mask, self_mask)
            client_updates.append((masked_params, num_samples, train_loss, cid))
            del pairwise_mask, self_mask, masked_params
            client_grad_norm_medians.append(client_global_grad_norm_median)
            _ = gate_info
            _ = sample_usage_stats
            _ = gen_losses
        
        round_global_grad_norm_median = np.median(client_grad_norm_medians) if client_grad_norm_medians else 0.0
        avg_train_loss = server.aggregate_models(client_updates, participating_clients=participating_client_ids)
        
        # NOTE,NOTE
        del client_updates
        if PERF_DEBUG:
            gc.collect()
        
        test_accuracy, test_loss = server.evaluate()
        
        # NOTE
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
        
        history['round'].append(round_idx)
        history['train_loss'].append(avg_train_loss)
        history['test_accuracy'].append(test_accuracy)
        history['test_loss'].append(test_loss)
        
        print(f"NOTE {round_idx} NOTE:")
        print(f"  NOTE: {avg_train_loss:.4f}")
        print(f"  NOTE: {test_accuracy:.2f}%")
        print(f"  NOTE: {best_accuracy:.2f}%")
        print(f"  NOTE: {test_loss:.4f}")
        print(f"  NOTE: {round_global_grad_norm_median:.4f}")
    
    # NOTE
    print("\n" + "=" * 60)
    print("GANNOTE!")
    print("=" * 60)
    
    print(f"\nNOTE: {history['test_accuracy'][-1]:.2f}%")
    print(f"NOTE: {history['test_loss'][-1]:.4f}")
    print(f"NOTE: ({EPSILON}, {DELTA})-DP")
    
    print("\nNOTE")
    print("\nNOTE!")


if __name__ == '__main__':
    main()

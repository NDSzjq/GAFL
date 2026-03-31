"""
安全聚合模块 - 基于成对掩码的安全聚合协议
参考论文: Practical Secure Aggregation for Privacy-Preserving Machine Learning (CCS'17)

核心思想:
1. 成对掩码: 每对客户端(u,v)协商共享种子，生成掩码
   - 用户u对用户v添加掩码: +PRG(s_{u,v})
   - 用户v对用户u减去掩码: -PRG(s_{v,u}) = -PRG(s_{u,v})
   - 聚合时掩码自动抵消

2. 双重掩码结构:
   - 成对掩码: 客户端之间相互抵消
   - 自身掩码: p_u = PRG(b_u)，服务器需要收集b_u才能去除

3. 与分布式差分隐私的结合:
   - 客户端本地添加DP噪声 -> 应用掩码 -> 安全聚合
   - 服务器只能看到聚合结果，无法获取单个客户端贡献

简化假设（模拟联邦学习环境）:
- 不处理用户掉线（假设所有用户都完成训练）
- 使用模拟的密钥协商（确定性种子）
- 不需要真实的网络通信
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional
import os


class SecureAggregator:
    """
    安全聚合器 - 服务器端
    
    负责协调安全聚合协议，收集掩码后的模型更新并计算聚合结果
    """
    
    def __init__(self, num_clients: int, threshold: int = None, seed: int = 42):
        """
        初始化安全聚合器
        
        参数:
            num_clients: 客户端总数
            threshold: 秘密共享阈值（用于处理掉线，这里简化处理）
            seed: 随机种子（用于确定性密钥生成）
        """
        self.num_clients = num_clients
        # 阈值默认为客户端数的一半+1
        self.threshold = threshold if threshold else (num_clients // 2 + 1)
        self.seed = seed
        
        # 存储客户端的公钥和掩码种子
        self.client_public_keys: Dict[int, bytes] = {}
        self.client_self_mask_seeds: Dict[int, bytes] = {}  # b_u 种子
        
        # 存储客户端的成对掩码种子（用于验证）
        self.pairwise_seeds: Dict[Tuple[int, int], bytes] = {}
        

    def generate_client_keys(self, client_id: int) -> Dict:
        """
        为客户端生成密钥对和种子
        
        参数:
            client_id: 客户端ID
        
        返回:
            keys: 包含私钥、公钥和种子的字典
        """
        # 使用确定性种子生成（模拟密钥协商）
        # 在真实场景中，这应该是Diffie-Hellman密钥交换
        np.random.seed(self.seed + client_id)
        
        # 生成私钥（模拟）
        private_key = np.random.bytes(32)
        
        # 生成公钥（模拟，实际是私钥的哈希）
        public_key = hashlib.sha256(private_key).digest()
        
        # 生成自身掩码种子 b_u
        self_mask_seed = np.random.bytes(32)
        
        # 存储公钥
        self.client_public_keys[client_id] = public_key
        self.client_self_mask_seeds[client_id] = self_mask_seed
        
        return {
            'client_id': client_id,
            'private_key': private_key,
            'public_key': public_key,
            'self_mask_seed': self_mask_seed
        }
    
    def compute_pairwise_seed(self, client_u: int, client_v: int) -> bytes:
        """
        计算客户端u和v之间的共享种子
        
        参数:
            client_u: 客户端u的ID
            client_v: 客户端v的ID
        
        返回:
            shared_seed: 共享种子（32字节）
        """
        # 确保u < v，保证种子对称
        u, v = min(client_u, client_v), max(client_u, client_v)
        
        # 检查缓存
        if (u, v) in self.pairwise_seeds:
            return self.pairwise_seeds[(u, v)]
        
        # 模拟Diffie-Hellman密钥协商
        # 在真实场景中，这应该是 DH(u_private, v_public) = DH(v_private, u_public)
        # 这里使用确定性种子生成
        seed_material = f"pairwise_seed_{u}_{v}_{self.seed}".encode()
        shared_seed = hashlib.sha256(seed_material).digest()
        
        # 缓存
        self.pairwise_seeds[(u, v)] = shared_seed
        
        return shared_seed
    
    def get_all_pairwise_seeds_for_client(self, client_id: int, other_clients: List[int]) -> Dict[int, bytes]:
        """
        获取某个客户端与所有其他客户端的成对种子
        
        参数:
            client_id: 当前客户端ID
            other_clients: 其他客户端ID列表
        
        返回:
            seeds: {other_client_id: shared_seed} 字典
        """
        seeds = {}
        for other_id in other_clients:
            if other_id != client_id:
                seeds[other_id] = self.compute_pairwise_seed(client_id, other_id)
        return seeds


class SecureAggregationClient:
    """
    安全聚合客户端
    
    负责生成掩码并应用于模型更新
    """
    
    def __init__(self, client_id: int, keys: Dict, num_clients: int):
        """
        初始化安全聚合客户端
        
        参数:
            client_id: 客户端ID
            keys: 密钥字典（包含private_key, public_key, self_mask_seed）
            num_clients: 客户端总数
        """
        self.client_id = client_id
        self.private_key = keys['private_key']
        self.public_key = keys['public_key']
        self.self_mask_seed = keys['self_mask_seed']
        self.num_clients = num_clients
        
        # 存储与其他客户端的成对种子
        self.pairwise_seeds: Dict[int, bytes] = {}
    
    def set_pairwise_seeds(self, seeds: Dict[int, bytes]):
        """
        设置与其他客户端的成对种子
        
        参数:
            seeds: {other_client_id: shared_seed} 字典
        """
        self.pairwise_seeds = seeds
    
    def _prg_expand(self, seed: bytes, shape: tuple, device: str = 'cpu', scale: float = 1e-6) -> torch.Tensor:
        """
        伪随机数生成器扩展
        
        将32字节种子扩展为指定形状的张量
        
        参数:
            seed: 32字节种子
            shape: 输出张量形状
            device: 计算设备
            scale: 掩码缩放因子（控制掩码幅度，设为1e-6使掩码几乎不影响模型）
        
        返回:
            tensor: 生成的伪随机张量
        """
        # 使用SHA-256作为PRG
        # 对于大张量，需要多次哈希
        total_elements = int(np.prod(shape))
        
        # 计算需要的哈希次数（每个哈希产生32字节 = 256位）
        bytes_per_element = 4  # float32
        total_bytes = total_elements * bytes_per_element
        num_hashes = (total_bytes + 31) // 32
        
        # 生成随机字节
        random_bytes = bytearray()
        for i in range(num_hashes):
            # 使用计数器模式
            hash_input = seed + i.to_bytes(4, 'big')
            random_bytes.extend(hashlib.sha256(hash_input).digest())
        
        # 将字节解释为uint32，然后归一化
        random_array = np.frombuffer(bytes(random_bytes[:total_bytes]), dtype=np.uint32)
        # 归一化到[0, 1]范围，然后映射到[-1, 1]
        random_array = random_array.astype(np.float64) / np.iinfo(np.uint32).max
        random_array = (random_array * 2 - 1).astype(np.float32)
        
        # 缩放掩码幅度，使用极小的scale使掩码几乎不影响模型精度
        random_array = random_array * scale
        
        # 转换为PyTorch张量
        tensor = torch.from_numpy(random_array.copy()).reshape(shape).to(device)
        
        return tensor
    
    def generate_mask(self, model_params: Dict[str, torch.Tensor], 
                      participating_clients: List[int],
                      device: str = 'cpu') -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        生成掩码
        
        参数:
            model_params: 模型参数字典
            participating_clients: 参与聚合的客户端ID列表
            device: 计算设备
        
        返回:
            pairwise_mask: 成对掩码
            self_mask: 自身掩码
        """
        # 初始化掩码
        pairwise_mask = {}
        self_mask = {}
        
        # 生成成对掩码
        # y_u = x_u + p_u + Σ_{v>u} PRG(s_{u,v}) - Σ_{v<u} PRG(s_{v,u})
        for key, param in model_params.items():
            if not torch.is_floating_point(param):
                continue
            
            shape = param.shape
            
            # 成对掩码初始化为0
            pairwise_mask[key] = torch.zeros(shape, device=device, dtype=param.dtype)
            
            # 遍历所有参与客户端
            for other_id in participating_clients:
                if other_id == self.client_id:
                    continue
                
                # 获取共享种子
                if other_id not in self.pairwise_seeds:
                    continue
                
                shared_seed = self.pairwise_seeds[other_id]
                
                # 生成掩码向量
                mask_vector = self._prg_expand(shared_seed, shape, device)
                
                # 应用符号：u < v 加，u > v 减
                if self.client_id < other_id:
                    pairwise_mask[key] = pairwise_mask[key] + mask_vector.to(param.dtype)
                else:
                    pairwise_mask[key] = pairwise_mask[key] - mask_vector.to(param.dtype)
            
            # 生成自身掩码
            self_mask[key] = self._prg_expand(self.self_mask_seed, shape, device).to(param.dtype)
        
        return pairwise_mask, self_mask
    
    def apply_mask(self, model_params: Dict[str, torch.Tensor],
                   pairwise_mask: Dict[str, torch.Tensor],
                   self_mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        应用掩码到模型参数
        
        参数:
            model_params: 原始模型参数
            pairwise_mask: 成对掩码
            self_mask: 自身掩码
        
        返回:
            masked_params: 掩码后的模型参数
        """
        masked_params = {}
        
        for key, param in model_params.items():
            if key in pairwise_mask and key in self_mask:
                # 应用双重掩码: y_u = x_u + p_u + pairwise_mask
                masked_params[key] = param + self_mask[key] + pairwise_mask[key]
            else:
                # 非浮点参数直接复制
                masked_params[key] = param.clone()
        
        return masked_params
    
    def get_self_mask_seed(self) -> bytes:
        """
        获取自身掩码种子（用于服务器去除掩码）
        
        返回:
            self_mask_seed: 自身掩码种子
        """
        return self.self_mask_seed


class SecureAggregationServer:
    """
    安全聚合服务器
    
    负责收集掩码后的更新，去除掩码并计算聚合结果
    """
    
    def __init__(self, aggregator: SecureAggregator):
        """
        初始化安全聚合服务器
        
        参数:
            aggregator: 安全聚合器实例
        """
        self.aggregator = aggregator
        
        # 存储客户端的自身掩码种子
        self.client_self_mask_seeds: Dict[int, bytes] = {}
        
        # 存储客户端的成对种子（用于处理掉线，这里简化）
        self.client_pairwise_seeds: Dict[int, Dict[int, bytes]] = {}
    
    def collect_self_mask_seed(self, client_id: int, seed: bytes):
        """
        收集客户端的自身掩码种子
        
        参数:
            client_id: 客户端ID
            seed: 自身掩码种子
        """
        self.client_self_mask_seeds[client_id] = seed
    
    def collect_pairwise_seeds(self, client_id: int, seeds: Dict[int, bytes]):
        """
        收集客户端的成对种子（用于处理掉线）
        
        参数:
            client_id: 客户端ID
            seeds: 成对种子字典
        """
        self.client_pairwise_seeds[client_id] = seeds
    
    def _prg_expand(self, seed: bytes, shape: tuple, device: str = 'cpu', scale: float = 1e-6) -> torch.Tensor:
        """
        伪随机数生成器扩展（与客户端相同）
        """
        total_elements = int(np.prod(shape))
        bytes_per_element = 4
        total_bytes = total_elements * bytes_per_element
        num_hashes = (total_bytes + 31) // 32
        
        random_bytes = bytearray()
        for i in range(num_hashes):
            hash_input = seed + i.to_bytes(4, 'big')
            random_bytes.extend(hashlib.sha256(hash_input).digest())
        
        # 将字节解释为uint32数组
        random_array = np.frombuffer(bytes(random_bytes[:total_bytes]), dtype=np.uint32)
        # 归一化到[0, 1]范围，然后映射到[-1, 1]
        random_array = random_array.astype(np.float64) / np.iinfo(np.uint32).max
        random_array = random_array * 2.0 - 1.0  # 映射到[-1, 1]
        random_array = random_array.astype(np.float32)
        
        # 缩放掩码幅度，使用极小的scale使掩码几乎不影响模型精度
        random_array = random_array * scale
        
        tensor = torch.from_numpy(random_array.copy()).reshape(shape).to(device)
        
        return tensor
    
    def remove_self_masks(self, aggregated_params: Dict[str, torch.Tensor],
                          participating_clients: List[int],
                          client_weights: Dict[int, float],
                          device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        从聚合结果中去除自身掩码
        
        参数:
            aggregated_params: 聚合后的参数
            participating_clients: 参与聚合的客户端ID列表
            client_weights: 每个客户端的聚合权重 {client_id: weight}
            device: 计算设备
        
        返回:
            clean_params: 去除自身掩码后的参数
        """
        clean_params = {}
        
        for key, param in aggregated_params.items():
            if not torch.is_floating_point(param):
                clean_params[key] = param.clone()
                continue
            
            # 初始化去除掩码后的参数
            clean_params[key] = param.clone()
            
            # 减去每个客户端的加权自身掩码
            for client_id in participating_clients:
                if client_id not in self.client_self_mask_seeds:
                    continue
                if client_id not in client_weights:
                    continue
                
                seed = self.client_self_mask_seeds[client_id]
                weight = client_weights[client_id]
                self_mask = self._prg_expand(seed, param.shape, device).to(param.dtype)
                # 减去加权后的自身掩码
                clean_params[key] = clean_params[key] - weight * self_mask
        
        return clean_params
    
    def aggregate_with_secure_masking(self, 
                                       masked_updates: List[Tuple[Dict[str, torch.Tensor], int, int]],
                                       participating_clients: List[int],
                                       device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        安全聚合：聚合掩码后的更新，然后去除掩码
        
        参数:
            masked_updates: 掩码后的更新列表 [(params, num_samples, client_id), ...]
            participating_clients: 参与聚合的客户端ID列表
            device: 计算设备
        
        返回:
            aggregated_params: 聚合后的参数（已去除掩码）
        """
        if not masked_updates:
            return {}
        
        # 计算总样本数
        total_samples = sum(num_samples for _, num_samples, _ in masked_updates)
        
        # 计算每个客户端的权重
        client_weights = {}
        for params, num_samples, client_id in masked_updates:
            client_weights[client_id] = num_samples / total_samples
        
        # 加权聚合
        aggregated_params = {}
        first_params = masked_updates[0][0]
        
        for key in first_params.keys():
            if not torch.is_floating_point(first_params[key]):
                # 非浮点参数直接复制
                aggregated_params[key] = first_params[key].clone()
                continue
            
            # 初始化聚合参数
            aggregated_params[key] = torch.zeros_like(first_params[key], device=device)
            
            # 加权求和
            for params, num_samples, client_id in masked_updates:
                weight = num_samples / total_samples
                aggregated_params[key] += weight * params[key].to(device)
        
        # 去除自身掩码（考虑权重）
        clean_params = self.remove_self_masks(aggregated_params, participating_clients, client_weights, device)
        
        return clean_params



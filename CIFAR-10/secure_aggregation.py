"""
NOTE - NOTE
NOTE: Practical Secure Aggregation for Privacy-Preserving Machine Learning (CCS'17)

NOTE:
1. NOTE: NOTE(u,v)NOTE,NOTE
   - NOTEuNOTEvNOTE: +PRG(s_{u,v})
   - NOTEvNOTEuNOTE: -PRG(s_{v,u}) = -PRG(s_{u,v})
   - NOTE

2. NOTE:
   - NOTE: NOTE
   - NOTE: p_u = PRG(b_u),NOTEb_uNOTE

3. NOTE:
   - NOTEDPNOTE -> NOTE -> NOTE
   - NOTE,NOTE

NOTE(NOTE):
- NOTE(NOTE)
- NOTE(NOTE)
- NOTE
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional
import os


class SecureAggregator:
    """
    NOTE - NOTE
    
    NOTE,NOTE
    """
    
    def __init__(self, num_clients: int, threshold: int = None, seed: int = 42):
        """
        NOTE
        
        NOTE:
            num_clients: NOTE
            threshold: NOTE(NOTE,NOTE)
            seed: NOTE(NOTE)
        """
        self.num_clients = num_clients
        # NOTE+1
        self.threshold = threshold if threshold else (num_clients // 2 + 1)
        self.seed = seed
        
        # NOTE
        self.client_public_keys: Dict[int, bytes] = {}
        self.client_self_mask_seeds: Dict[int, bytes] = {}  # b_u NOTE
        
        # NOTE(NOTE)
        self.pairwise_seeds: Dict[Tuple[int, int], bytes] = {}
        

    def generate_client_keys(self, client_id: int) -> Dict:
        """
        NOTE
        
        NOTE:
            client_id: NOTEID
        
        NOTE:
            keys: NOTE,NOTE
        """
        # NOTE(NOTE)
        # NOTE,NOTEDiffie-HellmanNOTE
        np.random.seed(self.seed + client_id)
        
        # NOTE(NOTE)
        private_key = np.random.bytes(32)
        
        # NOTE(NOTE,NOTE)
        public_key = hashlib.sha256(private_key).digest()
        
        # NOTE b_u
        self_mask_seed = np.random.bytes(32)
        
        # NOTE
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
        NOTEuNOTEvNOTE
        
        NOTE:
            client_u: NOTEuNOTEID
            client_v: NOTEvNOTEID
        
        NOTE:
            shared_seed: NOTE(32NOTE)
        """
        # NOTEu < v,NOTE
        u, v = min(client_u, client_v), max(client_u, client_v)
        
        # NOTE
        if (u, v) in self.pairwise_seeds:
            return self.pairwise_seeds[(u, v)]
        
        # NOTEDiffie-HellmanNOTE
        # NOTE,NOTE DH(u_private, v_public) = DH(v_private, u_public)
        # NOTE
        seed_material = f"pairwise_seed_{u}_{v}_{self.seed}".encode()
        shared_seed = hashlib.sha256(seed_material).digest()
        
        # NOTE
        self.pairwise_seeds[(u, v)] = shared_seed
        
        return shared_seed
    
    def get_all_pairwise_seeds_for_client(self, client_id: int, other_clients: List[int]) -> Dict[int, bytes]:
        """
        NOTE
        
        NOTE:
            client_id: NOTEID
            other_clients: NOTEIDNOTE
        
        NOTE:
            seeds: {other_client_id: shared_seed} NOTE
        """
        seeds = {}
        for other_id in other_clients:
            if other_id != client_id:
                seeds[other_id] = self.compute_pairwise_seed(client_id, other_id)
        return seeds


class SecureAggregationClient:
    """
    NOTE
    
    NOTE
    """
    
    def __init__(self, client_id: int, keys: Dict, num_clients: int):
        """
        NOTE
        
        NOTE:
            client_id: NOTEID
            keys: NOTE(NOTEprivate_key, public_key, self_mask_seed)
            num_clients: NOTE
        """
        self.client_id = client_id
        self.private_key = keys['private_key']
        self.public_key = keys['public_key']
        self.self_mask_seed = keys['self_mask_seed']
        self.num_clients = num_clients
        
        # NOTE
        self.pairwise_seeds: Dict[int, bytes] = {}
    
    def set_pairwise_seeds(self, seeds: Dict[int, bytes]):
        """
        NOTE
        
        NOTE:
            seeds: {other_client_id: shared_seed} NOTE
        """
        self.pairwise_seeds = seeds
    
    def _prg_expand(self, seed: bytes, shape: tuple, device: str = 'cpu', scale: float = 1e-6) -> torch.Tensor:
        """
        NOTE
        
        NOTE32NOTE
        
        NOTE:
            seed: 32NOTE
            shape: NOTE
            device: NOTE
            scale: NOTE(NOTE,NOTE1e-6NOTE)
        
        NOTE:
            tensor: NOTE
        """
        # NOTESHA-256NOTEPRG
        # NOTE,NOTE
        total_elements = int(np.prod(shape))
        
        # NOTE(NOTE32NOTE = 256NOTE)
        bytes_per_element = 4  # float32
        total_bytes = total_elements * bytes_per_element
        num_hashes = (total_bytes + 31) // 32
        
        # NOTE
        random_bytes = bytearray()
        for i in range(num_hashes):
            # NOTE
            hash_input = seed + i.to_bytes(4, 'big')
            random_bytes.extend(hashlib.sha256(hash_input).digest())
        
        # NOTEuint32,NOTE
        random_array = np.frombuffer(bytes(random_bytes[:total_bytes]), dtype=np.uint32)
        # NOTE[0, 1]NOTE,NOTE[-1, 1]
        random_array = random_array.astype(np.float64) / np.iinfo(np.uint32).max
        random_array = (random_array * 2 - 1).astype(np.float32)
        
        # NOTE,NOTEscaleNOTE
        random_array = random_array * scale
        
        # NOTEPyTorchNOTE
        tensor = torch.from_numpy(random_array.copy()).reshape(shape).to(device)
        
        return tensor
    
    def generate_mask(self, model_params: Dict[str, torch.Tensor], 
                      participating_clients: List[int],
                      device: str = 'cpu') -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        NOTE
        
        NOTE:
            model_params: NOTE
            participating_clients: NOTEIDNOTE
            device: NOTE
        
        NOTE:
            pairwise_mask: NOTE
            self_mask: NOTE
        """
        # NOTE
        pairwise_mask = {}
        self_mask = {}
        
        # NOTE
        # y_u = x_u + p_u + Σ_{v>u} PRG(s_{u,v}) - Σ_{v<u} PRG(s_{v,u})
        for key, param in model_params.items():
            if not torch.is_floating_point(param):
                continue
            
            shape = param.shape
            
            # NOTE0
            pairwise_mask[key] = torch.zeros(shape, device=device, dtype=param.dtype)
            
            # NOTE
            for other_id in participating_clients:
                if other_id == self.client_id:
                    continue
                
                # NOTE
                if other_id not in self.pairwise_seeds:
                    continue
                
                shared_seed = self.pairwise_seeds[other_id]
                
                # NOTE
                mask_vector = self._prg_expand(shared_seed, shape, device)
                
                # NOTE:u < v NOTE,u > v NOTE
                if self.client_id < other_id:
                    pairwise_mask[key] = pairwise_mask[key] + mask_vector.to(param.dtype)
                else:
                    pairwise_mask[key] = pairwise_mask[key] - mask_vector.to(param.dtype)
            
            # NOTE
            self_mask[key] = self._prg_expand(self.self_mask_seed, shape, device).to(param.dtype)
        
        return pairwise_mask, self_mask
    
    def apply_mask(self, model_params: Dict[str, torch.Tensor],
                   pairwise_mask: Dict[str, torch.Tensor],
                   self_mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        NOTE
        
        NOTE:
            model_params: NOTE
            pairwise_mask: NOTE
            self_mask: NOTE
        
        NOTE:
            masked_params: NOTE
        """
        masked_params = {}
        
        for key, param in model_params.items():
            if key in pairwise_mask and key in self_mask:
                # NOTE: y_u = x_u + p_u + pairwise_mask
                masked_params[key] = param + self_mask[key] + pairwise_mask[key]
            else:
                # NOTE
                masked_params[key] = param.clone()
        
        return masked_params
    
    def get_self_mask_seed(self) -> bytes:
        """
        NOTE(NOTE)
        
        NOTE:
            self_mask_seed: NOTE
        """
        return self.self_mask_seed


class SecureAggregationServer:
    """
    NOTE
    
    NOTE,NOTE
    """
    
    def __init__(self, aggregator: SecureAggregator):
        """
        NOTE
        
        NOTE:
            aggregator: NOTE
        """
        self.aggregator = aggregator
        
        # NOTE
        self.client_self_mask_seeds: Dict[int, bytes] = {}
        
        # NOTE(NOTE,NOTE)
        self.client_pairwise_seeds: Dict[int, Dict[int, bytes]] = {}
    
    def collect_self_mask_seed(self, client_id: int, seed: bytes):
        """
        NOTE
        
        NOTE:
            client_id: NOTEID
            seed: NOTE
        """
        self.client_self_mask_seeds[client_id] = seed
    
    def collect_pairwise_seeds(self, client_id: int, seeds: Dict[int, bytes]):
        """
        NOTE(NOTE)
        
        NOTE:
            client_id: NOTEID
            seeds: NOTE
        """
        self.client_pairwise_seeds[client_id] = seeds
    
    def _prg_expand(self, seed: bytes, shape: tuple, device: str = 'cpu', scale: float = 1e-6) -> torch.Tensor:
        """
        NOTE(NOTE)
        """
        total_elements = int(np.prod(shape))
        bytes_per_element = 4
        total_bytes = total_elements * bytes_per_element
        num_hashes = (total_bytes + 31) // 32
        
        random_bytes = bytearray()
        for i in range(num_hashes):
            hash_input = seed + i.to_bytes(4, 'big')
            random_bytes.extend(hashlib.sha256(hash_input).digest())
        
        # NOTEuint32NOTE
        random_array = np.frombuffer(bytes(random_bytes[:total_bytes]), dtype=np.uint32)
        # NOTE[0, 1]NOTE,NOTE[-1, 1]
        random_array = random_array.astype(np.float64) / np.iinfo(np.uint32).max
        random_array = random_array * 2.0 - 1.0  # NOTE[-1, 1]
        random_array = random_array.astype(np.float32)
        
        # NOTE,NOTEscaleNOTE
        random_array = random_array * scale
        
        tensor = torch.from_numpy(random_array.copy()).reshape(shape).to(device)
        
        return tensor
    
    def remove_self_masks(self, aggregated_params: Dict[str, torch.Tensor],
                          participating_clients: List[int],
                          client_weights: Dict[int, float],
                          device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        NOTE
        
        NOTE:
            aggregated_params: NOTE
            participating_clients: NOTEIDNOTE
            client_weights: NOTE {client_id: weight}
            device: NOTE
        
        NOTE:
            clean_params: NOTE
        """
        clean_params = {}
        
        for key, param in aggregated_params.items():
            if not torch.is_floating_point(param):
                clean_params[key] = param.clone()
                continue
            
            # NOTE
            clean_params[key] = param.clone()
            
            # NOTE
            for client_id in participating_clients:
                if client_id not in self.client_self_mask_seeds:
                    continue
                if client_id not in client_weights:
                    continue
                
                seed = self.client_self_mask_seeds[client_id]
                weight = client_weights[client_id]
                self_mask = self._prg_expand(seed, param.shape, device).to(param.dtype)
                # NOTE
                clean_params[key] = clean_params[key] - weight * self_mask
        
        return clean_params
    
    def aggregate_with_secure_masking(self, 
                                       masked_updates: List[Tuple[Dict[str, torch.Tensor], int, int]],
                                       participating_clients: List[int],
                                       device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        NOTE:NOTE,NOTE
        
        NOTE:
            masked_updates: NOTE [(params, num_samples, client_id), ...]
            participating_clients: NOTEIDNOTE
            device: NOTE
        
        NOTE:
            aggregated_params: NOTE(NOTE)
        """
        if not masked_updates:
            return {}
        
        # NOTE
        total_samples = sum(num_samples for _, num_samples, _ in masked_updates)
        
        # NOTE
        client_weights = {}
        for params, num_samples, client_id in masked_updates:
            client_weights[client_id] = num_samples / total_samples
        
        # NOTE
        aggregated_params = {}
        first_params = masked_updates[0][0]
        
        for key in first_params.keys():
            if not torch.is_floating_point(first_params[key]):
                # NOTE
                aggregated_params[key] = first_params[key].clone()
                continue
            
            # NOTE
            aggregated_params[key] = torch.zeros_like(first_params[key], device=device)
            
            # NOTE
            for params, num_samples, client_id in masked_updates:
                weight = num_samples / total_samples
                aggregated_params[key] += weight * params[key].to(device)
        
        # NOTE(NOTE)
        clean_params = self.remove_self_masks(aggregated_params, participating_clients, client_weights, device)
        
        return clean_params



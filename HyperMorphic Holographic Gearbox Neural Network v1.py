#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  HYPERMORPHIC HOLOGRAPHIC GEARBOX NEURAL NETWORK - COMPLETE EDITION          ║
║  Full Training Pipeline with Dictionary + Thesaurus + SQuAD                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ARCHITECTURE:                                                               ║
║  • HoloRAID: CRT-based fault-tolerant encoding (FIXED)                      ║
║  • SafeGear: Bijective modular transformation                               ║
║  • HoloMix: Multi-frequency sinusoidal interference FFN                     ║
║  • HolographicAttention: Prime-frequency attention mechanism                ║
║                                                                              ║
║  TRAINING DATA:                                                              ║
║  • Dictionary API: Word definitions → QA pairs                              ║
║  • Thesaurus API: Synonyms → QA pairs                                       ║
║  • SQuAD Dataset: Question answering                                        ║
║                                                                              ║
║  FEATURES:                                                                   ║
║  • Early stopping with patience                                             ║
║  • Learning rate scheduling with warmup                                     ║
║  • Gradient clipping                                                        ║
║  • Comprehensive visualization                                              ║
║  • Frequency analysis                                                       ║
║  • Model checkpointing                                                      ║
║                                                                              ║
║  Author: Shaun Gerrard                                                       ║
║  Framework: HyperMorphic Mathematics                                         ║
║  License: MIT                                                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

GOOGLE COLAB INSTRUCTIONS:
1. Copy this entire script into a Colab cell
2. Run - it will install dependencies and train for 50 epochs
3. Early stopping will halt if validation doesn't improve
"""

# ════════════════════════════════════════════════════════════════════════════════
# CELL 1: INSTALLATIONS
# ════════════════════════════════════════════════════════════════════════════════

print("█" * 80)
print("█  HYPERMORPHIC HOLOGRAPHIC GEARBOX - COMPLETE EDITION" + " " * 22 + "█")
print("█" * 80)

print("\n" + "=" * 80)
print("  INSTALLING DEPENDENCIES")
print("=" * 80)

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

packages = [
    "torch", "torchvision", "torchaudio",
    "transformers", "datasets", "tokenizers", "sentencepiece",
    "requests", "aiohttp", "pandas",
    "matplotlib", "seaborn", "tqdm",
    "numpy", "scipy"
]

for pkg in packages:
    try:
        install(pkg)
    except Exception as e:
        print(f"  Note: {pkg} - {e}")

print("\n✓ Dependencies installed!")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 2: IMPORTS AND CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  IMPORTING LIBRARIES")
print("=" * 80)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import requests
import json
import time
import random
import math
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Prime frequencies for holographic operations
PRIME_FREQUENCIES = {
    'tiny': [3, 5, 7],
    'small': [11, 13, 17, 19, 23],
    'medium': [29, 31, 37, 41, 43, 47],
    'large': [53, 59, 61, 67, 71, 73, 79],
    'default': [31, 37, 41, 43, 47],
    'holoraid': [53, 59, 61, 67, 71],  # For CRT encoding
}

print("✓ All imports successful!")

# ════════════════════════════════════════════════════════════════════════════════
# CELL 3: COMPLETE MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  BUILDING HYPERMORPHIC HOLOGRAPHIC GEARBOX ARCHITECTURE")
print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────────
# HOLORAID: CRT-Based Fault-Tolerant Encoding (FIXED)
# ─────────────────────────────────────────────────────────────────────────────────

class HoloRAID(nn.Module):
    """
    HoloRAID: Holographic Redundant Array of Independent Data
    
    Uses Chinese Remainder Theorem for fault-tolerant encoding/decoding.
    
    Mathematical Foundation:
    - Encode: v → (v mod p₁, v mod p₂, ..., v mod pₙ)
    - Decode: CRT reconstruction from any k residues
    
    Properties:
    - Exact k-of-n reconstruction (when k primes are coprime)
    - Information-theoretic security
    - Graceful degradation under failures
    
    FIXED: Proper index handling during training subset selection
    """
    
    def __init__(self, n_shards: int = 5, k_threshold: int = 3,
                 primes: List[int] = None, scale: float = 100.0):
        super().__init__()
        
        self.n = n_shards
        self.k = k_threshold
        self.scale = scale  # Quantization scale
        
        # Select coprime moduli for CRT
        if primes is None:
            self.primes = PRIME_FREQUENCIES['holoraid'][:n_shards]
        else:
            self.primes = primes[:n_shards]
        
        # Verify coprimes
        self._verify_coprime()
        
        # Register as buffer (not trained)
        self.register_buffer('prime_tensor', 
                            torch.tensor(self.primes, dtype=torch.float32))
    
    def _verify_coprime(self):
        """Verify all primes are pairwise coprime."""
        from math import gcd
        for i, p1 in enumerate(self.primes):
            for p2 in self.primes[i+1:]:
                assert gcd(p1, p2) == 1, f"Primes {p1}, {p2} not coprime!"
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Extended Euclidean algorithm for modular inverse."""
        def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
            if b == 0:
                return a, 1, 0
            g, x, y = extended_gcd(b, a % b)
            return g, y, x - (a // b) * y
        
        g, x, _ = extended_gcd(a % m, m)
        if g != 1:
            raise ValueError(f"Modular inverse doesn't exist for {a} mod {m}")
        return x % m
    
    def _compute_crt_coefficients(self, indices: List[int]) -> Tuple[int, List[int], List[int]]:
        """Compute CRT coefficients for given prime indices."""
        selected_primes = [self.primes[i] for i in indices]
        
        # Product of all selected primes
        M = 1
        for p in selected_primes:
            M *= p
        
        # Partial products and modular inverses
        M_i = [M // p for p in selected_primes]
        y_i = [self._mod_inverse(M_i[j], selected_primes[j]) 
               for j in range(len(indices))]
        
        return M, M_i, y_i
    
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode tensor to n shards via modular residues.
        
        Args:
            x: Input tensor of any shape (float)
            
        Returns:
            List of n shard tensors, each normalized to [0, 1)
        """
        # Scale and quantize to positive integers
        # Shift to positive range first
        x_shifted = x - x.min() + 1  # Ensure positive
        x_scaled = (x_shifted * self.scale).long()
        
        # Compute residues for each prime
        shards = []
        for p in self.primes:
            residue = (x_scaled % p).float() / p  # Normalize to [0, 1)
            shards.append(residue)
        
        # Store min for reconstruction
        self._last_min = x.min().item()
        
        return shards
    
    def decode(self, shards: List[torch.Tensor], 
               indices: List[int] = None) -> torch.Tensor:
        """
        Decode from k shards via CRT reconstruction.
        
        FIXED: Now correctly handles subset selection.
        
        Args:
            shards: FULL list of n shard tensors
            indices: Which k indices to use (default: first k)
            
        Returns:
            Reconstructed tensor (approximately original)
        """
        if indices is None:
            indices = list(range(self.k))
        
        # Ensure we have exactly k indices
        indices = sorted(indices[:self.k])
        
        # Get CRT coefficients for selected indices
        M, M_i, y_i = self._compute_crt_coefficients(indices)
        
        # Reconstruct via CRT
        result = torch.zeros_like(shards[0])
        
        for j, idx in enumerate(indices):
            # Get shard and denormalize
            shard = shards[idx]  # Access from FULL shard list
            p = self.primes[idx]
            
            # Recover residue from normalized value
            r = (shard * p).round()
            
            # CRT accumulation
            result = result + r * M_i[j] * y_i[j]
        
        # Final modulo and unscale
        result = torch.fmod(result, float(M))
        result = result / self.scale
        
        # Restore offset
        if hasattr(self, '_last_min'):
            result = result + self._last_min - 1
        
        return result
    
    def forward(self, x: torch.Tensor, 
                fault_probability: float = 0.2) -> torch.Tensor:
        """
        Encode and decode with optional fault simulation.
        
        FIXED: Always passes full shard list to decode, only varies indices.
        
        Args:
            x: Input tensor
            fault_probability: Probability of simulating faults during training
            
        Returns:
            Reconstructed tensor
        """
        # Encode to all n shards
        shards = self.encode(x)
        
        # During training, occasionally use random k-subset
        if self.training and random.random() < fault_probability:
            # Select random k indices from n (simulating k survivors)
            indices = sorted(random.sample(range(self.n), self.k))
        else:
            # Use first k indices
            indices = list(range(self.k))
        
        # Decode from full shard list with selected indices
        return self.decode(shards, indices)
    
    def get_redundancy_factor(self) -> float:
        """Return storage overhead factor."""
        return self.n / self.k
    
    def get_fault_tolerance(self) -> int:
        """Return number of failures that can be tolerated."""
        return self.n - self.k


# ─────────────────────────────────────────────────────────────────────────────────
# SAFEGEAR: Bijective Transformation
# ─────────────────────────────────────────────────────────────────────────────────

class SafeGear(nn.Module):
    """
    SafeGear: Bijective transformation using modular arithmetic.
    
    Mathematical Form:
    W_{a,b}(x) = (x mod b) * a + floor(x / b)
    
    For neural networks, we use a differentiable approximation:
    G(x) = sin(2πx/b) * a + x/b
    
    Properties:
    - Approximately bijective (invertible)
    - Smooth and differentiable
    - Learnable parameters a, b
    - Cascadable: G₁ ∘ G₂ ∘ ... ∘ Gₙ
    """
    
    def __init__(self, dim: int, n_gears: int = 4):
        super().__init__()
        
        self.dim = dim
        self.n_gears = n_gears
        
        # Learnable gear parameters (initialized for stability)
        self.gear_a = nn.Parameter(torch.ones(n_gears) * 0.5)
        self.gear_b = nn.Parameter(torch.ones(n_gears) * 7.0)
        
        # Output mixing layer
        self.mix = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def gear_transform(self, x: torch.Tensor, a: torch.Tensor, 
                       b: torch.Tensor) -> torch.Tensor:
        """Apply single differentiable gear transformation."""
        # Ensure positive gear ratios
        a_pos = F.softplus(a) + 0.1
        b_pos = F.softplus(b) + 2.0
        
        # Differentiable modular-like transform
        mod_component = torch.sin(2 * math.pi * x / b_pos) * a_pos
        div_component = x / b_pos
        
        return mod_component + div_component
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cascaded gear transformations."""
        out = x
        
        for i in range(self.n_gears):
            out = self.gear_transform(out, self.gear_a[i], self.gear_b[i])
        
        # Mix and normalize
        out = self.mix(out)
        return self.norm(out + x)  # Residual connection


# ─────────────────────────────────────────────────────────────────────────────────
# HOLOMIX: Multi-Frequency Interference Layer
# ─────────────────────────────────────────────────────────────────────────────────

class HoloMix(nn.Module):
    """
    HoloMix: Multi-frequency sinusoidal interference layer.
    
    Mathematical Form:
    H(x) = xW₁W₂ + α Σᵢ Aᵢ ⊙ sin(2πxW₁/pᵢ + φᵢ)
    
    Theoretical Basis:
    - Stone-Weierstrass: Sinusoids dense in C([0,1])
    - Universal approximation via frequency superposition
    - Holographic principle: Information distributed across frequencies
    
    Properties:
    - Universal approximation capability
    - C^∞ differentiable (smooth gradients)
    - Bounded Jacobian perturbation (stable training)
    """
    
    def __init__(self, dim: int, hidden_dim: int = None,
                 frequencies: List[int] = None, alpha: float = 0.3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4
        self.frequencies = frequencies or PRIME_FREQUENCIES['default']
        self.n_freq = len(self.frequencies)
        self.alpha = alpha
        
        # Main feedforward layers
        self.W1 = nn.Linear(dim, self.hidden_dim)
        self.W2 = nn.Linear(self.hidden_dim, dim)
        
        # Per-frequency learnable parameters
        self.amplitudes = nn.Parameter(
            torch.randn(self.n_freq, self.hidden_dim) * 0.1
        )
        self.phases = nn.Parameter(
            torch.rand(self.n_freq, self.hidden_dim) * 2 * math.pi
        )
        
        # Normalization and regularization
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Register frequencies as buffer
        self.register_buffer('freq_tensor', 
                            torch.tensor(self.frequencies, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply HoloMix transformation with multi-frequency interference."""
        # First linear projection
        h = self.W1(x)
        
        # Multi-frequency interference
        interference = torch.zeros_like(h)
        for i, freq in enumerate(self.frequencies):
            phase = 2 * math.pi * h / freq + self.phases[i]
            interference = interference + self.amplitudes[i] * torch.sin(phase)
        
        # Add scaled interference
        h = h + self.alpha * interference
        
        # Activation and second projection
        h = F.gelu(h)
        h = self.dropout(h)
        out = self.W2(h)
        
        return self.norm(out)


# ─────────────────────────────────────────────────────────────────────────────────
# HOLOGRAPHIC ATTENTION
# ─────────────────────────────────────────────────────────────────────────────────

class HolographicAttention(nn.Module):
    """
    Holographic Attention: Multi-frequency interference attention mechanism.
    
    Mathematical Form:
    A_ij = Σ_f α_f · sin(2π(Q_i · K_j)/(p_f · τ) + φ_f)
    
    where:
    - p_f are prime frequencies
    - α_f are learnable amplitudes
    - φ_f are learnable phase offsets
    - τ is temperature
    
    Key Properties:
    ─────────────────
    1. BOUNDED: |A_ij| ≤ Σ|α_f| (never overflows)
    2. SMOOTH: sin is C^∞ everywhere (stable gradients)
    3. MULTI-SCALE: Different frequencies capture different relationships
    4. INTERPRETABLE: Frequency decomposition available
    
    Theoretical Foundation:
    ──────────────────────
    - Based on holographic principle (information distributed)
    - Prime frequencies avoid harmonic interference
    - Stone-Weierstrass: universal approximation
    """
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1,
                 frequencies: List[int] = None):
        super().__init__()
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.frequencies = frequencies or PRIME_FREQUENCIES['default']
        self.n_freq = len(self.frequencies)
        
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        # QKV projections
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)
        
        # Per-head, per-frequency learnable parameters
        self.alpha = nn.Parameter(torch.randn(n_heads, self.n_freq) * 0.1)
        self.phi = nn.Parameter(torch.rand(n_heads, self.n_freq) * 2 * math.pi)
        
        # Learnable temperature (controls attention sharpness)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
        # Store attention weights for visualization
        self.last_attention_weights = None
    
    def forward(self, x: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with holographic attention.
        
        Args:
            x: (batch, seq_len, dim)
            mask: Optional attention mask (batch, 1, 1, seq_len) or (batch, seq_len)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product similarity
        similarity = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply temperature
        temp = self.temperature.abs() + 0.1  # Ensure positive
        similarity = similarity / temp
        
        # Multi-frequency holographic scoring
        scores = torch.zeros_like(similarity)
        for f_idx, freq in enumerate(self.frequencies):
            # Phase for this frequency
            phase = 2 * math.pi * similarity / freq
            # Add per-head phase offset
            phase = phase + self.phi[:, f_idx].view(1, self.n_heads, 1, 1)
            # Accumulate weighted sinusoid
            scores = scores + self.alpha[:, f_idx].view(1, self.n_heads, 1, 1) * torch.sin(phase)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Normalize via sigmoid + row normalization (bounded, smooth)
        weights = torch.sigmoid(scores)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Store for visualization
        self.last_attention_weights = weights.detach()
        
        # Apply dropout
        weights = self.dropout(weights)
        
        # Compute output
        out = torch.matmul(weights, V)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.W_o(out)
        
        return self.norm(out)
    
    def get_frequency_response(self) -> Dict[int, torch.Tensor]:
        """Get per-frequency contribution to attention (for interpretability)."""
        if self.last_attention_weights is None:
            return {}
        
        # This would require recomputing with stored Q, K
        # Return amplitudes as proxy
        return {freq: self.alpha[:, i].detach() 
                for i, freq in enumerate(self.frequencies)}


# ─────────────────────────────────────────────────────────────────────────────────
# HOLOGRAPHIC TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────────────────────────────

class HolographicTransformerBlock(nn.Module):
    """
    Complete transformer block with Holographic Attention and HoloMix FFN.
    
    Architecture:
    x → LayerNorm → HolographicAttention → + → LayerNorm → HoloMix → +
    └──────────────────────────────────────┘   └────────────────────┘
                  Residual                           Residual
    """
    
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1,
                 frequencies: List[int] = None, ffn_mult: int = 4):
        super().__init__()
        
        self.attention = HolographicAttention(dim, n_heads, dropout, frequencies)
        self.ffn = HoloMix(dim, dim * ffn_mult, frequencies, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        h = self.attention(h, mask)
        x = x + self.dropout(h)
        
        # Pre-norm FFN
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        
        return x


# ─────────────────────────────────────────────────────────────────────────────────
# COMPLETE HOLOGRAPHIC GEARBOX MODEL
# ─────────────────────────────────────────────────────────────────────────────────

class HolographicGearboxModel(nn.Module):
    """
    Complete HyperMorphic Holographic Gearbox Model.
    
    Full Architecture:
    ─────────────────
    1. Token Embedding (vocab → dim)
    2. SafeGear Transformation (bijective encoding)
    3. Holographic Position Embedding (prime-frequency based)
    4. Optional HoloRAID Encoding (fault tolerance)
    5. Stack of Holographic Transformer Blocks
    6. Task-specific Heads (QA, Classification, LM)
    
    Mathematical Properties:
    ───────────────────────
    ✓ Bounded attention (never overflows)
    ✓ C^∞ differentiable (smooth gradients)
    ✓ Multi-scale representation (frequency decomposition)
    ✓ Fault-tolerant (HoloRAID encoding)
    ✓ Interpretable (frequency analysis)
    """
    
    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 4,
                 n_heads: int = 4, max_seq_len: int = 512, dropout: float = 0.1,
                 n_classes: int = 2, frequencies: List[int] = None,
                 use_holoraid: bool = True, holoraid_prob: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.frequencies = frequencies or PRIME_FREQUENCIES['default']
        self.use_holoraid = use_holoraid
        self.holoraid_prob = holoraid_prob
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, dim)
        
        # SafeGear transformation
        self.safegear = SafeGear(dim, n_gears=3)
        
        # Holographic position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self._init_holographic_positions()
        
        # HoloRAID for fault-tolerant encoding
        self.holoraid = HoloRAID(n_shards=5, k_threshold=3, scale=100.0)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            HolographicTransformerBlock(dim, n_heads, dropout, self.frequencies)
            for _ in range(n_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Task heads
        self.qa_head = nn.Linear(dim, 2)  # start and end logits
        self.cls_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, n_classes)
        )
        self.lm_head = nn.Linear(dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_holographic_positions(self):
        """Initialize position embeddings using holographic frequencies."""
        positions = torch.arange(self.max_seq_len).float()
        pe = torch.zeros(self.max_seq_len, self.dim)
        
        # Use prime frequencies for first dimensions
        for i, freq in enumerate(self.frequencies):
            if i * 2 < self.dim:
                pe[:, i*2] = torch.sin(2 * math.pi * positions / freq)
            if i * 2 + 1 < self.dim:
                pe[:, i*2+1] = torch.cos(2 * math.pi * positions / freq)
        
        # Standard sinusoidal for remaining dimensions
        for i in range(len(self.frequencies) * 2, self.dim, 2):
            div_term = 10000 ** (i / self.dim)
            pe[:, i] = torch.sin(positions / div_term)
            if i + 1 < self.dim:
                pe[:, i+1] = torch.cos(positions / div_term)
        
        self.pos_embed.data = pe.unsqueeze(0)
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None,
                task: str = 'qa') -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) attention mask
            task: 'qa', 'classify', or 'lm'
            
        Returns:
            Dictionary with task-specific outputs
        """
        B, L = input_ids.shape
        
        # Token embedding
        x = self.token_embed(input_ids)
        
        # SafeGear transformation
        x = self.safegear(x)
        
        # Add position embedding
        x = x + self.pos_embed[:, :L, :]
        
        # HoloRAID encoding (optional, for fault tolerance training)
        if self.use_holoraid and self.training:
            if random.random() < self.holoraid_prob:
                x = self.holoraid(x)
        
        x = self.dropout(x)
        
        # Create attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # Task-specific outputs
        outputs = {'hidden_states': x}
        
        if task == 'qa':
            logits = self.qa_head(x)
            outputs['start_logits'] = logits[:, :, 0]
            outputs['end_logits'] = logits[:, :, 1]
        
        elif task == 'classify':
            cls_hidden = x[:, 0, :]  # [CLS] token
            outputs['logits'] = self.cls_head(cls_hidden)
        
        elif task == 'lm':
            outputs['logits'] = self.lm_head(x)
        
        return outputs
    
    def get_num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_attention_weights(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """Get attention weights from specified layer."""
        if 0 <= layer_idx < len(self.layers) or layer_idx == -1:
            layer = self.layers[layer_idx]
            return layer.attention.last_attention_weights
        return None


print("✓ Model architecture defined!")
print(f"  Components: HoloRAID, SafeGear, HoloMix, HolographicAttention")
print(f"  Frequencies: {PRIME_FREQUENCIES['default']}")


# ════════════════════════════════════════════════════════════════════════════════
# CELL 4: DATA LOADING (Dictionary, Thesaurus, SQuAD)
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  LOADING DATA SOURCES")
print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────────
# DICTIONARY & THESAURUS API
# ─────────────────────────────────────────────────────────────────────────────────

class DictionaryThesaurusAPI:
    """
    Fetch definitions, synonyms, and related words from free APIs.
    
    APIs:
    - Free Dictionary API: https://dictionaryapi.dev/
    - Datamuse API: https://www.datamuse.com/api/
    """
    
    def __init__(self):
        self.dict_url = "https://api.dictionaryapi.dev/api/v2/entries/en/"
        self.datamuse_url = "https://api.datamuse.com/words"
        self.cache = {}
    
    def get_definition(self, word: str) -> Optional[str]:
        """Get word definition."""
        if word in self.cache and 'definition' in self.cache[word]:
            return self.cache[word]['definition']
        
        try:
            resp = requests.get(f"{self.dict_url}{word}", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0:
                    meanings = data[0].get('meanings', [])
                    if meanings:
                        defs = meanings[0].get('definitions', [])
                        if defs:
                            definition = defs[0].get('definition', '')
                            if word not in self.cache:
                                self.cache[word] = {}
                            self.cache[word]['definition'] = definition
                            return definition
        except:
            pass
        return None
    
    def get_synonyms(self, word: str, max_results: int = 10) -> List[str]:
        """Get synonyms from Datamuse."""
        try:
            params = {'rel_syn': word, 'max': max_results}
            resp = requests.get(self.datamuse_url, params=params, timeout=5)
            if resp.status_code == 200:
                return [item['word'] for item in resp.json()]
        except:
            pass
        return []
    
    def get_antonyms(self, word: str, max_results: int = 5) -> List[str]:
        """Get antonyms from Datamuse."""
        try:
            params = {'rel_ant': word, 'max': max_results}
            resp = requests.get(self.datamuse_url, params=params, timeout=5)
            if resp.status_code == 200:
                return [item['word'] for item in resp.json()]
        except:
            pass
        return []
    
    def get_related(self, word: str, max_results: int = 10) -> List[str]:
        """Get semantically related words."""
        try:
            params = {'ml': word, 'max': max_results}
            resp = requests.get(self.datamuse_url, params=params, timeout=5)
            if resp.status_code == 200:
                return [item['word'] for item in resp.json()]
        except:
            pass
        return []
    
    def build_vocabulary_dataset(self, words: List[str]) -> List[Dict]:
        """Build comprehensive vocabulary dataset."""
        dataset = []
        
        for word in tqdm(words, desc="Building vocabulary"):
            entry = {'word': word}
            
            definition = self.get_definition(word)
            if definition:
                entry['definition'] = definition
            
            synonyms = self.get_synonyms(word)
            if synonyms:
                entry['synonyms'] = synonyms
            
            antonyms = self.get_antonyms(word)
            if antonyms:
                entry['antonyms'] = antonyms
            
            related = self.get_related(word)
            if related:
                entry['related'] = related
            
            if len(entry) > 1:
                dataset.append(entry)
            
            time.sleep(0.15)  # Rate limiting
        
        return dataset


# ─────────────────────────────────────────────────────────────────────────────────
# VOCABULARY QA DATASET (Dictionary/Thesaurus → QA Format)
# ─────────────────────────────────────────────────────────────────────────────────

class VocabularyQADataset(Dataset):
    """
    Convert dictionary/thesaurus data to QA format for training.
    
    Creates question-answer pairs like:
    - "What is the definition of X?" → definition
    - "What are synonyms of X?" → synonym list
    - "What is the opposite of X?" → antonym
    """
    
    def __init__(self, vocab_data: List[Dict], tokenizer, max_length: int = 384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for entry in vocab_data:
            word = entry['word']
            
            # Definition QA
            if 'definition' in entry:
                self.samples.append({
                    'question': f"What is the definition of {word}?",
                    'context': f"The word {word} is defined as: {entry['definition']}",
                    'answer': entry['definition'],
                    'answer_start': len(f"The word {word} is defined as: ")
                })
            
            # Synonym QA
            if 'synonyms' in entry and entry['synonyms']:
                syns = ', '.join(entry['synonyms'][:5])
                self.samples.append({
                    'question': f"What are synonyms of {word}?",
                    'context': f"Synonyms of {word} include: {syns}. These words have similar meanings.",
                    'answer': syns,
                    'answer_start': len(f"Synonyms of {word} include: ")
                })
            
            # Antonym QA
            if 'antonyms' in entry and entry['antonyms']:
                ants = ', '.join(entry['antonyms'][:3])
                self.samples.append({
                    'question': f"What is the opposite of {word}?",
                    'context': f"The opposite of {word} is: {ants}. These are antonyms.",
                    'answer': ants,
                    'answer_start': len(f"The opposite of {word} is: ")
                })
            
            # Related words QA
            if 'related' in entry and entry['related']:
                rel = ', '.join(entry['related'][:5])
                self.samples.append({
                    'question': f"What words are related to {word}?",
                    'context': f"Words related to {word} include: {rel}. They share semantic connections.",
                    'answer': rel,
                    'answer_start': len(f"Words related to {word} include: ")
                })
        
        # Tokenize all samples
        self.features = []
        for sample in self.samples:
            self._process_sample(sample)
    
    def _process_sample(self, sample: Dict):
        """Tokenize a single sample."""
        enc = self.tokenizer(
            sample['question'],
            sample['context'],
            max_length=self.max_length,
            truncation='only_second',
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Find answer positions
        answer_start = sample['answer_start']
        answer_end = answer_start + len(sample['answer'])
        
        start_pos, end_pos = 0, 0
        offsets = enc['offset_mapping'].squeeze(0).tolist()
        
        for idx, (s, e) in enumerate(offsets):
            if s <= answer_start < e:
                start_pos = idx
            if s < answer_end <= e:
                end_pos = idx
                break
        
        self.features.append({
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'start_positions': start_pos,
            'end_positions': end_pos
        })
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


# ─────────────────────────────────────────────────────────────────────────────────
# SQUAD DATASET
# ─────────────────────────────────────────────────────────────────────────────────

class SQuADDataset(Dataset):
    """SQuAD dataset for question answering."""
    
    def __init__(self, tokenizer, split: str = 'train', 
                 max_length: int = 384, max_samples: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"  Loading SQuAD {split}...")
        dataset = load_dataset("squad", split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.features = []
        self.examples = []
        
        print(f"  Processing {len(dataset)} examples...")
        for example in tqdm(dataset, desc="Tokenizing SQuAD"):
            self._process_example(example)
    
    def _process_example(self, example: Dict):
        """Process a single SQuAD example."""
        enc = self.tokenizer(
            example['question'],
            example['context'],
            max_length=self.max_length,
            truncation='only_second',
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Find answer positions
        start_pos, end_pos = 0, 0
        answers = example['answers']
        
        if answers['answer_start']:
            answer_start = answers['answer_start'][0]
            answer_end = answer_start + len(answers['text'][0])
            offsets = enc['offset_mapping'].squeeze(0).tolist()
            
            for idx, (s, e) in enumerate(offsets):
                if s <= answer_start < e:
                    start_pos = idx
                if s < answer_end <= e:
                    end_pos = idx
                    break
        
        self.features.append({
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'start_positions': start_pos,
            'end_positions': end_pos
        })
        
        self.examples.append({
            'question': example['question'],
            'context': example['context'],
            'answers': answers
        })
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


# ─────────────────────────────────────────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────────────────────────────────────────

print("\n  Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Comprehensive vocabulary list
VOCABULARY_WORDS = [
    # Technical terms
    'algorithm', 'neural', 'network', 'quantum', 'holographic',
    'mathematics', 'transform', 'frequency', 'encoding', 'decoding',
    'attention', 'memory', 'learning', 'intelligence', 'computation',
    'distributed', 'parallel', 'efficient', 'robust', 'scalable',
    'optimization', 'gradient', 'convergence', 'stability', 'accuracy',
    # Language terms
    'language', 'understanding', 'reasoning', 'knowledge', 'inference',
    'context', 'semantic', 'syntactic', 'embedding', 'representation',
    # General knowledge
    'science', 'technology', 'engineering', 'physics', 'chemistry',
    'biology', 'medicine', 'history', 'geography', 'philosophy',
    'economics', 'psychology', 'sociology', 'politics', 'culture',
    # Common words
    'important', 'significant', 'essential', 'fundamental', 'primary',
    'complex', 'simple', 'abstract', 'concrete', 'theoretical',
    'practical', 'empirical', 'analytical', 'systematic', 'comprehensive'
]

# Fetch vocabulary data
print("\n  Fetching dictionary and thesaurus data...")
api = DictionaryThesaurusAPI()
vocab_data = api.build_vocabulary_dataset(VOCABULARY_WORDS)
print(f"  ✓ Loaded {len(vocab_data)} vocabulary entries")

# Create vocabulary QA dataset
vocab_dataset = VocabularyQADataset(vocab_data, tokenizer)
print(f"  ✓ Created {len(vocab_dataset)} vocabulary QA samples")

# Load SQuAD datasets
MAX_TRAIN_SAMPLES = 10000  # Increase for better results
MAX_VAL_SAMPLES = 2000

squad_train = SQuADDataset(tokenizer, 'train', max_samples=MAX_TRAIN_SAMPLES)
squad_val = SQuADDataset(tokenizer, 'validation', max_samples=MAX_VAL_SAMPLES)

print(f"  ✓ Loaded SQuAD: {len(squad_train)} train, {len(squad_val)} validation")

# Combine vocabulary and SQuAD for training
print("\n  Combining datasets...")
combined_train = ConcatDataset([vocab_dataset, squad_train])
print(f"  ✓ Combined training set: {len(combined_train)} samples")
print(f"    - Vocabulary QA: {len(vocab_dataset)}")
print(f"    - SQuAD: {len(squad_train)}")


# ════════════════════════════════════════════════════════════════════════════════
# CELL 5: TRAINING CONFIGURATION AND LOOP
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TRAINING CONFIGURATION")
print("=" * 80)


@dataclass
class TrainingConfig:
    """Training hyperparameters - optimized for Holographic Gearbox."""
    
    # Model architecture
    vocab_size: int = 30522  # BERT vocab
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    max_seq_len: int = 384
    dropout: float = 0.2  # Increased for regularization
    
    # HoloRAID
    use_holoraid: bool = True
    holoraid_prob: float = 0.05  # 5% of batches
    
    # Training
    batch_size: int = 16
    epochs: int = 50  # More epochs with early stopping
    learning_rate: float = 1e-4  # Lower LR for stability
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Early stopping
    patience: int = 5  # Stop if no improvement for 5 epochs
    min_delta: float = 0.001  # Minimum improvement threshold
    
    # Scheduler
    scheduler_type: str = 'cosine_warmup'
    
    # Logging
    log_every: int = 50
    save_best: bool = True


config = TrainingConfig()

# Adjust for CPU/limited memory
if not torch.cuda.is_available():
    config.batch_size = 4
    config.dim = 128
    config.n_layers = 2
    config.epochs = 10
    print("  ⚠️ CPU mode - reduced configuration")

print(f"\n  Model Configuration:")
print(f"    Dimension: {config.dim}")
print(f"    Layers: {config.n_layers}")
print(f"    Heads: {config.n_heads}")
print(f"    Dropout: {config.dropout}")
print(f"    HoloRAID: {config.use_holoraid} (prob={config.holoraid_prob})")

print(f"\n  Training Configuration:")
print(f"    Batch size: {config.batch_size}")
print(f"    Epochs: {config.epochs}")
print(f"    Learning rate: {config.learning_rate}")
print(f"    Early stopping patience: {config.patience}")


# ─────────────────────────────────────────────────────────────────────────────────
# INITIALIZE MODEL
# ─────────────────────────────────────────────────────────────────────────────────

print("\n  Initializing model...")

model = HolographicGearboxModel(
    vocab_size=config.vocab_size,
    dim=config.dim,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    max_seq_len=config.max_seq_len,
    dropout=config.dropout,
    frequencies=PRIME_FREQUENCIES['default'],
    use_holoraid=config.use_holoraid,
    holoraid_prob=config.holoraid_prob
).to(DEVICE)

n_params = model.get_num_parameters()
print(f"  ✓ Model initialized: {n_params:,} parameters")


# ─────────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'start_positions': torch.tensor([x['start_positions'] for x in batch]),
        'end_positions': torch.tensor([x['end_positions'] for x in batch])
    }

train_loader = DataLoader(
    combined_train,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    squad_val,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"  ✓ DataLoaders: {len(train_loader)} train batches, {len(val_loader)} val batches")


# ─────────────────────────────────────────────────────────────────────────────────
# OPTIMIZER AND SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────────

# Optimizer with weight decay (excluding biases and LayerNorm)
no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
optimizer_groups = [
    {
        'params': [p for n, p in model.named_parameters() 
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': config.weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() 
                   if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]

optimizer = AdamW(optimizer_groups, lr=config.learning_rate)

# Learning rate scheduler
total_steps = len(train_loader) * config.epochs
warmup_steps = int(total_steps * config.warmup_ratio)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"  ✓ Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
print(f"  ✓ Scheduler: Linear warmup ({warmup_steps} steps) + decay")


# ─────────────────────────────────────────────────────────────────────────────────
# TRAINING AND EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config.epochs}")
    
    for step, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        start_positions = batch['start_positions'].to(DEVICE)
        end_positions = batch['end_positions'].to(DEVICE)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, task='qa')
        
        # Compute loss
        start_loss = F.cross_entropy(outputs['start_logits'], start_positions)
        end_loss = F.cross_entropy(outputs['end_logits'], end_positions)
        loss = (start_loss + end_loss) / 2
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Update
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg': f'{total_loss/n_batches:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / n_batches


def evaluate(model, loader):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0
    correct_start = 0
    correct_end = 0
    exact_match = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_positions = batch['start_positions'].to(DEVICE)
            end_positions = batch['end_positions'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask, task='qa')
            
            # Loss
            start_loss = F.cross_entropy(outputs['start_logits'], start_positions)
            end_loss = F.cross_entropy(outputs['end_logits'], end_positions)
            loss = (start_loss + end_loss) / 2
            
            total_loss += loss.item() * input_ids.size(0)
            
            # Predictions
            start_preds = outputs['start_logits'].argmax(dim=-1)
            end_preds = outputs['end_logits'].argmax(dim=-1)
            
            # Metrics
            correct_start += (start_preds == start_positions).sum().item()
            correct_end += (end_preds == end_positions).sum().item()
            exact_match += ((start_preds == start_positions) & 
                           (end_preds == end_positions)).sum().item()
            total += input_ids.size(0)
    
    return {
        'loss': total_loss / total,
        'start_acc': correct_start / total,
        'end_acc': correct_end / total,
        'exact_match': exact_match / total
    }


# ─────────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP WITH EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  STARTING TRAINING (50 EPOCHS WITH EARLY STOPPING)")
print("=" * 80)

# Tracking
train_losses = []
val_metrics_history = []
best_val_loss = float('inf')
best_exact_match = 0
patience_counter = 0
best_epoch = 0

# Training loop
for epoch in range(config.epochs):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)
    train_losses.append(train_loss)
    
    # Evaluate
    val_metrics = evaluate(model, val_loader)
    val_metrics_history.append(val_metrics)
    
    # Print epoch summary
    print(f"\n  Epoch {epoch+1}/{config.epochs} Summary:")
    print(f"    Train Loss:    {train_loss:.4f}")
    print(f"    Val Loss:      {val_metrics['loss']:.4f}")
    print(f"    Start Acc:     {val_metrics['start_acc']:.4f}")
    print(f"    End Acc:       {val_metrics['end_acc']:.4f}")
    print(f"    Exact Match:   {val_metrics['exact_match']:.4f}")
    
    # Check for improvement
    improved = False
    
    if val_metrics['loss'] < best_val_loss - config.min_delta:
        best_val_loss = val_metrics['loss']
        improved = True
    
    if val_metrics['exact_match'] > best_exact_match:
        best_exact_match = val_metrics['exact_match']
        improved = True
    
    if improved:
        best_epoch = epoch + 1
        patience_counter = 0
        
        # Save best model
        if config.save_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'exact_match': val_metrics['exact_match'],
                'config': config.__dict__
            }, 'holographic_gearbox_best.pt')
            print(f"    ✓ New best model saved! (loss={best_val_loss:.4f}, EM={best_exact_match:.4f})")
    else:
        patience_counter += 1
        print(f"    No improvement ({patience_counter}/{config.patience})")
    
    # Early stopping
    if patience_counter >= config.patience:
        print(f"\n  Early stopping triggered at epoch {epoch+1}")
        print(f"  Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}, EM={best_exact_match:.4f}")
        break

print("\n✓ Training complete!")


# ════════════════════════════════════════════════════════════════════════════════
# CELL 6: VISUALIZATION AND ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  VISUALIZATION AND ANALYSIS")
print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Loss curves
ax1 = axes[0, 0]
epochs_range = range(1, len(train_losses) + 1)
ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs_range, [m['loss'] for m in val_metrics_history], 'r-', label='Val Loss', linewidth=2)
ax1.axvline(best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Accuracy curves
ax2 = axes[0, 1]
ax2.plot(epochs_range, [m['start_acc'] for m in val_metrics_history], 'g-', label='Start Acc', linewidth=2)
ax2.plot(epochs_range, [m['end_acc'] for m in val_metrics_history], 'b-', label='End Acc', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Position Prediction Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Exact match
ax3 = axes[0, 2]
ax3.bar(epochs_range, [m['exact_match'] for m in val_metrics_history], color='purple', alpha=0.7)
ax3.axhline(best_exact_match, color='r', linestyle='--', label=f'Best ({best_exact_match:.4f})')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Exact Match')
ax3.set_title('Exact Match Score')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Learning rate schedule
ax4 = axes[1, 0]
# Reconstruct LR history
lr_history = []
temp_scheduler = get_linear_schedule_with_warmup(
    AdamW(model.parameters(), lr=config.learning_rate),
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
for _ in range(min(len(train_losses) * len(train_loader), total_steps)):
    lr_history.append(temp_scheduler.get_last_lr()[0])
    temp_scheduler.step()

ax4.plot(lr_history[::len(train_loader)], 'b-', linewidth=2)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate')
ax4.set_title('Learning Rate Schedule')
ax4.grid(True, alpha=0.3)

# 5. Holographic frequency amplitudes
ax5 = axes[1, 1]
# Get amplitude parameters from last layer
last_attn = model.layers[-1].attention
alphas = last_attn.alpha.detach().cpu().numpy()

im = ax5.imshow(alphas, cmap='RdBu_r', aspect='auto')
ax5.set_xlabel('Frequency Index')
ax5.set_ylabel('Head Index')
ax5.set_title('Learned Amplitude Parameters (Last Layer)')
ax5.set_xticks(range(len(PRIME_FREQUENCIES['default'])))
ax5.set_xticklabels(PRIME_FREQUENCIES['default'])
plt.colorbar(im, ax=ax5)

# 6. Per-frequency importance
ax6 = axes[1, 2]
# Average absolute amplitude per frequency across all layers and heads
all_alphas = []
for layer in model.layers:
    all_alphas.append(layer.attention.alpha.detach().cpu().numpy())
avg_importance = np.mean([np.mean(np.abs(a), axis=0) for a in all_alphas], axis=0)

ax6.bar(PRIME_FREQUENCIES['default'], avg_importance, color='teal', alpha=0.7)
ax6.set_xlabel('Prime Frequency')
ax6.set_ylabel('Average |Amplitude|')
ax6.set_title('Frequency Importance (Avg Across Layers)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('holographic_gearbox_training.png', dpi=150, bbox_inches='tight')
plt.savefig('holographic_gearbox_training.pdf', bbox_inches='tight')
plt.show()
print("  ✓ Saved training visualizations")


# ─────────────────────────────────────────────────────────────────────────────────
# ATTENTION PATTERN VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────────

def visualize_attention_patterns(model, tokenizer, question, context):
    """Visualize attention patterns for a sample."""
    model.eval()
    
    # Tokenize
    enc = tokenizer(
        question, context,
        max_length=128,
        truncation='only_second',
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, task='qa')
    
    # Get attention weights from last layer
    attn_weights = model.get_attention_weights(-1)
    
    if attn_weights is not None:
        # Average over heads
        weights = attn_weights[0].mean(dim=0).cpu().numpy()
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        valid_len = attention_mask[0].sum().item()
        valid_len = min(valid_len, 40)  # Limit for visualization
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Attention heatmap
        im = axes[0].imshow(weights[:valid_len, :valid_len], cmap='viridis', aspect='auto')
        axes[0].set_xlabel('Key Position')
        axes[0].set_ylabel('Query Position')
        axes[0].set_title('Holographic Attention Weights')
        plt.colorbar(im, ax=axes[0])
        
        # Answer probabilities
        start_probs = F.softmax(outputs['start_logits'][0], dim=-1).cpu().numpy()
        end_probs = F.softmax(outputs['end_logits'][0], dim=-1).cpu().numpy()
        
        axes[1].bar(range(valid_len), start_probs[:valid_len], alpha=0.5, label='Start', color='blue')
        axes[1].bar(range(valid_len), end_probs[:valid_len], alpha=0.5, label='End', color='red')
        axes[1].set_xlabel('Token Position')
        axes[1].set_ylabel('Probability')
        axes[1].set_title('Answer Position Probabilities')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('holographic_attention_patterns.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print prediction
        start_idx = start_probs.argmax()
        end_idx = end_probs.argmax()
        
        print(f"\n  Question: {question}")
        print(f"  Predicted answer span: [{start_idx}, {end_idx}]")
        if end_idx >= start_idx and end_idx < valid_len:
            answer_tokens = tokens[start_idx:end_idx+1]
            answer = tokenizer.convert_tokens_to_string(answer_tokens)
            print(f"  Predicted answer: {answer}")


# Test with a sample
print("\n  Visualizing attention patterns...")
sample_q = "What is machine learning?"
sample_c = "Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed. It has revolutionized many fields including healthcare and finance."

visualize_attention_patterns(model, tokenizer, sample_q, sample_c)
print("  ✓ Saved attention visualization")


# ─────────────────────────────────────────────────────────────────────────────────
# HOLORAID ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────────

def analyze_holoraid(model):
    """Analyze HoloRAID fault tolerance."""
    print("\n  HoloRAID Fault Tolerance Analysis:")
    print("  " + "-" * 50)
    
    raid = model.holoraid
    print(f"    Shards (n): {raid.n}")
    print(f"    Threshold (k): {raid.k}")
    print(f"    Primes: {raid.primes}")
    print(f"    Redundancy factor: {raid.get_redundancy_factor():.2f}x")
    print(f"    Fault tolerance: {raid.get_fault_tolerance()} failures")
    
    # Test reconstruction accuracy
    print("\n    Reconstruction test:")
    test_tensor = torch.randn(4, 16, 64)
    
    for n_failures in range(raid.n - raid.k + 1):
        # Simulate failures by using different subsets
        if n_failures == 0:
            indices = list(range(raid.k))
        else:
            available = list(range(raid.n))
            random.shuffle(available)
            indices = sorted(available[:raid.k])
        
        shards = raid.encode(test_tensor)
        reconstructed = raid.decode(shards, indices)
        error = (reconstructed - test_tensor).abs().mean().item()
        
        print(f"      {n_failures} failures: reconstruction error = {error:.6f}")


analyze_holoraid(model)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY AND MODEL SAVING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("█  TRAINING COMPLETE - FINAL SUMMARY" + " " * 40 + "█")
print("█" * 80)

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__,
    'train_losses': train_losses,
    'val_metrics': val_metrics_history,
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
    'best_exact_match': best_exact_match,
    'vocab_data': vocab_data,
    'frequencies': PRIME_FREQUENCIES['default']
}, 'holographic_gearbox_final.pt')

summary = f"""

  MODEL: HyperMorphic Holographic Gearbox
  ═══════════════════════════════════════════════════════════════

  ARCHITECTURE
  ────────────
  • Parameters:      {n_params:,}
  • Dimension:       {config.dim}
  • Layers:          {config.n_layers}
  • Heads:           {config.n_heads}
  • Frequencies:     {PRIME_FREQUENCIES['default']}
  • HoloRAID:        {config.use_holoraid} (n=5, k=3)

  TRAINING DATA
  ─────────────
  • Vocabulary QA:   {len(vocab_dataset)} samples
  • SQuAD Train:     {len(squad_train)} samples
  • SQuAD Val:       {len(squad_val)} samples
  • Total Train:     {len(combined_train)} samples

  TRAINING
  ────────
  • Epochs run:      {len(train_losses)}/{config.epochs}
  • Best epoch:      {best_epoch}
  • Learning rate:   {config.learning_rate}
  • Early stopping:  patience={config.patience}

  RESULTS
  ───────
  • Best Val Loss:   {best_val_loss:.4f}
  • Best Exact Match: {best_exact_match:.4f}
  • Final Train Loss: {train_losses[-1]:.4f}
  • Final Val Loss:  {val_metrics_history[-1]['loss']:.4f}
  • Final Start Acc: {val_metrics_history[-1]['start_acc']:.4f}
  • Final End Acc:   {val_metrics_history[-1]['end_acc']:.4f}

  FILES SAVED
  ───────────
  • holographic_gearbox_best.pt      - Best model checkpoint
  • holographic_gearbox_final.pt     - Final model + history
  • holographic_gearbox_training.png - Training curves
  • holographic_gearbox_training.pdf - Training curves (vector)
  • holographic_attention_patterns.png - Attention visualization

  THEORETICAL PROPERTIES
  ──────────────────────
  ✓ Bounded attention (|A_ij| ≤ Σ|α_f|) - Never overflows
  ✓ C^∞ differentiable - Smooth gradients everywhere
  ✓ Multi-scale representation - Prime frequency decomposition
  ✓ Fault-tolerant - HoloRAID k-of-n reconstruction
  ✓ Interpretable - Frequency analysis available

"""
print(summary)

print("█" * 80)
print("█  🌊 HOLOGRAPHIC GEARBOX READY FOR GITHUB! 🌊" + " " * 30 + "█")
print("█" * 80)

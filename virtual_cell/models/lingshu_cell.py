"""
Lingshu-Cell — 掩码离散扩散细胞世界模型 (Masked Discrete Diffusion Cell World Model)

基于 arXiv:2603.25240 论文实现的完整PyTorch架构。

核心创新：
- 281级离散token化（UMI counts → discrete bins）
- 掩码离散扩散模型（MDDM）用于全转录组建模
- Embedding-space序列压缩（18000 genes → ~563 tokens）
- Classifier-Free Guidance 条件生成
- 生物先验注入（下调基因保持固定）

架构：13层双向Transformer + RoPE + RMSNorm + SwiGLU
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ================================================================
# 1. Quantizer — 281级离散token化
# ================================================================

class Quantizer:
    """281级离散token化，将UMI counts转换为离散token。

    分箱方案：
    - 0-99: 精确保留 (100 bins, tokens 0-99)
    - 100-999: 对数分箱 (90 bins, 步长=10, tokens 100-189)
    - 1000-9999: 对数分箱 (90 bins, 步长=100, tokens 190-279)
    - OVF: >9999 的溢出token (token 280)
    - [MASK] token: 281 (不在此类处理，由模型使用)
    """

    OVF_TOKEN = 280
    MASK_TOKEN = 281
    VOCAB_SIZE = 281
    C = 9999

    def encode(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=np.float64)
        tokens = np.zeros_like(counts, dtype=np.int64)

        mask_low = counts < 100
        tokens[mask_low] = np.clip(counts[mask_low], 0, 99).astype(np.int64)

        mask_mid = (~mask_low) & (counts <= self.C)
        if mask_mid.any():
            x = counts[mask_mid]
            k = np.floor(np.log10(np.maximum(x, 1))).astype(np.int64)
            r = np.floor((x - 10**k) / np.maximum(10**(k - 1), 1)).astype(np.int64)
            tokens[mask_mid] = np.clip(100 + 90 * np.maximum(0, k - 2) + r, 0, 279).astype(np.int64)

        mask_ovf = counts > self.C
        tokens[mask_ovf] = self.OVF_TOKEN

        return tokens

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        counts = np.zeros_like(tokens, dtype=np.float64)

        mask_low = tokens < 100
        counts[mask_low] = tokens[mask_low]

        mask_s1 = (tokens >= 100) & (tokens < 190)
        if mask_s1.any():
            offset = tokens[mask_s1] - 100
            r = offset % 90
            counts[mask_s1] = 100.0 + r * 10.0

        mask_s2 = (tokens >= 190) & (tokens < 280)
        if mask_s2.any():
            offset = tokens[mask_s2] - 100
            r = offset % 90
            counts[mask_s2] = 1000.0 + r * 100.0

        mask_ovf = tokens == self.OVF_TOKEN
        counts[mask_ovf] = self.C + 1

        return counts


# ================================================================
# 2. RMSNorm
# ================================================================

if HAS_TORCH:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x / rms * self.weight


# ================================================================
# 3. SwiGLU FFN
# ================================================================

if HAS_TORCH:
    class SwiGLU(nn.Module):
        def __init__(self, dim: int, hidden_dim: int):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ================================================================
# 4. RoPE + Multi-Head Attention
# ================================================================

if HAS_TORCH:
    class MultiHeadAttention(nn.Module):
        def __init__(self, dim: int, n_heads: int, max_seq_len: int = 20000):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            assert dim % n_heads == 0

            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.o_proj = nn.Linear(dim, dim, bias=False)

            # RoPE频率
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            t = torch.arange(max_seq_len).float()
            freqs = torch.outer(t, inv_freq)
            self.register_buffer("_rope_cos", torch.cat([freqs.cos(), freqs.cos()], dim=-1))
            self.register_buffer("_rope_sin", torch.cat([freqs.sin(), freqs.sin()], dim=-1))

        def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
            seq_len = x.shape[2]
            cos = self._rope_cos[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = self._rope_sin[:seq_len].unsqueeze(0).unsqueeze(0)
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return x * cos + torch.cat([-x2, x1], dim=-1) * sin

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            B, S, D = x.shape
            q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

            q = self._apply_rope(q)
            k = self._apply_rope(k)

            scale = math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale

            if mask is not None:
                attn_mask = mask[:, None, None, :].bool()
                attn = attn.masked_fill(~attn_mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
            return self.o_proj(out)


# ================================================================
# 5. Transformer Block
# ================================================================

if HAS_TORCH:
    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, n_heads: int, intermediate_dim: int, max_seq_len: int = 20000):
            super().__init__()
            self.attn_norm = RMSNorm(dim)
            self.attn = MultiHeadAttention(dim, n_heads, max_seq_len)
            self.ffn_norm = RMSNorm(dim)
            self.ffn = SwiGLU(dim, intermediate_dim)

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            x = x + self.attn(self.attn_norm(x), mask)
            x = x + self.ffn(self.ffn_norm(x))
            return x


# ================================================================
# 6. Sequence Compressor
# ================================================================

if HAS_TORCH:
    class SequenceCompressor(nn.Module):
        def __init__(self, hidden_dim: int, patch_size: int = 32):
            super().__init__()
            self.patch_size = patch_size
            self.w_down = nn.Linear(patch_size * hidden_dim, hidden_dim, bias=False)
            self.w_up = nn.Linear(hidden_dim, patch_size * hidden_dim, bias=False)

        def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
            B, S, D = embeddings.shape
            pad_len = (self.patch_size - S % self.patch_size) % self.patch_size
            if pad_len > 0:
                embeddings = F.pad(embeddings, (0, 0, 0, pad_len))
            n_patches = embeddings.shape[1] // self.patch_size
            patches = embeddings.view(B, n_patches, self.patch_size * D)
            return self.w_down(patches)

        def decompress(self, compressed: torch.Tensor, target_len: int) -> torch.Tensor:
            B, N, _ = compressed.shape
            patches_flat = self.w_up(compressed)
            patches = patches_flat.view(B, N * self.patch_size, -1)
            return patches[:, :target_len, :]


# ================================================================
# 7. LingshuCell — 完整模型
# ================================================================

if HAS_TORCH:
    class LingshuCell(nn.Module):
        """Lingshu-Cell：掩码离散扩散细胞世界模型。

        13层双向Transformer + RoPE + RMSNorm + SwiGLU
        支持条件生成（cell type + perturbation）
        支持序列压缩（18000 genes → 563 tokens）
        支持CFG分类器引导
        """

        def __init__(
            self,
            n_genes: int = 18000,
            vocab_size: int = 281,
            hidden_dim: int = 1024,
            n_heads: int = 16,
            n_layers: int = 13,
            intermediate_dim: int = 2752,
            patch_size: int = 32,
            n_cell_types: int = 100,
            n_perturbations: int = 500,
            max_seq_len: int = 20000,
            dropout: float = 0.0,
        ):
            super().__init__()

            self.n_genes = n_genes
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.patch_size = patch_size
            self.mask_token = vocab_size  # [MASK] = vocab_size

            # Token embedding (vocab + 1 for MASK + conditions)
            total_vocab = vocab_size + 1 + n_cell_types + n_perturbations
            self.token_embedding = nn.Embedding(total_vocab, hidden_dim)

            # Gene embedding (可学习的基因位置编码)
            self.gene_embedding = nn.Embedding(n_genes, hidden_dim)

            # Condition token IDs
            self.cell_type_offset = vocab_size + 1
            self.perturbation_offset = vocab_size + 1 + n_cell_types

            # Sequence compressor
            self.compressor = SequenceCompressor(hidden_dim, patch_size)

            # Transformer layers
            self.layers = nn.ModuleList([
                TransformerBlock(hidden_dim, n_heads, intermediate_dim, max_seq_len)
                for _ in range(n_layers)
            ])

            # Output norm + projection
            self.norm = RMSNorm(hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)

            # Dropout
            self.drop = nn.Dropout(dropout)

        def forward(
            self,
            x_t: torch.Tensor,
            condition_cell_type: Optional[torch.Tensor] = None,
            condition_perturbation: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """前向传播。

            Args:
                x_t: (batch, n_genes) — 部分mask的token序列，值0-281
                condition_cell_type: (batch,) — cell type索引
                condition_perturbation: (batch,) — perturbation索引

            Returns:
                logits: (batch, n_genes, vocab_size) — 每个位置的token概率分布
            """
            B, G = x_t.shape

            # Token embedding
            tok_emb = self.token_embedding(x_t)  # (B, G, D)

            # Gene embedding
            gene_ids = torch.arange(G, device=x_t.device).unsqueeze(0).expand(B, -1)
            gene_emb = self.gene_embedding(gene_ids)  # (B, G, D)

            # Combine
            h = self.drop(tok_emb + gene_emb)

            # Prepend condition tokens
            if condition_cell_type is not None or condition_perturbation is not None:
                cond_tokens = []
                if condition_cell_type is not None:
                    ct = condition_cell_type + self.cell_type_offset
                    cond_tokens.append(self.token_embedding(ct).unsqueeze(1))
                if condition_perturbation is not None:
                    pt = condition_perturbation + self.perturbation_offset
                    cond_tokens.append(self.token_embedding(pt).unsqueeze(1))

                cond_emb = torch.cat(cond_tokens, dim=1)  # (B, n_cond, D)
                h = torch.cat([cond_emb, h], dim=1)  # (B, n_cond + G, D)

            # Sequence compression → Transformer → decompress
            compressed = self.compressor.compress(h)
            for layer in self.layers:
                compressed = layer(compressed)
            h = self.compressor.decompress(compressed, h.shape[1])

            # Remove condition tokens from output
            n_cond = h.shape[1] - G
            if n_cond > 0:
                h = h[:, n_cond:, :]

            h = self.norm(h)
            logits = self.output_proj(h)  # (B, G, vocab_size)
            return logits

        def compute_loss(
            self,
            x_0: torch.Tensor,
            condition_cell_type: Optional[torch.Tensor] = None,
            condition_perturbation: Optional[torch.Tensor] = None,
            t: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """计算MDDM训练损失。

            Args:
                x_0: (batch, n_genes) — 完整的token序列（未mask）
                condition_cell_type: (batch,) — 条件
                condition_perturbation: (batch,) — 条件
                t: (batch,) — 扩散时间步，如不指定则随机采样

            Returns:
                loss: 标量
            """
            B, G = x_0.shape

            if t is None:
                t = torch.rand(B, device=x_0.device)  # U(0,1)

            # Forward process: mask tokens with probability t[i]
            mask_probs = t.unsqueeze(1).expand(-1, G)  # (B, G)
            mask = torch.bernoulli(mask_probs).bool()  # True = masked

            x_t = x_0.clone()
            x_t[mask] = self.mask_token  # Replace with [MASK]

            # Predict
            logits = self.forward(x_t, condition_cell_type, condition_perturbation)

            # Loss only on masked positions
            loss = F.cross_entropy(
                logits[mask],
                x_0[mask],
                reduction='mean',
            )
            return loss

        @torch.no_grad()
        def sample(
            self,
            batch_size: int = 1,
            n_steps: int = 128,
            condition_cell_type: Optional[torch.Tensor] = None,
            condition_perturbation: Optional[torch.Tensor] = None,
            guidance_scale: float = 2.5,
            prior_genes: Optional[np.ndarray] = None,
            prior_value: int = 1,
            device: str = 'cpu',
        ) -> torch.Tensor:
            """从全mask序列生成表达谱。

            Args:
                batch_size: 批大小
                n_steps: 反向扩散步数
                condition_cell_type: (batch,) — cell type
                condition_perturbation: (batch,) — perturbation
                guidance_scale: CFG引导权重
                prior_genes: 下调基因索引（生物先验注入）
                prior_value: 先验基因的初始token值
                device: 设备

            Returns:
                generated: (batch, n_genes) — 生成的token序列
            """
            G = self.n_genes

            # Start fully masked
            x = torch.full((batch_size, G), self.mask_token, dtype=torch.long, device=device)

            # Apply biological prior
            if prior_genes is not None:
                for gene_idx in prior_genes:
                    if 0 <= gene_idx < G:
                        x[:, gene_idx] = prior_value

            # Iterative denoising
            for step in range(n_steps):
                t_curr = 1.0 - step / n_steps
                t_next = 1.0 - (step + 1) / n_steps

                # Conditional logits
                logits_cond = self.forward(x, condition_cell_type, condition_perturbation)

                # Unconditional logits (for CFG)
                if guidance_scale > 0:
                    logits_uncond = self.forward(x, None, None)
                    # CFG: a~ = a_uncond + (w+1)(a_cond - a_uncond)
                    logits = logits_uncond + (guidance_scale + 1) * (logits_cond - logits_uncond)
                else:
                    logits = logits_cond

                # Sample from predicted distribution (logits: B,G,V → sample vocab token per gene)
                probs = F.softmax(logits, dim=-1)
                predicted = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, G)

                # Keep some tokens masked (remask with probability t_next/t_curr)
                if step < n_steps - 1:
                    remask_prob = t_next / max(t_curr, 1e-8)
                    remask = torch.bernoulli(torch.full((batch_size, G), remask_prob, device=device)).bool()

                    # Don't remask prior genes
                    if prior_genes is not None:
                        for gene_idx in prior_genes:
                            if 0 <= gene_idx < G:
                                remask[:, gene_idx] = False

                    x = torch.where(remask, self.mask_token, predicted)
                else:
                    x = predicted

            return x


# ================================================================
# 8. Trainer
# ================================================================

if HAS_TORCH:
    class LingshuTrainer:
        """Lingshu-Cell训练器。"""

        def __init__(
            self,
            model: LingshuCell,
            lr: float = 1e-4,
            weight_decay: float = 0.01,
            device: str = 'cpu',
        ):
            self.model = model.to(device)
            self.device = device
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay,
            )
            self.step_count = 0

        def train_step(self, batch: torch.Tensor, condition_ct=None, condition_pt=None) -> float:
            """单步训练。

            Args:
                batch: (batch, n_genes) token序列
                condition_ct: (batch,) cell type
                condition_pt: (batch,) perturbation

            Returns:
                loss值
            """
            self.model.train()
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            if condition_ct is not None:
                condition_ct = condition_ct.to(self.device)
            if condition_pt is not None:
                condition_pt = condition_pt.to(self.device)

            loss = self.model.compute_loss(batch, condition_ct, condition_pt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.step_count += 1

            return loss.item()

        def save_checkpoint(self, path: str):
            torch.save({
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'step': self.step_count,
            }, path)

        def load_checkpoint(self, path: str):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.step_count = ckpt.get('step', 0)

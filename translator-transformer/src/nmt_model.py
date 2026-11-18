from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.transformer_layers import (
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    SwiGLU,
    scaled_dot_product_attention,
)


def _split_heads(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    d_head: int,
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    projected = x @ weight.T
    return projected.reshape(batch_size, seq_len, num_heads, d_head).transpose(1, 2)


def _merge_heads(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, d_head = x.shape
    merged = x.transpose(1, 2).reshape(batch_size, seq_len, num_heads * d_head)
    return merged @ weight.T


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        use_rope: bool = True,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope
        self.rope = (
            RoPE(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)
            if use_rope
            else None
        )
        self.w_q = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.w_k = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.w_v = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.w_o = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        std = math.sqrt(1.0 / d_model)
        for weight in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        query = _split_heads(x, self.w_q, self.num_heads, self.d_head)
        key = _split_heads(x, self.w_k, self.num_heads, self.d_head)
        value = _split_heads(x, self.w_v, self.num_heads, self.d_head)

        if self.rope is not None:
            if positions is None:
                positions = torch.arange(seq_len, device=x.device)
            query = self.rope(query, positions)
            key = self.rope(key, positions)

        mask = torch.ones(
            batch_size,
            1,
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=x.device,
        )
        if key_padding_mask is not None:
            mask = mask & key_padding_mask[:, None, None, :]
        if causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            )
            mask = mask & causal_mask[None, None, :, :]

        attended = scaled_dot_product_attention(query, key, value, mask)
        return _merge_heads(attended, self.w_o)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.w_q = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.w_k = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.w_v = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.w_o = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        std = math.sqrt(1.0 / d_model)
        for weight in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = _split_heads(x, self.w_q, self.num_heads, self.d_head)
        key = _split_heads(memory, self.w_k, self.num_heads, self.d_head)
        value = _split_heads(memory, self.w_v, self.num_heads, self.d_head)
        mask = memory_mask[:, None, None, :]
        attended = scaled_dot_product_attention(query, key, value, mask)
        return _merge_heads(attended, self.w_o)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        use_rope: bool,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.self_attention = MultiheadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.feed_forward = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(
            self.norm1(x),
            key_padding_mask=source_mask,
            causal=False,
        )
        return x + self.feed_forward(self.norm2(x))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        use_rope: bool,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.self_attention = MultiheadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.cross_attention = MultiheadCrossAttention(
            d_model,
            num_heads,
            device=device,
            dtype=dtype,
        )
        self.norm3 = RMSNorm(d_model, device=device, dtype=dtype)
        self.feed_forward = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attention(
            self.norm1(x),
            key_padding_mask=target_mask,
            causal=True,
        )
        x = x + self.cross_attention(self.norm2(x), memory, source_mask)
        return x + self.feed_forward(self.norm3(x))


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 80,
        use_rope: bool = True,
        theta: float = 10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.source_embedding = Embedding(
            source_vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.target_embedding = Embedding(
            target_vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len,
                    use_rope,
                    theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len,
                    use_rope,
                    theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.encoder_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.decoder_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output = Linear(d_model, target_vocab_size, device=device, dtype=dtype)

    def encode(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        x = self.source_embedding(source_ids)
        for layer in self.encoder_layers:
            x = layer(x, source_mask)
        return self.encoder_norm(x)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.target_embedding(decoder_input_ids)
        for layer in self.decoder_layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.output(self.decoder_norm(x))

    def forward(
        self,
        source_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(source_ids, source_mask)
        return self.decode(decoder_input_ids, memory, source_mask, target_mask)

    @torch.no_grad()
    def generate(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int = 64,
        repetition_penalty: float = 1.0,
        blocked_ids: Optional[list[int]] = None,
    ) -> list[int]:
        self.eval()
        memory = self.encode(source_ids, source_mask)
        generated: list[int] = []
        decoder_ids = torch.tensor([[bos_id]], dtype=torch.long, device=source_ids.device)
        for _ in range(max_new_tokens):
            logits = self.decode(
                decoder_ids,
                memory,
                source_mask,
                torch.ones_like(decoder_ids, dtype=torch.bool),
            )[0, -1, :].float().clone()
            logits[pad_id] = float("-inf")
            logits[bos_id] = float("-inf")
            if blocked_ids:
                logits[blocked_ids] = float("-inf")
            if generated and repetition_penalty > 1.0:
                for token_id in set(generated):
                    logits[token_id] /= repetition_penalty
            next_id = int(torch.argmax(logits).item())
            if next_id == eos_id:
                break
            generated.append(next_id)
            decoder_ids = torch.cat(
                [
                    decoder_ids,
                    torch.tensor([[next_id]], dtype=torch.long, device=source_ids.device),
                ],
                dim=1,
            )
        return generated

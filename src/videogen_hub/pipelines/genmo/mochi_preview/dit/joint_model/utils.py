from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def pool_tokens(x: torch.Tensor, mask: torch.Tensor, *, keepdim=False) -> torch.Tensor:
    """
    Pool tokens in x using mask.

    NOTE: We assume x does not require gradients.

    Args:
        x: (B, L, D) tensor of tokens.
        mask: (B, L) boolean tensor indicating which tokens are not padding.

    Returns:
        pooled: (B, D) tensor of pooled tokens.
    """
    assert x.size(1) == mask.size(1)  # Expected mask to have same length as tokens.
    assert x.size(0) == mask.size(0)  # Expected mask to have same batch size as tokens.
    mask = mask[:, :, None].to(dtype=x.dtype)
    mask = mask / mask.sum(dim=1, keepdim=True).clamp(min=1)
    pooled = (x * mask).sum(dim=1, keepdim=keepdim)
    return pooled


class AttentionPool(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            spatial_dim (int): Number of tokens in sequence length.
            embed_dim (int): Dimensionality of input tokens.
            num_heads (int): Number of attention heads.
            output_dim (int): Dimensionality of output tokens. Defaults to embed_dim.
        """
        super().__init__()
        self.num_heads = num_heads
        self.to_kv = nn.Linear(embed_dim, 2 * embed_dim, device=device)
        self.to_q = nn.Linear(embed_dim, embed_dim, device=device)
        self.to_out = nn.Linear(embed_dim, output_dim or embed_dim, device=device)

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): (B, L, D) tensor of input tokens.
            mask (torch.Tensor): (B, L) boolean tensor indicating which tokens are not padding.

        NOTE: We assume x does not require gradients.

        Returns:
            x (torch.Tensor): (B, D) tensor of pooled tokens.
        """
        D = x.size(2)

        # Construct attention mask, shape: (B, 1, num_queries=1, num_keys=1+L).
        attn_mask = mask[:, None, None, :].bool()  # (B, 1, 1, L).
        attn_mask = F.pad(attn_mask, (1, 0), value=True)  # (B, 1, 1, 1+L).

        # Average non-padding token features. These will be used as the query.
        x_pool = pool_tokens(x, mask, keepdim=True)  # (B, 1, D)

        # Concat pooled features to input sequence.
        x = torch.cat([x_pool, x], dim=1)  # (B, L+1, D)

        # Compute queries, keys, values. Only the mean token is used to create a query.
        kv = self.to_kv(x)  # (B, L+1, 2 * D)
        q = self.to_q(x[:, 0])  # (B, D)

        # Extract heads.
        head_dim = D // self.num_heads
        kv = kv.unflatten(2, (2, self.num_heads, head_dim))  # (B, 1+L, 2, H, head_dim)
        kv = kv.transpose(1, 3)  # (B, H, 2, 1+L, head_dim)
        k, v = kv.unbind(2)  # (B, H, 1+L, head_dim)
        q = q.unflatten(1, (self.num_heads, head_dim))  # (B, H, head_dim)
        q = q.unsqueeze(2)  # (B, H, 1, head_dim)

        # Compute attention.
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)  # (B, H, 1, head_dim)

        # Concatenate heads and run output.
        x = x.squeeze(2).flatten(1, 2)  # (B, D = H * head_dim)
        x = self.to_out(x)
        return x


class PadSplitXY(torch.autograd.Function):
    """
    Merge heads, pad and extract visual and text tokens,
    and split along the sequence length.
    """

    @staticmethod
    def forward(
        ctx,
        xy: torch.Tensor,
        indices: torch.Tensor,
        B: int,
        N: int,
        L: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xy: Packed tokens. Shape: (total <= B * (N + L), num_heads * head_dim).
            indices: Valid token indices out of unpacked tensor. Shape: (total,)

        Returns:
            x: Visual tokens. Shape: (B, N, num_heads * head_dim).
            y: Text tokens. Shape: (B, L, num_heads * head_dim).
        """
        ctx.save_for_backward(indices)
        ctx.B, ctx.N, ctx.L = B, N, L
        D = xy.size(1)

        # Pad sequences to (B, N + L, dim).
        assert indices.ndim == 1
        output = torch.zeros(B * (N + L), D, device=xy.device, dtype=dtype)
        indices = indices.unsqueeze(1).expand(-1, D)  # (total,) -> (total, num_heads * head_dim)
        output.scatter_(0, indices, xy)
        xy = output.view(B, N + L, D)

        # Split visual and text tokens along the sequence length.
        return torch.tensor_split(xy, (N,), dim=1)


def pad_and_split_xy(xy, indices, B, N, L, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    return PadSplitXY.apply(xy, indices, B, N, L, dtype)


class UnifyStreams(torch.autograd.Function):
    """Unify visual and text streams."""

    @staticmethod
    def forward(
        ctx,
        q_x: torch.Tensor,
        k_x: torch.Tensor,
        v_x: torch.Tensor,
        q_y: torch.Tensor,
        k_y: torch.Tensor,
        v_y: torch.Tensor,
        indices: torch.Tensor,
    ):
        """
        Args:
            q_x: (B, N, num_heads, head_dim)
            k_x: (B, N, num_heads, head_dim)
            v_x: (B, N, num_heads, head_dim)
            q_y: (B, L, num_heads, head_dim)
            k_y: (B, L, num_heads, head_dim)
            v_y: (B, L, num_heads, head_dim)
            indices: (total <= B * (N + L))

        Returns:
            qkv: (total <= B * (N + L), 3, num_heads, head_dim)
        """
        ctx.save_for_backward(indices)
        B, N, num_heads, head_dim = q_x.size()
        ctx.B, ctx.N, ctx.L = B, N, q_y.size(1)
        D = num_heads * head_dim

        q = torch.cat([q_x, q_y], dim=1)
        k = torch.cat([k_x, k_y], dim=1)
        v = torch.cat([v_x, v_y], dim=1)
        qkv = torch.stack([q, k, v], dim=2).view(B * (N + ctx.L), 3, D)

        indices = indices[:, None, None].expand(-1, 3, D)
        qkv = torch.gather(qkv, 0, indices)  # (total, 3, num_heads * head_dim)
        return qkv.unflatten(2, (num_heads, head_dim))


def unify_streams(q_x, k_x, v_x, q_y, k_y, v_y, indices) -> torch.Tensor:
    return UnifyStreams.apply(q_x, k_x, v_x, q_y, k_y, v_y, indices)

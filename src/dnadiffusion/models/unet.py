# unet_dna_embeddings.py
# Minimal adaptation of the original one-hot UNet to support higher-dim sequence embeddings.
# Changes:
#   - Parameterize spatial geometry (H = embed_dim, W = seq_len).
#   - Keep time embedding size small (time_dim = hidden_dim * 4), as in original.
#   - Add ONE linear projection (to_time_map) to form an (H, W) time map for cross-attention
#     instead of relying on time_dim == H*W.
#   - Input expected as (B, D, L); we unsqueeze channel to (B, 1, D, L) internally.
#
# Everything else (UNet trunk, FiLM conditioning, cross-attn over (B, H, W) with feature dim = W)
# stays as close as possible to the original.

from functools import partial
import torch
import torch.nn as nn
from memory_efficient_attention_pytorch import Attention as EfficientAttention

from dnadiffusion.models.layers import (
    Attention,
    Downsample,
    LearnedSinusoidalPosEmb,
    LinearAttention,
    PreNorm,
    Residual,
    ResnetBlock,
    Upsample,
)
from dnadiffusion.utils.utils import default


class UNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,                     # renamed from `dim` for clarity (base UNet width)
        init_hidden_dim: int | None = None,  # renamed from `init_dim`
        dim_mults: list = [1, 2, 4],
        channels: int = 1,                   # keep single-channel like original
        resnet_block_groups: int = 8,
        learned_sinusoidal_dim: int = 18,
        num_classes: int | None = 10,
        output_attention: bool = False,
        # NEW: parameterize geometry (default matches original one-hot shape if embed_dim=4, seq_len=200)
        embed_dim: int = 4,                  # H (rows) = per-token embedding size (e.g., 4 one-hot or 512 FM)
        seq_len: int = 200,                  # W (cols) = sequence length
    ) -> None:
        super().__init__()

        # --- geometry and bookkeeping ---
        self.H = embed_dim                   # rows
        self.W = seq_len                     # cols
        self.channels = channels
        self.output_attention = output_attention

        # --- stem & channel widths ---
        init_hidden_dim = default(init_hidden_dim, hidden_dim)
        self.init_conv = nn.Conv2d(self.channels, init_hidden_dim, kernel_size=7, padding=3)
        widths = [init_hidden_dim, *(hidden_dim * m for m in dim_mults)]

        in_out = list(zip(widths[:-1], widths[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # --- time / class embeddings (same size strategy as original) ---
        time_dim = hidden_dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.label_emb = nn.Embedding(num_classes, time_dim) if num_classes is not None else None

        # NEW: project the small time vector into an (H*W) map for cross-attention
        self.to_time_map = nn.Linear(time_dim, self.H * self.W, bias=False)

        # --- encoder / decoder stacks (unchanged structure) ---
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = widths[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(hidden_dim * 2, hidden_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(hidden_dim, self.channels, 1)

        # --- cross attention over (B, H, W) with feature dim = W (seq_len) ---
        self.cross_attn = EfficientAttention(
            dim=self.W,           # feature dim (last axis)
            dim_head=64,
            heads=1,
            memory_efficient=True,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )

        # normalize flattened H*W tokens before cross-attn (kept similar to original)
        self.norm_to_cross = nn.LayerNorm(self.H * self.W)

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor | None = None):
        """
        x: (B, D, L)  where D = embed_dim (H), L = seq_len (W)
           (we internally unsqueeze channel -> (B, 1, H, W))
        time: (B,)
        classes: (B,) or None
        """
        # ensure 4D for Conv2d; keep original single-channel assumption
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        assert x.shape[1] == self.channels, f"Expected channels={self.channels}, got {x.shape[1]}"
        assert x.shape[-2:] == (self.H, self.W), f"Expected spatial {(self.H, self.W)}, got {tuple(x.shape[-2:])}"

        # stem
        x = self.init_conv(x)
        r = x.clone()

        # time embeddings (small) + optional class conditioning
        t_start = self.time_mlp(time)    # (B, time_dim)
        t_mid   = t_start.clone()
        t_end   = t_start.clone()
        t_cross = t_start.clone()

        if self.label_emb is not None and classes is not None:
            lbl = self.label_emb(classes)
            t_start = t_start + lbl
            t_mid   = t_mid   + lbl
            t_end   = t_end   + lbl
            t_cross = t_cross + lbl

        # encoder
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_start)
            h.append(x)

            x = block2(x, t_start)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)

        # decoder
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)
            x = attn(x)

            x = upsample(x)

        # final
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t_end)
        x = self.final_conv(x)  # (B, 1, H, W) if channels==1

        B = x.shape[0]

        # map for cross-attention
        x_map = x.view(B, self.H, self.W)                               # (B, H, W)
        t_map = self.to_time_map(t_cross).view(B, self.H, self.W)       # (B, H, W)

        # LayerNorm over all H*W tokens, then cross-attn with time map as context
        x_norm = self.norm_to_cross(x_map.view(B, self.H * self.W)).view(B, self.H, self.W)
        cross_out = self.cross_attn(x_norm, context=t_map)              # (B, H, W)

        # residual merge and restore CNN layout
        x_out = (x_map + cross_out).view(B, self.channels, self.H, self.W)

        if self.output_attention:
            return x_out, cross_out.view(B, 1, self.H, self.W)
        return x_out

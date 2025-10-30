"""
Adaptation of DNA-Diffusion UNet for 512-dim foundation model embeddings.

CORRECTED UNDERSTANDING:
- Input: (B, 1, embedding_dim, seq_len) e.g., (B, 1, 512, 200)
- For cross-attention, we want: seq_len tokens, each with embedding_dim features
- So we need: (B, seq_len, embedding_dim) = (B, 200, 512)
- This requires TRANSPOSING the spatial dimensions

For size invariance:
- Time embedding should match embedding_dim (the feature dimension)
- No scaling with seq_len or spatial size
"""

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
        dim: int,  # Base UNet channel width (e.g., 64)
        init_dim: int | None = None,
        dim_mults: list = [1, 2, 4],
        channels: int = 1,
        resnet_block_groups: int = 8,
        learned_sinusoidal_dim: int = 18,
        num_classes: int = 10,
        output_attention: bool = False,
        # NEW: parameterize input spatial dimensions
        embedding_dim: int = 20,  # per-position embedding size (was 4 for one-hot)
        seq_len: int = 200,  # sequence length (number of positions)
    ) -> None:
        super().__init__()

        # Store spatial dimensions
        # Convention: input shape is (B, C, embedding_dim, seq_len)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.channels = channels
        self.output_attention = output_attention

        # Initial convolution
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, (7, 7), padding=3)

        # Channel dimensions for down/upsampling
        dims = [init_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Time embeddings (standard: 4x the base dim)
        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class conditioning
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        # Project time to match embedding_dim for cross-attention context
        # This is size-invariant: only depends on embedding_dim, not seq_len
        self.time_to_embedding = nn.Linear(time_dim, embedding_dim)

        # Encoder (downsampling path)
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Decoder (upsampling path)
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        # Final layers
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

        # Cross-attention: seq_len tokens, each with embedding_dim features
        self.cross_attn = EfficientAttention(
            dim=embedding_dim,  # feature dimension
            dim_head=64,
            heads=1,
            memory_efficient=True,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )

        # LayerNorm over embedding_dim (feature dimension)
        self.norm_to_cross = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor | None = None):
        """
        Args:
            x: Input tensor (B, C, embedding_dim, seq_len)
            time: Timestep tensor (B,)
            classes: Class labels (B,) or None
        """
        # Initial convolution and store residual
        x = self.init_conv(x)
        residual = x.clone()

        # Compute time embedding
        time_emb = self.time_mlp(time)  # (B, time_dim)

        # Add class conditioning if provided
        if classes is not None and hasattr(self, 'label_emb'):
            time_emb = time_emb + self.label_emb(classes)

        # Encoder
        skip_connections = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            skip_connections.append(x)

            x = block2(x, time_emb)
            x = attn(x)
            skip_connections.append(x)

            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        # Decoder
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block1(x, time_emb)

            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block2(x, time_emb)
            x = attn(x)

            x = upsample(x)

        # Final convolution
        x = torch.cat((x, residual), dim=1)
        x = self.final_res_block(x, time_emb)
        x = self.final_conv(x)  # (B, C, embedding_dim, seq_len)

        # Cross-attention with time conditioning
        batch_size = x.shape[0]

        # Reshape and transpose for cross-attention
        # (B, C, embedding_dim, seq_len) -> (B, embedding_dim, seq_len) -> (B, seq_len, embedding_dim)
        x_spatial = x.reshape(batch_size, self.embedding_dim, self.seq_len)
        x_transposed = x_spatial.transpose(1, 2)  # (B, seq_len, embedding_dim)

        # Project time embedding to embedding_dim and broadcast to all sequence positions
        # (B, time_dim) -> (B, embedding_dim) -> (B, 1, embedding_dim) -> (B, seq_len, embedding_dim)
        time_features = self.time_to_embedding(time_emb)  # (B, embedding_dim)
        time_context = time_features.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, seq_len, embedding_dim)

        # Apply layer norm
        x_normalized = self.norm_to_cross(x_transposed)  # (B, seq_len, embedding_dim)

        # Cross-attention: sequence tokens attend to time context
        # Both are (B, seq_len, embedding_dim)
        cross_attn_out = self.cross_attn(x_normalized, context=time_context)

        # Transpose back and reshape to original format
        # (B, seq_len, embedding_dim) -> (B, embedding_dim, seq_len) -> (B, C, embedding_dim, seq_len)
        cross_attn_out = cross_attn_out.transpose(1, 2).reshape(
            batch_size, self.channels, self.embedding_dim, self.seq_len
        )

        # Residual connection
        x = x + cross_attn_out

        if self.output_attention:
            return x, cross_attn_out
        return x


def load_from_checkpoint(checkpoint_path,**kwargs):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Instantiate the UNet model
    model = UNet(**kwargs)

    # Load the saved weights
    model.load_state_dict(checkpoint['model'])

    return model

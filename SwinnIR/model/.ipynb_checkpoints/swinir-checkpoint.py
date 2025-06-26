import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

# ✅ Window-based Multi-head Self Attention (W-MSA)
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_index = relative_coords[:, :, 0] * (2 * window_size - 1) + relative_coords[:, :, 1]
        self.register_buffer("relative_position_index", relative_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        q = q * self.scale  # ✅ Fixed: Avoid in-place operation

        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].clone()  # ✅ Fix: Clone view
        relative_position_bias = relative_position_bias.view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(x)

# ✅ Swin Transformer Block (Shifted Window Attention)
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + shortcut  # ✅ Out-of-place addition

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + shortcut  # ✅ Out-of-place addition

        return x

# ✅ Patch Embedding (Converts image to patch tokens)
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x

# ✅ SwinIR Model for Image Restoration
class SwinIR(nn.Module):
    def __init__(self, upscale=1, in_chans=1, img_size=256, window_size=8, img_range=1.0,
                 depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2):
        super(SwinIR, self).__init__()

        self.upscale = upscale
        self.img_range = img_range
        self.mean = torch.tensor(0.5).view(1, 1, 1, 1)

        # ✅ Patch Embedding
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)

        # ✅ Transformer Blocks
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            self.layers.append(SwinTransformerBlock(
                dim=embed_dim, num_heads=num_heads[i], window_size=window_size,
                shift_size=window_size // 2 if i % 2 == 1 else 0, mlp_ratio=mlp_ratio
            ))

        # ✅ Reconstruction Head
        self.reconstruction = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = (x - self.mean.to(x.device)) * self.img_range
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)

        B, N, C = x.shape
        H = W = int(N ** 0.5)  # Recover original spatial resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        x = self.reconstruction(x)
        x = x / self.img_range + self.mean.to(x.device)
        return x

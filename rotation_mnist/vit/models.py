"""
LightViT model ("ViT_Tiny_P4_28") for the Rotated-MNIST permutation test.

A tiny Vision Transformer for 1-channel 28x28 MNIST: patch size 4 (-> 49
patches), embed dim 96, depth 6, 3 heads, learnable positional embeddings and a
CLS token. Produces binary (treated vs. untreated) logits.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            "Image dimensions must be divisible by patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)    # (B, N, embed_dim)
        return x


class ViT_Tiny_P4_28(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, num_classes=2,
                 embed_dim=96, depth=6, num_heads=3, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches  # 49 for 28x28, p=4

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _interp_pos_embed(pos_embed, new_hw, embed_dim):
        cls = pos_embed[:, :1, :]
        patch = pos_embed[:, 1:, :]
        old_N = patch.shape[1]
        old_hw = int(math.sqrt(old_N))
        patch = patch.reshape(1, old_hw, old_hw, embed_dim).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, size=new_hw, mode="bicubic", align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, new_hw[0] * new_hw[1], embed_dim)
        return torch.cat([cls, patch], dim=1)

    def forward_features(self, x):
        B, C, H, W = x.shape  # expect (B,1,28,28)
        x = self.patch_embed(x)
        ps_h = H // self.patch_embed.patch_size[0]
        ps_w = W // self.patch_embed.patch_size[1]
        expected = ps_h * ps_w
        if expected != (self.pos_embed.shape[1] - 1):
            pe = self._interp_pos_embed(self.pos_embed, (ps_h, ps_w), self.embed_dim)
        else:
            pe = self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + pe
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LightVisionTransformer(nn.Module):
    def __init__(self, num_classes=2, in_chans=1):
        super(LightVisionTransformer, self).__init__()
        self.model = ViT_Tiny_P4_28(
            img_size=28, patch_size=4, in_chans=in_chans, num_classes=num_classes,
            embed_dim=96, depth=6, num_heads=3, mlp_ratio=4.0, drop=0.1, attn_drop=0.1,
        )

    def forward(self, x):
        return self.model(x)

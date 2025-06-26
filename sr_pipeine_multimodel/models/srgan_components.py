import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class Generator(nn.Module):
    """Text-conditioned SRGAN Generator"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text conditioning branch
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_dim, config.text_proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.text_proj_dim, 64*64)  # Spatial dimension for concatenation
        )

        # Initial processing of LR image + text features
        self.initial = nn.Sequential(
            nn.Conv2d(config.img_channels + 1, 64, 3, padding=1),  # +1 text channel
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(config.num_res_blocks)]
        )

        # Upsampling blocks
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),  # 128x128 → 256x256
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),  # 256x256 → 512x512
            nn.ReLU(inplace=True),
            nn.Conv2d(64, config.img_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, lr_img, text_emb):
        # Process text embedding
        text_feat = self.text_proj(text_emb)
        text_feat = text_feat.view(-1, 1, 64, 64)  # Reshape to spatial features
        
        # Concatenate with upsampled LR image
        x = F.interpolate(lr_img, scale_factor=2, mode='bilinear')  # 128→256
        x = torch.cat([x, text_feat], dim=1)  # Concatenate along channel dim
        
        # Main processing
        x = self.initial(x)
        x = self.res_blocks(x)
        return self.upsample(x)

class Discriminator(nn.Module):
    """PatchGAN discriminator"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Input: 512x512 RGB
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 256x256
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128x128
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 64x64
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 32x32
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, hr_img):
        return self.model(hr_img)
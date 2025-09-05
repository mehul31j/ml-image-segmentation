import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Fixed SimpleUNet with proper size handling
class SimpleUNet(nn.Module):
    """Simplified U-Net with robust size handling"""
    def __init__(self, n_channels, n_classes):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_layer(n_channels, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        
        # Decoder
        self.dec2 = self._make_layer(256 + 128, 128)
        self.dec1 = self._make_layer(128 + 64, 64)
        
        self.final = nn.Conv2d(64, n_classes, 1)
        
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _resize_and_concat(self, upsampled, skip):
        """Resize upsampled to match skip connection size and concatenate"""
        # Get target size from skip connection
        target_size = skip.shape[2:]
        
        # Resize upsampled to match skip connection
        upsampled_resized = F.interpolate(
            upsampled, 
            size=target_size, 
            mode='bilinear', 
            align_corners=True
        )
        
        return torch.cat([upsampled_resized, skip], dim=1)
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)                    # Full resolution
        e2 = self.enc2(F.max_pool2d(e1, 2))  # 1/2 resolution
        e3 = self.enc3(F.max_pool2d(e2, 2))  # 1/4 resolution
        
        # Decoder with proper resizing
        d2 = F.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self._resize_and_concat(d2, e2)  # Ensure sizes match
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self._resize_and_concat(d1, e1)  # Ensure sizes match
        d1 = self.dec1(d1)
        
        return self.final(d1)

# Even simpler U-Net without skip connections for testing
class BasicUNet(nn.Module):
    """Ultra-simple U-Net without skip connections"""
    def __init__(self, n_channels, n_classes):
        super(BasicUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Simple encoder-decoder without skip connections
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.final(x)

# Minimal CNN for quick testing
class SimpleCNN(nn.Module):
    """Minimal CNN for segmentation"""
    def __init__(self, n_channels, n_classes):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1)
        )

    def forward(self, x):
        return self.features(x)

# Keep the other classes for compatibility
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.model = SimpleUNet(n_channels, n_classes)

    def forward(self, x):
        return self.model(x)

class FastUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_channels=32):
        super(FastUNet, self).__init__()
        self.model = BasicUNet(n_channels, n_classes)

    def forward(self, x):
        return self.model(x)

class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LightUNet, self).__init__()
        self.model = SimpleCNN(n_channels, n_classes)

    def forward(self, x):
        return self.model(x)

class ImprovedUNet(nn.Module):
    """Improved U-Net with better architecture"""
    def __init__(self, n_channels, n_classes):
        super(ImprovedUNet, self).__init__()
        
        # Encoder with residual connections
        self.enc1 = self._make_encoder_block(n_channels, 32)
        self.enc2 = self._make_encoder_block(32, 64)
        self.enc3 = self._make_encoder_block(64, 128)
        self.enc4 = self._make_encoder_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(256, 512)
        
        # Decoder
        self.dec4 = self._make_decoder_block(512 + 256, 256)
        self.dec3 = self._make_decoder_block(256 + 128, 128)
        self.dec2 = self._make_decoder_block(128 + 64, 64)
        self.dec1 = self._make_decoder_block(64 + 32, 32)
        
        self.final = nn.Sequential(
            nn.Conv2d(32, n_classes, 1),
            nn.Dropout2d(0.1)
        )
        
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def _safe_concat(self, upsampled, skip):
        """Safely concatenate tensors with size matching"""
        target_size = skip.shape[2:]
        upsampled_resized = F.interpolate(
            upsampled, size=target_size, mode='bilinear', align_corners=True
        )
        return torch.cat([upsampled_resized, skip], dim=1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.dec4(self._safe_concat(d4, e4))
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.dec3(self._safe_concat(d3, e3))
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.dec2(self._safe_concat(d2, e2))
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.dec1(self._safe_concat(d1, e1))
        
        return self.final(d1)
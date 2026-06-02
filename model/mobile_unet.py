import torch
import torch.nn as nn

# (이전 Generator와 동일한 SEBlock, MobileConvBlock 사용)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                            stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU(inplace=True) 
        
        self.se = SEBlock(in_channels) if use_se else nn.Identity()
        
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)
        
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        out = self.dw(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        out = self.se(out)
        
        out = self.pw(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        if self.use_res_connect:
            return x + out
        return out

class OptimizedMaskSegmentation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_c=16):
        super().__init__()
        
        # 채널 크기 기본값이 16이므로, base_c=32이면 동일 구조를 더 넓은 채널로 확장
        c1 = base_c
        c2 = base_c * 2
        c3 = base_c * 4
        
        # [Input] 384x384 RGB 영상
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True)
        )
        
        # [Encoder - MobileNetV3-Small 스타일]
        self.enc1 = MobileConvBlock(c1, c2 // 2, stride=1)
        self.enc2 = MobileConvBlock(c2 // 2, c2, stride=2)
        self.enc3 = MobileConvBlock(c2, c3, stride=2)
        
        # [Bottleneck]
        self.bottleneck = nn.Sequential(
            MobileConvBlock(c3, c3, stride=1),
            MobileConvBlock(c3, c3, stride=1)
        )
        
        # [Light Decoder]
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec1 = MobileConvBlock(c3, c2, stride=1)
        
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec2 = MobileConvBlock(c2, c2 // 2, stride=1)
        
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec3 = MobileConvBlock(c2 // 2, c1, stride=1)
        
        # [Refinement]
        self.dec4 = MobileConvBlock(c1, c1, stride=1)
        
        # [Output]
        self.final_conv = nn.Conv2d(c1, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: [B, 3, 384, 384]
        x_init = self.init_conv(x)  # [16, 192, 192]
        
        e1 = self.enc1(x_init)      # [24, 192, 192]
        e2 = self.enc2(e1)          # [48, 96, 96]
        e3 = self.enc3(e2)          # [96, 48, 48]
        
        b = self.bottleneck(e3)     # [96, 48, 48]
        
        # Decoder
        d1 = self.up1(b)            # [96, 96, 96]
        d1 = self.dec1(d1)          # [48, 96, 96]
        d1 = d1 + e2                # Skip connection (48 채널)
        
        d2 = self.up2(d1)           # [48, 192, 192]
        d2 = self.dec2(d2)          # [24, 192, 192]
        d2 = d2 + e1                # Skip connection (24 채널)
        
        d3 = self.up3(d2)           # [24, 384, 384]
        d3 = self.dec3(d3)          # [16, 384, 384]
        # x_init(16, 192, 192)과 해상도가 달라 여기서는 Skip Connection 생략 
        # (혹은 up4 직전에 연결할 수 있으나, 마스킹 모델 특성상 생략해도 충분함)
        
        d4 = self.dec4(d3)          # [16, 384, 384]
        
        out = self.final_conv(d4)   # [1, 384, 384]
        return out
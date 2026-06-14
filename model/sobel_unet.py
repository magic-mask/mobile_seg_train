# model/sobel_unet.py

import torch
import torch.nn as nn
import numpy as np

# ==========================================
# 1. NPU 최적화 엣지 추출 레이어 (가중치 고정)
# ==========================================
class SobelEdgeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 1채널(Grayscale), 출력 2채널(Sobel X, Y)
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1, bias=False)
        
        # 필터 행렬 정의
        sobel_x = np.array([[-1,  0,  1],
                            [-2,  0,  2],
                            [-1,  0,  1]], dtype=np.float32)
                            
        sobel_y = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=np.float32)
        
        # PyTorch 가중치 형태로 변환: [out_channels, in_channels, height, width]
        weight = np.stack([sobel_x, sobel_y]).reshape(2, 1, 3, 3)
        self.conv.weight = nn.Parameter(torch.from_numpy(weight))
        
        # 🔥 핵심: 학습되지 않도록 가중치 동결 (NPU에 상수 행렬로 태우기 위함)
        self.conv.weight.requires_grad = False 

    def forward(self, x):
        return self.conv(x)

# ==========================================
# 2. 기본 합성곱 블록 (기존 코드 유지)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# ==========================================
# 3. 엣지-인식 최적화 모델 (HOG 실험의 결론 반영)
# ==========================================
class OptimizedMaskSegmentation(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_c=32):
        super().__init__()
        
        # 엣지 추출 모듈 초기화
        self.sobel = SobelEdgeLayer()
        
        # 🔥 입력 채널 변경: 원본 RGB(3) + 엣지 X,Y(2) = 5채널
        combined_channels = in_channels + 2 
        
        self.inc = DoubleConv(combined_channels, base_c)

        # 인코더 (기존 구조 동일)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c, base_c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 2, base_c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 4, base_c * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 8, base_c * 16))

        # 디코더 (기존 구조 동일)
        self.up1 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_c * 16, base_c * 8)

        self.up2 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_c * 8, base_c * 4)

        self.up3 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_c * 4, base_c * 2)

        self.up4 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_c * 2, base_c)

        self.outc = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. RGB 이미지를 내부적으로 Grayscale로 변환 (텐서 연산)
        # x 차원: [Batch, Channel, Height, Width]
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]
        x_gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        # 2. 고정된 소벨 필터를 통해 구조적 특징 추출
        edges = self.sobel(x_gray)  # 결과: [B, 2, H, W]
        
        # 3. 원본 입력과 엣지 맵 병합 -> 5채널 입력 생성
        x_combined = torch.cat([x, edges], dim=1)  # 결과: [B, 5, H, W]
        
        # 4. 결합된 피처를 U-Net에 태움
        x1 = self.inc(x_combined)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        up = self.up1(x5)
        up = torch.cat([up, x4], dim=1)
        up = self.conv_up1(up)

        up = self.up2(up)
        up = torch.cat([up, x3], dim=1)
        up = self.conv_up2(up)

        up = self.up3(up)
        up = torch.cat([up, x2], dim=1)
        up = self.conv_up3(up)

        up = self.up4(up)
        up = torch.cat([up, x1], dim=1)
        up = self.conv_up4(up)

        return self.outc(up)
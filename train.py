# train.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UltraLightUNet
from model import OptimizedMaskSegmentation

# ==========================================
# 0. Dice Loss & Metrics
# ==========================================
class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


def dice_score(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(pred, target, smooth=1e-6):
    """Calculate Intersection over Union (IoU)"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


# ==========================================
# 1. 실제 디스크 경로에서 데이터를 읽는 Custom Dataset
# ==========================================
class RealSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # 이미지 폴더 내 파일 목록 정렬 (파일명이 매칭된다고 가정)
        self.image_filenames = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        # Training/Validation 구분: Train은 augmentation 적용, Val는 기본 전처리만
        if is_train:
            # Training Augmentation: 더 강력한 데이터 증강
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(p=0.3),
                A.MotionBlur(blur_limit=5, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # Validation: 기본 전처리만 (Augmentation 없음)
            self.transform = A.Compose([
                A.Resize(384, 384),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # 이미지와 마스크의 파일 이름이 같다고 가정하되, 확장자가 다를 경우를 처리
        mask_name = img_name
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            # 마스크 확장자가 다를 경우(예: 이미지 .jpg -> 마스크 .png) 대응 로직
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = os.path.join(self.masks_dir, base_name + ext)
                if os.path.exists(potential_path):
                    mask_path = potential_path
                    break

        # 이미지 및 마스크 로드
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 1채널 흑백 마스크로 변환
        
        # Numpy 변환
        image = np.array(image)
        mask = np.array(mask)
        
        # Albumentations 적용 (이미지와 마스크에 동일한 변환 적용)
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # 마스크 이진화 유효성 확보
        mask = (mask > 0.5).float() if isinstance(mask, torch.Tensor) else torch.from_numpy((mask > 0.5)).float()
        
        # 마스크의 채널 차원 확보 (C, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask


# ==========================================
# 3. 메인 훈련 루프 실행 파트
# ==========================================
def train_model():
    # ========== Seed 고정 (재현성 확보) ==========
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    
    # 장치 할당 (CUDA > CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터셋 경로 지정
    #IMAGES_DIR = "bucket/images"
    #MASKS_DIR = "bucket/masks"
    IMAGES_DIR = "all_dataset/sd_images_954/images"
    MASKS_DIR = "all_dataset/sd_images_954/masks"
    
    # 데이터셋 인스턴스 생성
    full_dataset = RealSegmentationDataset(images_dir=IMAGES_DIR, masks_dir=MASKS_DIR, is_train=True)
    dataset_size = len(full_dataset)
    print(f"Total dataset size found: {dataset_size} images.")
    
    if dataset_size == 0:
        raise ValueError(f"No images found in '{IMAGES_DIR}'. Please check the folder paths.")

    # 8:2 비율로 Train / Validation 데이터셋 분할 (Seed 고정으로 재현성 확보)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Validation dataset은 augmentation 없이 생성
    val_dataset_no_aug = RealSegmentationDataset(images_dir=IMAGES_DIR, masks_dir=MASKS_DIR, is_train=False)
    _, val_dataset = random_split(val_dataset_no_aug, [train_size, val_size], generator=generator)

    # DataLoader 생성 (Batch Size는 A17 Pro 이외 학습 환경에 맞춰 조절 가능)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 선언 (base_c=32로 3.5M 미만 경량화 세팅 유지)
    model = OptimizedMaskSegmentation(in_channels=3, out_channels=1, base_c=32).to(device)
    
    # 하이퍼파라미터 및 학습 객체 설정
    epochs = 150
    learning_rate = 1.5e-3 #1e-3
    
    # ========== Loss Functions (BCE + Dice) ==========
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(smooth=1e-6)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # ========== Mixed Precision (CUDA 가속화) ==========
    scaler = GradScaler()
    
    best_dice = 0.0  # Best model 저장 기준을 Dice로 변경

    # 에폭 루프
    for epoch in range(epochs):
        # [Train Phase]
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # ========== Mixed Precision Training ==========
            with autocast():
                outputs = model(images)
                # Combined Loss: 50% BCE + 50% Dice
                loss = 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_dataset)

        # [Validation Phase]
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                with autocast():
                    outputs = model(images)
                    loss = 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                val_dice += dice_score(outputs, masks).item() * images.size(0)
                val_iou += iou_score(outputs, masks).item() * images.size(0)
                
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_dice = val_dice / len(val_dataset)
        avg_val_iou = val_iou / len(val_dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Dice: {avg_val_dice:.4f} | "
              f"IoU: {avg_val_iou:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # ========== Best Model 저장 기준 변경: Dice 기준 ==========
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), "weights/sobel_unet_20260614.pth")
            print(f"--> Saved new best model with Dice: {best_dice:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train_model()
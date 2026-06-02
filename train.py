import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torchvision.transforms as transforms

from model import UltraLightUNet

# ==========================================
# 1. 실제 디스크 경로에서 데이터를 읽는 Custom Dataset
# ==========================================
class RealSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # 이미지 폴더 내 파일 목록 정렬 (파일명이 매칭된다고 가정)
        self.image_filenames = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        # 이미지 전처리: 384x384 리사이즈 및 텐서 변환
        # (Tip: CoreML 변환 시 전처리를 모델 내부에 포함할 수 있도록 0~1 정규화만 우선 적용)
        self.image_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(), # 이미지를 0.0 ~ 1.0 사이의 값으로 변환
        ])
        
        # 마스크 전처리: Nearest Neighbor 보간법으로 리사이즈하여 픽셀 값 붕괴 방지
        self.mask_transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(), # 0.0 ~ 1.0 사이 float 텐서로 변환
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
        mask = Image.open(mask_path).convert("L") # 1채널 흑백 마스크로 변환

        # 전처리 적용
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        # 이진화 유효성 확보 (경계면 보간 등으로 인해 생길 수 있는 중간값을 0 또는 1로 고정)
        mask = (mask > 0.5).float()

        return image, mask


# ==========================================
# 3. 메인 훈련 루프 실행 파트
# ==========================================
def train_model():
    # 장치 할당 (CUDA > MPS > CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터셋 경로 지정
    IMAGES_DIR = "bucket/images"
    MASKS_DIR = "bucket/masks"
    
    # 데이터셋 인스턴스 생성
    full_dataset = RealSegmentationDataset(images_dir=IMAGES_DIR, masks_dir=MASKS_DIR)
    dataset_size = len(full_dataset)
    print(f"Total dataset size found: {dataset_size} images.")
    
    if dataset_size == 0:
        raise ValueError(f"No images found in '{IMAGES_DIR}'. Please check the folder paths.")

    # 8:2 비율로 Train / Validation 데이터셋 분할
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # DataLoader 생성 (Batch Size는 A17 Pro 이외 학습 환경에 맞춰 조절 가능)
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 선언 (base_c=32로 3.5M 미만 경량화 세팅 유지)
    model = UltraLightUNet(in_channels=3, out_channels=1, base_c=32).to(device)
    
    # 하이퍼파라미터 및 학습 객체 설정
    epochs = 50
    learning_rate = 1e-3
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    # 에폭 루프
    for epoch in range(epochs):
        # [Train Phase]
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_dataset)

        # [Validation Phase]
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
        avg_val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 최고 성능 모델 저장 (Early Saving)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "weights/ultralight_unet_20260603.pth")
            print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train_model()
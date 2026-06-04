import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from model import OptimizedMaskSegmentation



def run_video_inference(
    video_path,
    model_path="best_ultralight_unet.pth",
    threshold=0.5,
    image_size=384,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        base_c = checkpoint.get("base_c", 32)

        model = OptimizedMaskSegmentation(
            in_channels=3,
            out_channels=1,
            base_c=base_c,
        ).to(device)

        model.load_state_dict(
            checkpoint["state_dict"]
        )
    else:
        model = OptimizedMaskSegmentation(
            in_channels=3,
            out_channels=1,
            base_c=32,
        ).to(device)

        model.load_state_dict(checkpoint)

    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # 여기서 cap.read()로 가져온 frame을 우리가 학습한 모델 384*384로 변환함
        tensor = transform(rgb)  # 모델 = 384*384
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)

        mask = probs[0, 0].cpu().numpy()

        binary_mask = (
            mask > threshold
        ).astype(np.uint8) * 255

        binary_mask = cv2.resize(
            binary_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_bgr = cv2.cvtColor(
            binary_mask,
            cv2.COLOR_GRAY2BGR,
        )

        overlay = frame.copy()

        overlay[binary_mask > 0] = (
            0,
            255,
            0,
        )

        overlay = cv2.addWeighted(
            frame,
            0.7,
            overlay,
            0.3,
            0,
        )

        # 최종으로 원본 프레임, 바이너리 마스크, 오버레이를 나란히 합치는 부분
        # 일단 잘리니까 frame, mask_bgr, overlay 해상도를 줄여놓고 확인
        frame = cv2.resize(frame, (854, 480))
        mask_bgr = cv2.resize(mask_bgr, (854, 480))
        overlay = cv2.resize(overlay, (854, 480))
        # 세 개의 이미지를 가로로 이어붙임
        combined = np.hstack([
            frame,
            mask_bgr,
            overlay,
        ])

        cv2.imshow(
            "Original | Mask | Overlay",
            combined,
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    VIDEO_PATH = "sample/magiggun.webm"

    run_video_inference(
        video_path=VIDEO_PATH,
        model_path="weights/mobile_unet_20260604.pth",
        threshold=0.5,
        image_size=384,
    )
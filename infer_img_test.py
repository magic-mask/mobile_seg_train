import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from model import OptimizedMaskSegmentation


def run_inference(
    image_path,
    model_path="weights/best_ultralight_unet.pth",
    output_dir="output",
    threshold=0.5,
    image_size=384,
):
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
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

        model.load_state_dict(checkpoint["state_dict"])

    else:
        model = OptimizedMaskSegmentation(
            in_channels=3,
            out_channels=1,
            base_c=32,
        ).to(device)

        model.load_state_dict(checkpoint)

    model.eval()

    # --------------------------------------------------
    # Transform
    # --------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------
    image = Image.open(image_path).convert("RGB")

    original_w, original_h = image.size

    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)

    mask = probs[0, 0].cpu().numpy()

    binary_mask = (
        mask > threshold
    ).astype(np.uint8) * 255

    binary_mask = cv2.resize(
        binary_mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # --------------------------------------------------
    # Save mask
    # --------------------------------------------------
    stem = Path(image_path).stem

    mask_path = os.path.join(
        output_dir,
        f"{stem}_mask.png",
    )

    cv2.imwrite(mask_path, binary_mask)

    # --------------------------------------------------
    # Save probability map
    # --------------------------------------------------
    prob_map = (mask * 255).astype(np.uint8)

    prob_map = cv2.resize(
        prob_map,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR,
    )

    prob_path = os.path.join(
        output_dir,
        f"{stem}_prob.png",
    )

    cv2.imwrite(prob_path, prob_map)

    # --------------------------------------------------
    # Save overlay
    # --------------------------------------------------
    original_bgr = cv2.imread(image_path)

    overlay = original_bgr.copy()

    overlay[binary_mask > 0] = (0, 255, 0)

    result = cv2.addWeighted(
        original_bgr,
        0.7,
        overlay,
        0.3,
        0,
    )

    overlay_path = os.path.join(
        output_dir,
        f"{stem}_overlay.jpg",
    )

    cv2.imwrite(
        overlay_path,
        result,
    )

    print()
    print(f"Mask     : {mask_path}")
    print(f"Prob Map : {prob_path}")
    print(f"Overlay  : {overlay_path}")
    print()


if __name__ == "__main__":

    IMAGE_PATH = "sample/karina_mask.jpg"

    run_inference(
        image_path=IMAGE_PATH,
        model_path="weights/mobile_unet_20260604.pth",
        output_dir="output",
        threshold=0.5,
        image_size=384,
    )
from pathlib import Path
import shutil

# 여러 데이터셋 폴더들이 들어있는 상위 디렉토리
ROOT_DIR = Path("all_dataset")

# 출력 버킷
BUCKET_DIR = Path("bucket")
BUCKET_IMAGES = BUCKET_DIR / "images"
BUCKET_MASKS = BUCKET_DIR / "masks"

BUCKET_IMAGES.mkdir(parents=True, exist_ok=True)
BUCKET_MASKS.mkdir(parents=True, exist_ok=True)


def copy_with_unique_name(src: Path, dst_dir: Path):
    dst = dst_dir / src.name

    if not dst.exists():
        shutil.copy2(src, dst)
        return

    stem = src.stem
    suffix = src.suffix
    idx = 1

    while True:
        candidate = dst_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return
        idx += 1


for dataset_dir in ROOT_DIR.iterdir():
    if not dataset_dir.is_dir():
        continue

    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    if images_dir.exists():
        for file in images_dir.iterdir():
            if file.is_file():
                copy_with_unique_name(file, BUCKET_IMAGES)

    if masks_dir.exists():
        for file in masks_dir.iterdir():
            if file.is_file():
                copy_with_unique_name(file, BUCKET_MASKS)

print("완료")
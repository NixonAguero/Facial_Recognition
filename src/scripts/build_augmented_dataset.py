import os
import cv2
import shutil
from pathlib import Path
import albumentations as A


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create_folder(path: Path | str):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_light_augmentation_pipeline():
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.6
        ),

        A.Rotate(
            limit=8,
            p=0.4
        ),

        A.GaussianBlur(
            blur_limit=(3, 3),
            p=0.15
        ),

        A.GaussNoise(
            p=0.15
        ),

        A.ImageCompression(
            quality_range=(80, 100),
            p=0.25
        )
    ])


def get_image_paths_from_folder(folder: Path) -> list[Path]:
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []

    for ext in extensions:
        image_paths.extend(folder.rglob(ext))

    return image_paths


def build_augmented_dataset(
    person_name: str,
    augmentations_per_image: int = 1,
    include_celeba: bool = True,
    include_own_frames: bool = True,
    clean_output: bool = True
):
    celeba_dir = Path(PROJECT_ROOT) / "dataset" / "img_align_celeba"
    own_frames_dir = Path(PROJECT_ROOT) / "dataset" / "own_faces" / "frames" / person_name
    output_dir = Path(PROJECT_ROOT) / "dataset" / "training_augmented"

    if clean_output and output_dir.exists():
        print(f"[INFO] Cleaning output folder: {output_dir}")
        shutil.rmtree(output_dir)

    create_folder(output_dir)

    source_folders = []

    if include_celeba:
        if celeba_dir.exists():
            source_folders.append(("celeba", celeba_dir))
        else:
            print(f"[WARNING] CelebA folder does not exist: {celeba_dir}")

    if include_own_frames:
        if own_frames_dir.exists():
            source_folders.append((person_name, own_frames_dir))
        else:
            print(f"[WARNING] Own frames folder does not exist: {own_frames_dir}")

    if not source_folders:
        print("[ERROR] No source folders found.")
        return

    transform = build_light_augmentation_pipeline()

    total_originals_saved = 0
    total_augmented_saved = 0

    for source_label, source_dir in source_folders:
        print(f"\n[SOURCE] {source_label}: {source_dir}")

        image_paths = get_image_paths_from_folder(source_dir)

        if not image_paths:
            print(f"[WARNING] No images found in {source_dir}")
            continue

        source_output_dir = output_dir / source_label
        create_folder(source_output_dir)

        for image_path in image_paths:
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[WARNING] Could not read image: {image_path}")
                continue

            # Original copy
            original_name = f"{image_path.stem}_original{image_path.suffix.lower()}"
            original_output_path = source_output_dir / original_name

            if cv2.imwrite(str(original_output_path), image):
                total_originals_saved += 1

            # Augmented copies
            for i in range(augmentations_per_image):
                augmented = transform(image=image)
                augmented_image = augmented["image"]

                output_name = f"{image_path.stem}_aug_{i + 1}.jpg"
                output_path = source_output_dir / output_name

                if cv2.imwrite(str(output_path), augmented_image):
                    total_augmented_saved += 1

    print("\n[DONE] Augmented dataset created.")
    print(f"[OUTPUT] {output_dir}")
    print(f"[ORIGINALS SAVED] {total_originals_saved}")
    print(f"[AUGMENTED SAVED] {total_augmented_saved}")
    print(f"[TOTAL] {total_originals_saved + total_augmented_saved}")


if __name__ == "__main__":
    ##build_augmented_dataset(
        person_name="ian",
        augmentations_per_image=1,
        include_celeba=True,
        include_own_frames=True,
        clean_output=True
    )
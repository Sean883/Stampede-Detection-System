"""train_yolo.py — Fine-tune YOLOv8n on crowd-specific data.

This script fine-tunes the YOLOv8 nano model on the CrowdHuman dataset
(or any YOLO-format dataset) so it detects heavily occluded and dense-crowd
persons much better than the stock COCO-trained model.

Usage:
    1. Download CrowdHuman and convert to YOLO format (see prepare_crowdhuman() below)
    2. Run:  python train_yolo.py
    3. The best weights are saved to  runs/detect/crowd_finetune/weights/best.pt
    4. Copy best.pt to this folder and update detector.py:
         YOLO_MODEL_NAME = "best.pt"

Requirements:
    pip install ultralytics
"""

import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR      = os.path.join(os.path.dirname(__file__), "datasets", "crowdhuman")
YOLO_BASE_MODEL  = "yolov8n.pt"        # start from pre-trained nano
EPOCHS           = 50                   # 50 is usually enough for fine-tuning
IMGSZ            = 640
BATCH            = 8                    # lower if you run out of VRAM/RAM
DEVICE           = "cpu"                # "0" for GPU, "cpu" for CPU-only
PROJECT_NAME     = "runs/detect"
RUN_NAME         = "crowd_finetune"


def create_dataset_yaml():
    """Create the dataset.yaml that ultralytics needs."""
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")

    # Expected folder structure:
    # datasets/crowdhuman/
    #   images/
    #     train/    ← training images
    #     val/      ← validation images
    #   labels/
    #     train/    ← YOLO-format .txt files (one per image)
    #     val/      ← YOLO-format .txt files

    content = f"""# CrowdHuman dataset for YOLOv8 fine-tuning
path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val

# Single class: person
names:
  0: person

nc: 1
"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Dataset YAML written to: {yaml_path}")
    return yaml_path


def prepare_crowdhuman():
    """Instructions for preparing the CrowdHuman dataset.

    CrowdHuman is not in YOLO format by default.  You need to convert it.

    Steps:
    1. Download from:  https://www.crowdhuman.org/
       - CrowdHuman_train01.zip, CrowdHuman_train02.zip, CrowdHuman_train03.zip
       - CrowdHuman_val.zip
       - annotation_train.odgt, annotation_val.odgt

    2. Run the converter below (or use an existing tool):
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              CrowdHuman Dataset Preparation                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Download CrowdHuman from:                                    ║
║     https://www.crowdhuman.org/                                  ║
║                                                                  ║
║  2. Place files in:                                              ║
║     datasets/crowdhuman/                                         ║
║       images/train/   ← training images (.jpg)                   ║
║       images/val/     ← validation images (.jpg)                 ║
║       annotation_train.odgt                                      ║
║       annotation_val.odgt                                        ║
║                                                                  ║
║  3. Run:  python train_yolo.py --convert                         ║
║     This converts .odgt annotations to YOLO format               ║
║                                                                  ║
║  4. Then run:  python train_yolo.py                              ║
║     to start fine-tuning.                                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def convert_odgt_to_yolo(odgt_path, images_dir, labels_dir):
    """Convert CrowdHuman .odgt annotation to YOLO format.

    YOLO format: one .txt per image, each line:
        class_id  center_x  center_y  width  height
    All values normalized to 0-1 relative to image dimensions.
    """
    import json
    from PIL import Image

    os.makedirs(labels_dir, exist_ok=True)
    count = 0

    with open(odgt_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            img_id = data["ID"]

            # Find the image to get dimensions
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(images_dir, img_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                continue

            img = Image.open(img_path)
            img_w, img_h = img.size

            label_path = os.path.join(labels_dir, img_id + ".txt")
            lines = []

            for box_info in data.get("gtboxes", []):
                if box_info.get("tag") != "person":
                    continue
                # CrowdHuman uses "fbox" (full body box): [x, y, w, h]
                fb = box_info.get("fbox")
                if fb is None:
                    continue

                x, y, w, h = fb
                # Convert to YOLO format (center_x, center_y, w, h) normalized
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                # Clamp to [0, 1]
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0.001, min(1, nw))
                nh = max(0.001, min(1, nh))

                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if lines:
                with open(label_path, "w") as lf:
                    lf.write("\n".join(lines))
                count += 1

    print(f"Converted {count} annotations from {odgt_path} → {labels_dir}")


def train():
    """Run the fine-tuning."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.  Run:  pip install ultralytics")
        sys.exit(1)

    yaml_path = create_dataset_yaml()

    # Verify dataset exists
    train_imgs = os.path.join(DATASET_DIR, "images", "train")
    if not os.path.isdir(train_imgs):
        print(f"ERROR: Training images not found at {train_imgs}")
        prepare_crowdhuman()
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Fine-tuning {YOLO_BASE_MODEL} on CrowdHuman")
    print(f"  Epochs: {EPOCHS}  |  Image size: {IMGSZ}  |  Batch: {BATCH}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    model = YOLO(YOLO_BASE_MODEL)
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        # Fine-tuning specific settings
        lr0=0.001,          # lower learning rate (don't destroy pre-trained features)
        lrf=0.01,           # final LR = lr0 * lrf
        warmup_epochs=3,    # gentle warmup
        mosaic=0.8,         # mosaic augmentation (great for crowd scenes)
        mixup=0.1,          # mild mixup
        degrees=5.0,        # slight rotation augmentation
        scale=0.5,          # scale augmentation
        flipud=0.0,         # no vertical flip (unnatural for crowds)
        fliplr=0.5,         # horizontal flip
        patience=15,        # early stopping patience
        save=True,
        save_period=10,     # save checkpoint every 10 epochs
        plots=True,         # generate training plots
        verbose=True,
    )

    best_path = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best weights: {best_path}")
    print(f"")
    print(f"  To use in the dashboard, update detector.py:")
    print(f'    YOLO_MODEL_NAME = "{os.path.abspath(best_path)}"')
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    if "--convert" in sys.argv:
        # Convert CrowdHuman annotations to YOLO format
        for split in ("train", "val"):
            odgt = os.path.join(DATASET_DIR, f"annotation_{split}.odgt")
            imgs = os.path.join(DATASET_DIR, "images", split)
            lbls = os.path.join(DATASET_DIR, "labels", split)
            if os.path.exists(odgt):
                convert_odgt_to_yolo(odgt, imgs, lbls)
            else:
                print(f"Annotation file not found: {odgt}")
        print("\nConversion done. Now run:  python train_yolo.py")
    elif "--help" in sys.argv:
        prepare_crowdhuman()
    else:
        train()

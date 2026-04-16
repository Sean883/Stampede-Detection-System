"""density_model.py — Lightweight crowd density estimator (CSRNet-lite).

A small CNN that predicts a density heatmap from a crowd image.
Summing the heatmap = estimated person count.
The spatial distribution shows WHERE the crowd is packed.

Architecture:
    VGG-16 front-end (first 10 layers, pre-trained on ImageNet)
        → extracts hierarchical features
    Dilated convolution back-end (6 layers)
        → preserves spatial resolution while seeing large context
    1×1 conv → density map (1 channel)

Training data:
    ShanghaiTech Part A or Part B, or custom annotated frames.
    Each image has dot annotations (head positions).
    Ground truth = sum of Gaussians placed at each head.

Usage:
    # Train:
    python density_model.py --train --data datasets/shanghaitech_a

    # Test on image:
    python density_model.py --predict --image crowd.jpg

    # Use from dashboard:
    from density_model import DensityEstimator
    estimator = DensityEstimator("density_model_best.pth")
    density_map, count = estimator.predict(frame)

Requirements:
    pip install torch torchvision
"""

import os
import sys
import numpy as np
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join(os.path.dirname(__file__), "density_model_best.pth")
INPUT_SIZE       = (512, 512)   # resize input for training/inference
DENSITY_SIGMA    = 4            # Gaussian sigma for ground truth generation (at output resolution)
DENSITY_SCALE    = 100.0        # scale factor to make density map values learnable
LEARNING_RATE    = 1e-4
BATCH_SIZE       = 4
EPOCHS           = 100
DEVICE           = "cuda" if (lambda: __import__('torch').cuda.is_available())() else "cpu"

# ── Try importing torch ──────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class CSRNetLite(nn.Module):
        """Lightweight CSRNet for crowd density estimation.

        Front-end: first 10 layers of VGG-16 (conv1 through conv3, pre-trained).
        Back-end: 6 dilated convolution layers that maintain spatial resolution.
        Output: single-channel density map (quarter resolution of input).

        Why dilated convolutions?
        ─────────────────────────
        Normal convolutions with pooling shrink the feature map rapidly
        (256×256 → 16×16). We lose WHERE people are — only know HOW MANY.

        Dilated (atrous) convolutions insert gaps between kernel elements:
        - A 3×3 kernel with dilation=2 covers a 5×5 area
        - A 3×3 kernel with dilation=4 covers a 9×9 area
        This means each neuron "sees" a large context window (67×67 pixels)
        while the feature map stays at 64×64 resolution.

        Result: the output density map has enough spatial detail to show
        exactly which 10×10 pixel regions are dangerously packed.
        """

        def __init__(self, pretrained=True):
            super().__init__()

            # ── Front-end: VGG-16 layers up to conv3_3 ────────────────────────
            # These extract low → mid-level features:
            #   conv1: edges, gradients
            #   conv2: textures, small patterns
            #   conv3: body parts, head shapes
            vgg = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
            # [:16] stops BEFORE the 3rd MaxPool, so output is 1/4 of input.
            # (Using [:17] would include pool3, giving 1/8 — too coarse for
            # dense crowds where ~500 people must be localised.)
            features = list(vgg.features.children())[:16]  # up to relu3_3
            self.frontend = nn.Sequential(*features)

            # Freeze early layers (they already know edges/textures)
            for i, param in enumerate(self.frontend.parameters()):
                if i < 6:  # freeze conv1 and conv2
                    param.requires_grad = False

            # ── Back-end: dilated convolutions ────────────────────────────────
            # Input: 256 channels from VGG conv3
            # Each layer uses dilation to increase receptive field without
            # reducing spatial resolution.
            self.backend = nn.Sequential(
                # Layer 1: dilation=2, receptive field grows to 5×5
                nn.Conv2d(256, 128, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),

                # Layer 2: dilation=2, receptive field 9×9
                nn.Conv2d(128, 128, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),

                # Layer 3: dilation=4, receptive field 17×17
                nn.Conv2d(128, 64, 3, padding=4, dilation=4),
                nn.ReLU(inplace=True),

                # Layer 4: dilation=4, receptive field 33×33
                nn.Conv2d(64, 64, 3, padding=4, dilation=4),
                nn.ReLU(inplace=True),

                # Layer 5: dilation=8, receptive field 49×49
                nn.Conv2d(64, 32, 3, padding=8, dilation=8),
                nn.ReLU(inplace=True),

                # Layer 6: 1×1 conv to squeeze to density map
                nn.Conv2d(32, 1, 1),
                nn.ReLU(inplace=True),  # density must be non-negative
            )

            # Initialize backend weights
            for m in self.backend.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x):
            """
            Input:  x = (batch, 3, H, W) RGB image tensor
            Output: density_map = (batch, 1, H/4, W/4) density per pixel
                    Sum of density_map ≈ person count
            """
            x = self.frontend(x)   # (batch, 256, H/4, W/4)
            x = self.backend(x)    # (batch, 1, H/4, W/4)
            return x


# ══════════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_density_map(image_shape, points, output_shape, sigma=DENSITY_SIGMA):
    """Create a ground truth density map from head annotation points.

    Each annotated head becomes a 2D Gaussian on the map.
    The integral (sum) of the map equals the number of people.

    Parameters
    ----------
    image_shape : (H, W) of the ORIGINAL image (for scaling point coords)
    points : list of (x, y) head annotation coordinates in ORIGINAL image space
    output_shape : (dh, dw) the desired density-map resolution
    sigma : Gaussian spread in OUTPUT pixels

    Returns
    -------
    density_map : np.ndarray of shape output_shape
    """
    h, w = image_shape[:2]
    dh, dw = output_shape
    density = np.zeros((dh, dw), dtype=np.float32)

    if len(points) == 0:
        return density

    # Scale points from original-image coords to output-map coords
    scale_x = dw / w
    scale_y = dh / h

    for px, py in points:
        x = int(px * scale_x)
        y = int(py * scale_y)

        # Place a Gaussian at this point
        radius = max(int(3 * sigma), 3)

        y_min = max(0, y - radius)
        y_max = min(dh, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(dw, x + radius + 1)

        if y_min >= y_max or x_min >= x_max:
            continue

        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

        # Normalize so the Gaussian sums to 1.0 (one person)
        g_sum = gaussian.sum()
        if g_sum > 0:
            gaussian /= g_sum

        density[y_min:y_max, x_min:x_max] += gaussian

    return density


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_shanghaitech(data_dir):
    """Load ShanghaiTech dataset (Part A or Part B).

    Expected structure:
        data_dir/
            train_data/
                images/       ← .jpg files
                ground-truth/ ← GT_IMG_*.mat files
            test_data/
                images/
                ground-truth/

    Each .mat file contains a variable 'image_info' with head positions.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        print("ERROR: scipy required for ShanghaiTech.  pip install scipy")
        sys.exit(1)

    samples = []
    for split in ("train_data", "test_data"):
        img_dir = os.path.join(data_dir, split, "images")
        gt_dir = os.path.join(data_dir, split, "ground-truth")
        if not os.path.isdir(img_dir):
            continue

        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(img_dir, img_name)

            # Ground truth file: GT_IMG_1.mat for IMG_1.jpg
            base = os.path.splitext(img_name)[0]
            gt_name = f"GT_{base}.mat"
            gt_path = os.path.join(gt_dir, gt_name)

            if not os.path.exists(gt_path):
                continue

            mat = loadmat(gt_path)
            # ShanghaiTech stores points in image_info[0][0][0][0][0]
            points = mat["image_info"][0][0][0][0][0]  # (N, 2) array
            points = [(float(p[0]), float(p[1])) for p in points]

            # Read image shape once (needed for density map generation)
            shape_img = cv2.imread(img_path)
            if shape_img is None:
                continue
            img_h, img_w = shape_img.shape[:2]

            samples.append({
                "image_path": img_path,
                "points": points,
                "count": len(points),
                "image_shape": (img_h, img_w),
                "split": "train" if "train" in split else "test",
            })

    print(f"Loaded {len(samples)} samples from {data_dir}")
    train = [s for s in samples if s["split"] == "train"]
    test = [s for s in samples if s["split"] == "test"]
    print(f"  Train: {len(train)}  |  Test: {len(test)}")
    return train, test


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(data_dir):
    """Train the CSRNet-lite density estimator."""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not installed.  pip install torch torchvision")
        sys.exit(1)

    device = torch.device(DEVICE)
    print(f"Training on device: {device}")

    # Load dataset
    train_samples, test_samples = load_shanghaitech(data_dir)
    if not train_samples:
        print("ERROR: No training data found.")
        sys.exit(1)

    # Create model
    model = CSRNetLite(pretrained=True).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable:        {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}\n")

    best_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        np.random.shuffle(train_samples)

        for i in range(0, len(train_samples), BATCH_SIZE):
            batch = train_samples[i:i + BATCH_SIZE]

            images = []
            targets = []

            for sample in batch:
                # Load and preprocess image
                img = cv2.imread(sample["image_path"])
                if img is None:
                    continue
                img = cv2.resize(img, INPUT_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                # Normalize with ImageNet stats
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img - mean) / std
                img = img.transpose(2, 0, 1)  # HWC → CHW
                images.append(img)

                # Generate density map directly at output resolution
                # (INPUT_SIZE // 4) so sigma is meaningful in output pixels.
                orig_h, orig_w = sample["image_shape"]  # cached below
                out_h, out_w = INPUT_SIZE[1] // 4, INPUT_SIZE[0] // 4
                dmap = generate_density_map(
                    (orig_h, orig_w), sample["points"], (out_h, out_w))
                # Scale up so values are learnable (divide back at inference)
                dmap = dmap * DENSITY_SCALE
                targets.append(dmap)

            if not images:
                continue

            x = torch.FloatTensor(np.array(images)).to(device)
            y = torch.FloatTensor(np.array(targets)).unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred = model(x)

            # Resize prediction to match target if needed
            if pred.shape[2:] != y.shape[2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=y.shape[2:], mode="bilinear", align_corners=False)

            # Spatial MSE: teaches the model WHERE density goes
            mse_loss = criterion(pred, y)

            # Count loss: forces the total mass (count) to match the target.
            # Without this, MSE collapses to near-zero predictions because
            # sparse targets make "predict zero everywhere" a strong local min.
            pred_sum = pred.sum(dim=(1, 2, 3))   # (B,)
            targ_sum = y.sum(dim=(1, 2, 3))      # (B,)
            count_loss = torch.abs(pred_sum - targ_sum).mean()

            loss = mse_loss + 0.01 * count_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Evaluate on test set
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            maes = []
            with torch.no_grad():
                for sample in test_samples[:50]:  # evaluate on subset
                    img = cv2.imread(sample["image_path"])
                    if img is None:
                        continue
                    img = cv2.resize(img, INPUT_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = (img - mean) / std
                    img = img.transpose(2, 0, 1)
                    x = torch.FloatTensor(img).unsqueeze(0).to(device)

                    pred = model(x)
                    pred_count = pred.sum().item() / DENSITY_SCALE
                    actual_count = sample["count"]
                    maes.append(abs(pred_count - actual_count))

            mae = np.mean(maes) if maes else 0
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={epoch_loss / max(len(train_samples), 1):.6f}  "
                  f"MAE={mae:.1f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"  → Saved best model (MAE={mae:.1f})")

    print(f"\nTraining complete. Best MAE: {best_mae:.1f}")
    print(f"Model saved to: {MODEL_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE WRAPPER (used by dashboard)
# ══════════════════════════════════════════════════════════════════════════════

class DensityEstimator:
    """Wrapper for using the trained density model in the dashboard.

    Usage:
        estimator = DensityEstimator("density_model_best.pth")
        density_map, count = estimator.predict(bgr_frame)
        # density_map: (H/4, W/4) float32 heatmap
        # count: estimated total person count
    """

    def __init__(self, model_path=None):
        self.available = False
        if not TORCH_AVAILABLE:
            print("[density_model] PyTorch not available")
            return

        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            print(f"[density_model] Model not found: {path}")
            print("  Train first:  python density_model.py --train --data <path>")
            return

        self.device = torch.device(DEVICE)
        self.model = CSRNetLite(pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.available = True
        print(f"[density_model] Loaded density estimator from {path}")

    def predict(self, bgr_frame):
        """Predict density map and count from a BGR frame.

        Returns (density_map, estimated_count)
        """
        if not self.available:
            return None, 0

        img = cv2.resize(bgr_frame, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)

        with torch.no_grad():
            x = torch.FloatTensor(img).unsqueeze(0).to(self.device)
            pred = self.model(x)
            density_map = pred.squeeze().cpu().numpy() / DENSITY_SCALE
            count = float(density_map.sum())

        return density_map, max(0, count)

    def predict_for_roi(self, bgr_frame, roi_coords):
        """Predict density within a specific ROI.

        Parameters
        ----------
        roi_coords : (y1, y2, x1, x2) in full-frame pixels

        Returns (density_in_roi, count_in_roi, full_density_map)
        """
        density_map, total_count = self.predict(bgr_frame)
        if density_map is None:
            return 0.0, 0, None

        fh, fw = bgr_frame.shape[:2]
        dh, dw = density_map.shape
        y1, y2, x1, x2 = roi_coords

        # Map ROI to density map coordinates
        ry1 = int(y1 / fh * dh)
        ry2 = int(y2 / fh * dh)
        rx1 = int(x1 / fw * dw)
        rx2 = int(x2 / fw * dw)

        roi_map = density_map[ry1:ry2, rx1:rx2]
        count_in_roi = float(roi_map.sum())

        return count_in_roi, int(round(count_in_roi)), density_map


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION (for testing)
# ══════════════════════════════════════════════════════════════════════════════

def predict_and_show(image_path):
    """Run prediction on a single image and display results."""
    estimator = DensityEstimator()
    if not estimator.available:
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return

    density_map, count = estimator.predict(img)

    print(f"Image: {image_path}")
    print(f"Estimated count: {count:.1f}")

    # Create heatmap visualization
    dmap_norm = density_map / (density_map.max() + 1e-6)
    heatmap = cv2.applyColorMap(
        (dmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Blend with original
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.putText(overlay, f"Count: {count:.0f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Density Estimation", overlay)
    cv2.imshow("Density Map", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--train" in sys.argv:
        data_dir = None
        for i, arg in enumerate(sys.argv):
            if arg == "--data" and i + 1 < len(sys.argv):
                data_dir = sys.argv[i + 1]
            if arg == "--epochs" and i + 1 < len(sys.argv):
                EPOCHS = int(sys.argv[i + 1])
        if data_dir is None:
            print("Usage: python density_model.py --train --data <shanghaitech_dir>")
            print("\nDownload ShanghaiTech from:")
            print("  https://github.com/desenzhou/ShanghaiTechDataset")
            print("\nExpected structure:")
            print("  <dir>/train_data/images/")
            print("  <dir>/train_data/ground-truth/")
            print("  <dir>/test_data/images/")
            print("  <dir>/test_data/ground-truth/")
            sys.exit(1)
        train_model(data_dir)

    elif "--predict" in sys.argv:
        img_path = None
        for i, arg in enumerate(sys.argv):
            if arg == "--image" and i + 1 < len(sys.argv):
                img_path = sys.argv[i + 1]
        if img_path is None:
            print("Usage: python density_model.py --predict --image <path>")
            sys.exit(1)
        predict_and_show(img_path)

    else:
        print("""
Crowd Density Estimator (CSRNet-lite)
═════════════════════════════════════

Commands:
  python density_model.py --train --data <shanghaitech_dir>
      Train the model on ShanghaiTech dataset

  python density_model.py --predict --image <image.jpg>
      Run prediction on a single image

  python density_model.py --help
      Show this message

From Python:
  from density_model import DensityEstimator
  est = DensityEstimator("density_model_best.pth")
  density_map, count = est.predict(frame)
""")

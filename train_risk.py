"""train_risk.py — Learn optimal risk formula weights from labeled data.

Instead of hardcoding risk = density^0.8 + 0.15*motion*density + 0.20*bneck,
this script:

  1. Runs the analysis pipeline on your video(s) and extracts features
     (density, motion, bottleneck) per frame.
  2. Opens a simple labeling GUI where YOU label each sampled frame as
     NORMAL (0), CONGESTED (1), or ALERT (2).
  3. Trains a small model to learn the mapping:
     [density, motion, bottleneck] → risk label
  4. Exports the learned weights back into test8.py.

Usage:
    python train_risk.py                      # label + train
    python train_risk.py --video path.mp4     # use specific video
    python train_risk.py --retrain            # retrain from existing labels

Requirements:
    pip install scikit-learn
"""

import cv2
import numpy as np
import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test8 import (
    PROCESS_SCALE, calculate_density_index_lower_roi,
    compute_flow_direction_map, detect_bottleneck,
    BOTTLENECK_SCORE_THRESH, BOTTLENECK_TEMPORAL_LEN,
    ROI_Y1_FRAC, ROI_Y2_FRAC, ROI_X1_FRAC, ROI_X2_FRAC,
)

# Try importing person-aware functions
try:
    from detector import init_detector, detect_persons, YOLO_AVAILABLE
    from test8 import (calculate_density_from_detections,
                       compute_person_velocities, detect_bottleneck_persons)
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

LABELS_FILE = os.path.join(os.path.dirname(__file__), "risk_labels.json")
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "risk_weights.json")
SAMPLE_INTERVAL = 15  # label every Nth frame (keep labeling manageable)


def extract_features(video_path, max_frames=2000):
    """Run the analysis pipeline and extract per-frame features.

    Returns list of dicts:
        [{frame, density, motion, bottleneck, image(BGR)}]
    """
    print(f"Extracting features from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diag = np.sqrt(dw ** 2 + dh ** 2)

    # Init YOLO if available
    use_yolo = False
    if _HAS_YOLO:
        init_detector()
        from detector import YOLO_AVAILABLE
        use_yolo = YOLO_AVAILABLE
        if use_yolo:
            print("  Using YOLO person detection")
        else:
            print("  YOLO not available, using legacy pipeline")

    prev_gray = None
    prev_dets = []
    running_scores = []
    features = []
    frame_n = 0

    while frame_n < max_frames:
        ret, frame_full = cap.read()
        if not ret:
            break
        frame_n += 1

        if frame_n % SAMPLE_INTERVAL != 0:
            # Still need to update prev_gray for optical flow continuity
            if PROCESS_SCALE != 1.0:
                proc = cv2.resize(frame_full, (0, 0),
                                  fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                                  interpolation=cv2.INTER_AREA)
            else:
                proc = frame_full.copy()
            prev_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            continue

        # Process this frame
        if PROCESS_SCALE != 1.0:
            proc = cv2.resize(frame_full, (0, 0),
                              fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                              interpolation=cv2.INTER_AREA)
        else:
            proc = frame_full.copy()
        proc_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        _, flow_motion, flow_data = compute_flow_direction_map(prev_gray, proc_gray)

        fh, fw = frame_full.shape[:2]
        roi = (int(fh * ROI_Y1_FRAC), int(fh * ROI_Y2_FRAC),
               int(fw * ROI_X1_FRAC), int(fw * ROI_X2_FRAC))

        if use_yolo:
            detections = detect_persons(proc, PROCESS_SCALE, track=True)
            density, pc, _ = calculate_density_from_detections(
                detections, roi, frame_full.shape)
            _, _, motion_m = compute_person_velocities(
                detections, prev_dets, fps, frame_diag=frame_diag)
            if motion_m == 0.0 and flow_motion > 0.0:
                motion_m = flow_motion
            bneck_score, _, _ = detect_bottleneck_persons(
                detections, flow_data, roi, frame_full.shape)
            prev_dets = detections
        else:
            density, roi = calculate_density_index_lower_roi(frame_full)
            motion_m = flow_motion
            bneck_score, _, _ = detect_bottleneck(proc_gray, flow_data, roi)

        running_scores.append(bneck_score)
        if len(running_scores) > BOTTLENECK_TEMPORAL_LEN:
            running_scores.pop(0)
        smoothed = float(np.mean(running_scores))
        bneck = 1.0 if smoothed >= BOTTLENECK_SCORE_THRESH else 0.0

        # Resize frame for display in labeling GUI
        display = cv2.resize(frame_full, (480, 270))

        features.append({
            "frame": frame_n,
            "density": float(density),
            "motion": float(motion_m),
            "bottleneck": float(bneck),
            "image": display,  # small BGR for labeling GUI
        })

        prev_gray = proc_gray.copy()

        if frame_n % 100 == 0:
            print(f"  Processed frame {frame_n}...")

    cap.release()
    print(f"  Extracted {len(features)} sample frames")
    return features


class LabelingGUI:
    """Simple GUI for labeling frames as NORMAL / CONGESTED / ALERT."""

    def __init__(self, features, existing_labels=None):
        self.features = features
        self.labels = existing_labels or {}
        self.idx = 0
        self.done = False

        self.root = tk.Tk()
        self.root.title("Risk Label Tool — Label each frame")
        self.root.configure(bg="#1a1a2e")

        # Frame display
        self.canvas = tk.Label(self.root, bg="black")
        self.canvas.pack(padx=10, pady=10)

        # Info
        self.info_lbl = tk.Label(self.root, text="",
                                  font=("Consolas", 11),
                                  fg="#cccccc", bg="#1a1a2e")
        self.info_lbl.pack()

        # Feature display
        self.feat_lbl = tk.Label(self.root, text="",
                                  font=("Consolas", 10),
                                  fg="#888899", bg="#1a1a2e")
        self.feat_lbl.pack(pady=(0, 8))

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#1a1a2e")
        btn_frame.pack(pady=10)

        bs = {"font": ("Consolas", 12, "bold"), "relief": "flat",
              "padx": 20, "pady": 8, "cursor": "hand2", "bd": 0}

        tk.Button(btn_frame, text="NORMAL (N)", bg="#0a4a0a", fg="white",
                  command=lambda: self._label(0), **bs).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="CONGESTED (C)", bg="#4a3a0a", fg="white",
                  command=lambda: self._label(1), **bs).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="ALERT (A)", bg="#4a0a0a", fg="white",
                  command=lambda: self._label(2), **bs).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Skip (S)", bg="#2a2a3a", fg="#888899",
                  command=lambda: self._skip(), **bs).pack(side=tk.LEFT, padx=4)

        # Keyboard bindings
        self.root.bind("n", lambda e: self._label(0))
        self.root.bind("c", lambda e: self._label(1))
        self.root.bind("a", lambda e: self._label(2))
        self.root.bind("s", lambda e: self._skip())
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Escape>", lambda e: self._finish())

        # Progress
        self.prog_lbl = tk.Label(self.root, text="",
                                  font=("Consolas", 9),
                                  fg="#555566", bg="#1a1a2e")
        self.prog_lbl.pack(pady=(0, 10))

        self._show_frame()
        self.root.mainloop()

    def _show_frame(self):
        if self.idx >= len(self.features):
            self._finish()
            return

        f = self.features[self.idx]
        bgr = f["image"]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self._photo)

        frame_key = str(f["frame"])
        existing = self.labels.get(frame_key, "?")
        label_names = {0: "NORMAL", 1: "CONGESTED", 2: "ALERT", "?": "unlabeled"}
        existing_str = label_names.get(existing, "unlabeled")

        self.info_lbl.config(
            text=f"Frame {f['frame']}  |  Current label: {existing_str}")
        self.feat_lbl.config(
            text=f"density={f['density']:.3f}  motion={f['motion']:.3f}  "
                 f"bottleneck={f['bottleneck']:.0f}")
        labeled = sum(1 for v in self.labels.values() if v != "?")
        self.prog_lbl.config(
            text=f"Progress: {self.idx + 1}/{len(self.features)}  |  "
                 f"Labeled: {labeled}  |  Keys: N/C/A/S  |  Esc=done")

    def _label(self, value):
        f = self.features[self.idx]
        self.labels[str(f["frame"])] = value
        self.idx += 1
        self._show_frame()

    def _skip(self):
        self.idx += 1
        self._show_frame()

    def _prev(self):
        self.idx = max(0, self.idx - 1)
        self._show_frame()

    def _finish(self):
        self.done = True
        self.root.destroy()


def save_labels(labels, features, video_path):
    """Save labels + features to JSON."""
    data = {
        "video": video_path,
        "timestamp": datetime.now().isoformat(),
        "sample_interval": SAMPLE_INTERVAL,
        "labels": {},
    }

    feat_map = {str(f["frame"]): f for f in features}
    for frame_key, label in labels.items():
        if label == "?":
            continue
        f = feat_map.get(frame_key, {})
        data["labels"][frame_key] = {
            "label": label,
            "density": f.get("density", 0),
            "motion": f.get("motion", 0),
            "bottleneck": f.get("bottleneck", 0),
        }

    with open(LABELS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data['labels'])} labels to {LABELS_FILE}")


def train_risk_model():
    """Train a risk model from labeled data and export weights."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import classification_report
    except ImportError:
        print("ERROR: scikit-learn not installed.  Run:  pip install scikit-learn")
        sys.exit(1)

    if not os.path.exists(LABELS_FILE):
        print(f"ERROR: No labels file found at {LABELS_FILE}")
        print("Run this script without --retrain first to create labels.")
        sys.exit(1)

    with open(LABELS_FILE, "r") as f:
        data = json.load(f)

    labels_data = data["labels"]
    if len(labels_data) < 20:
        print(f"WARNING: Only {len(labels_data)} labeled frames. "
              "At least 50-100 recommended for reliable training.")

    # Build feature matrix
    X = []
    y = []
    for frame_key, info in labels_data.items():
        density = info["density"]
        motion = info["motion"]
        bneck = info["bottleneck"]
        label = info["label"]
        # Features: [density, motion, bottleneck, density*motion, density^2]
        X.append([density, motion, bneck, density * motion, density ** 2])
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"\nTraining on {len(X)} samples")
    print(f"  NORMAL:    {(y == 0).sum()}")
    print(f"  CONGESTED: {(y == 1).sum()}")
    print(f"  ALERT:     {(y == 2).sum()}")

    # Train logistic regression (interpretable + fast)
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    print(f"\nTraining accuracy: {(y_pred == y).mean():.1%}")
    print("\nClassification report:")
    print(classification_report(y, y_pred,
                                target_names=["NORMAL", "CONGESTED", "ALERT"]))

    # Extract learned weights
    # The model has 3 sets of coefficients (one per class)
    # For a continuous risk score, we use the ALERT-class probability
    coefs = model.coef_.tolist()
    intercepts = model.intercept_.tolist()

    weights = {
        "model_type": "logistic_regression",
        "feature_names": ["density", "motion", "bottleneck",
                          "density*motion", "density^2"],
        "coefficients": {
            "NORMAL": coefs[0],
            "CONGESTED": coefs[1],
            "ALERT": coefs[2],
        },
        "intercepts": {
            "NORMAL": intercepts[0],
            "CONGESTED": intercepts[1],
            "ALERT": intercepts[2],
        },
        "n_samples": len(X),
        "accuracy": float((y_pred == y).mean()),
        "trained_at": datetime.now().isoformat(),
    }

    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\nWeights saved to: {WEIGHTS_FILE}")

    # Print human-readable interpretation
    print("\n" + "=" * 60)
    print("  LEARNED RISK WEIGHTS")
    print("=" * 60)
    alert_coefs = coefs[2]  # ALERT class
    feat_names = ["density", "motion", "bottleneck",
                  "density*motion", "density^2"]
    print("\n  ALERT class coefficients (higher = more risk):")
    for name, coef in zip(feat_names, alert_coefs):
        print(f"    {name:>16s}: {coef:+.4f}")
    print(f"    {'intercept':>16s}: {intercepts[2]:+.4f}")

    print("\n  To use these weights, the dashboard will load them from")
    print(f"  {WEIGHTS_FILE} at startup.")
    print("=" * 60)

    return weights


def main():
    if "--retrain" in sys.argv:
        train_risk_model()
        return

    # Get video path
    video_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--video" and i + 1 < len(sys.argv):
            video_path = sys.argv[i + 1]

    if video_path is None:
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="Select video for labeling",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"),
                       ("All files", "*.*")])
        root.destroy()
        if not video_path:
            print("No video selected. Exiting.")
            return

    # Extract features
    features = extract_features(video_path)
    if not features:
        print("No features extracted. Exiting.")
        return

    # Load existing labels if available
    existing_labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            data = json.load(f)
        existing_labels = {k: v["label"] for k, v in data.get("labels", {}).items()}
        print(f"Loaded {len(existing_labels)} existing labels")

    # Open labeling GUI
    print(f"\nOpening labeling GUI for {len(features)} frames...")
    print("  Keys: N=Normal, C=Congested, A=Alert, S=Skip, Esc=Done")
    gui = LabelingGUI(features, existing_labels)

    if gui.labels:
        save_labels(gui.labels, features, video_path)

        labeled_count = sum(1 for v in gui.labels.values() if v != "?")
        if labeled_count >= 20:
            print("\nTraining risk model...")
            train_risk_model()
        else:
            print(f"\nOnly {labeled_count} frames labeled. "
                  "Label at least 20 for training.")
            print("Run again to label more, then:  python train_risk.py --retrain")


if __name__ == "__main__":
    main()

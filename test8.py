
import cv2
import numpy as np
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

# ========== User config ==========
VIDEO_FILE = r'C:\Users\seans\Desktop\project_videos\my_crowd_video.mp4'
WINDOW_NAME = "RTSDA Dashboard"

# Performance / real-time tweaks
PROCESS_SCALE = 0.6      # processing scale (0.5..1.0). Lower -> faster processing
ENABLE_FRAME_SKIPPING = True

# Panel width (requested change)
PANEL_WIDTH_FRAC = 0.40  # 40% of video width (wider side panel)

# UI / thresholds
RISK_ALERT_THRESHOLD = 0.6
PLAY_MODE_AUTOSTART = True

# Bottleneck detection
BOTTLENECK_SCORE_THRESH  = 0.32   # smoothed score (0–1) required to declare a bottleneck
BOTTLENECK_TEMPORAL_LEN  = 30     # frames to smooth over (~1.2 s at 25 fps)

# Lower ROI (focus on human area to avoid roof/structure)
ROI_Y1_FRAC = 0.55
ROI_Y2_FRAC = 0.95
ROI_X1_FRAC = 0.15
ROI_X2_FRAC = 0.85

# Person-aware analysis (requires detector.py + ultralytics)
DENSITY_SATURATION  = 25     # person count at which density = 1.0 (tune per scene)
SPEED_SATURATION    = 200.0  # px/s at which motion_metric = 1.0
MAX_PERSON_SPEED    = 500.0  # cap per-person speed to suppress track-ID swaps
YOLO_SKIP_FRAMES    = 2      # run YOLO every Nth frame (1 = every frame)

# Colors
COLOR_NORMAL = (0, 255, 0)
COLOR_CONGESTED = (0, 165, 255)
COLOR_ALERT = (0, 0, 255)
COLOR_PERSON_BOX = (0, 200, 100)  # green boxes around detected persons
PANEL_BG_COLOR = (30, 30, 30)

# ========== Helper functions ==========

def calculate_density_index_lower_roi(frame_full):
    """Brightness-based density proxy inside lower ROI (full-res)."""
    gray = cv2.cvtColor(frame_full, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    y1 = int(h * ROI_Y1_FRAC)
    y2 = int(h * ROI_Y2_FRAC)
    x1 = int(w * ROI_X1_FRAC)
    x2 = int(w * ROI_X2_FRAC)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, (y1, y2, x1, x2)
    density_index = np.mean(roi) / 255.0
    return np.clip(density_index * 1.5, 0.0, 1.0), (y1, y2, x1, x2)

# ========== Person-aware functions (used when YOLO is available) ==========

def calculate_density_from_detections(detections, roi_coords, frame_shape):
    """Count persons whose center falls inside the ROI.

    Returns (density_index: float 0..1, person_count: int, roi_coords)
    """
    y1, y2, x1, x2 = roi_coords
    count = 0
    for d in detections:
        cx, cy = d["center"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            count += 1
    density = float(np.clip(count / max(DENSITY_SATURATION, 1), 0.0, 1.0))
    return density, count, roi_coords


def compute_person_velocities(current_dets, prev_dets, fps,
                              frame_diag=None):
    """Compute per-person speed from tracked detections across two frames.

    Parameters
    ----------
    frame_diag : float or None
        Diagonal of the full-resolution frame in pixels.  When provided,
        speeds are normalised as a fraction of the diagonal before being
        compared to SPEED_SATURATION.  This makes the motion metric
        resolution-independent (a person walking at the same real-world
        speed produces the same metric at 480p and 1080p).

    Returns (avg_speed_norm, max_speed_norm, motion_metric 0..1)
    """
    if not current_dets or not prev_dets or fps <= 0:
        return 0.0, 0.0, 0.0

    prev_map = {}
    for d in prev_dets:
        tid = d.get("track_id")
        if tid is not None:
            prev_map[tid] = d["center"]

    speeds = []
    for d in current_dets:
        tid = d.get("track_id")
        if tid is not None and tid in prev_map:
            px, py = prev_map[tid]
            cx, cy = d["center"]
            disp = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            speed = disp * fps  # px/sec
            speed = min(speed, MAX_PERSON_SPEED)  # cap outliers from ID swaps
            speeds.append(speed)

    if not speeds:
        return 0.0, 0.0, 0.0

    avg_s = float(np.mean(speeds))
    max_s = float(np.max(speeds))

    # Normalise by frame diagonal so metric is resolution-independent
    if frame_diag and frame_diag > 0:
        avg_s_norm = avg_s / frame_diag
        max_s_norm = max_s / frame_diag
        # SPEED_SATURATION is now interpreted as fraction-of-diagonal per second
        # e.g. 0.15 means 15% of diagonal per second ≈ brisk walk
        motion_metric = float(np.clip(avg_s_norm / 0.15, 0.0, 1.0))
    else:
        motion_metric = float(np.clip(avg_s / SPEED_SATURATION, 0.0, 1.0))
        avg_s_norm = avg_s
        max_s_norm = max_s

    return avg_s_norm, max_s_norm, motion_metric


# ── Weidmann fundamental diagram anomaly (from Dufour et al. 2025) ───────────
# Weidmann parameters (standard pedestrian dynamics literature)
_WEIDMANN_V0    = 1.34   # free-flow speed m/s
_WEIDMANN_GAMMA = 1.913  # shape parameter m^-2
_WEIDMANN_RMAX  = 5.4    # jam density ped/m^2

def weidmann_expected_speed(density_ped_m2):
    """Expected walking speed at a given density (Weidmann 1993).

    v(rho) = v0 * [1 - exp(-gamma * (1/rho - 1/rho_max))]
    Returns speed in m/s.
    """
    rho = max(density_ped_m2, 0.01)
    if rho >= _WEIDMANN_RMAX:
        return 0.0
    return _WEIDMANN_V0 * (1.0 - np.exp(-_WEIDMANN_GAMMA * (1.0 / rho - 1.0 / _WEIDMANN_RMAX)))


def compute_speed_anomaly(density_norm, motion_metric, density_saturation=None):
    """Compare observed speed against Weidmann's fundamental diagram.

    Maps our normalised density (0-1) back to an approximate ped/m^2 using
    DENSITY_SATURATION and a reference area, then checks if the observed
    motion deviates from what the fundamental diagram predicts.

    Returns speed_anomaly in 0..1:
      - 0.0 = behaviour matches the fundamental diagram (normal)
      - ~0.5 = moderately abnormal (people slower OR faster than expected)
      - 1.0 = extreme deviation (frozen crowd or panicked running)
    """
    sat = density_saturation or DENSITY_SATURATION
    # Rough conversion: at DENSITY_SATURATION persons, density ~2 ped/m^2
    ROI_AREA_M2 = sat / 2.0  # e.g. sat=25 -> 12.5 m^2 -> 25/12.5=2.0 ped/m^2 at max
    person_count_est = density_norm * sat
    density_ped_m2 = person_count_est / ROI_AREA_M2

    if density_ped_m2 < 0.3:
        return 0.0  # too sparse for meaningful comparison

    expected_speed = weidmann_expected_speed(density_ped_m2)
    # Normalise expected speed the same way as motion_metric (fraction of free-flow)
    expected_metric = expected_speed / _WEIDMANN_V0

    # Deviation: how far is observed from expected (signed)
    # Negative = slower than expected (stagnation/crush risk)
    # Positive = faster than expected (panic/running)
    deviation = motion_metric - expected_metric

    # Dead zone: people routinely walk 40% slower than Weidmann predicts
    # (social groups, strolling, chatting — noted in Dufour et al. 2025 Fig. 8)
    DEAD_ZONE = 0.40

    if deviation < 0:
        # Slower than expected — only flag if MUCH slower
        excess = max(0.0, abs(deviation) - DEAD_ZONE)
        anomaly = excess * 2.0  # scale up: remaining deviation is serious
    else:
        # Faster than expected — possible panic (less common)
        excess = max(0.0, deviation - 0.15)
        anomaly = excess * 1.5

    return float(np.clip(anomaly, 0.0, 1.0))


# ── Velocity variance / counterflow detection ───────────────────────────────

def compute_counterflow_metric(current_dets, prev_dets, fps,
                               frame_diag=None):
    """Measure how much individual velocities deviate from the mean flow.

    Inspired by Dufour et al. 2025: high velocity variance relative to the
    mean velocity field indicates counter-walking pedestrians, which is a
    key collision and crush risk factor.

    Returns (counterflow_metric 0..1, n_counter, n_total):
      - counterflow_metric: 0 = everyone moving the same way, 1 = chaotic
      - n_counter: number of people moving against the dominant flow
      - n_total: total matched people
    """
    if not current_dets or not prev_dets or fps <= 0:
        return 0.0, 0, 0

    prev_map = {}
    for d in prev_dets:
        tid = d.get("track_id")
        if tid is not None:
            prev_map[tid] = d["center"]

    velocities = []
    for d in current_dets:
        tid = d.get("track_id")
        if tid is not None and tid in prev_map:
            px, py = prev_map[tid]
            cx, cy = d["center"]
            vx = (cx - px) * fps
            vy = (cy - py) * fps
            speed = np.sqrt(vx ** 2 + vy ** 2)
            if speed > MAX_PERSON_SPEED:
                continue  # ID swap — skip
            velocities.append((vx, vy))

    n_total = len(velocities)
    if n_total < 3:
        return 0.0, 0, n_total

    vels = np.array(velocities, dtype=np.float32)

    # Mean velocity (the "base flow" from the paper)
    mean_vx = float(np.mean(vels[:, 0]))
    mean_vy = float(np.mean(vels[:, 1]))
    mean_speed = np.sqrt(mean_vx ** 2 + mean_vy ** 2)

    if mean_speed < 1e-3:
        # Everyone is essentially stationary — no counterflow concept
        return 0.0, 0, n_total

    # Velocity variance: mean squared deviation from the mean flow
    deviations = vels - np.array([mean_vx, mean_vy])
    var_v = float(np.mean(np.sum(deviations ** 2, axis=1)))

    # Normalise by frame diagonal if available
    norm = (frame_diag * fps) if (frame_diag and frame_diag > 0) else SPEED_SATURATION
    var_normalised = var_v / (norm ** 2 + 1e-6)

    # Count counter-walkers: dot product with mean flow < 0
    mean_dir = np.array([mean_vx, mean_vy]) / (mean_speed + 1e-6)
    dots = vels @ mean_dir
    n_counter = int(np.sum(dots < 0))

    # Counterflow metric blends variance and fraction of counter-walkers
    counter_frac = n_counter / n_total
    # Scale variance to 0..1 (empirical: var_normalised > 0.01 is very chaotic)
    var_score = float(np.clip(var_normalised / 0.01, 0.0, 1.0))

    metric = float(np.clip(0.5 * var_score + 0.5 * counter_frac, 0.0, 1.0))
    return metric, n_counter, n_total


# ── Trained risk model (loaded once, if available) ────────────────────────────
_trained_risk_model = None
_trained_risk_loaded = False

def _load_trained_risk():
    """Try to load learned risk weights from risk_weights.json."""
    global _trained_risk_model, _trained_risk_loaded
    if _trained_risk_loaded:
        return
    _trained_risk_loaded = True
    import json
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "risk_weights.json")
    if not os.path.exists(weights_path):
        return
    try:
        from sklearn.linear_model import LogisticRegression
        with open(weights_path, "r") as f:
            data = json.load(f)
        coefs = [data["coefficients"]["NORMAL"],
                 data["coefficients"]["CONGESTED"],
                 data["coefficients"]["ALERT"]]
        intercepts = [data["intercepts"]["NORMAL"],
                      data["intercepts"]["CONGESTED"],
                      data["intercepts"]["ALERT"]]
        model = LogisticRegression(max_iter=1000)
        model.classes_ = np.array([0, 1, 2])
        model.coef_ = np.array(coefs)
        model.intercept_ = np.array(intercepts)
        _trained_risk_model = model
        print(f"[risk] Loaded trained risk model from {weights_path} "
              f"(accuracy={data.get('accuracy', '?')})")
    except Exception as exc:
        print(f"[risk] Could not load trained model: {exc}")


def compute_risk_score(density, motion_metric, bottleneck_active,
                       speed_anomaly=0.0, counterflow=0.0):
    """Risk score — uses trained model if available, else handcrafted formula.

    Used by BOTH the YOLO path and the legacy fallback path so that
    the same inputs always produce the same risk level.

    Parameters
    ----------
    speed_anomaly : float 0..1
        Weidmann fundamental diagram deviation (0 = normal, 1 = extreme).
    counterflow : float 0..1
        Velocity variance / counterflow metric (0 = uniform, 1 = chaotic).

    When risk_weights.json exists (produced by train_risk.py), uses the
    learned logistic regression to predict P(ALERT) as the risk score,
    then adds anomaly/counterflow boosts on top.
    Otherwise falls back to the handcrafted formula.
    """
    _load_trained_risk()

    if _trained_risk_model is not None:
        bneck_val = 1.0 if bottleneck_active else 0.0
        features = np.array([[density, motion_metric, bneck_val,
                               density * motion_metric, density ** 2]])
        probs = _trained_risk_model.predict_proba(features)[0]
        risk = float(probs[1] * 0.5 + probs[2] * 1.0)
        # Layer research-paper signals on top of learned base
        risk += 0.08 * speed_anomaly * density
        risk += 0.10 * counterflow * density
        return float(np.clip(risk, 0.0, 1.0))

    # Handcrafted fallback
    base = density ** 0.8                                 # concave — rises fast
    motion_boost = 0.15 * motion_metric * density         # motion matters when dense
    bneck_boost  = 0.20 if bottleneck_active else 0.0
    anomaly_boost = 0.10 * speed_anomaly * density        # Weidmann deviation
    cflow_boost   = 0.12 * counterflow * density          # counterflow danger
    risk = float(np.clip(base + motion_boost + bneck_boost
                         + anomaly_boost + cflow_boost, 0.0, 1.0))
    return risk


def detect_bottleneck_persons(detections, flow_data, roi_coords,
                              frame_shape, process_scale=PROCESS_SCALE):
    """Person-position-based bottleneck detection.

    1. Project person centres onto the X-axis within the ROI.
    2. Sort by X; compute gaps between adjacent persons.
    3. If the largest gap is below a threshold the passage is blocked.
    4. Combine with flow compression from optical flow for early warning.

    Returns (score 0-1, bbox, details) — same shape as detect_bottleneck.
    """
    y1, y2, x1, x2 = roi_coords
    roi_w = max(x2 - x1, 1)
    roi_h = max(y2 - y1, 1)

    # Filter detections inside ROI
    xs = []
    for d in detections:
        cx, cy = d["center"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            xs.append(cx)

    n_persons = len(xs)
    if n_persons < 2:
        return 0.0, (x1, y1, x2, y2), {"score": 0.0, "persons_in_roi": n_persons}

    xs.sort()

    # Gaps between adjacent persons + edges
    gaps = [xs[0] - x1]  # left edge to first person
    for i in range(1, len(xs)):
        gaps.append(xs[i] - xs[i - 1])
    gaps.append(x2 - xs[-1])  # last person to right edge

    max_gap = max(gaps)
    # Gap score: smaller max gap = more blocked
    # If max_gap < 15% of ROI width → heavily congested
    gap_score = float(np.clip(1.0 - max_gap / (roi_w * 0.4), 0.0, 1.0))

    # Crowd packing: how many people per unit width (same saturation as density)
    packing_score = float(np.clip(n_persons / max(DENSITY_SATURATION, 1), 0.0, 1.0))

    # Flow compression signal (reuse from optical flow)
    compression_score = 0.0
    if flow_data is not None and flow_data[0] is not None:
        mag, ang = flow_data
        ph, pw = mag.shape
        s = process_scale
        py1 = int(np.clip(y1 * s, 0, ph))
        py2 = int(np.clip(y2 * s, 0, ph))
        px1 = int(np.clip(x1 * s, 0, pw))
        px2 = int(np.clip(x2 * s, 0, pw))
        if py2 - py1 > 4 and px2 - px1 > 4:
            ang_roi = ang[py1:py2, px1:px2]
            mag_roi = mag[py1:py2, px1:px2]
            ang_rad = np.deg2rad(ang_roi)
            fx = (mag_roi * np.cos(ang_rad)).astype(np.float32)
            fy = (mag_roi * np.sin(ang_rad)).astype(np.float32)
            dfx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
            dfy = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)
            div_neg = np.clip(-(dfx + dfy), 0.0, None)
            p95 = float(np.percentile(div_neg, 95)) + 1e-6
            compression_score = float(np.clip(np.mean(div_neg) / p95, 0.0, 1.0))

    # Blend: person gaps + packing + flow compression
    score = float(np.clip(
        0.40 * gap_score +
        0.30 * packing_score +
        0.30 * compression_score,
        0.0, 1.0))

    # Bbox around the densest band (narrowest gap region)
    min_gap_idx = int(np.argmin(gaps))
    if min_gap_idx == 0:
        bx1 = x1
        bx2 = int(xs[0]) if xs else x2
    elif min_gap_idx >= len(xs):
        bx1 = int(xs[-1]) if xs else x1
        bx2 = x2
    else:
        bx1 = int(xs[min_gap_idx - 1])
        bx2 = int(xs[min_gap_idx])
    # Expand bbox a bit for visibility
    pad = int(roi_w * 0.05)
    bx1 = max(x1, bx1 - pad)
    bx2 = min(x2, bx2 + pad)

    bbox = (bx1, y1, bx2, y2)
    details = {
        "score": score,
        "gap_score": gap_score,
        "packing_score": packing_score,
        "compression_score": compression_score,
        "max_gap_px": max_gap,
        "persons_in_roi": n_persons,
    }
    return score, bbox, details


def draw_person_boxes(frame, detections, roi_coords=None):
    """Draw thin bounding boxes + optional track IDs on the frame (in-place)."""
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON_BOX, 1)
        tid = d.get("track_id")
        if tid is not None:
            cv2.putText(frame, str(tid), (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COLOR_PERSON_BOX, 1, cv2.LINE_AA)


# ========== Legacy / fallback functions ==========

def compute_flow_direction_map(prev_gray_p, curr_gray_p):
    """
    Farneback optical flow on processed grayscale images.
    Returns: flow_bgr (processed resolution), motion_metric (0..1), (mag, ang)
    """
    h, w = curr_gray_p.shape
    if prev_gray_p is None:
        return np.zeros((h, w, 3), dtype=np.uint8), 0.0, None

    flow = cv2.calcOpticalFlowFarneback(prev_gray_p, curr_gray_p, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Motion metric on processed coords (map ROI to processed size)
    ph, pw = curr_gray_p.shape
    fy1 = int(ph * ROI_Y1_FRAC)
    fy2 = int(ph * ROI_Y2_FRAC)
    fx1 = int(pw * ROI_X1_FRAC)
    fx2 = int(pw * ROI_X2_FRAC)
    mag_roi = mag[fy1:fy2, fx1:fx2] if mag is not None else np.zeros((1,1))
    motion_metric = np.clip(np.mean(mag_roi) / 10.0, 0.0, 1.0)

    return flow_bgr, motion_metric, (mag, ang)

def detect_bottleneck(proc_gray, flow_data, roi_coords,
                      process_scale=PROCESS_SCALE):
    """
    Physics-based crowd bottleneck detector using three complementary signals:

      1. Texture density  — Sobel gradient magnitude over the ROI.
                            Works for stationary crowds (unlike MOG2 foreground masks
                            which adapt away from still people after a few seconds).

      2. Stagnation       — texture density weighted by (1 - normalised speed).
                            High density + low optical-flow speed = crowd pressure
                            building up; the most dangerous pre-crush indicator.

      3. Flow compression — negative divergence of the optical-flow field.
                            div(F) < 0 means flow vectors are converging inward,
                            i.e. the crowd is being physically compressed at that
                            location.  This is the direct fluid-mechanics signature
                            of a bottleneck.

    The three maps are blended into a single bottleneck map.  The overall score is
    its mean; the bounding box is placed around the column band with the highest
    per-column average score (the actual congestion hotspot, not free space).

    Args:
        proc_gray    : uint8 grayscale frame at processed resolution
        flow_data    : (mag, ang) arrays in processed-resolution coords, or None
        roi_coords   : (y1, y2, x1, x2) in FULL-resolution pixels
        process_scale: factor used when downsampling (default = PROCESS_SCALE)

    Returns:
        score   : float 0–1, higher = stronger bottleneck signal
        bbox    : (bx1, by1, bx2, by2) in full-resolution pixels, around hotspot
        details : dict with signal maps and compatibility fields
    """
    y1_f, y2_f, x1_f, x2_f = roi_coords
    s = process_scale

    # Scale ROI to processed resolution and clamp to frame bounds
    ph, pw = proc_gray.shape
    y1 = int(np.clip(y1_f * s, 0, ph))
    y2 = int(np.clip(y2_f * s, 0, ph))
    x1 = int(np.clip(x1_f * s, 0, pw))
    x2 = int(np.clip(x2_f * s, 0, pw))
    roi_h, roi_w = y2 - y1, x2 - x1

    if roi_h < 4 or roi_w < 4:
        return 0.0, (x1_f, y1_f, x2_f, y2_f), {"score": 0.0, "widest_clearance": 0}

    # ── Signal 1: Texture density ─────────────────────────────────────────────
    gray_roi = proc_gray[y1:y2, x1:x2].astype(np.float32)
    gx = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    p95 = float(np.percentile(grad_mag, 95)) + 1e-6
    density_map = np.clip(grad_mag / p95, 0.0, 1.0)

    # ── Signals 2 & 3: Flow-based ─────────────────────────────────────────────
    stagnation_map  = np.zeros((roi_h, roi_w), dtype=np.float32)
    compression_map = np.zeros((roi_h, roi_w), dtype=np.float32)

    if flow_data is not None and flow_data[0] is not None:
        mag, ang = flow_data
        mag_roi = mag[y1:y2, x1:x2]
        ang_roi = ang[y1:y2, x1:x2]

        # Stagnation: dense texture where people are barely moving
        speed_norm     = np.clip(mag_roi / 8.0, 0.0, 1.0)   # 8 px/frame ≈ brisk walk
        stagnation_map = density_map * (1.0 - speed_norm)

        # Divergence: negative means vectors converging → physical compression
        ang_rad = np.deg2rad(ang_roi)
        fx = (mag_roi * np.cos(ang_rad)).astype(np.float32)
        fy = (mag_roi * np.sin(ang_rad)).astype(np.float32)
        dfx = cv2.Sobel(fx, cv2.CV_32F, 1, 0, ksize=3)
        dfy = cv2.Sobel(fy, cv2.CV_32F, 0, 1, ksize=3)
        compression_raw = np.clip(-(dfx + dfy), 0.0, None)
        p95_c = float(np.percentile(compression_raw, 95)) + 1e-6
        compression_map = np.clip(compression_raw / p95_c, 0.0, 1.0)

        bneck_map = (0.45 * stagnation_map +
                     0.35 * compression_map +
                     0.20 * density_map)
    else:
        # First frame — no flow yet; fall back to density only
        bneck_map = density_map

    # ── Overall score ─────────────────────────────────────────────────────────
    score = float(np.mean(bneck_map))

    # ── Localise the hotspot (peak-score column band) ─────────────────────────
    col_scores = np.mean(bneck_map, axis=0).astype(np.float32)
    ksize      = min(21, (roi_w // 4) | 1)          # odd, ≤ 21
    col_smooth = cv2.GaussianBlur(
        col_scores.reshape(1, -1), (1, ksize), 0).flatten()
    peak_col   = int(np.argmax(col_smooth))
    half_w     = max(int(roi_w * 0.15), 4)

    bx1_p = x1 + max(0,    peak_col - half_w)
    bx2_p = x1 + min(roi_w, peak_col + half_w)

    # Scale bbox back to full resolution
    inv_s = 1.0 / s
    bbox = (int(bx1_p * inv_s), y1_f,
            int(bx2_p * inv_s), y2_f)

    details = {
        "score":           score,
        "density_map":     density_map,
        "stagnation_map":  stagnation_map,
        "compression_map": compression_map,
        "bneck_map":       bneck_map,
        # Legacy compatibility field for dashboard
        "widest_clearance": int(roi_w * max(0.0, 1.0 - score)),
    }

    return score, bbox, details

def draw_side_panel(canvas, risk_score, density_val, motion_val, status_color, status_text, bottleneck=False):
    """
    Draw a right-side panel with metrics. PANEL_WIDTH_FRAC is used to set width.
    Pass combined_mask (full-res) to render a small inset showing the moving-foreground mask.
    """
    h, w = canvas.shape[:2]
    panel_w = int(w * PANEL_WIDTH_FRAC)
    out = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    out[:, :w] = canvas
    out[:, w:] = np.full((h, panel_w, 3), PANEL_BG_COLOR, dtype=np.uint8)

    start_x = w + 18
    y = 40
    line_h = 44
    title_font = cv2.FONT_HERSHEY_DUPLEX
    metric_font = cv2.FONT_HERSHEY_TRIPLEX
    small_font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(out, f"STATUS: {status_text}", (start_x, y), title_font, 1.1, status_color, 3, cv2.LINE_AA)
    y += line_h
    cv2.putText(out, f"LOCATION: Platform / Gate", (start_x, y), small_font, 0.85, (220, 220, 220), 2, cv2.LINE_AA)
    y += int(line_h * 0.9)
    cv2.putText(out, f"SCORE: {risk_score:.3f}", (start_x, y), metric_font, 0.95, (255, 255, 255), 3, cv2.LINE_AA)
    y += int(line_h * 1.0)
    cv2.putText(out, "Density:", (start_x, y), metric_font, 0.85, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(out, f"{density_val:.2f}", (start_x + 160, y), metric_font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    y += line_h
    cv2.putText(out, "Motion:", (start_x, y), metric_font, 0.85, (200, 200, 200), 2, cv2.LINE_AA)
    cv2.putText(out, f"{motion_val:.2f}", (start_x + 160, y), metric_font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    y += int(line_h * 0.9)

    if bottleneck:
        cv2.putText(out, "!! BOTTLENECK !!", (start_x, y), title_font, 0.85, COLOR_ALERT, 3, cv2.LINE_AA)
        y += int(line_h * 0.9)

    return out

# ========== Simple report-plot function ==========

def generate_report_plot(metrics):
    """
    Show interactive plots:
      - risk over frames
      - density & motion over frames
      - bottleneck timeline (step)
    This opens matplotlib windows (blocking) and returns when closed.
    """
    if not metrics:
        print("No metrics to plot.")
        return

    frames = [m["frame"] for m in metrics]
    density = [m["density"] for m in metrics]
    motion = [m["motion"] for m in metrics]
    risk = [m["risk"] for m in metrics]
    bott = [1 if m["bottleneck"] else 0 for m in metrics]

    plt.figure(figsize=(10, 3))
    plt.plot(frames, risk, label="Risk", linewidth=2)
    plt.ylim(0, 1)
    plt.xlabel("Frame")
    plt.ylabel("Risk")
    plt.title("Risk over time")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(10, 3))
    plt.plot(frames, density, label="Density", linewidth=1.5)
    plt.plot(frames, motion, label="Motion", linewidth=1.5)
    plt.ylim(0, 1)
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title("Density and Motion")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(10, 2))
    plt.step(frames, bott, where='post')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ["No", "Yes"])
    plt.xlabel("Frame")
    plt.title("Bottleneck detected (timeline)")
    plt.grid(True)

    plt.show()  # blocking until windows closed

# ========== Main analysis loop ==========

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay_ms = int(1000.0 / fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_w = width
    display_h = height

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    prev_gray_proc  = None
    running_scores  = []   # temporal smoothing for bottleneck score
    metrics         = []

    play_mode = PLAY_MODE_AUTOSTART
    ALERT_TRIGGERED = False
    frame_count = 0

    # wall-clock for skipping
    start_time_wall = time.time()

    while True:
        loop_start = time.time()
        if play_mode:
            ret, frame_full = cap.read()
            if not ret:
                break
            frame_count += 1
        else:
            # paused: reuse last frame if available
            if 'frame_full' not in locals():
                ret, frame_full = cap.read()
                if not ret:
                    break

        # Downscale for processing
        if PROCESS_SCALE != 1.0:
            proc = cv2.resize(frame_full, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE, interpolation=cv2.INTER_AREA)
        else:
            proc = frame_full.copy()
        proc_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        # Flow on processed frame
        flow_vis_proc, motion_metric, flow_data = compute_flow_direction_map(prev_gray_proc, proc_gray)
        flow_vis_full = cv2.resize(flow_vis_proc, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

        # Density on full-res lower ROI
        density_index, roi_coords = calculate_density_index_lower_roi(frame_full)

        # Bottleneck detection (physics-based: stagnation + compression + density)
        bneck_score, bbox, details = detect_bottleneck(proc_gray, flow_data, roi_coords)

        running_scores.append(bneck_score)
        if len(running_scores) > BOTTLENECK_TEMPORAL_LEN:
            running_scores.pop(0)
        smoothed_score = float(np.mean(running_scores)) if running_scores else bneck_score
        final_bneck    = smoothed_score >= BOTTLENECK_SCORE_THRESH

        # Compose overlay — real frame at full weight, subtle flow tint
        combined = cv2.addWeighted(frame_full, 1.0, flow_vis_full, 0.25, 0)

        # Draw bottleneck bounding box if flagged
        bx1, by1, bx2, by2 = bbox
        if final_bneck:
            cv2.rectangle(combined, (bx1, by1), (bx2, by2), COLOR_ALERT, 4)
            cv2.putText(combined, "BOTTLENECK", (bx1 + 5, by1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, COLOR_ALERT, 3, cv2.LINE_AA)

        # Risk score
        risk_score = 0.4 * density_index + 0.6 * motion_metric + 0.5 * (density_index * motion_metric)
        risk_score = np.clip(risk_score, 0.0, 1.0)

        status_text = "NORMAL"
        status_color = COLOR_NORMAL
        if risk_score >= RISK_ALERT_THRESHOLD:
            status_text = "ALERT"
            status_color = COLOR_ALERT
            if not ALERT_TRIGGERED:
                print(f"!!! ALERT at frame {frame_count}, score {risk_score:.3f} !!!")
                ALERT_TRIGGERED = True
        elif risk_score >= 0.5:
            status_text = "CONGESTED"
            status_color = COLOR_CONGESTED
            ALERT_TRIGGERED = False
        else:
            ALERT_TRIGGERED = False

        composed = draw_side_panel(combined, risk_score, density_index, motion_metric, status_color, status_text, bottleneck=final_bneck)

        # show
        cv2.imshow(WINDOW_NAME, composed)

        # record metrics for plotting later
        metrics.append({
            "timestamp": datetime.now().isoformat(),
            "frame": frame_count,
            "density": float(density_index),
            "motion": float(motion_metric),
            "risk": float(risk_score),
            "bottleneck": bool(final_bneck),
            "bneck_score": float(smoothed_score)
        })

        prev_gray_proc = proc_gray.copy()

        # pacing: compute elapsed and wait accordingly
        loop_elapsed = time.time() - loop_start
        wait_ms = max(1, int(frame_delay_ms - loop_elapsed * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF

        # frame skipping if lagging behind
        if ENABLE_FRAME_SKIPPING:
            expected_frames = int((time.time() - start_time_wall) * fps)
            current_read = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if expected_frames - current_read > 1:
                skip_count = min(expected_frames - current_read, 5)
                for _ in range(skip_count):
                    ret_skip, _ = cap.read()
                    if ret_skip:
                        frame_count += 1
                    else:
                        break

        # keyboard handling
        if key == ord('q'):
            print("Quitting — showing report plots...")
            generate_report_plot(metrics)
            break
        elif key == ord(' '):
            play_mode = not play_mode
        elif key == ord('r'):
            print("Showing report plots (on-demand)...")
            generate_report_plot(metrics)
            print("Returned from report plots — continuing analysis.")

    # cleanup
    cap.release()
    cv2.destroyAllWindows()

    # final report plot on exit if metrics exist
    if metrics:
        print("Analysis finished — showing final report plots...")
        generate_report_plot(metrics)
    else:
        print("No metrics collected — no plots to show.")

    print("Done. Frames processed:", len(metrics))

# ========== Run ==========

if __name__ == "__main__":
    if not os.path.exists(VIDEO_FILE):
        print("ERROR: Video file not found. Update VIDEO_FILE variable to your video path and re-run.")
    else:
        analyze_video(VIDEO_FILE)

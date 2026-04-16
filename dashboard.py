"""dashboard.py — RTSDA Live Dashboard (Tkinter)

Supports video files AND live sources (webcam indices, RTSP, HTTP streams).

Features:
    - Live video feed with fullscreen (double-click)
    - Resizable window
    - Draggable ROI zone (right-click drag on video; double-right-click to reset)
    - Multi-camera 2x2 grid view
    - Audio alert on ALERT status transitions
    - Timestamped event log panel
    - Snapshot (save current frame as PNG)

Dependencies (beyond what test8 already uses):
    pip install Pillow

Run:
    python dashboard.py
"""

import cv2
import numpy as np
import threading
import queue
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
from datetime import datetime
import os
import sys

# Audio alert (Windows-only; graceful fallback on other OSes)
try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

# ── Import everything from test8 ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test8 import (
    VIDEO_FILE, PROCESS_SCALE, ENABLE_FRAME_SKIPPING,
    RISK_ALERT_THRESHOLD,
    BOTTLENECK_SCORE_THRESH, BOTTLENECK_TEMPORAL_LEN,
    ROI_Y1_FRAC, ROI_Y2_FRAC, ROI_X1_FRAC, ROI_X2_FRAC,
    DENSITY_SATURATION, YOLO_SKIP_FRAMES,
    COLOR_ALERT,
    # Legacy (fallback when YOLO unavailable)
    calculate_density_index_lower_roi,
    compute_flow_direction_map,
    detect_bottleneck,
    # Person-aware (primary when YOLO available)
    calculate_density_from_detections,
    compute_person_velocities,
    compute_risk_score,
    detect_bottleneck_persons,
    draw_person_boxes,
    # Research-paper signals (Dufour et al. 2025)
    compute_speed_anomaly,
    compute_counterflow_metric,
)
from detector import init_detector, detect_persons, YOLO_AVAILABLE

# Density map estimator (optional — needs trained model)
try:
    from density_model import DensityEstimator
    _HAS_DENSITY_MODEL = True
except ImportError:
    _HAS_DENSITY_MODEL = False

# ── Dashboard settings ────────────────────────────────────────────────────────
CHART_HISTORY    = 150   # frames of history shown in live charts
POLL_MS          = 40    # GUI poll interval ms  (~25 Hz)
VIDEO_W          = 700   # initial video panel pixel width
VIDEO_H          = 420   # initial video panel pixel height
RECONNECT_DELAY  = 2.0   # seconds to wait before reconnecting a dropped live source

SNAPSHOT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
ALERT_BEEP_FREQ  = 800   # Hz
ALERT_BEEP_DUR   = 350   # ms

# ── Auto-calibration ─────────────────────────────────────────────────────────
CALIBRATION_FRAMES = 75   # observe first N frames (~3 sec at 25 fps)
CALIBRATION_MIN    = 5    # minimum saturation (avoid div-by-zero on empty scenes)
CALIBRATION_PAD    = 1.3  # multiply observed max by this (headroom)

# ── Color palette ─────────────────────────────────────────────────────────────
BG_MAIN  = "#1a1a2e"
BG_PANEL = "#16213e"
BG_DARK  = "#0f3460"
FG_TEXT  = "#e0e0e0"
FG_DIM   = "#888899"

COL_RISK  = "#e94560"
COL_DEN   = "#4cc9f0"
COL_MOT   = "#f7b731"
COL_OK    = "#00e676"
COL_WARN  = "#ffa726"
COL_ALERT = "#f44336"
COL_LIVE  = "#ff3366"

STATUS_STYLE = {
    "NORMAL":    (COL_OK,    "#0a2a1a"),
    "CONGESTED": (COL_WARN,  "#2a1a0a"),
    "ALERT":     (COL_ALERT, "#2a0a0a"),
}


# ── Source helpers ────────────────────────────────────────────────────────────
def _is_live(src) -> bool:
    """True for camera indices (int or digit string) and streaming URLs."""
    if isinstance(src, int):
        return True
    s = str(src).strip()
    if s.isdigit():
        return True
    return any(s.lower().startswith(p)
               for p in ("rtsp://", "rtmp://", "http://", "https://"))


def _open_source(src):
    """Return an opened cv2.VideoCapture from int, digit-string, or URL/path."""
    if isinstance(src, int):
        return cv2.VideoCapture(src)
    s = str(src).strip()
    if s.isdigit():
        return cv2.VideoCapture(int(s))
    return cv2.VideoCapture(s)


def _label_for(src) -> str:
    """Short human-readable label for a source."""
    if isinstance(src, int) or (isinstance(src, str) and str(src).strip().isdigit()):
        return f"Camera {src}"
    s = str(src)
    if any(s.lower().startswith(p) for p in ("rtsp://", "rtmp://")):
        return f"Stream: {s[:40]}"
    return os.path.basename(s)


# ── Analysis worker thread ────────────────────────────────────────────────────
class AnalysisWorker(threading.Thread):
    """
    Runs the RTSDA analysis pipeline in a background thread.
    source can be: int (camera index), digit-string ("0"), file path, or URL.
    """

    def __init__(self, source, out_q: queue.Queue, pause_evt: threading.Event):
        super().__init__(daemon=True, name="AnalysisWorker")
        self.source    = source
        self.is_live   = _is_live(source)
        self.q         = out_q
        self.pause_evt = pause_evt   # set = playing, clear = paused
        self._stop     = threading.Event()
        self.saved_metrics: list = []
        self.roi_frac  = None  # (y1_frac, y2_frac, x1_frac, x2_frac) or None

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            self._run_capture()
            if not self.is_live or self._stop.is_set():
                break
            # Live source dropped — attempt reconnect
            self._push({"type": "reconnecting"})
            time.sleep(RECONNECT_DELAY)

    def _run_capture(self):
        cap = _open_source(self.source)
        if not cap.isOpened():
            self._push({"type": "error",
                        "msg": f"Cannot open: {self.source}"})
            return

        if self.is_live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialise YOLO detector (first worker to call this loads the model)
        import detector as _det
        if not _det.YOLO_AVAILABLE and _det._model is None:
            self._push({"type": "info", "msg": "Loading YOLO model..."})
            init_detector()
        use_yolo = _det.YOLO_AVAILABLE

        # Initialise density map estimator (optional)
        density_est = None
        if _HAS_DENSITY_MODEL:
            density_est = DensityEstimator()
            if not density_est.available:
                density_est = None

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay  = 1.0 / fps
        dw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        dh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = -1 if self.is_live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        prev_gray      = None
        prev_dets      = []       # previous frame's detections (for velocity)
        last_dets      = []       # most recent YOLO detections (for skip-frames)
        running_scores = []       # temporal smoothing for bottleneck
        frame_n        = 0
        wall0          = time.time()
        last_overlay   = None
        frame_diag     = np.sqrt(dw ** 2 + dh ** 2)  # for resolution-independent speed
        last_risk      = 0.0      # for adaptive YOLO skip

        # ── Auto-calibration state ────────────────────────────────────────────
        calibrating      = use_yolo         # only calibrate when YOLO is active
        calib_max_count  = 0                # peak person count observed
        density_sat      = DENSITY_SATURATION  # will be overwritten after calib

        while not self._stop.is_set():
            # ── Paused ────────────────────────────────────────────────────────
            if not self.pause_evt.is_set():
                if last_overlay is not None:
                    self._push({"type": "frame_only", "overlay": last_overlay,
                                "frame": frame_n, "total": total_frames,
                                "is_live": self.is_live,
                                "source_label": _label_for(self.source)})
                time.sleep(0.05)
                continue

            t0 = time.time()

            # For live sources flush stale buffered frames first
            if self.is_live:
                cap.grab()

            ret, frame_full = cap.read()
            if not ret:
                if not self.is_live:
                    self._push({"type": "done", "metrics": self.saved_metrics})
                cap.release()
                return
            frame_n += 1

            # ── Downscale for processing ──────────────────────────────────────
            if PROCESS_SCALE != 1.0:
                proc = cv2.resize(frame_full, (0, 0),
                                  fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                                  interpolation=cv2.INTER_AREA)
            else:
                proc = frame_full.copy()

            proc_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

            # ── Optical flow (always — used for overlay + compression) ────────
            flow_vis_p, flow_motion, flow_data = compute_flow_direction_map(
                prev_gray, proc_gray)
            flow_vis = cv2.resize(flow_vis_p, (dw, dh),
                                   interpolation=cv2.INTER_LINEAR)

            # ── ROI coords ────────────────────────────────────────────────────
            custom_roi = self.roi_frac
            if custom_roi is not None:
                y1f, y2f, x1f, x2f = custom_roi
            else:
                y1f, y2f, x1f, x2f = ROI_Y1_FRAC, ROI_Y2_FRAC, ROI_X1_FRAC, ROI_X2_FRAC
            fh, fw = frame_full.shape[:2]
            roi = (int(fh * y1f), int(fh * y2f),
                   int(fw * x1f), int(fw * x2f))

            # ── Person detection (YOLO) or fallback ───────────────────────────
            if use_yolo:
                # Adaptive YOLO skip: every frame when risk is high, skip when calm
                if last_risk >= 0.5:
                    skip = 1  # every frame during congestion/alert
                else:
                    skip = YOLO_SKIP_FRAMES
                run_yolo = (frame_n % skip == 1) or skip <= 1
                if run_yolo:
                    detections = detect_persons(proc, PROCESS_SCALE, track=True)
                    last_dets = detections
                else:
                    detections = last_dets

                density, _pc, _ = calculate_density_from_detections(
                    detections, roi, frame_full.shape)

                # ── Auto-calibration: learn DENSITY_SATURATION ────────────────
                if calibrating:
                    calib_max_count = max(calib_max_count, _pc)
                    if frame_n >= CALIBRATION_FRAMES:
                        calibrating = False
                        density_sat = max(CALIBRATION_MIN,
                                          int(calib_max_count * CALIBRATION_PAD))
                        import test8 as _t8
                        _t8.DENSITY_SATURATION = density_sat
                        self._push({"type": "info",
                                    "msg": f"Auto-calibrated: DENSITY_SATURATION={density_sat} "
                                           f"(observed max {calib_max_count} persons)"})

                # ── Density map estimator (blend if available) ────────────────
                if density_est is not None:
                    dm_count_roi, _, _ = density_est.predict_for_roi(
                        frame_full, roi)
                    dm_density = float(np.clip(
                        dm_count_roi / max(density_sat, 1), 0.0, 1.0))
                    # Blend: 60% YOLO counting + 40% density map regression
                    # YOLO is precise for sparse crowds; density map wins in
                    # dense/occluded scenes where YOLO misses people
                    density = 0.6 * density + 0.4 * dm_density

                _, _, motion_m = compute_person_velocities(
                    detections, prev_dets, fps, frame_diag=frame_diag)
                # If no tracked velocities yet, fall back to flow metric
                if motion_m == 0.0 and flow_motion > 0.0:
                    motion_m = flow_motion

                # Weidmann speed anomaly + counterflow (research paper signals)
                speed_anom = compute_speed_anomaly(density, motion_m, density_sat)
                cflow_m, n_counter, n_cflow_total = compute_counterflow_metric(
                    detections, prev_dets, fps, frame_diag=frame_diag)

                bneck_score, bbox, details = detect_bottleneck_persons(
                    detections, flow_data, roi, frame_full.shape)

                prev_dets = detections
            else:
                # ── Legacy fallback (no YOLO) ─────────────────────────────────
                # Use density map estimator if available
                if density_est is not None:
                    dm_count_roi, _, _ = density_est.predict_for_roi(
                        frame_full, roi)
                    density = float(np.clip(
                        dm_count_roi / max(density_sat, 1), 0.0, 1.0))
                else:
                    density, roi = calculate_density_index_lower_roi(frame_full)
                    if custom_roi is not None:
                        roi = (int(fh * y1f), int(fh * y2f),
                               int(fw * x1f), int(fw * x2f))
                        gray = cv2.cvtColor(frame_full, cv2.COLOR_BGR2GRAY)
                        crop = gray[roi[0]:roi[1], roi[2]:roi[3]]
                        density = float(np.clip(np.mean(crop) / 255.0 * 1.5, 0.0, 1.0)) if crop.size > 0 else 0.0
                motion_m = flow_motion
                bneck_score, bbox, details = detect_bottleneck(
                    proc_gray, flow_data, roi)
                detections = []
                speed_anom = 0.0
                cflow_m = 0.0
                n_counter = 0
                n_cflow_total = 0

            # ── Bottleneck temporal smoothing ─────────────────────────────────
            running_scores.append(bneck_score)
            if len(running_scores) > BOTTLENECK_TEMPORAL_LEN:
                running_scores.pop(0)
            smoothed_score = float(np.mean(running_scores)) if running_scores else bneck_score
            bneck = smoothed_score >= BOTTLENECK_SCORE_THRESH

            # ── Risk score (unified formula for both paths) ───────────────────
            risk = compute_risk_score(density, motion_m, bneck,
                                      speed_anomaly=speed_anom,
                                      counterflow=cflow_m)
            last_risk = risk
            status = ("ALERT"     if risk >= RISK_ALERT_THRESHOLD else
                      "CONGESTED" if risk >= 0.5 else
                      "NORMAL")

            # ── Overlay ───────────────────────────────────────────────────────
            overlay = cv2.addWeighted(frame_full, 1.0, flow_vis, 0.25, 0)
            y1, y2, x1, x2 = roi
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 80, 80), 1)

            if use_yolo and detections:
                draw_person_boxes(overlay, detections)

            if bneck:
                bx1, by1, bx2, by2 = bbox
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), COLOR_ALERT, 3)
                cv2.putText(overlay, "BOTTLENECK", (bx1 + 4, by1 + 28),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_ALERT, 2,
                            cv2.LINE_AA)

            # ── Push result ───────────────────────────────────────────────────
            m = {
                "type":             "update",
                "overlay":          overlay,
                "frame":            frame_n,
                "total":            total_frames,
                "is_live":          self.is_live,
                "source_label":     _label_for(self.source),
                "density":          density,
                "motion":           float(motion_m),
                "risk":             risk,
                "status":           status,
                "bottleneck":       bneck,
                "bneck_score":      float(smoothed_score),
                "speed_anomaly":    float(speed_anom),
                "counterflow":      float(cflow_m),
                "n_counterflow":    n_counter,
                "timestamp":        datetime.now().isoformat(),
            }
            self.saved_metrics.append(
                {k: v for k, v in m.items() if k not in ("overlay", "type")})
            last_overlay = overlay
            self._push(m)

            prev_gray = proc_gray.copy()

            # ── Frame skipping (file mode only) ───────────────────────────────
            if not self.is_live and ENABLE_FRAME_SKIPPING:
                exp  = int((time.time() - wall0) * fps)
                cur  = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                skip = min(exp - cur, 5)
                for _ in range(max(0, skip - 1)):
                    rs, _ = cap.read()
                    if rs:
                        frame_n += 1
                    else:
                        break

            # ── Pacing (file mode only; live runs as-fast-as-possible) ────────
            if not self.is_live:
                sleep_t = max(0.0, frame_delay - (time.time() - t0))
                if sleep_t > 0:
                    time.sleep(sleep_t)

        cap.release()

    def _push(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            pass


# ── Multi-camera cell ─────────────────────────────────────────────────────────
class CameraCell:
    """One cell in the multi-camera grid."""

    def __init__(self, parent, row, col):
        self.frame = tk.Frame(parent, bg="#111122", bd=1, relief="solid")
        self.frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)

        self.canvas = tk.Canvas(self.frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_lbl = tk.Label(self.frame, text="No source — double-click to add",
                                   font=("Consolas", 9), fg=FG_DIM,
                                   bg="#111122", pady=3)
        self.status_lbl.pack(fill=tk.X)

        self.q = queue.Queue(maxsize=30)
        self.pause_evt = threading.Event()
        self.pause_evt.set()
        self.worker = None
        self._photo = None
        self._w = 320
        self._h = 240
        self.canvas.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self._w = max(event.width, 1)
        self._h = max(event.height, 1)

    def set_source(self, source):
        self.stop()
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.worker = AnalysisWorker(source, self.q, self.pause_evt)
        self.worker.start()
        self.status_lbl.config(text=_label_for(source), fg=FG_TEXT)

    def poll(self):
        """Drain up to 3 items; render only the latest."""
        latest = None
        for _ in range(3):
            try:
                latest = self.q.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return

        t = latest.get("type")
        if t in ("update", "frame_only"):
            overlay = latest.get("overlay")
            if overlay is not None:
                rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                sh, sw = overlay.shape[:2]
                scale = min(self._w / max(sw, 1), self._h / max(sh, 1))
                nw = max(1, int(sw * scale))
                nh = max(1, int(sh * scale))
                img = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
                self._photo = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.create_image(self._w // 2, self._h // 2,
                                         anchor="center", image=self._photo)
            if t == "update":
                status = latest.get("status", "")
                fg, _ = STATUS_STYLE.get(status, (FG_DIM, "#111122"))
                src_label = _label_for(self.worker.source) if self.worker else "?"
                self.status_lbl.config(
                    text=f"{src_label}  |  {status}  R:{latest['risk']:.2f}",
                    fg=fg)
        elif t == "error":
            self.status_lbl.config(text=f"Error: {latest['msg']}", fg=COL_ALERT)

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker = None


# ── Multi-camera window ──────────────────────────────────────────────────────
class MultiCamWindow:
    """2x2 camera grid window."""

    def __init__(self, root):
        self.win = tk.Toplevel(root)
        self.win.title("RTSDA — Multi-Camera View")
        self.win.configure(bg=BG_MAIN)
        self.win.geometry("960x680")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        grid = tk.Frame(self.win, bg=BG_MAIN)
        grid.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        for i in range(2):
            grid.rowconfigure(i, weight=1)
            grid.columnconfigure(i, weight=1)

        self.cells = []
        for r in range(2):
            for c in range(2):
                cell = CameraCell(grid, r, c)
                cell.canvas.bind("<Double-Button-1>",
                                 lambda e, cl=cell: self._pick_source(cl))
                self.cells.append(cell)

        tk.Label(self.win,
                 text="Double-click a cell to assign a video source",
                 font=("Consolas", 9), fg=FG_DIM, bg=BG_MAIN
                 ).pack(pady=(0, 6))

        self._polling = True
        self._poll()

    def _poll(self):
        if not self._polling:
            return
        for cell in self.cells:
            cell.poll()
        self.win.after(POLL_MS, self._poll)

    def _pick_source(self, cell):
        """Simple source picker for a grid cell."""
        win = tk.Toplevel(self.win)
        win.title("Select Source")
        win.configure(bg=BG_MAIN)
        win.resizable(False, False)
        win.grab_set()

        def use(src):
            win.destroy()
            cell.set_source(src)

        tk.Label(win, text="Select a source for this cell:",
                 font=("Consolas", 10), fg=FG_TEXT, bg=BG_MAIN
                 ).pack(padx=14, pady=(10, 6))

        bs = {"font": ("Consolas", 10), "relief": "flat", "padx": 12,
              "pady": 4, "cursor": "hand2", "bg": "#1e3a6e", "fg": "white"}

        for idx in range(4):
            tk.Button(win, text=f"Camera {idx}",
                      command=lambda i=idx: use(i), **bs
                      ).pack(fill=tk.X, padx=14, pady=2)

        tk.Button(win, text=f"Video: {os.path.basename(VIDEO_FILE)}",
                  command=lambda: use(VIDEO_FILE), **bs
                  ).pack(fill=tk.X, padx=14, pady=2)

        def _browse():
            path = filedialog.askopenfilename(
                parent=win, title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                           ("All files", "*.*")])
            if path:
                use(path)

        tk.Button(win, text="Browse video file...",
                  command=_browse, **bs
                  ).pack(fill=tk.X, padx=14, pady=2)

        # URL entry
        url_frame = tk.Frame(win, bg=BG_MAIN)
        url_frame.pack(fill=tk.X, padx=14, pady=6)
        url_var = tk.StringVar(value="rtsp://")
        tk.Entry(url_frame, textvariable=url_var, width=30,
                 font=("Consolas", 10), bg="#0d0d2a", fg=FG_TEXT,
                 insertbackground=FG_TEXT, relief="flat", bd=4
                 ).pack(side=tk.LEFT)
        tk.Button(url_frame, text="Use",
                  command=lambda: use(url_var.get().strip()), **bs
                  ).pack(side=tk.LEFT, padx=(6, 0))

        tk.Button(win, text="Cancel", command=win.destroy,
                  font=("Consolas", 10), bg="#2a1a1a", fg=FG_DIM,
                  relief="flat", padx=14, pady=4
                  ).pack(pady=(4, 10))

    def _on_close(self):
        self._polling = False
        for cell in self.cells:
            cell.stop()
        self.win.destroy()


# ── Dashboard GUI ─────────────────────────────────────────────────────────────
class Dashboard:
    def __init__(self, root: tk.Tk, initial_source=None):
        self.root = root
        self.root.title("RTSDA Live Dashboard")
        self.root.configure(bg=BG_MAIN)
        self.root.resizable(True, True)
        self.root.minsize(1050, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.source     = initial_source or VIDEO_FILE
        self.q          = queue.Queue(maxsize=60)
        self.pause_evt  = threading.Event()
        self.pause_evt.clear()  # Start PAUSED (user must click Play)
        self.playing    = False

        self.hist_risk  = deque(maxlen=CHART_HISTORY)
        self.hist_den   = deque(maxlen=CHART_HISTORY)
        self.hist_mot   = deque(maxlen=CHART_HISTORY)
        self.hist_f     = deque(maxlen=CHART_HISTORY)
        self.all_metrics: list = []
        self._chart_tick = 0
        self._canvas_w   = VIDEO_W
        self._canvas_h   = VIDEO_H
        self._live_blink = False   # toggled for pulsing LIVE dot

        # Fullscreen state
        self._fs_win    = None
        self._fs_canvas = None
        self._fs_photo  = None
        self._fs_w      = 1
        self._fs_h      = 1

        # Audio alert state
        self._muted      = False
        self._last_status = "NORMAL"

        # Snapshot state
        self._last_overlay = None   # most recent BGR overlay frame

        # ROI drag state
        self._roi_dragging = False
        self._roi_start    = None   # (canvas_x, canvas_y)
        self._roi_rect_id  = None   # canvas rectangle item id

        # Multi-cam reference
        self._multi_cam_win = None

        # Worker (started only when user clicks Play)
        self.worker = None

        self._build_ui()
        # Don't start worker here — wait for user to click Play
        # self._start_worker(self.source)
        self.root.after(POLL_MS, self._poll)
        self.root.after(500, self._pulse_live)

    # ── Worker management ─────────────────────────────────────────────────────
    def _start_worker(self, source):
        self.source = source
        self.worker = AnalysisWorker(source, self.q, self.pause_evt)
        self.worker.start()

    def _switch_source(self, source):
        """Stop the current worker, drain the queue, start a new one."""
        if self.worker is not None:
            self.worker.stop()
            # Drain stale frames
            while True:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break

        # Clear accumulated metrics so new video starts fresh
        self.all_metrics.clear()
        self.hist_risk.clear()
        self.hist_den.clear()
        self.hist_mot.clear()
        self.hist_f.clear()
        self._chart_tick = 0

        # Pause the new source (user must click Play)
        self.pause_evt.clear()

        # Start the new worker (will be paused until user clicks Play)
        self._start_worker(source)
        label = _label_for(source)
        self.info_lbl.config(text=f"Ready. Click Play to start analyzing {label}")
        self._log_event(f"Source: {label}")

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        main = tk.Frame(self.root, bg=BG_MAIN)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 4))
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)

        # ── Left column: video feed ───────────────────────────────────────────
        left = tk.Frame(main, bg=BG_MAIN)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        # Header row: "LIVE FEED" label + source tag
        hdr = tk.Frame(left, bg=BG_MAIN)
        hdr.pack(anchor="w", fill=tk.X)
        tk.Label(hdr, text="LIVE FEED", font=("Consolas", 9),
                 fg=FG_DIM, bg=BG_MAIN).pack(side=tk.LEFT)
        self.source_tag = tk.Label(hdr, text=f"  {_label_for(self.source)}",
                                    font=("Consolas", 9, "italic"),
                                    fg="#6688cc", bg=BG_MAIN)
        self.source_tag.pack(side=tk.LEFT)

        # Progress / live bar anchored to bottom so canvas fills the rest
        prog_row = tk.Frame(left, bg=BG_MAIN)
        prog_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        self.prog_label = tk.Label(prog_row, text="Progress:",
                                    font=("Consolas", 9),
                                    fg=FG_DIM, bg=BG_MAIN)
        self.prog_label.pack(side=tk.LEFT)
        self.prog_bar = tk.Canvas(prog_row, height=8, bg="#111122",
                                   highlightthickness=0)
        self.prog_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        # Hint below progress bar
        tk.Label(left, text="Dbl-click: fullscreen  |  Right-drag: set ROI  |  Dbl-right-click: reset ROI",
                 font=("Consolas", 7), fg="#444466", bg=BG_MAIN
                 ).pack(side=tk.BOTTOM, anchor="e")

        # Canvas fills remaining vertical space
        self.vid_canvas = tk.Canvas(left, width=VIDEO_W,
                                    bg="#000000", highlightthickness=1,
                                    highlightbackground="#333355")
        self.vid_canvas.pack(fill=tk.BOTH, expand=True)
        self.vid_canvas.bind("<Configure>", self._on_canvas_resize)
        self.vid_canvas.bind("<Double-Button-1>", self._toggle_fullscreen)

        # ROI drag bindings (right mouse button)
        self.vid_canvas.bind("<ButtonPress-3>", self._roi_press)
        self.vid_canvas.bind("<B3-Motion>", self._roi_motion)
        self.vid_canvas.bind("<ButtonRelease-3>", self._roi_release)
        self.vid_canvas.bind("<Double-Button-3>", self._reset_roi)

        # ── Right column: metrics + charts + event log ────────────────────────
        right = tk.Frame(main, bg=BG_MAIN)
        right.grid(row=0, column=1, sticky="nsew")

        self.status_lbl = tk.Label(right, text="STATUS: ---",
                                    font=("Consolas", 17, "bold"),
                                    fg=COL_OK, bg="#0a2a1a",
                                    anchor="center", pady=9)
        self.status_lbl.pack(fill=tk.X, pady=(0, 10))

        cards = tk.Frame(right, bg=BG_MAIN)
        cards.pack(fill=tk.X, pady=(0, 8))
        for i in range(3):
            cards.columnconfigure(i, weight=1)

        self.card_risk = self._metric_card(cards, "RISK",    "0.000", COL_RISK, 0)
        self.card_den  = self._metric_card(cards, "DENSITY", "0.00",  COL_DEN,  1)
        self.card_mot  = self._metric_card(cards, "MOTION",  "0.00",  COL_MOT,  2)

        self.bneck_lbl = tk.Label(right, text="Bottleneck: ---",
                                   font=("Consolas", 12, "bold"),
                                   fg=FG_DIM, bg=BG_PANEL, pady=6)
        self.bneck_lbl.pack(fill=tk.X, pady=(0, 4))

        # ── Research-paper metrics (Weidmann + counterflow) ──────────────
        research_frame = tk.Frame(right, bg=BG_PANEL)
        research_frame.pack(fill=tk.X, pady=(0, 8))
        for i in range(2):
            research_frame.columnconfigure(i, weight=1)

        self.anomaly_lbl = tk.Label(research_frame,
                                     text="Speed Anomaly: 0.00",
                                     font=("Consolas", 10),
                                     fg=FG_DIM, bg=BG_PANEL, pady=4)
        self.anomaly_lbl.grid(row=0, column=0, sticky="ew", padx=4)

        self.cflow_lbl = tk.Label(research_frame,
                                    text="Counterflow: 0.00",
                                    font=("Consolas", 10),
                                    fg=FG_DIM, bg=BG_PANEL, pady=4)
        self.cflow_lbl.grid(row=0, column=1, sticky="ew", padx=4)

        tk.Label(right, text="LIVE CHARTS", font=("Consolas", 9),
                 fg=FG_DIM, bg=BG_MAIN).pack(anchor="w")

        self.fig = Figure(figsize=(5.6, 4.4), facecolor=BG_MAIN)
        self.fig.subplots_adjust(left=0.09, right=0.97,
                                  top=0.92, bottom=0.10, hspace=0.60)
        self.ax_r = self.fig.add_subplot(211)
        self.ax_d = self.fig.add_subplot(212)
        self._style_axes()

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Event log ──────────────────────────────────────────────────────────
        tk.Label(right, text="EVENT LOG", font=("Consolas", 9),
                 fg=FG_DIM, bg=BG_MAIN).pack(anchor="w", pady=(6, 0))

        log_frame = tk.Frame(right, bg="#0d0d1a")
        log_frame.pack(fill=tk.X, pady=(0, 4))

        self.event_log = tk.Text(log_frame, height=5, width=40,
                                  font=("Consolas", 8), fg="#99aabb",
                                  bg="#0d0d1a", wrap=tk.WORD, bd=0,
                                  highlightthickness=0, state=tk.DISABLED)
        log_scroll = tk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                   command=self.event_log.yview)
        self.event_log.config(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.event_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Bottom control bar ────────────────────────────────────────────────
        bar = tk.Frame(self.root, bg="#0d0d1a", pady=7)
        bar.pack(fill=tk.X, padx=12, pady=(0, 10))

        bs = {"font": ("Consolas", 11, "bold"), "relief": "flat",
              "padx": 16, "pady": 5, "cursor": "hand2", "bd": 0}

        self.play_btn = tk.Button(bar, text=">  Play",
                                   bg="#1e3a6e", fg="white",
                                   command=self.toggle_play, **bs)
        self.play_btn.pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(bar, text="  Source",
                  bg="#3a2060", fg="white",
                  command=self._source_dialog, **bs).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(bar, text="  Report",
                  bg="#1a4a2a", fg="white",
                  command=self.show_report, **bs).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(bar, text="  Snap",
                  bg="#1a3a4a", fg="white",
                  command=self._take_snapshot, **bs).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(bar, text="  Multi-Cam",
                  bg="#2a2a4a", fg="white",
                  command=self._open_multi_cam, **bs).pack(side=tk.LEFT, padx=(0, 6))

        self.mute_btn = tk.Button(bar, text="  Mute",
                                   bg="#3a3a2a", fg="white",
                                   command=self._toggle_mute, **bs)
        self.mute_btn.pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(bar, text="X  Quit",
                  bg="#4a1a1a", fg="white",
                  command=self.on_close, **bs).pack(side=tk.LEFT)

        self.info_lbl = tk.Label(bar, text="Initializing...",
                                  font=("Consolas", 10),
                                  fg=FG_DIM, bg="#0d0d1a")
        self.info_lbl.pack(side=tk.RIGHT, padx=8)

        # Initial log entry
        self._log_event("Dashboard started")

    # ── Source dialog ─────────────────────────────────────────────────────────
    def _source_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Video Source")
        win.configure(bg=BG_MAIN)
        win.resizable(False, False)
        win.grab_set()

        pad = {"padx": 14, "pady": 6}

        def section(title):
            tk.Label(win, text=title, font=("Consolas", 10, "bold"),
                     fg=FG_DIM, bg=BG_MAIN).pack(anchor="w", padx=14, pady=(12, 2))
            tk.Frame(win, bg="#333355", height=1).pack(fill=tk.X, padx=14)

        def use_btn(parent, src):
            def _use():
                win.destroy()
                self._switch_source(src)
            return tk.Button(parent, text="Use", font=("Consolas", 10, "bold"),
                              bg="#1e3a6e", fg="white", relief="flat",
                              padx=10, pady=3, cursor="hand2", command=_use)

        # ── Camera indices ────────────────────────────────────────────────────
        section("Webcam / USB Camera")
        cam_frame = tk.Frame(win, bg=BG_MAIN)
        cam_frame.pack(fill=tk.X, **pad)

        self._cam_status = {}
        for idx in range(4):
            row = tk.Frame(cam_frame, bg=BG_MAIN)
            row.pack(fill=tk.X, pady=2)
            status_lbl = tk.Label(row, text=f"Camera {idx}",
                                   font=("Consolas", 10), fg=FG_TEXT,
                                   bg=BG_MAIN, width=12, anchor="w")
            status_lbl.pack(side=tk.LEFT)
            self._cam_status[idx] = status_lbl
            use_btn(row, idx).pack(side=tk.RIGHT, padx=(6, 0))
            tk.Button(row, text="Test", font=("Consolas", 9),
                       bg="#2a2a4a", fg=FG_DIM, relief="flat",
                       padx=8, pady=2, cursor="hand2",
                       command=lambda i=idx, lbl=status_lbl: self._test_cam(i, lbl)
                       ).pack(side=tk.RIGHT)

        # ── Stream URL ────────────────────────────────────────────────────────
        section("Stream URL  (RTSP / RTMP / HTTP)")
        url_frame = tk.Frame(win, bg=BG_MAIN)
        url_frame.pack(fill=tk.X, **pad)

        url_var = tk.StringVar(value="rtsp://")
        url_entry = tk.Entry(url_frame, textvariable=url_var, width=38,
                              font=("Consolas", 10),
                              bg="#0d0d2a", fg=FG_TEXT,
                              insertbackground=FG_TEXT,
                              relief="flat", bd=4)
        url_entry.pack(side=tk.LEFT)

        def _use_url():
            u = url_var.get().strip()
            if u:
                win.destroy()
                self._switch_source(u)
        tk.Button(url_frame, text="Use", font=("Consolas", 10, "bold"),
                   bg="#1e3a6e", fg="white", relief="flat",
                   padx=10, pady=3, cursor="hand2",
                   command=_use_url).pack(side=tk.LEFT, padx=(6, 0))

        # ── Current video file ────────────────────────────────────────────────
        section("Video File")
        file_frame = tk.Frame(win, bg=BG_MAIN)
        file_frame.pack(fill=tk.X, **pad)
        short = os.path.basename(VIDEO_FILE)
        tk.Label(file_frame, text=short, font=("Consolas", 10),
                  fg=FG_TEXT, bg=BG_MAIN).pack(side=tk.LEFT)
        use_btn(file_frame, VIDEO_FILE).pack(side=tk.RIGHT)

        # Browse for any video file
        def _browse_file():
            path = filedialog.askopenfilename(
                parent=win, title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                           ("All files", "*.*")])
            if path:
                win.destroy()
                self._switch_source(path)

        tk.Button(win, text="Browse video file...",
                  font=("Consolas", 10, "bold"), bg="#1e3a6e", fg="white",
                  relief="flat", padx=10, pady=3, cursor="hand2",
                  command=_browse_file).pack(fill=tk.X, padx=14, pady=(6, 0))

        tk.Button(win, text="Cancel", font=("Consolas", 10),
                   bg="#2a1a1a", fg=FG_DIM, relief="flat",
                   padx=14, pady=5, cursor="hand2",
                   command=win.destroy).pack(pady=(10, 14))

    def _test_cam(self, idx: int, lbl: tk.Label):
        """Quick non-blocking camera availability test."""
        lbl.config(text=f"Camera {idx}  testing...", fg=COL_WARN)
        lbl.update()

        def check():
            cap = cv2.VideoCapture(idx)
            ok  = cap.isOpened()
            cap.release()
            self.root.after(0, lambda: lbl.config(
                text=f"Camera {idx}  {'available' if ok else 'not found'}",
                fg=COL_OK if ok else COL_ALERT
            ))

        threading.Thread(target=check, daemon=True).start()

    # ── Progress / live indicator ─────────────────────────────────────────────
    def _pulse_live(self):
        """Toggle blink state every 500 ms for the LIVE dot."""
        self._live_blink = not self._live_blink
        self.root.after(500, self._pulse_live)

    def _draw_progress(self, frame: int, total: int, is_live: bool):
        pw = self.prog_bar.winfo_width() or 300
        self.prog_bar.delete("all")
        if is_live:
            self.prog_label.config(text="")
            self.prog_bar.create_rectangle(0, 0, pw, 8,
                                            fill="#1a0a0a", outline="")
            dot_color = COL_LIVE if self._live_blink else "#550011"
            self.prog_bar.create_oval(2, 1, 9, 7, fill=dot_color, outline="")
            self.prog_bar.create_text(16, 4, anchor="w",
                                       text="LIVE", fill=COL_LIVE,
                                       font=("Consolas", 7, "bold"))
        else:
            self.prog_label.config(text="Progress:")
            pct = frame / max(total, 1)
            self.prog_bar.create_rectangle(0, 0, int(pw * pct), 8,
                                            fill=BG_DARK, outline="")

    # ── Metric card ───────────────────────────────────────────────────────────
    def _metric_card(self, parent, title: str, init_val: str,
                     color: str, col: int):
        pad_left = 0 if col == 0 else 6
        frame = tk.Frame(parent, bg=BG_PANEL, padx=10, pady=8)
        frame.grid(row=0, column=col, sticky="nsew", padx=(pad_left, 0))
        tk.Label(frame, text=title, font=("Consolas", 9),
                 fg=FG_DIM, bg=BG_PANEL).pack()
        val = tk.Label(frame, text=init_val,
                       font=("Consolas", 20, "bold"),
                       fg=color, bg=BG_PANEL)
        val.pack()
        return val

    def _on_canvas_resize(self, event):
        self._canvas_w = max(event.width, 1)
        self._canvas_h = max(event.height, 1)

    # ── Fullscreen video ───────────────────────────────────────────────────────
    def _toggle_fullscreen(self, event=None):
        """Open fullscreen window on first double-click; close it on the second."""
        if self._fs_win is not None:
            try:
                if self._fs_win.winfo_exists():
                    self._fs_win.destroy()
            except tk.TclError:
                pass
            self._fs_win = None
            return

        self._fs_win = tk.Toplevel(self.root)
        self._fs_win.title("RTSDA — Fullscreen")
        self._fs_win.configure(bg="black")
        self._fs_win.attributes("-fullscreen", True)
        self._fs_win.protocol("WM_DELETE_WINDOW", self._toggle_fullscreen)

        self._fs_canvas = tk.Canvas(self._fs_win, bg="black",
                                    highlightthickness=0)
        self._fs_canvas.pack(fill=tk.BOTH, expand=True)
        self._fs_canvas.bind("<Configure>", self._on_fs_resize)
        self._fs_canvas.bind("<Double-Button-1>", self._toggle_fullscreen)
        self._fs_win.bind("<Escape>", self._toggle_fullscreen)

        hint = tk.Label(self._fs_win,
                        text="  Double-click or Esc to exit fullscreen  ",
                        font=("Consolas", 11), fg="#aaaacc", bg="#22223a",
                        pady=4)
        hint.place(relx=0.5, rely=0.97, anchor="s")
        self._fs_win.after(3000, lambda: hint.place_forget())

    def _on_fs_resize(self, event):
        self._fs_w = max(event.width, 1)
        self._fs_h = max(event.height, 1)

    # ── ROI drag (right-click) ─────────────────────────────────────────────────
    def _roi_press(self, event):
        self._roi_dragging = True
        self._roi_start = (event.x, event.y)
        if self._roi_rect_id:
            self.vid_canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    def _roi_motion(self, event):
        if not self._roi_dragging or self._roi_start is None:
            return
        if self._roi_rect_id:
            self.vid_canvas.delete(self._roi_rect_id)
        x0, y0 = self._roi_start
        self._roi_rect_id = self.vid_canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#00ffff", width=2, dash=(4, 4))

    def _roi_release(self, event):
        if not self._roi_dragging or self._roi_start is None:
            self._roi_dragging = False
            return
        self._roi_dragging = False
        x0, y0 = self._roi_start
        x1, y1 = event.x, event.y
        if self._roi_rect_id:
            self.vid_canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

        # Ignore tiny drags (likely accidental)
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            return

        # Convert canvas pixels to 0-1 fractions
        cw, ch = self._canvas_w, self._canvas_h
        fx0 = min(x0, x1) / cw
        fx1 = max(x0, x1) / cw
        fy0 = min(y0, y1) / ch
        fy1 = max(y0, y1) / ch
        fx0 = max(0.0, min(1.0, fx0))
        fx1 = max(0.0, min(1.0, fx1))
        fy0 = max(0.0, min(1.0, fy0))
        fy1 = max(0.0, min(1.0, fy1))

        if self.worker is not None:
            self.worker.roi_frac = (fy0, fy1, fx0, fx1)
            self._log_event(f"Custom ROI set: x[{fx0:.0%}-{fx1:.0%}] y[{fy0:.0%}-{fy1:.0%}]")

    def _reset_roi(self, event=None):
        """Double-right-click resets ROI to default."""
        if self.worker is not None:
            self.worker.roi_frac = None
            self._log_event("ROI reset to default")

    # ── Snapshot ──────────────────────────────────────────────────────────────
    def _take_snapshot(self):
        if self._last_overlay is None:
            self.info_lbl.config(text="No frame to snapshot yet")
            return
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SNAPSHOT_DIR, f"rtsda_snap_{ts}.png")
        cv2.imwrite(path, self._last_overlay)
        self.info_lbl.config(text=f"Snapshot saved: {os.path.basename(path)}")
        self._log_event(f"Snapshot saved: {os.path.basename(path)}")

    # ── Audio alert ───────────────────────────────────────────────────────────
    def _toggle_mute(self):
        self._muted = not self._muted
        self.mute_btn.config(text="  Unmute" if self._muted else "  Mute")
        self._log_event("Audio muted" if self._muted else "Audio unmuted")

    def _beep_alert(self):
        """Play a short beep in a background thread (non-blocking)."""
        if self._muted or not _HAS_WINSOUND:
            return
        threading.Thread(target=lambda: winsound.Beep(ALERT_BEEP_FREQ, ALERT_BEEP_DUR),
                         daemon=True).start()

    # ── Event log ─────────────────────────────────────────────────────────────
    def _log_event(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.event_log.config(state=tk.NORMAL)
        self.event_log.insert(tk.END, line)
        self.event_log.see(tk.END)
        self.event_log.config(state=tk.DISABLED)

    # ── Multi-camera grid ─────────────────────────────────────────────────────
    def _open_multi_cam(self):
        if self._multi_cam_win is not None:
            try:
                if self._multi_cam_win.win.winfo_exists():
                    self._multi_cam_win.win.lift()
                    return
            except tk.TclError:
                pass
        self._multi_cam_win = MultiCamWindow(self.root)
        self._log_event("Multi-camera view opened")

    def _style_axes(self):
        for ax, title, col in [(self.ax_r, "Risk Score", COL_RISK),
                                (self.ax_d, "Density / Motion", COL_DEN)]:
            ax.set_facecolor("#0d0d1a")
            ax.set_title(title, color=col, fontsize=8, pad=4)
            ax.tick_params(colors="#555566", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#223344")
            ax.set_ylim(0, 1)

    # ── Poll loop ─────────────────────────────────────────────────────────────
    def _poll(self):
        drained = 0
        while drained < 6:
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            drained += 1

            t = item.get("type")
            if t == "error":
                self.info_lbl.config(text=f"ERROR: {item['msg']}")
                self._log_event(f"ERROR: {item['msg']}")
            elif t == "done":
                self.info_lbl.config(text="Video finished.")
                self._log_event("Video playback finished")
            elif t == "reconnecting":
                self.info_lbl.config(text=f"Reconnecting to {_label_for(self.source)}...")
                self._log_event(f"Reconnecting to {_label_for(self.source)}")
            elif t == "info":
                self.info_lbl.config(text=item.get("msg", ""))
                self._log_event(item.get("msg", ""))
            elif t in ("update", "frame_only"):
                self._render(item)

        self.root.after(POLL_MS, self._poll)

    # ── Aspect-ratio-preserving resize ───────────────────────────────────────
    @staticmethod
    def _fit_size(src_w, src_h, box_w, box_h):
        """Return (new_w, new_h) that fits src inside box, preserving aspect ratio."""
        if src_w <= 0 or src_h <= 0:
            return box_w, box_h
        scale = min(box_w / src_w, box_h / src_h)
        return max(1, int(src_w * scale)), max(1, int(src_h * scale))

    # ── Render one frame ──────────────────────────────────────────────────────
    def _render(self, item: dict):
        overlay = item.get("overlay")
        if overlay is not None:
            self._last_overlay = overlay
            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            src_h, src_w = overlay.shape[:2]

            # Main canvas — fit inside canvas bounds, centre with black bars
            fit_w, fit_h = self._fit_size(src_w, src_h,
                                          self._canvas_w, self._canvas_h)
            img = Image.fromarray(rgb).resize((fit_w, fit_h), Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(img)
            cx = self._canvas_w // 2
            cy = self._canvas_h // 2
            self.vid_canvas.delete("all")
            self.vid_canvas.create_image(cx, cy, anchor="center",
                                         image=self._photo)

            # Mirror to fullscreen window if open
            if self._fs_win is not None:
                try:
                    if self._fs_win.winfo_exists():
                        fs_fw, fs_fh = self._fit_size(src_w, src_h,
                                                      self._fs_w, self._fs_h)
                        fs_img = Image.fromarray(rgb).resize(
                            (fs_fw, fs_fh), Image.BILINEAR)
                        self._fs_photo = ImageTk.PhotoImage(fs_img)
                        self._fs_canvas.delete("all")
                        self._fs_canvas.create_image(
                            self._fs_w // 2, self._fs_h // 2,
                            anchor="center", image=self._fs_photo)
                except tk.TclError:
                    self._fs_win = None

        if item.get("type") == "frame_only":
            return

        risk    = item["risk"]
        density = item["density"]
        motion  = item["motion"]
        status  = item["status"]
        bneck   = item["bottleneck"]
        frame   = item["frame"]
        total   = item.get("total", -1)
        is_live = item.get("is_live", False)
        label   = item.get("source_label", "")

        self.all_metrics.append(
            {k: v for k, v in item.items() if k not in ("overlay", "type")})

        # Source tag
        self.source_tag.config(text=f"  {label}")

        # Status banner
        fg, bg = STATUS_STYLE[status]
        self.status_lbl.config(text=f"STATUS: {status}", fg=fg, bg=bg)

        # Audio alert + event log on status transition
        if status != self._last_status:
            self._log_event(f"Status: {self._last_status} -> {status}  (risk={risk:.3f})")
            if status == "ALERT":
                self._beep_alert()
            self._last_status = status

        # Research-paper metrics
        sa = item.get("speed_anomaly", 0.0)
        cf = item.get("counterflow", 0.0)
        nc = item.get("n_counterflow", 0)

        # Speed anomaly / counterflow event logging (on transition above threshold)
        if sa >= 0.4:
            prev_sa = (self.all_metrics[-2].get("speed_anomaly", 0.0)
                       if len(self.all_metrics) >= 2 else 0.0)
            if prev_sa < 0.4:
                self._log_event(f"Speed ANOMALY detected ({sa:.2f}) — "
                                f"crowd deviates from expected flow")
        if cf >= 0.3:
            prev_cf = (self.all_metrics[-2].get("counterflow", 0.0)
                       if len(self.all_metrics) >= 2 else 0.0)
            if prev_cf < 0.3:
                self._log_event(f"COUNTERFLOW detected ({cf:.2f}) — "
                                f"{nc} people against main flow")

        # Bottleneck event logging
        if bneck:
            # Only log on transition (avoid spamming)
            prev_bneck = (len(self.all_metrics) >= 2 and
                          self.all_metrics[-2].get("bottleneck", False))
            if not prev_bneck:
                self._log_event(f"Bottleneck DETECTED at frame {frame}")

        # Metric cards
        self.card_risk.config(text=f"{risk:.3f}")
        self.card_den.config(text=f"{density:.2f}")
        self.card_mot.config(text=f"{motion:.2f}")

        # Bottleneck label
        if bneck:
            self.bneck_lbl.config(text="!! BOTTLENECK DETECTED",
                                   fg=COL_ALERT, bg="#2a0a0a")
        else:
            self.bneck_lbl.config(text="Passage: clear",
                                   fg=COL_OK, bg="#0a2a1a")

        sa_color = COL_ALERT if sa >= 0.4 else (COL_WARN if sa >= 0.2 else FG_DIM)
        self.anomaly_lbl.config(text=f"Speed Anomaly: {sa:.2f}",
                                 fg=sa_color)

        cf_color = COL_ALERT if cf >= 0.3 else (COL_WARN if cf >= 0.15 else FG_DIM)
        self.cflow_lbl.config(text=f"Counterflow: {cf:.2f} ({nc})",
                                fg=cf_color)

        # Progress / LIVE indicator
        self._draw_progress(frame, total, is_live)

        # Info bar
        if is_live:
            self.info_lbl.config(
                text=f"LIVE  {label}  |  {datetime.now().strftime('%H:%M:%S')}")
        else:
            self.info_lbl.config(
                text=f"Frame {frame}/{total}  |  {datetime.now().strftime('%H:%M:%S')}")

        # Charts
        self.hist_f.append(frame)
        self.hist_risk.append(risk)
        self.hist_den.append(density)
        self.hist_mot.append(motion)

        self._chart_tick = (self._chart_tick + 1) % 3
        if self._chart_tick == 0 and len(self.hist_f) >= 2:
            self._update_charts()

    def _update_charts(self):
        fs = list(self.hist_f)
        rs = list(self.hist_risk)
        ds = list(self.hist_den)
        ms = list(self.hist_mot)

        self.ax_r.clear()
        self.ax_d.clear()
        self._style_axes()

        self.ax_r.plot(fs, rs, color=COL_RISK, linewidth=1.5)
        self.ax_r.fill_between(fs, rs, alpha=0.12, color=COL_RISK)
        self.ax_r.axhline(RISK_ALERT_THRESHOLD, color=COL_RISK,
                           linewidth=0.8, linestyle="--", alpha=0.5)

        self.ax_d.plot(fs, ds, color=COL_DEN, linewidth=1.5, label="Density")
        self.ax_d.plot(fs, ms, color=COL_MOT, linewidth=1.5, label="Motion")
        self.ax_d.fill_between(fs, ds, alpha=0.10, color=COL_DEN)
        self.ax_d.fill_between(fs, ms, alpha=0.10, color=COL_MOT)
        self.ax_d.legend(fontsize=7, loc="upper left",
                          labelcolor=FG_TEXT, facecolor=BG_MAIN,
                          edgecolor="#333355")

        self.chart_canvas.draw_idle()

    # ── Controls ──────────────────────────────────────────────────────────────
    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.pause_evt.set()
            self.play_btn.config(text="  Pause")
        else:
            self.pause_evt.clear()
            self.play_btn.config(text=">  Play")

    def show_report(self):
        if not self.all_metrics:
            return

        fs = [m["frame"]   for m in self.all_metrics]
        rs = [m["risk"]    for m in self.all_metrics]
        ds = [m["density"] for m in self.all_metrics]
        ms = [m["motion"]  for m in self.all_metrics]

        win = tk.Toplevel(self.root)
        win.title("RTSDA Analysis Report")
        win.configure(bg=BG_MAIN)

        fig = Figure(figsize=(11, 5), facecolor=BG_MAIN)
        fig.subplots_adjust(left=0.07, right=0.97,
                             top=0.88, bottom=0.09, hspace=0.55)

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        for ax, title, col in [(ax1, "Risk Score", COL_RISK),
                                (ax2, "Density / Motion", COL_DEN)]:
            ax.set_facecolor("#0d0d1a")
            ax.set_title(title, color=col, fontsize=9)
            ax.tick_params(colors="#555566", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#223344")

        ax1.plot(fs, rs, color=COL_RISK, linewidth=1.5)
        ax1.fill_between(fs, rs, alpha=0.15, color=COL_RISK)
        ax1.axhline(RISK_ALERT_THRESHOLD, color=COL_RISK,
                     linestyle="--", linewidth=0.8, alpha=0.5)
        ax1.set_ylim(0, 1)

        ax2.plot(fs, ds, color=COL_DEN, linewidth=1.5, label="Density")
        ax2.plot(fs, ms, color=COL_MOT, linewidth=1.5, label="Motion")
        ax2.fill_between(fs, ds, alpha=0.12, color=COL_DEN)
        ax2.fill_between(fs, ms, alpha=0.12, color=COL_MOT)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=7, facecolor=BG_MAIN,
                    labelcolor=FG_TEXT, edgecolor="#333355")

        peak_r  = max(rs) if rs else 0.0
        peak_f  = fs[rs.index(peak_r)] if rs else 0
        alert_p = sum(1 for r in rs if r >= RISK_ALERT_THRESHOLD) / max(len(rs), 1) * 100
        fig.text(
            0.5, 0.97,
            f"Frames: {len(fs)}   Peak risk: {peak_r:.3f} @ f{peak_f}   "
            f"Alert: {alert_p:.1f}%",
            ha="center", va="top", fontsize=8,
            color=FG_DIM, fontfamily="monospace",
        )

        c = FigureCanvasTkAgg(fig, master=win)
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        c.draw()

        tk.Button(win, text="Close", command=win.destroy,
                  font=("Consolas", 10), bg="#4a1a1a", fg="white",
                  relief="flat", padx=14, pady=4).pack(pady=8)

    def on_close(self):
        if self.worker is not None:
            self.worker.stop()
        # Close multi-cam if open
        if self._multi_cam_win is not None:
            try:
                self._multi_cam_win._on_close()
            except Exception:
                pass
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # hide until source is chosen

    # If the default video file exists, use it; otherwise ask the user to pick one
    initial_source = VIDEO_FILE
    if not os.path.exists(VIDEO_FILE) and not _is_live(VIDEO_FILE):
        chosen = filedialog.askopenfilename(
            title="Select a video to analyse",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                       ("All files", "*.*")])
        if not chosen:
            sys.exit(0)  # user cancelled
        initial_source = chosen

    root.deiconify()
    Dashboard(root, initial_source=initial_source)
    root.mainloop()

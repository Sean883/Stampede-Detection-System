"""detector.py — YOLO person detection + ByteTrack abstraction.

Provides a thin wrapper around ultralytics YOLOv8 with graceful fallback
when the library or model isn't available.

Usage:
    from detector import init_detector, detect_persons, YOLO_AVAILABLE
    init_detector()           # call once at startup
    dets = detect_persons(frame, process_scale=0.6)
"""

import numpy as np

# ── State ────────────────────────────────────────────────────────────────────
YOLO_AVAILABLE = False
_model = None

# ── Config ───────────────────────────────────────────────────────────────────
YOLO_MODEL_NAME  = "yolov8n.pt"    # nano — fast on CPU (~20 ms/frame)
YOLO_CONF        = 0.45            # minimum detection confidence
YOLO_IOU         = 0.45            # NMS IoU threshold
YOLO_PERSON_CLS  = 0               # COCO class 0 = person
YOLO_IMGSZ       = 640             # internal inference resolution


def init_detector(model_name: str = None, device: str = "cpu") -> bool:
    """Load the YOLO model.  Call once at startup.

    Returns True if the model loaded successfully, False otherwise.
    On failure the system falls back to the old pixel-based pipeline.
    """
    global YOLO_AVAILABLE, _model
    try:
        from ultralytics import YOLO
        name = model_name or YOLO_MODEL_NAME
        _model = YOLO(name)
        _model.to(device)
        # Warm-up inference (downloads weights on first run)
        _model.predict(np.zeros((64, 64, 3), dtype=np.uint8),
                       verbose=False, imgsz=64)
        YOLO_AVAILABLE = True
    except Exception as exc:
        print(f"[detector] YOLO init failed ({exc}); using fallback pipeline")
        YOLO_AVAILABLE = False
    return YOLO_AVAILABLE


def detect_persons(frame, process_scale: float = 1.0,
                   track: bool = True) -> list[dict]:
    """Run person detection (and optional tracking) on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR image at **process-scale** resolution.
    process_scale : float
        The scale factor that was used to produce *frame* from the
        full-resolution capture.  Returned bounding boxes are mapped
        back to full-resolution coordinates.
    track : bool
        If True use ByteTrack for persistent track IDs across frames.

    Returns
    -------
    list[dict]
        Each dict: {
            "bbox":     (x1, y1, x2, y2),   # full-res pixel coords
            "conf":     float,
            "track_id": int | None,
            "center":   (cx, cy),            # full-res pixel coords
        }
        Returns an empty list when YOLO is unavailable.
    """
    if not YOLO_AVAILABLE or _model is None:
        return []

    inv_s = 1.0 / process_scale if process_scale != 1.0 else 1.0

    if track:
        results = _model.track(
            frame, persist=True,
            classes=[YOLO_PERSON_CLS],
            conf=YOLO_CONF, iou=YOLO_IOU,
            imgsz=YOLO_IMGSZ,
            tracker="bytetrack.yaml",
            verbose=False,
        )
    else:
        results = _model.predict(
            frame,
            classes=[YOLO_PERSON_CLS],
            conf=YOLO_CONF, iou=YOLO_IOU,
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )

    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(float)

            # Scale back to full-res
            x1 *= inv_s; y1 *= inv_s; x2 *= inv_s; y2 *= inv_s

            conf = float(boxes.conf[i].cpu())
            tid = None
            if boxes.id is not None:
                tid = int(boxes.id[i].cpu())

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append({
                "bbox":     (x1, y1, x2, y2),
                "conf":     conf,
                "track_id": tid,
                "center":   (cx, cy),
            })

    return detections

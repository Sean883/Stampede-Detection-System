# RTSDA — Real-Time Surveillance and Density Analysis System

A desktop application for real-time crowd safety monitoring using computer vision and pedestrian dynamics theory.

## Features

- **Person Detection & Tracking** — YOLOv8n + ByteTrack multi-object tracking
- **Density Estimation** — Hybrid blend of YOLO counting (60%) and CSRNet-lite regression (40%)
- **Optical Flow Analysis** — Farneback algorithm for crowd motion quantification
- **Bottleneck Detection** — Multi-signal fusion (gap score, packing, flow compression)
- **Speed Anomaly Detection** — Weidmann fundamental diagram comparison with 40% dead zone
- **Counterflow Metric** — Velocity variance analysis for opposing crowd flows
- **Unified Risk Scoring** — 5-signal weighted combination (density, motion, bottleneck, anomaly, counterflow)
- **Real-Time Dashboard** — Tkinter GUI with live video, metric cards, charts, event log, audio alerts
- **Multi-Camera Support** — 2x2 grid view with independent analysis workers
- **Auto-Calibration** — Automatic density saturation from first 75 frames

## Project Structure

```
├── dashboard.py          # Main GUI application (Tkinter)
├── test8.py              # Core analysis algorithms (pure functions)
├── density_model.py      # CSRNet-lite density estimation model
├── detector.py           # YOLOv8 detection abstraction
├── train_yolo.py         # YOLO fine-tuning pipeline (CrowdHuman)
├── train_risk.py         # Risk labelling GUI + logistic regression training
├── risk_labels.json      # Training labels for risk classification
├── generate_report.js    # Project report generator (docx)
├── package.json          # Node.js dependencies for report generation
├── figures/              # Diagram images for the report
│   ├── weidmann_speed_density_curve.png
│   ├── yolov8_detection_pipeline.png
│   ├── rtsda_system_context_diagram.png
│   ├── density_blending_diagram_rtsda_1.png
│   ├── bottleneck_signal_fusion_1.png
│   ├── threading_architecture.png
│   ├── crowd_crush_fatalities.png
│   └── logo.png
├── make_weidmann_plot.py         # Generates Weidmann diagram
├── make_architecture_diagram.py  # Generates system architecture diagram
├── make_risk_flowchart.py        # Generates risk flowchart
├── make_dashboard_normal.py      # Generates dashboard (normal) mockup
├── make_dashboard_alert.py       # Generates dashboard (alert) mockup
└── .gitignore
```

## Requirements

- Python 3.9+ (recommended 3.11)
- Windows 10/11

### Python Dependencies

```bash
pip install opencv-python numpy Pillow matplotlib
pip install ultralytics          # YOLOv8 (recommended)
pip install torch torchvision    # For CSRNet-lite density model
pip install scikit-learn         # For risk classification training
```

## Usage

### Running the Dashboard

```bash
python dashboard.py
```

Select a video source (file, webcam index, or RTSP URL) when prompted.

### Generating the Report

```bash
npm install
node generate_report.js
```

### Regenerating Diagrams

```bash
python make_weidmann_plot.py
python make_architecture_diagram.py
python make_risk_flowchart.py
python make_dashboard_normal.py
python make_dashboard_alert.py
```

## Model Weights

YOLOv8n weights (`yolov8n.pt`) are not included due to size. They will be downloaded automatically by Ultralytics on first run, or download manually from [Ultralytics](https://github.com/ultralytics/assets/releases).

## Authors

- Bipasa Dutta (17562)
- Shivom Singh (17588)
- Teena Khorwal (17458)

**Supervisor:** Dr Sanjeev Kumar

Dronacharya Group of Institutions, Greater Noida
Dr. A. P. J. Abdul Kalam Technical University

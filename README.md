# Stampede Detection System (RTSDA)

Real-time crowd safety monitoring using computer vision, density estimation, and crowd dynamics analysis.

## Overview

This repository contains a desktop application and supporting tools for detecting crowd stampede risk, monitoring pedestrian density, and generating safety reports.

Key capabilities:
- Person detection and tracking with YOLOv8
- Crowd density estimation using a hybrid YOLO/CSRNet-lite model
- Motion and optical flow analysis for dynamic risk assessment
- Bottleneck, speed anomaly, and counterflow scoring
- Real-time Tkinter dashboard with alerts and event logging

## Features

- **Person Detection & Tracking** — YOLOv8n with ByteTrack-style tracking
- **Density Estimation** — Hybrid blend of YOLO counting and CSRNet-lite regression
- **Optical Flow Analysis** — Farneback-based crowd motion quantification
- **Bottleneck Detection** — Gap score, packing, and flow compression fusion
- **Speed Anomaly Detection** — Weidmann fundamental diagram comparison
- **Counterflow Metric** — Velocity variance analysis for opposing crowd flows
- **Unified Risk Scoring** — Weighted combination of five risk signals
- **Real-Time Dashboard** — Live video, charts, metrics, event log, and audio alerts
- **Multi-Camera Support** — Grid layout with independent worker analysis
- **Auto-Calibration** — Early saturation estimation from initial frames

## Repository Structure

```text
├── dashboard.py                # Main GUI application
├── test8.py                    # Core crowd analysis logic
├── density_model.py            # CSRNet-lite density model definition
├── detector.py                 # YOLOv8 detection wrapper
├── train_yolo.py               # YOLO training pipeline for CrowdHuman
├── train_risk.py               # Risk label training and model creation
├── risk_labels.json            # Risk label definitions
├── generate_report.js          # Word report generation script
├── package.json                # Node dependencies for report generation
├── figures/                    # Report and architecture figures
├── make_architecture_diagram.py # Diagram generation script
├── make_risk_flowchart.py      # Risk flowchart generator
├── make_weidmann_plot.py       # Weidmann diagram generator
├── make_dashboard_normal.py    # Dashboard normal-mode mockup generator
├── make_dashboard_alert.py     # Dashboard alert-mode mockup generator
├── .gitignore                  # Ignored files and folders
└── README.md                   # Project documentation
```

## Requirements

- Python 3.9+ (recommended 3.11)
- Windows 10/11

### Python Dependencies

```bash
pip install opencv-python numpy Pillow matplotlib
pip install ultralytics
pip install torch torchvision
pip install scikit-learn
```

### Node.js Dependencies

```bash
npm install
```

## Usage

### Run the dashboard

```bash
python dashboard.py
```

Choose a video source when prompted: file path, webcam index, or RTSP stream.

### Generate the report

```bash
npm install
node generate_report.js
```

### Regenerate diagrams

```bash
python make_architecture_diagram.py
python make_risk_flowchart.py
python make_weidmann_plot.py
python make_dashboard_normal.py
python make_dashboard_alert.py
```

## Notes

- The repository excludes generated output folders such as `plot/` and `snapshots/`.
- Large model files and datasets are not tracked in Git.
- `yolov8n.pt` is required for YOLO inference and is downloaded automatically by Ultralytics if not present.

## Authors

- Bipasa Dutta (17562)
- Shivom Singh (17588)
- Teena Khorwal (17458)

**Supervisor:** Dr Sanjeev Kumar

Dronacharya Group of Institutions, Greater Noida
Dr. A. P. J. Abdul Kalam Technical University

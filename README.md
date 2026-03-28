# YOLO MLOps Pipeline

A production-ready MLOps pipeline for real-time object detection using YOLOv8,
MLflow experiment tracking, and Docker containerization.

## Features
- YOLOv8 training with automated experiment tracking via MLflow
- Dockerized environment for reproducible training
- CI/CD pipeline via GitHub Actions
- Configurable training parameters via YAML

## Project Structure
```
yolo-mlops-pipeline/
├── src/
│   ├── train.py        # Training script with MLflow tracking
│   └── inference.py    # Inference script
├── configs/
│   └── config.yaml     # Training configuration
├── Dockerfile          # Container definition
├── requirements.txt    # Dependencies
└── .github/
    └── workflows/
        └── train.yml   # CI/CD pipeline
```

## Quick Start

### Local
```bash
pip install -r requirements.txt
python src/train.py
mlflow ui  # View results at http://127.0.0.1:5000
```

### Docker
```bash
docker build -t yolo-mlops .
docker run --rm yolo-mlops
```

## Results
| Metric | Value |
|--------|-------|
| mAP50 | 0.889 |
| Precision | 0.771 |
| Recall | 0.833 |
| mAP50-95 | 0.655 |

## Tech Stack
- YOLOv8 (Ultralytics)
- MLflow
- Docker
- GitHub Actions

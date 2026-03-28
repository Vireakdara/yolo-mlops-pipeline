import mlflow
from ultralytics import YOLO
from ultralytics import settings
import yaml

# Disable Ultralytics built-in MLflow callback
settings.update({"mlflow": False})

def train():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_experiment("yolo-mlops")

    with mlflow.start_run(run_name="yolo-training"):
        mlflow.log_params(config)

        model = YOLO(config["model"])
        results = model.train(
            data=config["data"],
            epochs=config["epochs"],
            imgsz=config["imgsz"],
        )

        metrics = results.results_dict
        mlflow.log_metrics({
            "mAP50": float(metrics.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(metrics.get("metrics/mAP50-95(B)", 0)),
            "precision": float(metrics.get("metrics/precision(B)", 0)),
            "recall": float(metrics.get("metrics/recall(B)", 0)),
        })

        print("Training complete. Metrics logged to MLflow.")

if __name__ == "__main__":
    train()
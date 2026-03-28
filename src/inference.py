import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import argparse

def run_inference(image_path: str, model_path: str = "yolov8n.pt"):
    mlflow.set_experiment("yolo-mlops")

    with mlflow.start_run(run_name="inference"):
        model = YOLO(model_path)
        results = model.predict(source=image_path, save=True)

        mlflow.log_param("image_path", image_path)
        mlflow.log_param("model_path", model_path)
        mlflow.log_metric("num_detections", len(results[0].boxes))

        print(f"Detections: {len(results[0].boxes)}")
        for box in results[0].boxes:
            print(f"  Class: {int(box.cls)} | Confidence: {float(box.conf):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--model", default="yolov8n.pt", help="Model path")
    args = parser.parse_args()

    run_inference(args.image, args.model)

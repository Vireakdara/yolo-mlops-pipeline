from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from ultralytics import settings
import uvicorn
import shutil
import uuid
import os

settings.update({"mlflow": False})

app = FastAPI(title="YOLO Object Detection API")

model = YOLO("yolov8n.pt")

@app.get("/")
def root():
    return {"message": "YOLO Object Detection API", "status": "running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    results = model.predict(source=temp_path)
    os.remove(temp_path)
    
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class_id": int(box.cls),
            "confidence": round(float(box.conf), 3),
            "bbox": box.xyxy[0].tolist()
        })
    
    return {
        "num_detections": len(detections),
        "detections": detections
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
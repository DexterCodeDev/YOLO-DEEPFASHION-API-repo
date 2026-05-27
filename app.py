import os
from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# 1. Load the DeepFashion2 Medium model
model = YOLO('fashion.pt')

# 2. The official DeepFashion2 13-category vocabulary
deepfashion_classes = [
    "short sleeve top", 
    "long sleeve top", 
    "short sleeve outwear", 
    "long sleeve outwear", 
    "vest", 
    "sling", 
    "shorts", 
    "trousers", 
    "skirt", 
    "short sleeve dress", 
    "long sleeve dress", 
    "vest dress", 
    "sling dress"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and decode the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run AI inference
    results = model(img, conf=0.25, imgsz=1024)
    
    # Extract results
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()
    
    # Map the AI's class ID numbers to the DeepFashion2 words
    class_names = [deepfashion_classes[int(c)] for c in classes]
    
    return {
        "boxes": boxes, 
        "classes": class_names, 
        "confidences": confidences
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

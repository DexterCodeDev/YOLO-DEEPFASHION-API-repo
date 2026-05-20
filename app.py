from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI(title="DeepFashion2 YOLO API")

# Load the YOLO model (Ensure your ~50MB .pt file is in the same directory)
# Change 'yolo_df2_m.pt' to your exact model filename
MODEL_PATH = "yolo_df2_m.pt" 

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image into memory
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Run inference
        results = model(image)
        
        # Parse results for the frontend
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0]] # [x1, y1, x2, y2]
                })
                
        return JSONResponse(content={"detections": detections})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Cloud Run needs to listen on the port defined by the PORT environment variable
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

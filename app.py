import os
import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Initializing API
app = FastAPI(
    title="Fashion Object Detection API",
    description="Serverless inference for apparel detection.",
    contact={"email": "support@sereneclothing.store"}
)

# Configuration
MODEL_ID = "yainage90/fashion-object-detection"
HF_TOKEN = os.environ.get("HF_TOKEN")
# Cloud Run doesn't use GPUs, so we explicitly lock to CPU to avoid overhead
DEVICE = torch.device('cpu') 

print(f"Loading {MODEL_ID} into memory...")
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, token=HF_TOKEN).to(DEVICE)
    print("Model ready!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise e

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "model": MODEL_ID, 
        "categories": list(model.config.id2label.values())
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.4):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file. Please upload an image.")

    try:
        # Load and convert image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prepare inputs
        inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Scale bounding boxes back to original image dimensions (requires [height, width])
        target_sizes = torch.tensor([image.size[::-1]]) 
        results = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        # Format the response payload
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "label": model.config.id2label[label.item()],
                "confidence": round(score.item(), 3),
                "bbox": [round(i, 2) for i in box.tolist()] # [xmin, ymin, xmax, ymax]
            })
            
        return JSONResponse(content={"detections": detections})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

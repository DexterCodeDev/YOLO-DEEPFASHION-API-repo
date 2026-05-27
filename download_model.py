import os
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = "yainage90/fashion-object-detection"
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Baking {MODEL_ID} into the Docker image cache...")
AutoImageProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
AutoModelForObjectDetection.from_pretrained(MODEL_ID, token=HF_TOKEN)
print("Caching complete.")

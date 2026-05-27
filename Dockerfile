# Use a lightweight python base
FROM python:3.11-slim

WORKDIR /app

# Ensure logs bypass buffering so they appear immediately in GCP Logs Explorer
ENV PYTHONUNBUFFERED=True \
    PORT=8080

# 1. Install CPU-only PyTorch FIRST to keep the image size small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Cache the Hugging Face model at build-time
COPY download_model.py .
# Accepts the token from Cloud Build if you configure it, though the model is public
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python download_model.py

# 4. Copy the server code
COPY app.py .

# Expose standard Cloud Run port
EXPOSE 8080

# 5. Boot Uvicorn 
# Note: Since Cloud Run handles scaling/concurrency via instances, 1 worker is standard.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system graphics dependencies and wget
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The External Download Trick
# IMPORTANT: Replace the URL in quotes with your actual direct download link!
RUN wget -O fashion.pt "https://huggingface.co/YOUR_LINK_HERE/fashion.pt"

COPY . .

CMD ["python", "app.py"]

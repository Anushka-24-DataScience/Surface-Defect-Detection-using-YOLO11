FROM python:3.10-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Railway sets PORT automatically — app reads it
ENV PORT=7860
ENV MODEL_PATH=models/best.pt
ENV CONF_THRESHOLD=0.25

EXPOSE 7860

CMD ["python", "app.py"]
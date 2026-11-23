# Use Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by OpenCV, Pillow, etc.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create instance folder for SQLite DB
RUN mkdir -p /app/backend/instance /app/backend/temp_uploads

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=backend/app.py
ENV FLASK_ENV=production

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Run with gunicorn on port 7860
# --timeout 300: allow 5 minutes for heavy ML model inference
# --workers 1: one worker to avoid memory issues on free tier
# --threads 4: handle multiple requests concurrently
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "--threads", "4", "backend.app:app"]

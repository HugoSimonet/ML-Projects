# Federated Learning System - Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints logs results plots data

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports for TensorBoard and potential server
EXPOSE 6006 8080

# Default command
CMD ["python", "train_federated.py", "--help"]

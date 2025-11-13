# Dockerfile for Recommendation System

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external models logs

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (run API)
CMD ["python", "-m", "src.api.app"]

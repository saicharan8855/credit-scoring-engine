# ============================================================
# Dockerfile
# Builds a container image for the Credit Scoring API.
#
# We use a slim Python image to keep the image size small.
# All dependencies are installed from requirements.txt.
# The API runs on port 8000 inside the container.
# ============================================================

# Use official Python 3.11 slim image as base
# We use 3.11 instead of 3.13 because it has better
# compatibility with all our ML libraries on Docker
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Set environment variables
# PYTHONDONTWRITEBYTECODE — prevents Python from writing .pyc files
# PYTHONUNBUFFERED — forces Python to print logs immediately without buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed by some Python packages
# libgomp1 is needed by LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer
# If requirements.txt does not change, Docker reuses this cached layer
# and skips reinstalling packages on every build
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose port 8000 so the API is accessible outside the container
EXPOSE 8000

# Command to run the API when the container starts
# host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
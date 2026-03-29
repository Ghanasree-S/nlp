# Use Python 3.11 slim
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by spaCy, torch, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies (no cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model (bundled in requirements.txt via URL, but install again to be safe)
RUN python -m spacy download en_core_web_sm || true

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Healthcheck so Render knows when the service is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

# System deps for native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create data directory
RUN mkdir -p data/cache models logs

# Ensure imports work
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "supervisor.py"]

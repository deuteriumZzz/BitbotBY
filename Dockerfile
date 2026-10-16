# Фронтенд приложения (MiniApp / десктоп): собирается здесь, дашборд отдаёт его по /app/
FROM node:20-alpine AS webapp
WORKDIR /webapp
COPY webapp/package.json webapp/package-lock.json ./
RUN npm ci
COPY webapp/ ./
RUN npm run build

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
# CPU-only torch — избегаем скачивания 1GB+ CUDA библиотек
RUN pip install --no-cache-dir --timeout 300 \
    "torch>=2.0.0" --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# Copy source
COPY . .
COPY --from=webapp /webapp/dist ./webapp/dist

# Create data directory
RUN mkdir -p data/cache models logs

# Ensure imports work
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "supervisor.py"]

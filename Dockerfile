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

# Зависимости отдельным слоем (кэш): список берётся из backend/pyproject.toml
COPY backend/pyproject.toml backend/pyproject.toml
# CPU-only torch — избегаем скачивания 1GB+ CUDA библиотек
RUN pip install --no-cache-dir --timeout 300 \
    "torch>=2.0.0" --extra-index-url https://download.pytorch.org/whl/cpu
RUN python -c "import tomllib; print('\\n'.join(tomllib.load(open('backend/pyproject.toml','rb'))['project']['dependencies']))" > /tmp/deps.txt \
    && pip install --no-cache-dir --timeout 300 -r /tmp/deps.txt

# Код: пакет bitbot ставится без зависимостей (они уже в слое выше)
COPY . .
RUN pip install --no-cache-dir --no-deps ./backend
COPY --from=webapp /webapp/dist ./webapp/dist

# Create data directory
RUN mkdir -p data/cache models logs

# В контейнере сервисы слушают все интерфейсы сети Docker; вне Docker — 127.0.0.1.
ENV BIND_HOST=0.0.0.0

EXPOSE 8080

CMD ["python", "-m", "bitbot.workers.supervisor"]

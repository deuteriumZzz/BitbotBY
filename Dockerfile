FROM python:3.12-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создание необходимых директорий
RUN mkdir -p logs data models

# Установка PYTHONPATH для корректных импортов
ENV PYTHONPATH=/app

CMD ["python", "run_bot.py"]

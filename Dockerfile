FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app:$PYTHONPATH

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY models/ ./models/

CMD ["python", "run_bot.py"]

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY run_bot.py /app/
COPY .env /app/

ENV PYTHONPATH=/app

CMD ["python", "run_bot.py"]

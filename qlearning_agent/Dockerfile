FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY qlearning_agent/qlearning_agent.py /app/qlearning_agent.py

ENV PYTHONUNBUFFERED=1
ENV MODELS_PATH=/app/models

CMD ["python", "qlearning_agent.py"]

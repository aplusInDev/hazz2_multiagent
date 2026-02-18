FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY human_client/human_client.py /app/human_client.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "human_client.py"]

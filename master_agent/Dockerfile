FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY master_agent/master_agent.py /app/master_agent.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "master_agent.py"]

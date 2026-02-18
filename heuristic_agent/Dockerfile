FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ /app/shared/
COPY heuristic_agent/heuristic_agent.py /app/heuristic_agent.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "heuristic_agent.py"]
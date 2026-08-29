FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY data/ ./data/
COPY src/ ./src/
COPY model_artifacts/ ./model_artifacts/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source code and data
COPY data/ ./data/
COPY src/ ./src/
COPY model_artifacts/ ./model_artifacts/

# Expose the API port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

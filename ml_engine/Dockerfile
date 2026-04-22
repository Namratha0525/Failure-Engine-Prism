FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy API code and calibrated models
COPY failure_api/ ./failure_api/
COPY failure_engine/models/ ./failure_engine/models/

# Set environment variable so the API knows where the models are
ENV MODEL_DIR=failure_engine/models

EXPOSE 8000

CMD ["uvicorn", "failure_api.main:app", "--host", "0.0.0.0", "--port", "8000"]

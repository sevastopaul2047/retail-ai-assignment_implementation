# Module C4 - Deployment
FROM python:3.11-slim

WORKDIR /code

# System deps needed by opencv-contrib-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY notebooks/ ./notebooks/
COPY data/ ./data/

ENV API_KEY=change-me-in-production
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

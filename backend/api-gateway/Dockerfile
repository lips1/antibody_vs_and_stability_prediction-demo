
FROM python:3.10-slim

WORKDIR /app

COPY backend/api-gateway/main.py /app/

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    python-multipart

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
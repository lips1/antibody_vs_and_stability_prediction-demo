
FROM python:3.10-slim-bullseye

WORKDIR /app

COPY main.py /app/

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    python-multipart \
    pandas

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
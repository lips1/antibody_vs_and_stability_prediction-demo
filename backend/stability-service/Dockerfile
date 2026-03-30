
FROM python:3.9-slim

WORKDIR /app

# Add CA certificates for HTTPS access (HuggingFace SSL requirements)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# Install dependencies using deepStabP-compatible pinned versions.
RUN pip install --no-cache-dir \
	"fastapi>=0.68.0,<0.69.0" \
	"pydantic>=1.8.0,<2.0.0" \
	"uvicorn>=0.15.0,<0.16.0" \
	"numpy==1.24.1" \
	"pandas==1.5.1" \
	"torch==1.13.0" \
	"transformers==4.24.0" \
	"Bio==1.5.3" \
	"pytorch_lightning==1.7.7" \
	"torchmetrics==0.10.3" \
	"requests==2.28.1" \
	"sentencepiece==0.1.97" \
	"python-multipart"

ENV HF_HOME=/app/hf_cache
ENV HF_HUB_DISABLE_SSL_VERIFICATION=1
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# deepStabP repo is mounted as volume at runtime

# Copy our service code
COPY backend/stability-service/main.py /app/

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

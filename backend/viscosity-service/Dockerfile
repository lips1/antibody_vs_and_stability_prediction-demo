FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    hmmer \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir \
    keras==2.11.0 \
    tensorflow-cpu==2.11.0 \
    scikit-learn==1.0.2 \
    numpy==1.26.4 \
    joblib \
    dill \
    fastapi==0.100.0 \
    uvicorn[standard]==0.23.0 \
    python-multipart==0.0.6 \
    biopython==1.81 \
    pandas==2.0.3

# Install ANARCI (binary name can vary by release: ANARCI/anarci/ANARCII)
RUN pip install --no-cache-dir anarci && \
    ANARCI_BIN="$(command -v ANARCI || command -v anarci || command -v ANARCII || true)" && \
    if [ -n "$ANARCI_BIN" ]; then ln -sf "$ANARCI_BIN" /usr/local/bin/ANARCI; fi && \
    command -v ANARCI || true

# DeepViscosity files are provided via volume mount at runtime (./DeepViscosity:/app/DeepViscosity)
RUN mkdir -p /app/DeepViscosity

# Copy only FastAPI service code
COPY main.py .

# Create upload directory
RUN mkdir -p /app/uploads

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]


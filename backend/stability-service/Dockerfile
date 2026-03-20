
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch first (CPU version, compatible with transformers)
RUN pip install --no-cache-dir torch==1.13.0 --index-url https://download.pytorch.org/whl/cpu

# Install deepStabP dependencies (without pytorch-lightning)
RUN pip install --no-cache-dir \
    transformers==4.24.0 \
    fastapi==0.100.0 \
    uvicorn[standard]==0.23.0 \
    requests==2.31.0 \
    biopython==1.81 \
    numpy==1.24.3 \
    pandas==2.0.3

# Clone deepStabP repo
RUN git clone https://github.com/CSBiology/deepStabP.git /app/deepStabP

# Copy service code
COPY predictor.py .
COPY main.py .

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
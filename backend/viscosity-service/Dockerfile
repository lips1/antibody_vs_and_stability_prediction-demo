FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install DeepViscosity dependencies  
RUN pip install --no-cache-dir \
    keras==2.11.0 \
    tensorflow-cpu==2.11.0 \
    scikit-learn==1.0.2 \
    numpy==1.26.4 \
    joblib \
    dill \
    fastapi==0.100.0 \
    uvicorn[standard]==0.23.0 \
    requests==2.31.0 \
    biopython==1.81 \
    pandas==2.0.3

# Install ANARCI with full path test
RUN pip install --no-cache-dir anarci && \
    which anarci || (python -m pip show anarci && echo "ANARCI installed but not in PATH")

# Clone DeepViscosity repo
RUN git clone https://github.com/Lailabcode/DeepViscosity.git /app/DeepViscosity

# Copy service code
COPY predictor.py .
COPY main.py .

EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]


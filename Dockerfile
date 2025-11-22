# Quantum Institutional Dockerfile
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# Use CPU-only PyTorch to reduce image size for inference
RUN pip install --user --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime libs (e.g. libgomp for torch/xgboost)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy Application Code
COPY . .

# Environment Setup
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default Command
CMD ["python", "start_bot.py"]

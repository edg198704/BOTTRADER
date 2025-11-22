FROM python:3.10-slim

# Set environment variables to prevent .pyc files and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building Python packages
# build-essential, gcc, python3-dev: Required for compiling numpy/pandas extensions
# libffi-dev: Required for cryptography/SSL
# curl: Useful for healthchecks
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# 1. Upgrade pip to ensure compatibility
# 2. Install requirements using the CPU-only index for PyTorch to save space/time
# 3. --no-cache-dir reduces image size
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application code
COPY . .

# Ensure scripts are executable
RUN chmod +x start_bot.sh setup.sh

# Define the command to run the bot
CMD ["python", "start_bot.py"]

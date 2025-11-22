FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
# 'git' is required for installing pandas-ta from source
# 'build-essential' is required for compiling some python extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# We use the CPU version of PyTorch to keep the image size manageable
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Create necessary directories for data persistence
RUN mkdir -p logs models backtest_data

# Command to run the bot
CMD ["python", "start_bot.py"]

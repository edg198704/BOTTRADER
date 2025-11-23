# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}
# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
# pandas-ta from zip to avoid git issues
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p logs models backtest_data

# Expose ports (optional, if your bot has a web dashboard)
# EXPOSE 8080

# Run the bot
CMD ["python", "start_bot.py"]
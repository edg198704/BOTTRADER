FROM python:3.10-slim

# System dependencies
# git: required for installing packages from git repositories
# gcc, build-essential: required for compiling python extensions
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command
CMD ["python", "start_bot.py"]

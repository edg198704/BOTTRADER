#!/bin/bash
# start_bot.sh
# Script to start the trading bot.

# Ensure the virtual environment is activated
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Set Python path to include the current directory for module imports
export PYTHONPATH=$(pwd):$PYTHONPATH

# Set logging level (optional, can also be set in config_enterprise.yaml)
# export LOG_LEVEL=INFO

echo "Starting BOTTRADER..."
python start_bot.py

echo "BOTTRADER stopped."

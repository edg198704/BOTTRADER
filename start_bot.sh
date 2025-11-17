#!/bin/bash

# Enterprise AI Trading Bot - Startup Script

echo "🚀 Starting Enterprise AI Trading Bot..."

# Activate virtual environment
source venv/bin/activate

# Validate environment
python3 utils_enterprise.py --validate-env

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Start the bot
python3 bot_ai_enterprise_refactored.py

echo "✅ Bot stopped"

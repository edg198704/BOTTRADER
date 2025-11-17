#!/bin/bash
# setup.sh
# Script to set up the Python virtual environment and install dependencies.

echo "Setting up Python virtual environment..."

# Check if python3 is available
if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Please install Python 3.8+."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment 'venv' created."
else
    echo "Virtual environment 'venv' already exists."
fi

# Activate the virtual environment
source venv/bin/activate
echo "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "All dependencies installed successfully."
else
    echo "Error installing dependencies. Please check requirements.txt and your internet connection."
    exit 1
fi

echo "Setup complete. You can now run the bot using ./start_bot.sh"
echo "To deactivate the virtual environment, run 'deactivate'."

#!/bin/bash

# Enterprise AI Trading Bot - Setup Script
# This script sets up the environment and dependencies for the enterprise trading bot

set -e  # Exit on any error

echo "🚀 Enterprise AI Trading Bot - Setup Script"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
            print_success "Python $PYTHON_VERSION is compatible"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.8+ is required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip..."
    
    if command -v pip3 &> /dev/null; then
        print_success "pip3 is available"
        PIP_CMD="pip3"
    elif command -v pip &> /dev/null; then
        print_success "pip is available"
        PIP_CMD="pip"
    else
        print_error "pip is not installed"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Upgrade pip
    $PIP_CMD install --upgrade pip
    
    # Install core dependencies
    $PIP_CMD install -r requirements_enterprise.txt
    
    print_success "Dependencies installed successfully"
}

# Setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    # Create config file if it doesn't exist
    if [ ! -f "config_enterprise.yaml" ]; then
        cp config_enterprise.yaml config_enterprise.yaml.bak 2>/dev/null || true
        print_warning "Config file not found. Using template."
    fi
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Exchange Configuration
EXCHANGE=binance
EXCHANGE_API_KEY=your_exchange_api_key_here
EXCHANGE_API_SECRET=your_exchange_api_secret_here

# Trading Parameters
DRY_RUN=true
SYMBOLS=BTC/USDT,ETH/USDT
INITIAL_CAPITAL=10000
MAX_POSITION_SIZE=0.1

# AI/ML Settings
USE_ENSEMBLE=true
TRAINING_SYMBOLS_LIMIT=50

# Monitoring
LOG_LEVEL=INFO
ENABLE_MONITORING=true

# InfluxDB (Optional)
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_influxdb_token
INFLUXDB_ORG=trading_bot
INFLUXDB_BUCKET=trading_metrics

# Telegram (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_ADMIN_IDS=123456789,987654321
EOF
        print_warning "Created .env file template - please update with your actual credentials"
    else
        print_success ".env file already exists"
    fi
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "logs"
        "data"
        "models"
        "reports"
        "config"
        "cache"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_success "Directory already exists: $dir"
        fi
    done
}

# Validate installation
validate_installation() {
    print_status "Validating installation..."
    
    # Test imports
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')

try:
    import pandas as pd
    import numpy as np
    import ccxt
    print('✅ Core dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation validation passed"
    else
        print_error "Installation validation failed"
        exit 1
    fi
}

# Run example
run_example() {
    print_status "Would you like to run the example? (y/n)"
    read -r RUN_EXAMPLE
    
    if [[ $RUN_EXAMPLE =~ ^[Yy]$ ]]; then
        print_status "Running example..."
        $PYTHON_CMD example_enterprise.py
    fi
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > start_bot.sh << 'EOF'
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
EOF
    
    chmod +x start_bot.sh
    print_success "Created startup script: start_bot.sh"
}

# Main setup function
main() {
    echo "Starting setup process..."
    echo ""
    
    check_python
    check_pip
    create_venv
    install_dependencies
    setup_config
    create_directories
    validate_installation
    create_startup_script
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Update .env file with your API credentials"
    echo "2. Review config_enterprise.yaml settings"
    echo "3. Run: ./start_bot.sh"
    echo "   Or: python3 example_enterprise.py"
    echo ""
    echo "📚 Documentation:"
    echo "- README_enterprise.md"
    echo "- config_enterprise.yaml"
    echo "- example_enterprise.py"
    echo ""
    echo "⚠️  Important:"
    echo "- Always test with DRY_RUN=true first"
    echo "- Never commit API keys to version control"
    echo "- Monitor the bot logs in the logs/ directory"
    echo ""
}

# Run main function
main "$@"
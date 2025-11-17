import os

class Config:
    # Exchange API Credentials
    API_KEY = os.getenv('BOTTRADER_API_KEY', 'YOUR_API_KEY')
    API_SECRET = os.getenv('BOTTRADER_API_SECRET', 'YOUR_API_SECRET')
    
    # Trading Parameters
    SYMBOL = os.getenv('BOTTRADER_SYMBOL', 'BTC/USDT')
    TRADE_QUANTITY = float(os.getenv('BOTTRADER_TRADE_QUANTITY', '0.001')) # e.g., 0.001 BTC
    STRATEGY_INTERVAL = int(os.getenv('BOTTRADER_STRATEGY_INTERVAL', '60')) # in seconds
    
    # Risk Management
    MAX_POSITION_SIZE = float(os.getenv('BOTTRADER_MAX_POSITION_SIZE', '0.01')) # e.g., 0.01 BTC
    STOP_LOSS_PERCENTAGE = float(os.getenv('BOTTRADER_STOP_LOSS_PERCENTAGE', '0.02')) # 2% stop loss
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('BOTTRADER_TAKE_PROFIT_PERCENTAGE', '0.03')) # 3% take profit
    
    # Other Settings
    LOG_LEVEL = os.getenv('BOTTRADER_LOG_LEVEL', 'INFO')
    DRY_RUN = os.getenv('BOTTRADER_DRY_RUN', 'True').lower() == 'true'

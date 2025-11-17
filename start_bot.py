#!/usr/bin/env python3
"""
Enterprise AI Trading Bot - Simple Start Script

A simplified entry point that loads configuration and starts the bot.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from bot_ai_enterprise_refactored import TradingConfig, EnterpriseTradingBot
    from config_loader import ConfigLoader
    from utils_enterprise import create_directory_structure, validate_environment
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run: pip install -r requirements_enterprise.txt")
    sys.exit(1)


async def main():
    """Main entry point"""
    try:
        print("🚀 Enterprise AI Trading Bot")
        print("=" * 40)
        
        # Create directories
        create_directory_structure()
        
        # Validate environment
        env_checks = validate_environment()
        print("\n🔍 Environment Check:")
        for check, result in env_checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check}")
        
        # Skip non-critical checks for demo
        if not env_checks.get('environment_vars', False):
            print("\n⚠️  Environment variables missing - this is OK for demo mode")
        
        # Load configuration
        print("\n⚙️  Loading configuration...")
        config_loader = ConfigLoader("config_enterprise.yaml")
        config_dict = config_loader.load_config()
        
        # Create TradingConfig
        config = TradingConfig(
            exchange=config_dict.get('exchange', {}).get('name', 'binance'),
            sandbox=config_dict.get('exchange', {}).get('sandbox', False),
            dry_run=True,  # Always start in dry-run mode for safety
            api_key=os.getenv('EXCHANGE_API_KEY'),
            api_secret=os.getenv('EXCHANGE_API_SECRET'),
            symbols=config_dict.get('trading', {}).get('symbols', ['BTC/USDT', 'ETH/USDT']),
            timeframe=config_dict.get('trading', {}).get('timeframe', '1h'),
            initial_capital=config_dict.get('trading', {}).get('initial_capital', 10000),
            max_position_size=config_dict.get('trading', {}).get('max_position_size', 0.1),
            stop_loss_pct=config_dict.get('trading', {}).get('stop_loss_pct', 0.02),
            take_profit_pct=config_dict.get('trading', {}).get('take_profit_pct', 0.04),
            use_ensemble=config_dict.get('ai_ml', {}).get('use_ensemble', True),
            log_level=config_dict.get('system', {}).get('log_level', 'INFO'),
            enable_debug_logging=config_dict.get('system', {}).get('enable_debug_logging', False)
        )
        
        print("✅ Configuration loaded")
        print(f"  Exchange: {config.exchange}")
        print(f"  Symbols: {', '.join(config.symbols)}")
        print(f"  Capital: ${config.initial_capital:,.2f}")
        print(f"  Dry Run: {config.dry_run}")
        
        # Create and initialize bot
        print("\n🤖 Initializing trading bot...")
        bot = EnterpriseTradingBot(config)
        
        # Start bot
        print("🚀 Starting bot...")
        if await bot.start():
            print("✅ Bot started successfully!")
            print("\n📊 Bot Status:")
            
            # Show initial status
            status = bot.get_status()
            print(f"  Status: {'Running' if status['is_running'] else 'Stopped'}")
            print(f"  Equity: ${status['current_equity']:,.2f}")
            print(f"  Memory: {status['memory_usage_mb']:.1f} MB")
            
            print("\n⏹️  Press Ctrl+C to stop the bot")
            print("-" * 40)
            
            # Keep running
            try:
                while bot.is_running:
                    await asyncio.sleep(10)
                    
                    # Show periodic status updates
                    if int(asyncio.get_event_loop().time()) % 60 == 0:
                        status = bot.get_status()
                        print(f"📈 Status Update - Equity: ${status['current_equity']:,.2f} | "
                              f"Positions: {status['open_positions']} | "
                              f"Memory: {status['memory_usage_mb']:.1f}MB")
            
            except KeyboardInterrupt:
                print("\n⏹️  Stopping bot...")
        
        else:
            print("❌ Failed to start bot")
        
        # Cleanup
        print("🧹 Cleaning up...")
        await bot.stop()
        print("✅ Bot stopped and cleaned up")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not available, skip
    
    # Run the bot
    asyncio.run(main())
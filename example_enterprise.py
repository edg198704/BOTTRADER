#!/usr/bin/env python3
"""
Enterprise AI Trading Bot - Example Usage

Demonstrates how to use the refactored enterprise trading bot.
"""

import asyncio
import os
import json
from datetime import datetime, timezone
from pathlib import Path

# Import our refactored components
from bot_ai_enterprise_refactored import (
    TradingConfig, EnterpriseTradingBot, EnterpriseLogger,
    ErrorHandler, ResourceManager
)
from config_loader import ConfigLoader
from utils_enterprise import (
    validate_environment, create_directory_structure,
    calculate_position_size, format_currency
)


class TradingBotExample:
    """Example implementation of the enterprise trading bot"""
    
    def __init__(self):
        self.config = None
        self.bot = None
        self.logger = None
    
    async def initialize(self):
        """Initialize the trading bot"""
        try:
            # Create directory structure
            create_directory_structure()
            
            # Load configuration
            config_loader = ConfigLoader("config_enterprise.yaml")
            config_dict = config_loader.load_config()
            
            # Create TradingConfig from loaded data
            self.config = TradingConfig(
                exchange=config_dict.get('exchange', {}).get('name', 'binance'),
                sandbox=config_dict.get('exchange', {}).get('sandbox', False),
                dry_run=config_dict.get('exchange', {}).get('dry_run', True),
                api_key=config_dict.get('exchange', {}).get('api_key'),
                api_secret=config_dict.get('exchange', {}).get('api_secret'),
                symbols=config_dict.get('trading', {}).get('symbols', ['BTC/USDT']),
                timeframe=config_dict.get('trading', {}).get('timeframe', '1h'),
                initial_capital=config_dict.get('trading', {}).get('initial_capital', 10000),
                max_position_size=config_dict.get('trading', {}).get('max_position_size', 0.1),
                stop_loss_pct=config_dict.get('trading', {}).get('stop_loss_pct', 0.02),
                take_profit_pct=config_dict.get('trading', {}).get('take_profit_pct', 0.04),
                use_ensemble=config_dict.get('ai_ml', {}).get('use_ensemble', True),
                log_level=config_dict.get('system', {}).get('log_level', 'INFO'),
                enable_debug_logging=config_dict.get('system', {}).get('enable_debug_logging', False)
            )
            
            # Validate environment
            env_checks = validate_environment()
            print("\n🔍 Environment Validation:")
            for check, result in env_checks.items():
                status = "✅" if result else "❌"
                print(f"  {status} {check}")
            
            # Initialize logger
            self.logger = EnterpriseLogger("TradingBotExample", self.config)
            
            # Initialize bot
            self.bot = EnterpriseTradingBot(self.config)
            
            # Register additional error handlers
            self._setup_custom_handlers()
            
            self.logger.info("trading_bot_example_initialized",
                           config_summary={
                               'exchange': self.config.exchange,
                               'symbols': len(self.config.symbols),
                               'initial_capital': self.config.initial_capital,
                               'dry_run': self.config.dry_run
                           })
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def _setup_custom_handlers(self):
        """Setup custom error handlers and callbacks"""
        
        # Custom error recovery strategy
        from bot_ai_enterprise_refactored import ExponentialBackoffRecovery
        
        self.bot.error_handler.register_recovery_strategy(
            "ValueError",
            ExponentialBackoffRecovery(max_attempts=2, base_delay=0.5)
        )
        
        # Custom cleanup callback
        async def custom_cleanup():
            """Custom cleanup operation"""
            self.logger.info("custom_cleanup_executed")
            # Add any custom cleanup logic here
        
        self.bot.resource_manager.register_cleanup_callback(
            "custom_cleanup",
            custom_cleanup,
            priority=9
        )
    
    async def run_example_scenarios(self):
        """Run example trading scenarios"""
        try:
            print("\n🚀 Starting Enterprise Trading Bot Examples...")
            
            # Start the bot
            if not await self.bot.start():
                print("❌ Failed to start bot")
                return
            
            print("✅ Bot started successfully")
            
            # Example 1: Basic Trading Loop
            await self._example_basic_trading()
            
            # Example 2: Risk Management Demo
            await self._example_risk_management()
            
            # Example 3: AI/ML Prediction Demo
            await self._example_ai_prediction()
            
            # Example 4: Monitoring Demo
            await self._example_monitoring()
            
            # Example 5: Error Handling Demo
            await self._example_error_handling()
            
            print("\n🎯 All example scenarios completed")
            
        except KeyboardInterrupt:
            print("\n⏹️  Example interrupted by user")
        except Exception as e:
            self.logger.error("example_scenarios_failed", error=str(e))
            print(f"❌ Example scenarios failed: {e}")
        finally:
            await self._cleanup()
    
    async def _example_basic_trading(self):
        """Example: Basic trading operations"""
        print("\n📈 Example 1: Basic Trading Operations")
        
        try:
            # Check exchange connection
            if self.bot.exchange_manager:
                ticker_result = await self.bot.exchange_manager.fetch_ticker(self.config.symbols[0])
                if ticker_result['success']:
                    print(f"  ✅ Exchange connected: {ticker_result['symbol']} = {format_currency(ticker_result['last'])}")
                else:
                    print(f"  ❌ Exchange connection failed")
            
            # Simulate position opening
            if self.bot.position_manager:
                symbol = self.config.symbols[0]
                entry_price = 50000.0  # Example BTC price
                size = 0.01  # 0.01 BTC
                
                success = await self.bot.position_manager.open_position(
                    symbol, 'buy', size, entry_price
                )
                
                if success:
                    print(f"  ✅ Position opened: {symbol} - {size} BTC @ {format_currency(entry_price)}")
                else:
                    print(f"  ❌ Position open failed")
            
            # Update positions with current price
            if self.bot.position_manager and self.bot.exchange_manager:
                for symbol in self.bot.position_manager.positions:
                    ticker_result = await self.bot.exchange_manager.fetch_ticker(symbol)
                    if ticker_result['success']:
                        await self.bot.position_manager.update_position(symbol, ticker_result['last'])
                        print(f"  📊 Position updated: {symbol} @ {format_currency(ticker_result['last'])}")
            
        except Exception as e:
            self.logger.error("basic_trading_example_failed", error=str(e))
    
    async def _example_risk_management(self):
        """Example: Risk management operations"""
        print("\n🛡️  Example 2: Risk Management")
        
        try:
            if self.bot.risk_manager and self.bot.position_manager:
                # Test risk limits
                symbol = self.config.symbols[0]
                can_trade, reason = await self.bot.risk_manager.check_risk_limits(
                    symbol, 'buy', 0.1, 50000.0
                )
                
                if can_trade:
                    print(f"  ✅ Risk check passed: {reason}")
                else:
                    print(f"  ⚠️  Risk check failed: {reason}")
                
                # Calculate position size
                position_size = calculate_position_size(
                    capital=self.config.initial_capital,
                    risk_per_trade=self.config.max_risk_per_trade,
                    entry_price=50000.0,
                    stop_loss_price=49000.0
                )
                print(f"  📏 Calculated position size: {position_size:.6f} BTC")
                
                # Assess portfolio risk
                risk_metrics = await self.bot.risk_manager.assess_portfolio_risk()
                print(f"  📊 Portfolio risk score: {risk_metrics.get('risk_score', 0)}/100")
            
        except Exception as e:
            self.logger.error("risk_management_example_failed", error=str(e))
    
    async def _example_ai_prediction(self):
        """Example: AI/ML prediction demo"""
        print("\n🤖 Example 3: AI/ML Predictions")
        
        try:
            if self.bot.ensemble_learner:
                # Check model status
                if self.bot.ensemble_learner.is_trained:
                    print("  ✅ Ensemble models are trained")
                    
                    # Simulate prediction
                    import pandas as pd
                    
                    # Create sample data
                    sample_data = pd.DataFrame({
                        'open': [50000, 50100, 50200],
                        'high': [50200, 50300, 50400],
                        'low': [49900, 50000, 50100],
                        'close': [50100, 50200, 50300],
                        'volume': [1000, 1100, 1200]
                    })
                    
                    # Make prediction
                    prediction = await self.bot.ensemble_learner.predict(sample_data)
                    
                    if prediction['success']:
                        print(f"  🎯 Prediction: {prediction['action']} with {prediction['confidence']:.1%} confidence")
                    else:
                        print(f"  ❌ Prediction failed: {prediction.get('error', 'Unknown error')}")
                else:
                    print("  ⏳ Ensemble models need training")
            
        except Exception as e:
            self.logger.error("ai_prediction_example_failed", error=str(e))
    
    async def _example_monitoring(self):
        """Example: Monitoring and health checks"""
        print("\n📊 Example 4: Monitoring & Health Checks")
        
        try:
            # Get bot status
            status = self.bot.get_status()
            print(f"  📈 Bot Status: {'Running' if status['is_running'] else 'Stopped'}")
            print(f"  💰 Current Equity: {format_currency(status['current_equity'])}")
            print(f"  📍 Open Positions: {status['open_positions']}")
            print(f"  🧠 Memory Usage: {status['memory_usage_mb']:.1f} MB")
            
            # Perform health check
            health = await self.bot._perform_health_check()
            print(f"  🏥 Health Status: {health['status'].upper()}")
            
            for check, result in health['checks'].items():
                status_icon = "✅" if result == "healthy" else "⚠️" if result == "warning" else "❌"
                print(f"    {status_icon} {check}: {result}")
            
            # Get error statistics
            error_stats = self.bot.error_handler.get_error_stats()
            print(f"  🔍 Total Errors: {error_stats.get('total_errors', 0)}")
            
        except Exception as e:
            self.logger.error("monitoring_example_failed", error=str(e))
    
    async def _example_error_handling(self):
        """Example: Error handling demonstration"""
        print("\n🚨 Example 5: Error Handling")
        
        try:
            # Demonstrate error logging
            self.logger.info("example_info_message", test_param="example_value")
            self.logger.warning("example_warning_message", warning_type="demonstration")
            
            # Demonstrate error recovery
            from bot_ai_enterprise_refactored import ExchangeError
            
            try:
                # Simulate an error
                raise ExchangeError("Simulated exchange error for demonstration")
            except ExchangeError as e:
                # This would be handled by the error handler
                self.logger.error("example_error_handled", error=str(e))
            
            print("  ✅ Error handling demonstration completed")
            
        except Exception as e:
            self.logger.error("error_handling_example_failed", error=str(e))
    
    async def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.bot:
                await self.bot.stop()
                print("  🧹 Bot stopped and cleaned up")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")


async def main():
    """Main example execution"""
    print("🚀 Enterprise AI Trading Bot - Example Usage")
    print("=" * 50)
    
    example = TradingBotExample()
    
    # Initialize
    if not await example.initialize():
        print("❌ Failed to initialize example")
        return
    
    # Run examples
    await example.run_example_scenarios()
    
    print("\n✅ Example execution completed")


if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault('EXCHANGE', 'binance')
    os.environ.setdefault('DRY_RUN', 'true')
    os.environ.setdefault('SYMBOLS', 'BTC/USDT,ETH/USDT')
    os.environ.setdefault('INITIAL_CAPITAL', '10000')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    # Run example
    asyncio.run(main())
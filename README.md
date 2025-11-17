# BOTTRADER - Algorithmic Trading Bot

This repository contains a refactored and production-ready algorithmic trading bot designed for robustness, modularity, and safety. The bot is built with Python and uses an event-driven (asyncio) architecture, allowing for flexible integration with various exchange APIs and trading strategies.

## Architecture Overview

The bot's architecture is designed with clear separation of concerns:

*   **`bot_core/`**: Contains the core logic of the trading bot.
    *   `exchange_api.py`: Defines an abstract interface for interacting with cryptocurrency exchanges. Includes a `MockExchangeAPI` for testing and a placeholder for `BinanceExchangeAPI`.
    *   `position_manager.py`: Manages open and closed trading positions, persisting them to an SQLite database (`position_ledger.db`).
    *   `risk_manager.py`: Implements pre-trade and post-trade risk checks, such as maximum position size, daily loss limits, and maximum open positions.
    *   `strategy.py`: Defines an abstract `TradingStrategy` interface. Concrete strategies (e.g., `SimpleMACrossoverStrategy`) implement the trading logic, decoupled from execution details.
    *   `bot.py`: The main orchestrator class (`TradingBot`) that ties together the exchange API, position manager, risk manager, and strategy. It runs the main trading loop.
*   **`config_loader.py`**: Handles loading and parsing of YAML configuration files, providing a structured way to manage bot settings.
*   **`config_enterprise.yaml`**: The primary configuration file for the bot, defining exchange credentials, strategy parameters, risk limits, and other operational settings.
*   **`start_bot.py`**: The main entry point for running the bot. It initializes all components based on the configuration and starts the trading loop.
*   **`requirements.txt`**: Lists all Python dependencies required for the project.
*   **`setup.sh`**: A utility script to set up the Python virtual environment and install dependencies.
*   **`start_bot.sh`**: A convenience script to activate the virtual environment and run `start_bot.py`.
*   **`position_ledger.db`**: SQLite database file for storing position history.

## Features

*   **Modular Design**: Easily swap out exchange APIs or trading strategies.
*   **Robust Risk Management**: Configurable limits for position size, daily loss, and open positions.
*   **Persistent Position Tracking**: All trades and positions are logged and managed in an SQLite database.
*   **Asynchronous Operations**: Built with `asyncio` for non-blocking I/O, crucial for low-latency trading.
*   **Comprehensive Logging**: Detailed logging for monitoring bot operations and debugging.
*   **Graceful Shutdown**: Handles `SIGINT` and `SIGTERM` signals for clean termination.
*   **Configurable**: All key parameters are managed via `config_enterprise.yaml`.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/edg198704/BOTTRADER.git
    cd BOTTRADER
    ```

2.  **Run the setup script:**
    This script will create a Python virtual environment and install all necessary dependencies.
    ```bash
    ./setup.sh
    ```

3.  **Configure the bot:**
    Open `config_enterprise.yaml` and update the settings according to your requirements:
    *   **`exchange`**: Choose `MockExchange` for testing or `Binance` for live trading. If using Binance, provide your `api_key` and `api_secret`. Set `testnet` to `true` for Binance testnet.
    *   **`strategy`**: Adjust `symbol`, `interval`, `trade_quantity`, and strategy-specific parameters.
    *   **`risk_management`**: Set your desired `max_position_size_usd`, `max_daily_loss_usd`, and `max_open_positions`.
    *   **`database`**: The path to your `position_ledger.db` file.

    **IMPORTANT**: Never hardcode sensitive information directly into the code. Use environment variables or a secure configuration management system for production. For this example, API keys are in `config_enterprise.yaml` for simplicity, but this is not recommended for production.

## Running the Bot

To start the trading bot:

```bash
./start_bot.sh
```

The bot will start its main loop, fetch market data, analyze trades based on the configured strategy, and manage positions.

## Development and Testing

### Mock Exchange

The `MockExchangeAPI` allows you to test your strategy logic without connecting to a real exchange. To use it, set `exchange.name` to `MockExchange` in `config_enterprise.yaml`.

### Adding New Strategies

To add a new trading strategy:
1.  Create a new Python file in `bot_core/strategy.py` (or a new file in `bot_core/strategies/` if you prefer a sub-package).
2.  Implement a class that inherits from `bot_core.strategy.TradingStrategy`.
3.  Implement the `analyze_market` and `manage_positions` abstract methods with your trading logic.
4.  Update `start_bot.py` to import and instantiate your new strategy class based on the `strategy.name` in `config_enterprise.yaml`.

### Database Inspection

You can inspect the `position_ledger.db` SQLite database using any SQLite browser to view historical positions and PnL.

## Error Handling and Resilience

The bot includes:
*   Structured logging for all operations and errors.
*   Graceful shutdown mechanisms.
*   Error handling around API calls and database operations.
*   Risk management to prevent excessive losses.

## Future Enhancements

*   **Real-time Data Streams**: Integrate WebSocket connections for lower-latency market data.
*   **Order Management System (OMS)**: More sophisticated order tracking, partial fills, and retry logic.
*   **Performance Monitoring**: Metrics collection (e.g., Prometheus) for real-time performance insights.
*   **Notification System**: Integrate with Telegram, Email, or Slack for alerts.
*   **Backtesting Framework**: A dedicated module for backtesting strategies against historical data.
*   **Deployment Automation**: Dockerization and Kubernetes deployment manifests.
*   **Advanced Risk Controls**: Dynamic position sizing, portfolio-level risk, circuit breakers.
*   **Configuration Validation**: Use libraries like Pydantic for strict configuration schema validation.

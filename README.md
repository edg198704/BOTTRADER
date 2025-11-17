# BOTTRADER - Modular Algorithmic Trading Bot

This repository contains a production-ready, modular algorithmic trading bot built with Python. The architecture is designed for robustness, extensibility, and clear separation of concerns, making it easy to develop, test, and deploy new trading strategies.

## Architecture Overview

The bot's architecture is centered around a core set of decoupled components located in the `bot_core/` directory:

*   **`bot.py`**: The main `TradingBot` class that orchestrates all components and runs the primary trading loop.
*   **`config.py`**: Defines Pydantic models for strict, type-safe configuration validation.
*   **`exchange_api.py`**: Provides an abstract interface for exchange interactions, with concrete implementations for a `MockExchangeAPI` (for testing) and `CCXTExchangeAPI` (for live/paper trading).
*   **`position_manager.py`**: Manages the state of all trading positions, persisting them to an SQLite database for durability using SQLAlchemy.
*   **`risk_manager.py`**: Implements pre-trade and portfolio-level risk controls, including dynamic position sizing, ATR-based stop loss, and a portfolio-wide circuit breaker.
*   **`strategy.py`**: Defines the `TradingStrategy` interface and provides concrete implementations like `SimpleMACrossoverStrategy` and the advanced `AIEnsembleStrategy`.

Key supporting files in the root directory include:

*   **`start_bot.py`**: The single entry point for running the bot. It handles initialization, dependency injection, and graceful shutdown.
*   **`config_loader.py`**: A utility to load and validate the `config_enterprise.yaml` file.
*   **`config_enterprise.yaml`**: The central configuration file for all bot parameters.
*   **`requirements.txt`**: A consolidated list of all Python dependencies.

## Key Features

*   **Modular & Extensible**: Easily swap out exchange APIs or trading strategies by modifying the configuration.
*   **Advanced AI Strategy**: Includes a powerful ensemble strategy using XGBoost, RandomForest, and other models for sophisticated signal generation.
*   **Robust Risk Management**: Features dynamic position sizing, ATR-based stop loss, multiple take-profit levels, and a portfolio-level circuit breaker to protect capital.
*   **Persistent State**: All trades are logged to an SQLite database using SQLAlchemy, ensuring state is not lost on restart.
*   **Asynchronous Core**: Built with `asyncio` for efficient, non-blocking I/O, critical for handling real-time market data and API requests.
*   **Type-Safe Configuration**: Uses Pydantic for configuration loading and validation, preventing runtime errors from incorrect settings.
*   **Graceful Shutdown**: Handles `SIGINT` and `SIGTERM` signals to ensure clean termination and resource cleanup.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/edg198704/BOTTRADER.git
    cd BOTTRADER
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure the bot:**
    Open `config_enterprise.yaml` and customize the settings:
    *   **`exchange`**: Set `name` to `MockExchange` for testing or a `ccxt`-supported exchange like `binance`. For a real exchange, provide your `api_key` and `api_secret` (it's highly recommended to use environment variables for these in production).
    *   **`strategy`**: Choose the `name` of the strategy to run (e.g., `AIEnsembleStrategy` or `SimpleMACrossoverStrategy`) and configure its parameters.
    *   **`risk_management`**: Define your risk tolerance with `max_position_size_usd`, `max_daily_loss_usd`, etc.

    **IMPORTANT**: For production, load API keys from environment variables or a secure vault, not directly from the YAML file.

## Running the Bot

To start the trading bot, simply run the main entry point:

```bash
python start_bot.py
```

The bot will initialize all components based on your configuration and begin its main trading loop. To stop the bot gracefully, press `Ctrl+C`.

## Development

### Adding a New Strategy

1.  Create a new class in `bot_core/strategy.py` that inherits from `TradingStrategy`.
2.  Implement the `analyze_market` and `manage_positions` abstract methods with your custom logic.
3.  Import your new strategy class in `start_bot.py`.
4.  Update the factory function `get_strategy` in `start_bot.py` to recognize and instantiate your new strategy by adding its class name to the `strategy_map` dictionary.
5.  Update `config_enterprise.yaml` to set `strategy.name` to your new strategy's class name.

### Database Inspection

You can inspect the `position_ledger.db` SQLite database using any standard SQLite browser to view historical positions and PnL.

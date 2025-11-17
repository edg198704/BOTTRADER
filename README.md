# BOTTRADER - Modular Algorithmic Trading Bot

This repository contains a production-ready, modular algorithmic trading bot built with Python. The architecture is designed for robustness, extensibility, and clear separation of concerns, making it easy to develop, test, and deploy new trading strategies.

## Architecture Overview

The bot's architecture is centered around a core set of decoupled components:

*   **`bot_core/`**: Contains the primary logic of the trading bot.
    *   `config.py`: Defines Pydantic models for strict, type-safe configuration validation.
    *   `exchange_api.py`: Provides an abstract interface for exchange interactions, with concrete implementations for a `MockExchangeAPI` (for testing) and `CCXTExchangeAPI` (for live/paper trading).
    *   `position_manager.py`: Manages the state of all open and closed trading positions, persisting them to an SQLite database for durability.
    *   `risk_manager.py`: Implements pre-trade and portfolio-level risk controls, including position size limits, daily loss thresholds, and a portfolio-wide circuit breaker.
    *   `strategy.py`: Defines the `TradingStrategy` interface and provides concrete implementations like `SimpleMACrossoverStrategy` and the more advanced `AIEnsembleStrategy`.
    *   `bot.py`: The main `TradingBot` class that orchestrates all components, running the primary trading loop.
*   **`config_loader.py`**: A utility to load and validate the `config_enterprise.yaml` file against the Pydantic models.
*   **`config_enterprise.yaml`**: The central configuration file for all bot parameters, including exchange credentials, strategy selection, and risk limits.
*   **`start_bot.py`**: The single entry point for running the bot. It handles initialization, dependency injection, and graceful shutdown.
*   **`requirements.txt`**: A consolidated list of all Python dependencies.
*   **`position_ledger.db`**: The SQLite database file where all trade and position history is stored.

## Key Features

*   **Modular & Extensible**: Easily swap out exchange APIs or trading strategies by modifying the configuration.
*   **Robust Risk Management**: Configurable limits for position size, daily loss, open positions, and a portfolio-level circuit breaker to protect capital.
*   **Persistent State**: All trades are logged to an SQLite database, ensuring state is not lost on restart.
*   **Asynchronous Core**: Built with `asyncio` for efficient, non-blocking I/O, critical for handling real-time market data and API requests.
*   **Strategy Support**: Comes with a simple MA Crossover strategy and a powerful AI Ensemble strategy out of the box.
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
    *   **`exchange`**: Set `name` to `MockExchange` for testing or a `ccxt`-supported exchange like `binance`. If using a real exchange, provide your `api_key` and `api_secret`. Set `testnet` to `true` for paper trading if supported.
    *   **`strategy`**: Choose the `name` of the strategy to run (e.g., `AIEnsembleStrategy`) and configure its parameters like `symbol` and `interval_seconds`.
    *   **`risk_management`**: Define your risk tolerance with `max_position_size_usd`, `max_daily_loss_usd`, etc.

    **IMPORTANT**: For production, it is strongly recommended to load API keys from environment variables or a secure vault, not directly from the YAML file.

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
3.  Update `config_enterprise.yaml` to set `strategy.name` to your new strategy's class name.
4.  Update the factory function `get_strategy` in `start_bot.py` to recognize and instantiate your new strategy.

### Database Inspection

You can inspect the `position_ledger.db` SQLite database using any standard SQLite browser to view historical positions and PnL.

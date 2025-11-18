# BOTTRADER - Modular Algorithmic Trading Bot

This repository contains a production-ready, modular algorithmic trading bot built with Python. The architecture is designed for robustness, extensibility, and a clear separation of concerns, making it easy to develop, test, and deploy new trading strategies.

## Architecture Overview

The bot's architecture is centered around a core set of decoupled components located in the `bot_core/` directory:

*   **`bot.py`**: The main `TradingBot` class that orchestrates all components and runs the primary trading loop for multiple symbols concurrently.
*   **`config.py`**: Defines Pydantic models for strict, type-safe configuration validation.
*   **`exchange_api.py`**: Provides an abstract interface for exchange interactions, with concrete implementations for a `MockExchangeAPI` (for testing) and `CCXTExchangeAPI` (for live/paper trading).
*   **`position_manager.py`**: Manages the state of all trading positions, persisting them to an SQLite database for durability using SQLAlchemy.
*   **`risk_manager.py`**: Implements pre-trade and portfolio-level risk controls, including dynamic position sizing, ATR-based stop loss, and a portfolio-wide circuit breaker.
*   **`strategy.py`**: Defines the `TradingStrategy` interface and provides concrete implementations like `SimpleMACrossoverStrategy` and the advanced `AIEnsembleStrategy`.
*   **`telegram_bot.py`**: An optional component for remote control and monitoring via Telegram.
*   **`logger.py`**: Configures a structured, context-aware logger for the application.

Key supporting files in the root directory include:

*   **`start_bot.py`**: The single entry point for running the bot. It handles initialization, dependency injection, and graceful shutdown.
*   **`config_loader.py`**: A utility to load and validate the `config_enterprise.yaml` file.
*   **`config_enterprise.yaml`**: The central configuration file for all bot parameters.
*   **`requirements.txt`**: A consolidated list of all Python dependencies.

## Key Features

*   **Multi-Symbol Trading**: Trade multiple currency pairs (e.g., BTC/USDT, ETH/USDT) concurrently from a single configuration.
*   **Modular & Extensible**: Easily swap out exchange APIs or trading strategies by modifying the configuration.
*   **Advanced AI Strategy**: Includes a powerful ensemble strategy using XGBoost, RandomForest, and other models for sophisticated signal generation.
*   **Robust Risk Management**: Features dynamic position sizing, ATR-based stop loss, and a portfolio-level circuit breaker to protect capital.
*   **Persistent State**: All trades are logged to an SQLite database using SQLAlchemy, ensuring state is not lost on restart.
*   **Remote Control via Telegram**: Includes an optional Telegram bot for real-time status monitoring and an emergency 'stop' command.
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
    Copy the `config_enterprise.yaml` file and customize the settings:
    *   **`exchange`**: Set `name` to `MockExchange` for testing or a `ccxt`-supported exchange like `binance`. For a real exchange, you will need to set environment variables for your API key and secret (see step 4).
    *   **`strategy`**: Choose the `name` of the strategy to run (e.g., `AIEnsembleStrategy`) and define the list of `symbols` to trade.
    *   **`risk_management`**: Define your risk tolerance.
    *   **`telegram`**: To enable remote control, create a bot with BotFather on Telegram, get the token, find your chat ID, and add them to the config and environment variables.

4.  **Set up Environment Variables:**
    For production, API keys and tokens **must** be loaded from environment variables. The bot uses Pydantic to automatically and securely load these values. They should **not** be written in `config_enterprise.yaml`.

    You can set them in your shell before running the bot:
    ```bash
    export BOT_EXCHANGE_API_KEY="your_exchange_api_key"
    export BOT_EXCHANGE_API_SECRET="your_exchange_api_secret"
    export BOT_TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
    ```

    Alternatively, for local development, you can create a `.env` file in the root directory (this file is git-ignored), and the bot will load it automatically on startup:
    ```
    BOT_EXCHANGE_API_KEY="your_exchange_api_key"
    BOT_EXCHANGE_API_SECRET="your_exchange_api_secret"
    BOT_TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
    ```

## Running the Bot

To start the trading bot, simply run the main entry point:

```bash
python start_bot.py
```

The bot will initialize all components based on your configuration and begin its main trading loop for all configured symbols. If configured, the Telegram bot will also start polling for commands. To stop the bot gracefully, press `Ctrl+C`.

## Development

### Adding a New Strategy

The bot is designed for easy strategy extension without modifying core application files.

1.  **Create the Strategy Class**: In `bot_core/strategy.py`, create a new class that inherits from `TradingStrategy`. Your new class will be automatically available to the bot.
    ```python
    # In bot_core/strategy.py
    class MyAwesomeStrategy(TradingStrategy):
        def __init__(self, config: StrategyConfig):
            super().__init__(config)
            # Add any strategy-specific initialization here.
            # You can add a config model for your strategy in bot_core/config.py

        async def analyze_market(self, symbol: str, df: pd.DataFrame, position: Optional[Position]) -> Optional[Dict[str, Any]]:
            # Implement your custom trading logic here
            # Return a signal dictionary or None
            pass
    ```

2.  **Configure the Bot**: Open `config_enterprise.yaml` and update the `strategy` section to use your new class.
    ```yaml
    strategy:
      name: "MyAwesomeStrategy" # Must match your class name exactly
      symbols: ["BTC/USDT"]
      interval_seconds: 10
      timeframe: "5m"
      # Add any custom config parameters for your strategy here
    ```

That's it. The bot will automatically find and load your strategy class from `bot_core/strategy.py` by its name. There is no need to modify `start_bot.py`.

### Database Inspection

You can inspect the `position_ledger.db` SQLite database using any standard SQLite browser to view historical positions and PnL.

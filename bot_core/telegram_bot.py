import asyncio
from typing import Dict, Any, List, Callable

from bot_core.logger import get_logger
from bot_core.config import TelegramConfig

# Safe import for telegram
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = get_logger(__name__)

class TelegramBot:
    def __init__(self, config: TelegramConfig, bot_state: Dict[str, Any]):
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot is not installed. Cannot start TelegramBot.")
        
        self.config = config
        self.bot_state = bot_state # Shared state with the main TradingBot
        if not config.bot_token or not config.admin_chat_ids:
            logger.warning("Telegram bot token or admin chat IDs not provided. Telegram bot disabled.")
            self.application = None
            return

        self.application = Application.builder().token(config.bot_token).build()
        self._setup_handlers()
        logger.info("TelegramBot initialized.")

    def _setup_handlers(self):
        if not self.application:
            return
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler("stop", CommandHandler("stop", self.stop_command))
        self.application.add_handler(CommandHandler("positions", self.positions_command))

    async def _is_admin(self, update: Update) -> bool:
        user_id = update.effective_user.id
        if user_id not in self.config.admin_chat_ids:
            await update.message.reply_text("You are not authorized to use this command.")
            logger.warning("Unauthorized access attempt from user", user_id=user_id)
            return False
        return True

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        await update.message.reply_text(
            "Welcome to the Trading Bot!\n" 
            "/status - Get bot status\n"
            "/positions - View open positions\n"
            "/stop - Halt all trading activity"
        )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        status = self.bot_state.get('status', 'UNKNOWN')
        equity = self.bot_state.get('equity', 0.0)
        await update.message.reply_text(f"Bot Status: {status}\nCurrent Equity: ${equity:,.2f}")

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        stop_bot_callback = self.bot_state.get('stop_bot_callback')
        if stop_bot_callback:
            asyncio.create_task(stop_bot_callback())
            await update.message.reply_text("Stop signal sent. The bot will shut down gracefully.")
        else:
            await update.message.reply_text("Could not send stop signal to the bot.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        positions = self.bot_state.get('open_positions', [])
        if not positions:
            await update.message.reply_text("No open positions.")
            return
        
        message = "Open Positions:\n"
        for pos in positions:
            message += f"- {pos.symbol} ({pos.side}): Qty {pos.quantity:.4f} @ ${pos.entry_price:,.2f}\n"
        await update.message.reply_text(message)

    async def run(self):
        if not self.application:
            return
        logger.info("Telegram bot is polling for commands...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

    async def stop(self):
        if self.application and self.application.updater.running:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram bot has been stopped.")

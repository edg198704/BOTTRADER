from typing import List, Optional, Dict
from bot_core.logger import get_logger
from bot_core.config import TelegramConfig

# Safe import for telegram library
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = get_logger(__name__)

class TelegramBot:
    def __init__(self, config: TelegramConfig, bot_instance):
        if not TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot is not installed. Telegram features are disabled.")
            self.enabled = False
            return

        self.config = config
        self.bot_instance = bot_instance
        self.application = None
        self.enabled = bool(self.config.bot_token and self.config.admin_chat_ids)

        if self.enabled:
            self.application = Application.builder().token(self.config.bot_token).build()
            self._register_handlers()
            logger.info("TelegramBot initialized.")
        else:
            logger.warning("TelegramBot disabled due to missing token or admin chat IDs.")

    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        self.application.add_handler(CommandHandler("resume", self.resume_command))
        self.application.add_handler(CommandHandler("positions", self.positions_command))

    async def _is_admin(self, update: Update) -> bool:
        is_admin = update.effective_chat.id in self.config.admin_chat_ids
        if not is_admin:
            await update.message.reply_text("You are not authorized to use this bot.")
        return is_admin

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        help_text = (
            "/status - Get the current status of the bot.\n"
            "/positions - View all open positions.\n"
            "/stop - Gracefully stop the bot (kill switch).\n"
            "/resume - Resume bot operations (if stopped)."
        )
        await update.message.reply_text(f"Welcome! Trading Bot is active.\n\n{help_text}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        status = "Running" if self.bot_instance.running else "Stopped"
        portfolio_value = self.bot_instance.position_manager.get_portfolio_value(self.bot_instance.latest_prices, self.bot_instance.config.initial_capital)
        await update.message.reply_text(f"Bot Status: {status}\nPortfolio Value: ${portfolio_value:,.2f}")

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        if not self.bot_instance.running:
            await update.message.reply_text("Bot is already stopped.")
            return
        self.bot_instance.running = False
        logger.critical("Bot stop command received from Telegram user.", user_id=update.effective_chat.id)
        await update.message.reply_text("Stop command received. The bot will shut down gracefully after the current cycle.")

    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        await update.message.reply_text("Resume functionality is not implemented. Please restart the bot application.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        open_positions = self.bot_instance.position_manager.get_all_open_positions()
        if not open_positions:
            await update.message.reply_text("No open positions.")
            return
        
        message = "Open Positions:\n"
        for symbol, pos in open_positions.items():
            message += f"- {symbol} ({pos.side}): {pos.quantity:.4f} @ ${pos.entry_price:,.2f}\n"
        await update.message.reply_text(message)

    async def run(self):
        if not self.enabled: return
        logger.info("Starting Telegram bot polling.")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

    async def stop(self):
        if not self.enabled or not self.application: return
        logger.info("Stopping Telegram bot polling.")
        if self.application.updater.running:
            await self.application.updater.stop()
        if self.application.running:
            await self.application.stop()
        await self.application.shutdown()

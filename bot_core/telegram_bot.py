import logging
from typing import List, Optional

from bot_core.config import TelegramConfig
from bot_core.bot import TradingBot

# Safe import for telegram library
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, config: TelegramConfig, trading_bot: TradingBot):
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot is not installed. Please install it to use the Telegram feature.")
        
        self.config = config
        self.trading_bot = trading_bot
        self.application: Optional[Application] = None

        if not self.config.bot_token or not self.config.admin_chat_ids:
            self.enabled = False
            logger.warning("Telegram bot token or admin chat IDs not configured. Telegram bot is disabled.")
        else:
            self.enabled = True
            self.application = Application.builder().token(self.config.bot_token).build()
            self._register_handlers()

    def _register_handlers(self):
        if not self.application:
            return
        handlers = [
            CommandHandler("start", self._start_command),
            CommandHandler("status", self._status_command),
            CommandHandler("stop", self._stop_command),
            CommandHandler("resume", self._resume_command),
            CommandHandler("positions", self._positions_command),
            CommandHandler("metrics", self._metrics_command),
        ]
        for handler in handlers:
            self.application.add_handler(handler)

    def _is_admin(self, update: Update) -> bool:
        return update.effective_chat.id in self.config.admin_chat_ids

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update):
            return
        help_text = (
            "🤖 Trading Bot Control Panel 🤖\n\n"
            "/status - Get current bot status and performance.\n"
            "/positions - View open positions.\n"
            "/metrics - Get detailed performance metrics.\n"
            "/stop - Halt all new trading activity.\n"
            "/resume - Resume trading activity."
        )
        await update.message.reply_text(help_text)

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update):
            return
        status = self.trading_bot.get_status()
        status_text = (
            f"**Bot Status**\n"
            f"- Running: {'✅' if status['is_running'] else '❌'}\n"
            f"- Halted: {'⚠️ YES' if status['is_halted'] else '✅ NO'}\n"
            f"- Symbol: `{self.trading_bot.symbol}`\n"
            f"- Uptime: {status['uptime']}\n\n"
            f"**Portfolio**\n"
            f"- Equity: `${status['equity']:.2f}`\n"
            f"- PnL (Unrealized): `${status['unrealized_pnl']:.2f}`\n"
            f"- Open Positions: `{status['open_positions_count']}`"
        )
        await update.message.reply_text(status_text, parse_mode='Markdown')

    async def _stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update):
            return
        self.trading_bot.halt_trading()
        logger.warning(f"Trading halted by admin {update.effective_chat.id}")
        await update.message.reply_text("🔴 Trading has been HALTED. No new positions will be opened.")

    async def _resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update):
            return
        self.trading_bot.resume_trading()
        logger.info(f"Trading resumed by admin {update.effective_chat.id}")
        await update.message.reply_text("🟢 Trading has been RESUMED.")

    async def _positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update):
            return
        summary = self.trading_bot.get_open_positions_summary()
        if not summary:
            await update.message.reply_text("No open positions.")
            return
        
        message = "**Open Positions**\n\n"
        for pos in summary:
            pnl_char = '🟢' if pos['pnl'] >= 0 else '🔴'
            message += (
                f"`{pos['symbol']}` ({pos['side']}) {pnl_char}\n"
                f"  Qty: `{pos['quantity']}`\n"
                f"  Entry: `${pos['entry_price']:.2f}`\n"
                f"  Current: `${pos['current_price']:.2f}`\n"
                f"  PnL: `${pos['pnl']:.2f}` (`{pos['pnl_pct']:.2f}%`)\n\n"
            )
        await update.message.reply_text(message, parse_mode='Markdown')

    async def _metrics_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update):
            return
        metrics = self.trading_bot.get_performance_metrics()
        message = (
            f"**Performance Metrics**\n"
            f"- Realized PnL (Daily): `${metrics['daily_realized_pnl']:.2f}`\n"
            f"- Total Unrealized PnL: `${metrics['total_unrealized_pnl']:.2f}`\n"
            f"- Portfolio Value: `${metrics['portfolio_value']:.2f}`\n"
            f"- Drawdown (Portfolio): `{metrics['portfolio_drawdown']:.2%}`\n"
        )
        await update.message.reply_text(message, parse_mode='Markdown')

    async def start(self):
        if not self.enabled or not self.application:
            return
        logger.info("Starting Telegram bot polling...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()

    async def stop(self):
        if not self.enabled or not self.application:
            return
        logger.info("Stopping Telegram bot...")
        if self.application.updater and self.application.updater.running:
            await self.application.updater.stop()
        if self.application.running:
            await self.application.stop()
        await self.application.shutdown()

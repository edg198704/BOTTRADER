import asyncio
from typing import Dict, Any, Callable, Awaitable
from datetime import datetime, timezone

from bot_core.logger import get_logger
from bot_core.config import TelegramConfig
from bot_core.strategy import AIEnsembleStrategy

# Safe import for telegram
try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import Application, CommandHandler, ContextTypes
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

        self.application = Application.builder().token(config.bot_token.get_secret_value()).build()
        self._setup_handlers()
        logger.info("TelegramBot initialized.")

    def _setup_handlers(self):
        if not self.application:
            return
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        self.application.add_handler(CommandHandler("positions", self.positions_command))
        self.application.add_handler(CommandHandler("ai", self.ai_command))

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
            "/status - Get detailed bot status and equity.\n"
            "/positions - View all open positions with live PnL.\n"
            "/ai <symbol> - View AI model details and top features.\n"
            "/stop - Send a shutdown signal to the bot."
        )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        
        status = self.bot_state.get('status', 'UNKNOWN')
        equity = self.bot_state.get('portfolio_equity', 0.0)
        daily_pnl = self.bot_state.get('daily_pnl', 0.0)
        pos_count = self.bot_state.get('open_positions_count', 0)
        start_time = self.bot_state.get('start_time')
        risk_manager = self.bot_state.get('risk_manager')
        
        uptime_str = "N/A"
        if start_time:
            uptime = datetime.now(timezone.utc) - start_time
            uptime_str = str(uptime).split('.')[0]

        risk_status = "OK"
        if risk_manager and risk_manager.is_halted:
            if risk_manager.circuit_breaker_halted:
                risk_status = "HALTED (Circuit Breaker)"
            elif risk_manager.daily_loss_halted:
                risk_status = "HALTED (Daily Loss Limit)"
            else:
                risk_status = "HALTED"

        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"

        message = (
            f"🤖 *Bot Status*\n\n"
            f"*Status*: `{status.upper()}`\n"
            f"*Uptime*: `{uptime_str}`\n"
            f"*Portfolio Equity*: `${equity:,.2f}`\n"
            f"*Today's Realized PnL*: {pnl_emoji} `${daily_pnl:,.2f}`\n"
            f"*Open Positions*: `{pos_count}`\n"
            f"*Risk Status*: `{risk_status}`"
        )
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

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
        
        position_manager = self.bot_state.get('position_manager')
        latest_prices = self.bot_state.get('latest_prices', {})

        if not position_manager:
            await update.message.reply_text("Position manager not available.")
            return

        # Await async DB call
        positions = await position_manager.get_all_open_positions()
        if not positions:
            await update.message.reply_text("No open positions.")
            return
        
        message = "📊 *Open Positions*\n\n"
        total_unrealized_pnl = 0.0

        for pos in positions:
            current_price = latest_prices.get(pos.symbol, pos.entry_price)
            pnl = (current_price - pos.entry_price) * pos.quantity
            if pos.side == 'SELL':
                pnl = -pnl
            total_unrealized_pnl += pnl
            
            pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 and pos.quantity > 0 else 0
            side_emoji = "🔼" if pos.side == 'BUY' else "🔽"
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"

            # Escape special characters for MarkdownV2
            symbol_md = pos.symbol.replace('-', '\-').replace('.', '\.')
            
            message += (
                f"{side_emoji} *{symbol_md}* \({pos.side}\)\n"
                f"Qty: `{pos.quantity}` | Entry: `{pos.entry_price}`\n"
                f"Current: `{current_price}`\n"
                f"PnL: {pnl_emoji} `${pnl:.2f}` \(`{pnl_pct:.2f}%`\)\n"
                f"-------------------\n"
            )

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    async def ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        
        if not context.args:
            await update.message.reply_text("Usage: /ai <symbol> (e.g., /ai BTC/USDT)")
            return

        symbol = context.args[0]
        strategy = self.bot_state.get('strategy')
        
        if not strategy or not isinstance(strategy, AIEnsembleStrategy):
            await update.message.reply_text("AI Strategy is not active.")
            return

        # Access the ensemble learner directly
        learner = strategy.ensemble_learner
        entry = learner.symbol_models.get(symbol)
        
        if not entry:
            await update.message.reply_text(f"No trained models found for {symbol}.")
            return

        meta = entry.get('meta', {})
        metrics = meta.get('metrics', {})
        top_features = meta.get('top_features', {})
        timestamp = meta.get('timestamp', 'N/A')

        # Format Metrics
        prec_buy = metrics.get('precision_buy', 0.0) * 100
        prec_sell = metrics.get('precision_sell', 0.0) * 100
        avg_prec = metrics.get('avg_action_precision', 0.0) * 100

        # Format Features
        features_str = ""
        for feat, imp in top_features.items():
            features_str += f"• `{feat}`: {imp:.4f}\n"

        message = (
            f"🧠 *AI Model Status: {symbol}*\n\n"
            f"*Last Retrained*: `{timestamp}`\n"
            f"*Avg Precision*: `{avg_prec:.1f}%`\n"
            f"*Buy Precision*: `{prec_buy:.1f}%`\n"
            f"*Sell Precision*: `{prec_sell:.1f}%`\n\n"
            f"*Top Contributing Features*:\n"
            f"{features_str}"
        )
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

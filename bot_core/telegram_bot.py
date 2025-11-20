import asyncio
from typing import Dict, Any, Callable, Awaitable, Optional
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
        self.config = config
        self.bot_state = bot_state
        self.application = None
        self._stop_event = asyncio.Event()
        
        if not TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot is not installed. Telegram integration disabled.")
            return

        if not config.bot_token or not config.admin_chat_ids:
            logger.warning("Telegram bot token or admin chat IDs not provided. Telegram bot disabled.")
            return

        try:
            self.application = Application.builder().token(config.bot_token.get_secret_value()).build()
            self._setup_handlers()
            logger.info("TelegramBot initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TelegramBot: {e}")
            self.application = None

    def _setup_handlers(self):
        if not self.application:
            return
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        self.application.add_handler(CommandHandler("positions", self.positions_command))
        self.application.add_handler(CommandHandler("ai", self.ai_command))

    async def run(self):
        """
        Runs the Telegram bot polling loop. 
        This method blocks until stop() is called, but yields control via asyncio.
        """
        if not self.application:
            return

        logger.info("Starting Telegram Bot polling loop...")
        self._stop_event.clear()

        try:
            # Initialize and start the application
            await self.application.initialize()
            await self.application.start()
            
            if self.application.updater:
                await self.application.updater.start_polling()
                logger.info("Telegram Bot is now polling for updates.")
            
            # Wait until the stop event is set
            await self._stop_event.wait()
            
            logger.info("Stopping Telegram Bot...")
            if self.application.updater:
                await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram Bot shutdown complete.")
            
        except Exception as e:
            logger.error(f"Telegram Bot runtime error: {e}", exc_info=True)

    async def stop(self):
        """Signals the run loop to terminate."""
        logger.info("Telegram Bot stop signal received.")
        self._stop_event.set()

    def create_alert_handler(self) -> Callable[[Dict[str, Any]], Awaitable[None]]:
        """Creates a callback to pipe system alerts to Telegram."""
        async def telegram_alert_handler(alert_data: Dict[str, Any]):
            if not self.application:
                return
            
            level = alert_data.get('level', 'INFO').upper()
            message = alert_data.get('message', '')
            details = alert_data.get('details', {})
            
            emoji_map = {
                'INFO': "ℹ️",
                'WARNING': "⚠️",
                'ERROR': "🛑",
                'CRITICAL': "🚨",
                'SUCCESS': "✅"
            }
            emoji = emoji_map.get(level, "ℹ️")
            
            # Escape Markdown V2 special characters
            def escape_md(text):
                special_chars = r"_*[]()~`>#+-=|{}.!"
                return "".join(f"\\{c}" if c in special_chars else c for c in str(text))

            formatted_msg = f"{emoji} *{escape_md(level)}*: {escape_md(message)}\n"
            
            if details:
                formatted_msg += "\n*Details:*\n"
                for k, v in details.items():
                    formatted_msg += f"• *{escape_md(k)}*: `{escape_md(v)}`\n"
            
            # Broadcast to all admins
            for chat_id in self.config.admin_chat_ids:
                try:
                    await self.application.bot.send_message(
                        chat_id=chat_id, 
                        text=formatted_msg, 
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
                except Exception as e:
                    logger.error(f"Failed to send Telegram alert to {chat_id}: {e}")
                    
        return telegram_alert_handler

    async def _is_admin(self, update: Update) -> bool:
        user_id = update.effective_user.id
        if user_id not in self.config.admin_chat_ids:
            await update.message.reply_text("⛔ You are not authorized to use this command.")
            logger.warning("Unauthorized access attempt", user_id=user_id)
            return False
        return True

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        await update.message.reply_text(
            "🤖 *Trading Bot Interface*\n\n" 
            "/status \- System health & equity\n"
            "/positions \- Open trades & PnL\n"
            "/ai \<symbol\> \- AI model insights\n"
            "/stop \- Emergency shutdown",
            parse_mode=ParseMode.MARKDOWN_V2
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

        # Escape for MarkdownV2
        def esc(t): return str(t).replace('.', '\.')

        message = (
            f"🤖 *Bot Status*\n\n"
            f"*Status*: `{status.upper()}`\n"
            f"*Uptime*: `{uptime_str}`\n"
            f"*Equity*: `${esc(f'{equity:,.2f}')}`\n"
            f"*Daily PnL*: {pnl_emoji} `${esc(f'{daily_pnl:,.2f}')}`\n"
            f"*Positions*: `{pos_count}`\n"
            f"*Risk*: `{risk_status}`"
        )
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        stop_bot_callback = self.bot_state.get('stop_bot_callback')
        if stop_bot_callback:
            await update.message.reply_text("🛑 Stop signal sent. Shutting down gracefully...")
            asyncio.create_task(stop_bot_callback())
        else:
            await update.message.reply_text("⚠️ Error: Stop callback not linked.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        
        position_manager = self.bot_state.get('position_manager')
        latest_prices = self.bot_state.get('latest_prices', {})

        if not position_manager:
            await update.message.reply_text("⚠️ Position manager unavailable.")
            return

        positions = await position_manager.get_all_open_positions()
        if not positions:
            await update.message.reply_text("🤷‍♂️ No open positions.")
            return
        
        message = "📊 *Open Positions*\n\n"
        
        def esc(t): 
            return str(t).replace('.', '\.').replace('-', '\-')

        for pos in positions:
            current_price = latest_prices.get(pos.symbol, pos.entry_price)
            pnl = (current_price - pos.entry_price) * pos.quantity
            if pos.side == 'SELL':
                pnl = -pnl
            
            pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 else 0
            side_emoji = "🔼" if pos.side == 'BUY' else "🔽"
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"

            message += (
                f"{side_emoji} *{esc(pos.symbol)}* \({pos.side}\)\n"
                f"Qty: `{esc(pos.quantity)}` @ `{esc(pos.entry_price)}`\n"
                f"PnL: {pnl_emoji} `${esc(f'{pnl:.2f}')}` \(`{esc(f'{pnl_pct:.2f}')}%`\)\n"
                f"-------------------\n"
            )

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    async def ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        
        if not context.args:
            await update.message.reply_text("Usage: `/ai <symbol>` (e.g., `/ai BTC/USDT`)", parse_mode=ParseMode.MARKDOWN_V2)
            return

        symbol = context.args[0]
        strategy = self.bot_state.get('strategy')
        
        if not strategy or not isinstance(strategy, AIEnsembleStrategy):
            await update.message.reply_text("⚠️ AI Strategy is not active.")
            return

        learner = strategy.ensemble_learner
        entry = learner.symbol_models.get(symbol)
        
        if not entry:
            await update.message.reply_text(f

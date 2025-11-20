import asyncio
from typing import Dict, Any, Callable, Awaitable, Optional
from datetime import datetime, timezone

from bot_core.logger import get_logger
from bot_core.config import TelegramConfig
from bot_core.strategy import AIEnsembleStrategy
from bot_core.utils import escape_markdown_v2

# Safe import for telegram
try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import Application, CommandHandler, ContextTypes, ApplicationBuilder
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = get_logger(__name__)

class TelegramBot:
    def __init__(self, config: TelegramConfig, bot_state: Dict[str, Any]):
        self.config = config
        self.bot_state = bot_state
        self.application: Optional['Application'] = None
        self._running = False
        
        if not TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot not installed. Telegram integration disabled.")
            return

        if not config.bot_token or not config.admin_chat_ids:
            logger.warning("Telegram credentials missing. Telegram integration disabled.")
            return

        try:
            self.application = ApplicationBuilder().token(config.bot_token.get_secret_value()).build()
            self._setup_handlers()
            logger.info("TelegramBot initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize TelegramBot: {e}")
            self.application = None

    def _setup_handlers(self):
        if not self.application: return
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        self.application.add_handler(CommandHandler("positions", self.positions_command))
        self.application.add_handler(CommandHandler("ai", self.ai_command))
        self.application.add_error_handler(self.error_handler)

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error(f"Telegram error: {context.error}", exc_info=context.error)

    def create_alert_handler(self) -> Callable[[Dict[str, Any]], Awaitable[None]]:
        """Returns an async function to handle alerts from the AlertSystem."""
        async def telegram_alert_handler(alert_data: Dict[str, Any]):
            if not self.application or not self._running:
                return
            
            level = alert_data.get('level', 'INFO').upper()
            message = alert_data.get('message', '')
            details = alert_data.get('details', {})
            
            emoji = "ℹ️"
            if level == 'WARNING': emoji = "⚠️"
            elif level == 'ERROR': emoji = "🛑"
            elif level == 'CRITICAL': emoji = "🚨"
            elif level == 'SUCCESS': emoji = "✅"
            
            # Escape content for MarkdownV2
            safe_level = escape_markdown_v2(level)
            safe_message = escape_markdown_v2(message)
            
            formatted_msg = f"{emoji} *{safe_level}*: {safe_message}\n"
            if details:
                formatted_msg += "```\n"
                for k, v in details.items():
                    safe_k = escape_markdown_v2(str(k))
                    safe_v = escape_markdown_v2(str(v))
                    formatted_msg += f"{safe_k}: {safe_v}\n"
                formatted_msg += "```"
            
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

    async def run(self):
        """Starts the Telegram bot polling loop."""
        if not self.application:
            return

        self._running = True
        logger.info("Starting Telegram Bot polling...")
        
        try:
            await self.application.initialize()
            await self.application.start()
            
            # Start polling updates in the background
            await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            
            # Keep the task alive until stopped
            while self._running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Telegram Bot runtime error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stops the Telegram bot."""
        if not self.application or not self._running:
            return
            
        logger.info("Stopping Telegram Bot...")
        self._running = False
        try:
            if self.application.updater.running:
                await self.application.updater.stop()
            if self.application.running:
                await self.application.stop()
            await self.application.shutdown()
            logger.info("Telegram Bot stopped.")
        except Exception as e:
            logger.error(f"Error stopping Telegram Bot: {e}")

    async def _is_admin(self, update: Update) -> bool:
        if not update.effective_user:
            return False
        user_id = update.effective_user.id
        if user_id not in self.config.admin_chat_ids:
            await update.message.reply_text("⛔ You are not authorized to use this command.")
            logger.warning("Unauthorized access attempt", user_id=user_id)
            return False
        return True

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        await update.message.reply_text(
            "🤖 *Welcome to the Trading Bot!*\n" 
            "/status \- Get detailed bot status and equity\.\n"
            "/positions \- View all open positions with live PnL\.\n"
            "/ai \<symbol\> \- View AI model details and top features\.\n"
            "/stop \- Send a shutdown signal to the bot\.",
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

        # Escape all dynamic values
        safe_status = escape_markdown_v2(status.upper())
        safe_uptime = escape_markdown_v2(uptime_str)
        safe_equity = escape_markdown_v2(f"${equity:,.2f}")
        safe_pnl = escape_markdown_v2(f"${daily_pnl:,.2f}")
        safe_pos_count = escape_markdown_v2(str(pos_count))
        safe_risk = escape_markdown_v2(risk_status)

        message = (
            f"🤖 *Bot Status*\n\n"
            f"*Status*: `{safe_status}`\n"
            f"*Uptime*: `{safe_uptime}`\n"
            f"*Portfolio Equity*: `{safe_equity}`\n"
            f"*Today's Realized PnL*: {pnl_emoji} `{safe_pnl}`\n"
            f"*Open Positions*: `{safe_pos_count}`\n"
            f"*Risk Status*: `{safe_risk}`"
        )
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        stop_bot_callback = self.bot_state.get('stop_bot_callback')
        if stop_bot_callback:
            await update.message.reply_text("🛑 Stop signal sent\. The bot will shut down gracefully\.", parse_mode=ParseMode.MARKDOWN_V2)
            asyncio.create_task(stop_bot_callback())
        else:
            await update.message.reply_text("⚠️ Could not send stop signal to the bot\.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        
        position_manager = self.bot_state.get('position_manager')
        latest_prices = self.bot_state.get('latest_prices', {})

        if not position_manager:
            await update.message.reply_text("⚠️ Position manager not available\.")
            return

        positions = await position_manager.get_all_open_positions()
        if not positions:
            await update.message.reply_text("ℹ️ No open positions\.")
            return
        
        message = "📊 *Open Positions*\n\n"

        for pos in positions:
            current_price = latest_prices.get(pos.symbol, pos.entry_price)
            pnl = (current_price - pos.entry_price) * pos.quantity
            if pos.side == 'SELL':
                pnl = -pnl
            
            pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 and pos.quantity > 0 else 0
            side_emoji = "🔼" if pos.side == 'BUY' else "🔽"
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"

            safe_symbol = escape_markdown_v2(pos.symbol)
            safe_side = escape_markdown_v2(pos.side)
            safe_qty = escape_markdown_v2(str(pos.quantity))
            safe_entry = escape_markdown_v2(str(pos.entry_price))
            safe_curr = escape_markdown_v2(str(current_price))
            safe_pnl = escape_markdown_v2(f"${pnl:.2f}")
            safe_pct = escape_markdown_v2(f"{pnl_pct:.2f}%")

            message += (
                f"{side_emoji} *{safe_symbol}* \({safe_side}\)\n"
                f"Qty: `{safe_qty}` \| Entry: `{safe_entry}`\n"
                f"Current: `{safe_curr}`\n"
                f"PnL: {pnl_emoji} `{safe_pnl}` \(`{safe_pct}`\)\n"
                f"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\n"
            )

        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

    async def ai_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._is_admin(update): return
        
        if not context.args:
            await update.message.reply_text("Usage: /ai \<symbol\> \(e\.g\., /ai BTC/USDT\)")
            return

        symbol = context.args[0]
        strategy = self.bot_state.get('strategy')
        
        if not strategy or not isinstance(strategy, AIEnsembleStrategy):
            await update.message.reply_text("⚠️ AI Strategy is not active\.")
            return

        learner = strategy.ensemble_learner
        entry = learner.symbol_models.get(symbol)
        
        if not entry:
            safe_sym = escape_markdown_v2(symbol)
            await update.message.reply_text(f"ℹ️ No trained models found for {safe_sym}\.", parse_mode=ParseMode.MARKDOWN_V2)
            return

        meta = entry.get('meta', {})
        metrics = meta.get('metrics', {})
        top_features = meta.get('top_features', {})
        timestamp = meta.get('timestamp', 'N/A')

        prec_buy = metrics.get('precision_buy', 0.0) * 100
        prec_sell = metrics.get('precision_sell', 0.0) * 100
        avg_prec = metrics.get('avg_action_precision', 0.0) * 100

        features_str = ""
        for feat, imp in top_features.items():
            safe_feat = escape_markdown_v2(feat)
            safe_imp = escape_markdown_v2(f"{imp:.4f}")
            features_str += f"• `{safe_feat}`: {safe_imp}\n"

        safe_symbol = escape_markdown_v2(symbol)
        safe_ts = escape_markdown_v2(timestamp)
        safe_avg = escape_markdown_v2(f"{avg_prec:.1f}%")
        safe_buy = escape_markdown_v2(f"{prec_buy:.1f}%")
        safe_sell = escape_markdown_v2(f"{prec_sell:.1f}%")

        message = (
            f"🧠 *AI Model Status: {safe_symbol}*\n\n"
            f"*Last Retrained*: `{safe_ts}`\n"
            f"*Avg Precision*: `{safe_avg}`\n"
            f"*Buy Precision*: `{safe_buy}`\n"
            f"*Sell Precision*: `{safe_sell}`\n\n"
            f"*Top Contributing Features*:\n"
            f"{features_str}"
        )
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN_V2)

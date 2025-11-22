import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

from bot_core.logger import get_logger, set_correlation_id
from bot_core.exchange_api import ExchangeAPI
from bot_core.position_manager import PositionManager, Position
from bot_core.risk_manager import RiskManager
from bot_core.strategy import TradingStrategy, AIEnsembleStrategy
from bot_core.config import BotConfig
from bot_core.data_handler import DataHandler
from bot_core.monitoring import HealthChecker, InfluxDBMetrics, AlertSystem, Watchdog
from bot_core.position_monitor import PositionMonitor
from bot_core.trade_executor import TradeExecutor
from bot_core.optimizer import StrategyOptimizer
from bot_core.event_system import EventBus, MarketDataEvent, TradeCompletedEvent
from bot_core.services import ServiceManager
from bot_core.tick_pipeline import TickPipeline, SymbolProcessor
from bot_core.order_lifecycle_manager import OrderLifecycleService

logger = get_logger(__name__)

class TradingBot:
    def __init__(self, config: BotConfig, exchange_api: ExchangeAPI, data_handler: DataHandler, 
                 strategy: TradingStrategy, position_manager: PositionManager, risk_manager: RiskManager,
                 health_checker: HealthChecker, position_monitor: PositionMonitor,
                 trade_executor: TradeExecutor, alert_system: AlertSystem, 
                 shared_latest_prices: Dict[str, float], event_bus: EventBus,
                 order_lifecycle_service: OrderLifecycleService,
                 metrics_writer: Optional[InfluxDBMetrics] = None,
                 shared_bot_state: Optional[Dict[str, Any]] = None):
        self.config = config
        self.exchange_api = exchange_api
        self.data_handler = data_handler
        self.strategy = strategy
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.health_checker = health_checker
        self.position_monitor = position_monitor
        self.trade_executor = trade_executor
        self.alert_system = alert_system
        self.event_bus = event_bus
        self.order_lifecycle_service = order_lifecycle_service
        self.metrics_writer = metrics_writer
        self.shared_bot_state = shared_bot_state if shared_bot_state is not None else {}
        
        self.optimizer = StrategyOptimizer(config, position_manager)
        self.service_manager = ServiceManager()
        self.start_time = datetime.now(timezone.utc)
        self.latest_prices = shared_latest_prices
        self.market_details: Dict[str, Dict[str, Any]] = {}
        
        self.watchdog = Watchdog(config.strategy.symbols, alert_system, self.stop, health_checker)
        self.tick_pipeline = TickPipeline(data_handler, position_manager, strategy, trade_executor, self.watchdog, metrics_writer, shared_latest_prices)
        self.processors: Dict[str, SymbolProcessor] = {}
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        self._initialize_shared_state()
        logger.info("TradingBot orchestrator initialized.")

    def _initialize_shared_state(self):
        self.shared_bot_state.update({
            'status': 'initializing', 'start_time': self.start_time,
            'position_manager': self.position_manager, 'risk_manager': self.risk_manager,
            'latest_prices': self.latest_prices, 'config': self.config, 'strategy': self.strategy
        })

    async def setup(self):
        # 1. Pre-flight Exchange Check
        logger.info("Performing pre-flight exchange checks...")
        try:
            await self.exchange_api.get_balance()
            logger.info("Exchange connection verified.")
        except Exception as e:
            logger.critical("Failed to connect to exchange during setup.", error=str(e))
            raise

        # 2. Component Setup
        if isinstance(self.strategy, AIEnsembleStrategy): self.strategy.data_fetcher = self.data_handler
        self.position_monitor.set_strategy(self.strategy)
        await self._load_market_details()
        await self.position_manager.reconcile_positions(self.exchange_api, self.latest_prices)
        await self.strategy.warmup(self.config.strategy.symbols)
        
        # 3. Initialize Processors
        for symbol in self.config.strategy.symbols:
            self.processors[symbol] = SymbolProcessor(symbol, self.tick_pipeline)
        
        # 4. Event Subscriptions
        self.event_bus.subscribe(MarketDataEvent, self.on_market_data)
        self.event_bus.subscribe(TradeCompletedEvent, self.strategy.on_trade_complete)

    async def _load_market_details(self):
        for symbol in self.config.strategy.symbols:
            try:
                details = await self.exchange_api.fetch_market_details(symbol)
                if details: self.market_details[symbol] = details
            except Exception: pass
        self.trade_executor.market_details = self.market_details

    async def run(self):
        self.shared_bot_state['status'] = 'running'
        await self.setup()
        for proc in self.processors.values(): await proc.start()

        # Register all background services
        self.service_manager.register("DataHandler", self.data_handler.run(), critical=True)
        self.service_manager.register("PositionMonitor", self.position_monitor.run(), critical=True)
        self.service_manager.register("OrderLifecycle", self.order_lifecycle_service.start(), critical=True)
        self.service_manager.register("Watchdog", self.watchdog.run(), critical=True)
        self.service_manager.register("RiskMetrics", self.risk_manager.run_metrics_loop(), critical=False)
        self.service_manager.register("PortfolioMonitor", self._portfolio_monitoring_service(), critical=False)
        self.service_manager.register("ModelMonitor", self._model_monitor_service(), critical=False)
        self.service_manager.register("Optimizer", self.optimizer.run(), critical=False)

        self.service_manager.start_all()
        await self.service_manager.monitor()

    async def on_market_data(self, event: MarketDataEvent):
        set_correlation_id()
        if event.symbol in self.processors: self.processors[event.symbol].on_tick()

    async def _portfolio_monitoring_service(self):
        """Service to monitor portfolio health and reconcile state periodically."""
        reconcile_counter = 0
        while True:
            try:
                reconcile_counter += 1
                if reconcile_counter >= 2:
                    await self.position_manager.reconcile_positions(self.exchange_api, self.latest_prices)
                    reconcile_counter = 0
                
                open_pos = await self.position_manager.get_all_open_positions()
                val = self.position_manager.get_portfolio_value(self.latest_prices, open_pos)
                pnl = await self.position_manager.get_daily_realized_pnl()
                await self.risk_manager.update_portfolio_risk(val, pnl)
                
                if self.risk_manager.liquidation_needed:
                    await self._liquidate_all_positions("Emergency Risk Halt")
                    self.risk_manager.liquidation_needed = False

                self.shared_bot_state.update({'portfolio_equity': val, 'open_positions_count': len(open_pos), 'daily_pnl': pnl})
                if self.metrics_writer:
                    await self.metrics_writer.write_metric('portfolio', fields={'equity': val, 'daily_pnl': pnl, 'open_positions': len(open_pos)})
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in PortfolioMonitor service", error=str(e))
            
            await asyncio.sleep(60)

    async def _liquidate_all_positions(self, reason: str):
        open_pos = await self.position_manager.get_all_open_positions()
        await asyncio.gather(*[self.trade_executor.close_position(pos, reason) for pos in open_pos])

    async def _model_monitor_service(self):
        """Service to check for updated model files and trigger hot-swaps."""
        if not isinstance(self.strategy, AIEnsembleStrategy):
            return
            
        logger.info("Model Monitor Service started.")
        interval = self.config.strategy.params.model_monitor_interval_seconds
        
        while True:
            try:
                for symbol in self.config.strategy.symbols:
                    if self.strategy.should_reload(symbol):
                        logger.info(f"New model detected for {symbol}. Initiating hot-swap...")
                        await self.strategy.reload_models(symbol)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in ModelMonitor service", error=str(e))
                await asyncio.sleep(interval)

    async def _close_position(self, position: Position, reason: str):
        await self.trade_executor.close_position(position, reason)

    async def stop(self):
        self.shared_bot_state['status'] = 'stopping'
        for proc in self.processors.values(): await proc.stop()
        await self.service_manager.stop_all()
        await self.order_lifecycle_service.stop()
        await self.watchdog.stop()
        await self.data_handler.stop()
        await self.position_monitor.stop()
        await self.risk_manager.stop()
        await self.optimizer.stop()
        await self.strategy.close()
        if self.exchange_api: await self.exchange_api.close()
        if self.position_manager: self.position_manager.close()
        if self.metrics_writer: await self.metrics_writer.close()
        self.process_executor.shutdown(wait=True)
        self.shared_bot_state['status'] = 'stopped'

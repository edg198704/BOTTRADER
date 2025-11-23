import asyncio
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

# FORCE FLUSH IMMEDIATELY
print("--- QUANTUM BOT TRAINING MODULE INITIALIZING ---", flush=True)

try:
    from config_loader import load_config
    from bot_core.logger import setup_logging, get_logger
    from bot_core.config import AIEnsembleStrategyParams
    from bot_core.exchange_api import CCXTExchangeAPI, MockExchangeAPI
    from bot_core.data_handler import DataHandler
    from bot_core.ai.ensemble_learner import train_ensemble_task
    from bot_core.ai.regime_detector import MarketRegimeDetector
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

logger = get_logger(__name__)

async def train_symbol(symbol: str, config, data_handler, regime_detector, executor):
    logger.info(f"Starting training for {symbol}...")
    print(f"[INFO] Fetching data for {symbol}...", flush=True)
    
    # 1. Fetch Data
    limit = config.strategy.params.training_data_limit
    df = await data_handler.fetch_full_history_for_symbol(symbol, limit)
    
    if df is None or df.empty:
        logger.error(f"No data found for {symbol}. Skipping.")
        print(f"[ERROR] No data for {symbol}", flush=True)
        return

    # 2. Detect Regime (Enrich Data)
    regime_res = await regime_detector.detect_regime(symbol, df)
    df_enriched = regime_res.get('enriched_df', df)

    # 3. Fetch Leader Data (if applicable)
    leader_df = None
    if config.strategy.params.market_leader_symbol:
        leader_df = await data_handler.fetch_full_history_for_symbol(
            config.strategy.params.market_leader_symbol, limit
        )

    # 4. Run Training Task
    loop = asyncio.get_running_loop()
    print(f"[INFO] Offloading training task for {symbol} to worker process...", flush=True)
    try:
        success = await loop.run_in_executor(
            executor, 
            train_ensemble_task, 
            symbol, 
            df_enriched, 
            config.strategy.params, 
            leader_df
        )

        if success:
            logger.info(f"Successfully trained model for {symbol}.")
            print(f"[SUCCESS] Model trained for {symbol}", flush=True)
        else:
            logger.error(f"Training failed for {symbol}.")
            print(f"[FAILURE] Training failed for {symbol}", flush=True)
    except Exception as e:
        logger.error(f"Worker process crashed for {symbol}: {e}")
        traceback.print_exc()

async def main():
    print("--- STARTING MAIN TRAINING LOOP ---", flush=True)
    try:
        config = load_config('config_enterprise.yaml')
        # Force console logging to ensure visibility in Docker
        setup_logging("INFO", "logs/training.log", False)
        
        if not isinstance(config.strategy.params, AIEnsembleStrategyParams):
            logger.error("Current strategy is not AIEnsembleStrategy. Nothing to train.")
            return

        logger.info("Initializing Training Pipeline...")
        
        # Initialize Exchange & DataHandler (Historical Mode)
        if config.exchange.name == "MockExchange":
            exchange_api = MockExchangeAPI()
        else:
            exchange_api = CCXTExchangeAPI(config.exchange)

        # We use a shared dictionary for prices, though not strictly needed for training
        latest_prices = {}
        data_handler = DataHandler(exchange_api, config, latest_prices)
        
        # Initialize Regime Detector
        regime_detector = MarketRegimeDetector(config.strategy.params)

        # Initialize Executor
        # We use ProcessPoolExecutor to avoid GIL contention during heavy training
        max_workers = min(os.cpu_count(), len(config.strategy.symbols))
        print(f"[INFO] Spawning {max_workers} worker processes...", flush=True)
        executor = ProcessPoolExecutor(max_workers=max_workers)

        try:
            # Pre-load data
            print("[INFO] Initializing data handler...", flush=True)
            await data_handler.initialize_data()

            tasks = []
            for symbol in config.strategy.symbols:
                tasks.append(train_symbol(symbol, config, data_handler, regime_detector, executor))
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.critical("Training pipeline crashed", error=str(e), exc_info=True)
            print(f"[CRITICAL] Pipeline crash: {e}", flush=True)
            traceback.print_exc()
        finally:
            executor.shutdown(wait=True)
            await exchange_api.close()
            await data_handler.stop()
            logger.info("Training Complete.")
            print("--- TRAINING COMPLETE ---", flush=True)

    except Exception as e:
        print(f"[FATAL] Uncaught exception in main: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

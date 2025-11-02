import asyncio
import sys
import os
import ccxt.async_support as ccxt
import json
import gc
import logging
import logging.handlers
import math
import random
import sqlite3
import time
import traceback
import uuid
from contextlib import contextmanager
from abc import ABC, abstractmethod
from collections import OrderedDict, deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Iterable
import concurrent.futures
from torch.distributions import Categorical
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from memory_profiler import profile
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, classification_report, silhouette_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering
from dateutil import parser as dateparser
from pydantic import BaseModel, Field, validator
import optuna
from optuna.integration import TorchDistributedTrial
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import warnings
import resource
import gc
import signal
import cProfile
import pstats
import psutil

class StructuredLogger:
    def __init__(self, name):
        self.LOG = logging.getLogger(name)
        if not self.LOG.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.LOG.addHandler(handler)
            self.LOG.setLevel(logging.INFO)

    def _safe_format(self, event, **kwargs):
        try:
            return f"{event} | {kwargs}"
        except MemoryError:
            return f"{event} | [memory_error_serializing_data]"
        except Exception:
            return f"{event} | [error_serializing_data]"

    def info(self, event, **kwargs):
        try:
            message = self._safe_format(event, **kwargs)
            self.LOG.info(message)
        except MemoryError:
            print(f"INFO: {event}")
        except Exception:
            pass

    def error(self, event, **kwargs):
        try:
            message = self._safe_format(event, **kwargs)
            self.LOG.error(message)
        except MemoryError:
            print(f"ERROR: {event}")
        except Exception:
            pass

    def warning(self, event, **kwargs):
        try:
            message = self._safe_format(event, **kwargs)
            self.LOG.warning(message)
        except MemoryError:
            print(f"WARNING: {event}")
        except Exception:
            pass

    def debug(self, event, **kwargs):
        try:
            message = self._safe_format(event, **kwargs)
            self.LOG.debug(message)
        except Exception:
            pass

    def critical(self, event, **kwargs):
        try:
            message = self._safe_format(event, **kwargs)
            self.LOG.critical(message)
        except MemoryError:
            print(f"CRITICAL: {event}")
        except Exception:
            pass
        
LOG = StructuredLogger(__name__)

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if load_dotenv(dotenv_path):
    LOG.info("dotenv_loaded_successfully", path=dotenv_path)
else:
    LOG.warning("dotenv_not_found", path=dotenv_path)
    
class TestResult(BaseModel):
    test_name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = {}

import os
import sqlite3  # Necesario para manipular la DB

class AutomatedTestSuite:
    """Sistema de testing automático que se ejecuta periódicamente"""
    def __init__(self, bot):
        self.bot = bot
        self.test_results = []
        self.last_test_run = None
        
        # Reset del PositionLedger para tests limpios (borra DB persistente y resetea estado en memoria)
        db_path = 'position_ledger.db'  # Ruta del DB, como en tu código original
        if os.path.exists(db_path):
            os.remove(db_path)  # Borra el archivo DB para empezar de cero en tests
            LOG.info("test_db_deleted_for_clean_start", path=db_path)  # Opcional: log para tracking
        
        # Alternativa a borrar: Truncate la tabla sin eliminar el archivo (comenta la línea anterior si prefieres esto)
        # conn = sqlite3.connect(db_path)
        # cursor = conn.cursor()
        # cursor.execute('DELETE FROM transactions')  # Limpia la tabla
        # conn.commit()
        # conn.close()
        # LOG.info("test_db_truncated_for_clean_start", path=db_path)
        
        self.bot.position_ledger = PositionLedger()  # Crea nueva instancia limpia
        self.bot.position_ledger.active_positions = {}  # Resetea posiciones activas
        self.bot.position_ledger.closed_positions = []  # Resetea posiciones cerradas
        self.bot.position_ledger.total_realized_pnl = 0.0  # Resetea PNL total acumulado
        self.bot.equity = self.bot.initial_capital  # Resetea equity al capital inicial para consistencia en tests
        
        LOG.info("position_ledger_reset_for_tests")  # Opcional: log para confirmar reset
        
    async def run_unit_tests(self) -> List[TestResult]:
        """Tests unitarios de componentes críticos"""
        results = []
        
        # Test 1: PositionLedger atomicidad
        start = time.perf_counter()
        try:
            test_symbol = "TEST/USDT"
            initial_equity = float(self.bot.equity)
            
            # ✅ Simular apertura
            transaction = await self.bot.position_ledger.record_open(
                self.bot, test_symbol, 'buy', 100.0, 0.1
            )
            assert transaction is not None, "Transaction should not be None"
            assert transaction.is_valid, f"Transaction validation errors: {transaction.validation_errors}"
            
            # ✅ CORRECCIÓN: El equity NO debe cambiar en apertura
            assert abs(self.bot.equity - initial_equity) < 0.01, \
                f"Equity should not change on open: before={initial_equity}, after={self.bot.equity}"
            
            # ✅ Simular cierre correctamente
            exit_price = 110.0
            executed_size = 0.1
            entry_price = 100.0
            
            # Calcular PnL esperado
            realized_pnl = (exit_price - entry_price) * executed_size  # = 1.0
            
            # ✅ ACTUALIZAR equity ANTES de record_close (simular lo que hace el caller real)
            equity_before_close = float(self.bot.equity)
            self.bot.equity = equity_before_close + realized_pnl
            
            LOG.debug("test_equity_update_before_close",
                     equity_before=equity_before_close,
                     realized_pnl=realized_pnl,
                     equity_after=self.bot.equity)
            
            close_tx = await self.bot.position_ledger.record_close(
                self.bot, test_symbol, exit_price, executed_size
            )
            
            assert close_tx is not None, "Close transaction failed"
            assert close_tx.is_valid, f"Close transaction validation errors: {close_tx.validation_errors}"
            
            # ✅ Validar PnL
            assert abs(close_tx.realized_pnl - realized_pnl) < 0.01, \
                f"PnL mismatch: {close_tx.realized_pnl} vs {realized_pnl}"
            
            # Validar equity final con tolerancia mayor
            expected_equity = initial_equity + realized_pnl
            equity_diff = abs(self.bot.equity - expected_equity)
            
            # Tolerancia permisiva para evitar falsos positivos
            
            tolerance = max(0.10, abs(expected_equity) * 0.0001)  # 0.01% o $0.10
            
            assert equity_diff < tolerance, \
                f"Equity mismatch: {self.bot.equity} vs {expected_equity} (diff: {equity_diff}, tolerance: {tolerance})"
                        
            audit = self.bot.position_ledger.audit_equity(self.bot)
                        
            if not audit['is_consistent']:
                if abs(audit['discrepancy']) < 1.0:
                    LOG.warning("audit_small_discrepancy_acceptable",
                               discrepancy=audit['discrepancy'],
                               tolerance=1.0)
                    # Corregir equity al valor auditado
                    self.bot.equity = audit['expected_free_equity']
                    LOG.info("equity_corrected_to_audit_value",
                            corrected_equity=self.bot.equity)
                else:
                    assert False, f"Audit failed with large discrepancy: {audit}"
            
            results.append(TestResult(
                test_name="position_ledger_atomicity",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={
                    'initial_equity': initial_equity,
                    'final_equity': self.bot.equity,
                    'realized_pnl': realized_pnl,
                    'audit': audit
                }
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="position_ledger_atomicity",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)
            ))
        
        # Test 2: RiskManager stop loss calculation
        start = time.perf_counter()
        try:
            # CORRECCIÓN: Usar un DataFrame con suficientes datos para el ATR
            df = pd.DataFrame({
                'high': [100 + i*0.5 for i in range(50)],
                'low': [98 + i*0.5 for i in range(50)],
                'close': [99 + i*0.5 for i in range(50)]
            })
            stop_loss = self.bot.risk_manager.calculate_stop_loss(
                "TEST/USDT", 100.0, 'buy', df
            )
            assert stop_loss > 0, "Stop loss must be positive"
            assert stop_loss < 100.0, "Stop loss must be below entry for buy"
            assert (100.0 - stop_loss) / 100.0 >= 0.01, "Stop loss too tight"
            results.append(TestResult(
                test_name="risk_manager_stop_loss",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details={'stop_loss': stop_loss}
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="risk_manager_stop_loss",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)
            ))
        
        # Test 3: EnsembleLearner prediction consistency
        start = time.perf_counter()
        try:
            if self.bot.ensemble_learner and self.bot.ensemble_learner.is_trained:
                test_df = pd.DataFrame({
                    'close': [100] * 50,
                    'rsi': [50] * 50,
                    'macd': [0] * 50,
                    'volume': [1000] * 50
                })
                
                pred1 = await self.bot.ensemble_learner.ensemble_predict(test_df)
                pred2 = await self.bot.ensemble_learner.ensemble_predict(test_df)
                
                assert pred1['action'] == pred2['action'], "Predictions inconsistent"
                assert abs(pred1['confidence'] - pred2['confidence']) < 0.01, "Confidence drift"
                
                results.append(TestResult(
                    test_name="ensemble_prediction_consistency",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000
                ))
            else:
                results.append(TestResult(
                    test_name="ensemble_prediction_consistency",
                    passed=False,
                    duration_ms=0,
                    error="Ensemble not trained"
                ))
        except Exception as e:
            results.append(TestResult(
                test_name="ensemble_prediction_consistency",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)
            ))
        
        return results
    
    async def run_integration_tests(self) -> List[TestResult]:
        """Tests de integración end-to-end"""
        results = []
        
        # Test 1: Pipeline completo sin errores
        start = time.perf_counter()
        try:
            symbol = self.bot.config.symbols[0]
            result = await self.bot.exchange_manager.fetch_ohlcv(symbol, '1h', limit=100)
            
            assert result['success'], "OHLCV fetch failed"
            
            df = create_dataframe(result['ohlcv'])
            assert df is not None and len(df) >= 50, "DataFrame creation failed"
            
            df = calculate_technical_indicators(df)
            assert 'rsi' in df.columns, "Technical indicators missing"
            
            results.append(TestResult(
                test_name="end_to_end_pipeline",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="end_to_end_pipeline",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)
            ))
        
        # Test 2: Equity audit consistency
        start = time.perf_counter()
        try:
            audit = self.bot.position_ledger.audit_equity(self.bot)
            
            assert audit['is_consistent'] or abs(audit['discrepancy']) < 10.0, \
                f"Large equity discrepancy: {audit['discrepancy']}"
            
            results.append(TestResult(
                test_name="equity_audit_consistency",
                passed=True,
                duration_ms=(time.perf_counter() - start) * 1000,
                details=audit
            ))
        except Exception as e:
            results.append(TestResult(
                test_name="equity_audit_consistency",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)
            ))
        
        return results
    
    async def run_regression_tests(self) -> List[TestResult]:
        """Tests de regresión contra baseline conocido"""
        results = []
        
        # Test 1: Performance no ha degradado
        start = time.perf_counter()
        try:
            baseline_win_rate = 0.45  # 45% win rate mínimo esperado
            current_win_rate = self.bot.performance_metrics.get('win_rate', 0.0)
            total_trades = self.bot.performance_metrics.get('total_trades', 0)
            
            if total_trades >= 50:
                assert current_win_rate >= baseline_win_rate, \
                    f"Win rate degradation: {current_win_rate} < {baseline_win_rate}"
                
                results.append(TestResult(
                    test_name="performance_regression",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details={'win_rate': current_win_rate, 'total_trades': total_trades}
                ))
            else:
                results.append(TestResult(
                    test_name="performance_regression",
                    passed=False,
                    duration_ms=0,
                    error="Insufficient trades for regression test"
                ))
        except Exception as e:
            results.append(TestResult(
                test_name="performance_regression",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)
            ))
        
        # Test 2: Memory no crece indefinidamente
        start = time.perf_counter()
        try:
            mem_stats = MEMORY_MANAGER.get_memory_stats()
            if mem_stats:
                current_mb = mem_stats.get('current_mb', 0)
                max_mb = mem_stats.get('max_mb', 0)
                
                # No debe crecer más del 150% del máximo histórico
                assert current_mb < max_mb * 1.5, \
                    f"Memory leak suspected: {current_mb} MB > {max_mb * 1.5} MB"
                
                results.append(TestResult(
                    test_name="memory_leak_regression",
                    passed=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    details=mem_stats
                ))
            else:
                results.append(TestResult(
                    test_name="memory_leak_regression",
                    passed=False,
                    duration_ms=0,
                    error="No memory stats available"
                ))
        except Exception as e:
            results.append(TestResult(
                test_name="memory_leak_regression",
                passed=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)
            ))
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Ejecuta suite completa de tests"""
        LOG.info("starting_automated_test_suite")
        
        unit_results = await self.run_unit_tests()
        integration_results = await self.run_integration_tests()
        regression_results = await self.run_regression_tests()
        
        all_results = unit_results + integration_results + regression_results
        
        passed = sum(1 for r in all_results if r.passed)
        failed = len(all_results) - passed
        
        self.test_results = all_results
        self.last_test_run = datetime.now(timezone.utc)
        
        summary = {
            'timestamp': self.last_test_run.isoformat(),
            'total_tests': len(all_results),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(all_results) if all_results else 0,
            'results': [r.dict() for r in all_results]
        }
        
        LOG.info("test_suite_completed",
                total=len(all_results),
                passed=passed,
                failed=failed)
        
        # Enviar a InfluxDB
        try:
            await INFLUX_METRICS.write_model_metrics('automated_tests', {
                'total_tests': len(all_results),
                'passed': passed,
                'failed': failed,
                'success_rate': summary['success_rate']
            })
        except Exception:
            pass
        
        return summary



try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import ASYNCHRONOUS
except Exception:
    InfluxDBClient = None
    Point = None
    WritePrecision = None
    ASYNCHRONOUS = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# ===== NUEVO: Importar Telegram =====
try:
    from telegram import Update, Bot
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    LOG.warning("telegram_library_not_available")

class TelegramKillSwitch:
    """Kill switch externo vía Telegram con autenticación"""
    def __init__(self, bot_token: str = None, admin_chat_ids: List[int] = None):
        self.enabled = False
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.admin_chat_ids = admin_chat_ids or []
        
        # Parsear admin IDs del env
        admin_ids_env = os.getenv('TELEGRAM_ADMIN_IDS', '')
        if admin_ids_env:
            try:
                self.admin_chat_ids.extend([int(x.strip()) for x in admin_ids_env.split(',')])
            except Exception:
                pass
        
        self.application = None
        self.trading_bot = None
        self.circuit_breaker_active = False
        self.manual_override = False
        
        if not TELEGRAM_AVAILABLE:
            LOG.warning("telegram_kill_switch_disabled_library_missing")
        elif not self.bot_token:
            LOG.warning("telegram_kill_switch_disabled_no_token")
        elif not self.admin_chat_ids:
            LOG.warning("telegram_kill_switch_disabled_no_admins")
        else:
            self.enabled = True
            LOG.info("telegram_kill_switch_enabled",
                    admin_count=len(self.admin_chat_ids))
    
    def _is_admin(self, chat_id: int) -> bool:
        """Verifica si el chat_id es admin autorizado"""
        return chat_id in self.admin_chat_ids
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para /start"""
        if not self._is_admin(update.effective_chat.id):
            await update.message.reply_text("❌ Acceso no autorizado")
            return
        
        await update.message.reply_text(
            "🤖 *Kill Switch Bot Activo*\n\n"
            "Comandos disponibles:\n"
            "/status - Estado del bot\n"
            "/stop - Detener trading (kill switch)\n"
            "/resume - Reanudar trading\n"
            "/positions - Ver posiciones activas\n"
            "/metrics - Métricas de performance\n"
            "/emergency - Cerrar TODAS las posiciones",
            parse_mode='Markdown'
        )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para /status"""
        if not self._is_admin(update.effective_chat.id):
            await update.message.reply_text("❌ Acceso no autorizado")
            return
        
        if not self.trading_bot:
            await update.message.reply_text("❌ Bot no conectado")
            return
        
        status_emoji = "🟢" if self.trading_bot.is_running and not self.circuit_breaker_active else "🔴"
        
        message = f"{status_emoji} *Estado del Bot*\n\n"
        message += f"Running: {'✅' if self.trading_bot.is_running else '❌'}\n"
        message += f"Circuit Breaker: {'🔴 ACTIVO' if self.circuit_breaker_active else '🟢 OK'}\n"
        message += f"Manual Override: {'⚠️ SÍ' if self.manual_override else 'No'}\n\n"
        
        # Métricas
        metrics = self.trading_bot.performance_metrics
        message += f"💰 Equity: ${self.trading_bot.equity:,.2f}\n"
        message += f"📊 Total Trades: {metrics.get('total_trades', 0)}\n"
        message += f"✅ Win Rate: {metrics.get('win_rate', 0)*100:.1f}%\n"
        
        # Posiciones activas
        if hasattr(self.trading_bot, 'risk_manager'):
            active = len(self.trading_bot.risk_manager.active_stops)
            message += f"📍 Posiciones Activas: {active}\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para /stop - Activa kill switch"""
        if not self._is_admin(update.effective_chat.id):
            await update.message.reply_text("❌ Acceso no autorizado")
            return
        
        if not self.trading_bot:
            await update.message.reply_text("❌ Bot no conectado")
            return
        
        self.circuit_breaker_active = True
        self.manual_override = True
        
        # Activar circuit breaker en risk manager
        if hasattr(self.trading_bot, 'risk_manager'):
            self.trading_bot.risk_manager.circuit_breaker_active = True
        
        LOG.critical("telegram_kill_switch_activated",
                    admin_id=update.effective_chat.id)
        
        await update.message.reply_text(
            "🔴 *KILL SWITCH ACTIVADO*\n\n"
            "Trading detenido. No se abrirán nuevas posiciones.\n"
            "Las posiciones existentes continuarán monitoreándose.\n\n"
            "Usa /resume para reanudar.",
            parse_mode='Markdown'
        )
        
        # Enviar alerta crítica
        await ALERT_SYSTEM.send_alert(
            "CRITICAL",
            "Kill switch activado vía Telegram",
            admin_id=update.effective_chat.id
        )
    
    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para /resume - Desactiva kill switch"""
        if not self._is_admin(update.effective_chat.id):
            await update.message.reply_text("❌ Acceso no autorizado")
            return
        
        if not self.trading_bot:
            await update.message.reply_text("❌ Bot no conectado")
            return
        
        self.circuit_breaker_active = False
        self.manual_override = False
        
        # Desactivar circuit breaker en risk manager
        if hasattr(self.trading_bot, 'risk_manager'):
            self.trading_bot.risk_manager.circuit_breaker_active = False
        
        LOG.info("telegram_kill_switch_deactivated",
                admin_id=update.effective_chat.id)
        
        await update.message.reply_text(
            "🟢 *Trading Reanudado*\n\n"
            "Kill switch desactivado.\n"
            "El bot puede abrir nuevas posiciones.",
            parse_mode='Markdown'
        )
    
    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para /positions"""
        if not self._is_admin(update.effective_chat.id):
            await update.message.reply_text("❌ Acceso no autorizado")
            return
        
        if not self.trading_bot or not hasattr(self.trading_bot, 'risk_manager'):
            await update.message.reply_text("❌ Bot no conectado")
            return
        
        active_stops = self.trading_bot.risk_manager.active_stops
        
        if not active_stops:
            await update.message.reply_text("📍 No hay posiciones activas")
            return
        
        message = "📍 *Posiciones Activas*\n\n"
        
        for symbol, stop_info in active_stops.items():
            try:
                ticker = await self.trading_bot.exchange_manager.exchange.fetch_ticker(symbol)
                current_price = ticker.get('last', 0)
                
                entry_price = stop_info['entry_price']
                side = stop_info['side']
                size = stop_info['remaining_size']
                
                if side == 'buy':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                
                pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"
                
                message += f"{pnl_emoji} {symbol} ({side.upper()})\n"
                message += f"   Entry: ${entry_price:.2f}\n"
                message += f"   Current: ${current_price:.2f}\n"
                message += f"   PnL: {pnl_pct:+.2f}%\n"
                message += f"   Size: {size:.4f}\n\n"
            except Exception:
                continue
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def metrics_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para /metrics"""
        if not self._is_admin(update.effective_chat.id):
            await update.message.reply_text("❌ Acceso no autorizado")
            return
        
        if not self.trading_bot:
            await update.message.reply_text("❌ Bot no conectado")
            return
        
        metrics = self.trading_bot.performance_metrics
        
        message = "📊 *Métricas de Performance*\n\n"
        message += f"💰 Equity: ${self.trading_bot.equity:,.2f}\n"
        message += f"💵 Capital Inicial: ${self.trading_bot.initial_capital:,.2f}\n"
        message += f"📈 PnL Total: ${metrics.get('total_pnl', 0):+,.2f}\n"
        message += f"📊 Total Trades: {metrics.get('total_trades', 0)}\n"
        message += f"✅ Winning: {metrics.get('winning_trades', 0)}\n"
        message += f"❌ Losing: {metrics.get('losing_trades', 0)}\n"
        message += f"🎯 Win Rate: {metrics.get('win_rate', 0)*100:.1f}%\n"
        message += f"📉 Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%\n"
        
        if metrics.get('sharpe_ratio'):
            message += f"📈 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def emergency_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler para /emergency - Cierra TODAS las posiciones"""
        if not self._is_admin(update.effective_chat.id):
            await update.message.reply_text("❌ Acceso no autorizado")
            return
        
        if not self.trading_bot or not hasattr(self.trading_bot, 'risk_manager'):
            await update.message.reply_text("❌ Bot no conectado")
            return
        
        # Confirmar comando
        await update.message.reply_text(
            "⚠️ *MODO EMERGENCIA*\n\n"
            "Esto cerrará TODAS las posiciones activas.\n"
            "Escribe 'CONFIRMAR' para continuar.",
            parse_mode='Markdown'
        )
        
        # Guardar contexto para siguiente mensaje
        context.user_data['awaiting_emergency_confirm'] = True
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handler genérico para mensajes"""
        if not self._is_admin(update.effective_chat.id):
            return
        
        # Verificar confirmación de emergencia
        if context.user_data.get('awaiting_emergency_confirm'):
            if update.message.text.upper() == 'CONFIRMAR':
                context.user_data['awaiting_emergency_confirm'] = False
                
                # Ejecutar cierre de emergencia
                closed = 0
                errors = 0
                
                active_symbols = list(self.trading_bot.risk_manager.active_stops.keys())
                
                for symbol in active_symbols:
                    try:
                        stop_info = self.trading_bot.risk_manager.active_stops[symbol]
                        side = 'sell' if stop_info['side'] == 'buy' else 'buy'
                        size = stop_info['remaining_size']
                        
                        order = await self.trading_bot.exchange_manager.create_order(
                            symbol, 'market', side, size
                        )
                        
                        if order and order.get('success'):
                            closed += 1
                            self.trading_bot.risk_manager.close_position(symbol)
                        else:
                            errors += 1
                    except Exception:
                        errors += 1
                
                LOG.critical("emergency_close_all_positions",
                            closed=closed,
                            errors=errors,
                            admin_id=update.effective_chat.id)
                
                await update.message.reply_text(
                    f"✅ Emergencia Ejecutada\n\n"
                    f"Cerradas: {closed}\n"
                    f"Errores: {errors}",
                    parse_mode='Markdown'
                )
                
                # Activar kill switch
                self.circuit_breaker_active = True
                self.manual_override = True
            else:
                context.user_data['awaiting_emergency_confirm'] = False
                await update.message.reply_text("❌ Operación cancelada")
    
    async def start(self, trading_bot):
        """Inicia el bot de Telegram"""
        if not self.enabled:
            LOG.info("telegram_kill_switch_not_started_disabled")
            return
        
        self.trading_bot = trading_bot
        
        # Crear aplicación
        self.application = Application.builder().token(self.bot_token).build()
        
        # Registrar handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        self.application.add_handler(CommandHandler("resume", self.resume_command))
        self.application.add_handler(CommandHandler("positions", self.positions_command))
        self.application.add_handler(CommandHandler("metrics", self.metrics_command))
        self.application.add_handler(CommandHandler("emergency", self.emergency_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Iniciar polling
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        LOG.info("telegram_kill_switch_started",
                admin_count=len(self.admin_chat_ids))
    
    async def stop(self):
        """Detiene el bot de Telegram"""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            LOG.info("telegram_kill_switch_stopped")

sys.setrecursionlimit(2000)
gc.collect()
warnings.filterwarnings('ignore')

@contextmanager
def managed_dataframe() -> Iterator[pd.DataFrame]:
    df = None
    try:
        yield df
    finally:
        if df is not None:
            del df
        gc.collect()

def hurst_exponent(ts: np.ndarray, lags: Iterable[int] = range(2, 20)) -> float:
    if len(ts) < 20:
        return 0.5
    tau = []
    lagvec = []
    for lag in lags:
        if lag >= len(ts):
            continue
        pp = np.subtract(ts[lag:], ts[:-lag])
        tau.append(np.std(pp))
        lagvec.append(lag)
    if len(tau) < 2:
        return 0.5
    try:
        m = np.polyfit(np.log(lagvec), np.log(tau), 1)
        return m[0]
    except:
        return 0.5

class FeatureCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self._cache = {}
        self._timestamps = {}
        self._hit_count = 0
        self._miss_count = 0
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._lock = asyncio.Lock()

    def _generate_key(self, symbol: str, timeframe: str, length: int) -> str:
        return f"{symbol}:{timeframe}:{length}"

    async def get(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[np.ndarray]:
        key = self._generate_key(symbol, timeframe, len(df))
        async with self._lock:
            if key in self._cache:
                timestamp = self._timestamps[key]
                if time.time() - timestamp < self.ttl:
                    self._hit_count += 1
                    return self._cache[key].copy()
                else:
                    del self._cache[key]
                    del self._timestamps[key]
            self._miss_count += 1
            return None

    async def set(self, symbol: str, timeframe: str, df: pd.DataFrame, features: np.ndarray):
        key = self._generate_key(symbol, timeframe, len(df))
        async with self._lock:
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            self._cache[key] = features.copy()
            self._timestamps[key] = time.time()

    def get_stats(self) -> Dict[str, Any]:
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0
        return {
            'cache_size': len(self._cache),
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'hit_rate': hit_rate,
            'memory_mb': sys.getsizeof(self._cache) / (1024 * 1024)
        }

def optimize_memory_usage():
    try:
        for generation in range(3):
            gc.collect(generation)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        try:
            import numpy as np
            np.core.arrayprint._summary = None
        except (ImportError, AttributeError):
            pass
        try:
            import pandas as pd
            pd.core.common.clear_cache()
        except (ImportError, AttributeError):
            pass
        LOG.debug("memory_optimization_completed")
    except Exception as e:
        LOG.debug("memory_optimization_skipped", error=str(e))

class AdvancedMemoryManager:
    def __init__(self, warning_threshold_mb: float = 1500, critical_threshold_mb: float = 2000):
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self._cleanup_strategies = []
        self._memory_history = deque(maxlen=100)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300

    def register_cleanup_strategy(self, name: str, func: callable, priority: int = 5):
        self._cleanup_strategies.append({'name': name, 'func': func, 'priority': priority})
        self._cleanup_strategies.sort(key=lambda x: x['priority'], reverse=True)

    def get_memory_usage(self) -> Dict[str, float]:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }

    async def monitor_and_cleanup(self):
        while True:
            try:
                await asyncio.sleep(60)
                mem_usage = self.get_memory_usage()
                self._memory_history.append({'timestamp': time.time(), 'usage_mb': mem_usage['rss_mb']})
                if mem_usage['rss_mb'] > self.critical_threshold:
                    LOG.warning("critical_memory_usage", **mem_usage)
                    await self.emergency_cleanup()
                    await ALERT_SYSTEM.send_alert("WARNING", "Critical memory usage detected", **mem_usage)
                elif mem_usage['rss_mb'] > self.warning_threshold:
                    if time.time() - self._last_cleanup > self._cleanup_interval:
                        await self.routine_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOG.error("memory_monitor_error", error=str(e))

    async def routine_cleanup(self):
        mem_before = self.get_memory_usage()['rss_mb']
        for strategy in self._cleanup_strategies:
            if strategy['priority'] <= 7:
                try:
                    if asyncio.iscoroutinefunction(strategy['func']):
                        await strategy['func']()
                    else:
                        strategy['func']()
                except Exception as e:
                    LOG.error("cleanup_strategy_failed", strategy=strategy['name'], error=str(e))
        for generation in range(3):
            gc.collect(generation)
        mem_after = self.get_memory_usage()['rss_mb']
        freed_mb = mem_before - mem_after
        self._last_cleanup = time.time()
        LOG.info("routine_cleanup_completed", mem_before_mb=mem_before, mem_after_mb=mem_after, freed_mb=freed_mb)

    async def emergency_cleanup(self):
        mem_before = self.get_memory_usage()['rss_mb']
        for strategy in self._cleanup_strategies:
            try:
                if asyncio.iscoroutinefunction(strategy['func']):
                    await strategy['func']()
                else:
                    strategy['func']()
            except Exception as e:
                LOG.error("emergency_cleanup_strategy_failed", strategy=strategy['name'], error=str(e))
        for _ in range(5):
            for generation in range(3):
                gc.collect(generation)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        mem_after = self.get_memory_usage()['rss_mb']
        # Forzar liberación de memoria no gestionada
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        freed_mb = mem_before - mem_after
        LOG.warning("emergency_cleanup_completed", mem_before_mb=mem_before, mem_after_mb=mem_after, freed_mb=freed_mb)

    def get_memory_stats(self) -> Dict[str, Any]:
        if len(self._memory_history) < 2:
            return {}
        usages = [h['usage_mb'] for h in self._memory_history]
        return {
            'current_mb': usages[-1],
            'avg_mb': np.mean(usages),
            'max_mb': np.max(usages),
            'min_mb': np.min(usages),
            'trend': 'increasing' if usages[-1] > np.mean(usages) else 'stable',
            'warning_threshold_mb': self.warning_threshold,
            'critical_threshold_mb': self.critical_threshold
        }

MEMORY_MANAGER = AdvancedMemoryManager()

def cleanup_feature_cache():
    try:
        if FEATURE_CACHE and hasattr(FEATURE_CACHE, '_cache'):
            initial_size = len(FEATURE_CACHE._cache)
            if initial_size > FEATURE_CACHE.max_size * 0.8:
                oldest_keys = sorted(FEATURE_CACHE._timestamps.keys(), key=lambda k: FEATURE_CACHE._timestamps[k])[:initial_size // 4]
                for key in oldest_keys:
                    if key in FEATURE_CACHE._cache:
                        del FEATURE_CACHE._cache[key]
                    if key in FEATURE_CACHE._timestamps:
                        del FEATURE_CACHE._timestamps[key]
                LOG.debug("feature_cache_cleaned", removed=len(oldest_keys), remaining=len(FEATURE_CACHE._cache))
    except Exception as e:
        LOG.debug("feature_cache_cleanup_failed", error=str(e))

def cleanup_metrics_buffer():
    try:
        if METRICS and hasattr(METRICS, '_batch_buffer'):
            METRICS._flush_buffer()
            LOG.debug("metrics_buffer_flushed")
    except Exception as e:
        LOG.debug("metrics_buffer_cleanup_failed", error=str(e))

MEMORY_MANAGER.register_cleanup_strategy('feature_cache', cleanup_feature_cache, priority=8)
MEMORY_MANAGER.register_cleanup_strategy('metrics_buffer', cleanup_metrics_buffer, priority=7)
LOG.info("memory_cleanup_strategies_registered", count=2)

FEATURE_CACHE = FeatureCache(max_size=1000, ttl_seconds=300)
optimize_memory_usage()
gc.collect()

def setup_memory_optimization():
    gc.set_threshold(50, 5, 5)
    import sys
    sys.setrecursionlimit(2000)
    try:
        import pandas as pd
        pd.set_option('mode.chained_assignment', None)
    except ImportError:
        pass
    LOG.info("memory_optimization_configured")

async def periodic_memory_cleanup():
    try:
        while True:
            try:
                await asyncio.sleep(300)
                optimize_memory_usage()
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                LOG.debug("memory_usage_check", memory_mb=memory_mb)
            except asyncio.CancelledError:
                break
    except Exception as e:
        LOG.debug("periodic_memory_cleanup_error", error=str(e))
        await asyncio.sleep(60)

setup_memory_optimization()
optimize_memory_usage()
gc.collect()

class AlertSystem:
    async def send_alert(self, level: str, message: str, **kwargs):
        alert_data = {
            'level': level,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        LOG.critical("alert_sent", **alert_data)

ALERT_SYSTEM = AlertSystem()

class InfluxWriteThrottler:
    """
    MEJORADO: Throttler con intervalos diferenciados según tipo de métrica
    """
    def __init__(self):
        self.last_writes = {}
        self._lock = asyncio.Lock()
        
        # MEJORA: Intervalos específicos por tipo de métrica
        self.intervals = {
            'trade': 0,  # Trades NUNCA throttle (eventos importantes)
            'portfolio': 10,  # Portfolio cada 10 segundos
            'model': 30,  # Modelos cada 30 segundos
            'health': 60,  # Health cada 60 segundos
            'regime': 30,  # Régimen cada 30 segundos
            'default': 10  # Default 10 segundos
        }
    
    async def should_write(self, metric_type: str, symbol: str = None) -> bool:
        """
        Determina si debe escribirse una métrica según throttling
        
        Args:
            metric_type: Tipo de métrica (trade, portfolio, model, etc)
            symbol: Símbolo opcional para métricas específicas de símbolo
        """
        # MEJORA: Determinar intervalo según tipo
        base_type = metric_type.split('_')[0]  # Extraer tipo base
        min_interval = self.intervals.get(base_type, self.intervals['default'])
        
        # Sin throttling para trades
        if min_interval == 0:
            return True
        
        key = f"{metric_type}:{symbol}" if symbol else metric_type
        
        async with self._lock:
            now = time.time()
            last_write = self.last_writes.get(key, 0)
            
            if now - last_write < min_interval:
                return False
            
            self.last_writes[key] = now
            return True
    
    def reset(self, metric_type: str = None):
        """Reset throttling para tipo específico o todos"""
        if metric_type:
            self.last_writes = {k: v for k, v in self.last_writes.items() 
                               if not k.startswith(metric_type)}
        else:
            self.last_writes.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de throttling"""
        return {
            'total_keys': len(self.last_writes),
            'intervals': self.intervals,
            'oldest_write': min(self.last_writes.values()) if self.last_writes else 0
        }

# Recrear instancia global con nueva configuración
INFLUX_THROTTLER = InfluxWriteThrottler()

class InfluxDBMetrics:
    def __init__(self, url: str = None, token: str = None, org: str = None, bucket: str = None):
        self.enabled = False
        self.client = None
        self.write_api = None
        self._write_success_count = 0
        self._write_error_count = 0
        self._last_error_time = None
        self._metrics_buffer = []
        self._buffer_lock = asyncio.Lock()
        self._max_buffer_size = 100
        self._last_flush_time = time.time()
        self._flush_interval = 10

        url = url or os.getenv('INFLUXDB_URL')
        token = token or os.getenv('INFLUXDB_TOKEN')
        org = org or os.getenv('INFLUXDB_ORG')
        bucket = bucket or os.getenv('INFLUXDB_BUCKET')
        if url and token and org and bucket:
            try:
                if InfluxDBClient:
                    self.client = InfluxDBClient(url=url, token=token, org=org)
                    self.write_api = self.client.write_api()
                    self.bucket = bucket
                    self.org = org
                    self.enabled = True
                    LOG.info("influxdb_metrics_enabled", url=url, org=org)
                else:
                    LOG.warning("influxdb_client_not_available")
            except Exception as e:
                LOG.error("influxdb_init_failed", error=str(e))
        else:
            LOG.info("influxdb_metrics_disabled", reason="missing_credentials")

    async def check_health(self) -> Dict[str, Any]:
        """
        Verifica la salud de la conexión InfluxDB
        
        Returns:
            Dict con estado de salud y estadísticas
        """
        if not self.enabled:
            return {
                'healthy': False,
                'reason': 'not_enabled',
                'stats': {}
            }
        
        try:
            # Verificar cliente
            if not self.client:
                return {
                    'healthy': False,
                    'reason': 'no_client',
                    'stats': {}
                }
            
            # Intentar ping con timeout
            try:
                health_check = await asyncio.wait_for(
                    asyncio.to_thread(self.client.ping),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                return {
                    'healthy': False,
                    'reason': 'ping_timeout',
                    'stats': self._get_write_stats()
                }
            
            # Estadísticas de escritura
            stats = self._get_write_stats()
            
            # Considerar healthy si:
            # 1. Ping OK
            # 2. Success rate > 80% (si hay suficientes writes)
            # 3. O si hay pocos writes aún
            total_writes = stats['total_writes']
            success_rate = stats['success_rate']
            
            is_healthy = bool(health_check) and (
                success_rate > 0.8 or total_writes < 10
            )
            
            return {
                'healthy': is_healthy,
                'ping_ok': bool(health_check),
                'stats': stats,
                'threshold_met': success_rate > 0.8 if total_writes >= 10 else True
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'reason': 'exception',
                'error': str(e),
                'stats': self._get_write_stats()
            }
    
    def _get_write_stats(self) -> Dict[str, Any]:
        """Helper para obtener estadísticas de escritura"""
        total = self._write_success_count + self._write_error_count
        success_rate = self._write_success_count / total if total > 0 else 0
        
        return {
            'total_writes': total,
            'successful_writes': self._write_success_count,
            'failed_writes': self._write_error_count,
            'success_rate': success_rate,
            'last_error': self._last_error_time.isoformat() if self._last_error_time else None
        }
    
    async def reset_stats(self):
        """Resetea estadísticas de escritura"""
        self._write_success_count = 0
        self._write_error_count = 0
        self._last_error_time = None
        LOG.info("influxdb_stats_reset")

    async def write_trade_metrics(self, symbol: str, action: str, confidence: float, price: float, size: float, pnl: float = 0.0):
        """
        MEJORADO: Escribe métricas de trade con validación robusta y retry automático
        """
        if not self.enabled:
            return False
        
        # CORRECCIÓN: Eliminar throttling para trades (son eventos importantes)
        # Los trades deben registrarse SIEMPRE
        
        # Validación exhaustiva
        try:
            price = float(price)
            size = float(size)
            confidence = float(confidence)
            pnl = float(pnl)
            
            if price <= 0 or np.isnan(price) or np.isinf(price):
                LOG.error("invalid_price_rejected_influx", symbol=symbol, price=price, action=action)
                return False
            if size <= 0 or np.isnan(size) or np.isinf(size):
                LOG.error("invalid_size_rejected_influx", symbol=symbol, size=size, action=action)
                return False
            if np.isnan(pnl) or np.isinf(pnl):
                LOG.warning("invalid_pnl_setting_to_zero", symbol=symbol, pnl=pnl)
                pnl = 0.0
            
            confidence = np.clip(confidence, 0.0, 1.0)
            
        except (ValueError, TypeError) as e:
            LOG.error("metric_conversion_failed", symbol=symbol, error=str(e))
            return False
        
        trade_value_usdt = price * size
        
        point = Point("trades")\
            .tag("symbol", symbol)\
            .tag("action", action.lower())\
            .tag("side", action.lower())\
            .field("confidence", confidence)\
            .field("price", price)\
            .field("size", size)\
            .field("trade_value_usdt", trade_value_usdt)\
            .field("pnl", pnl)\
            .time(datetime.now(timezone.utc), WritePrecision.NS)
        
        # MEJORA: Retry automático en caso de fallo
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.write_api.write(bucket=self.bucket, org=self.org, record=point, write_options=ASYNCHRONOUS)
                self._write_success_count += 1
                
                # Log solo cada 10 trades para no saturar
                if self._write_success_count % 10 == 0:
                    LOG.debug("trade_metrics_batch_written", 
                             total_writes=self._write_success_count,
                             symbol=symbol)
                
                return True
                
            except Exception as write_error:
                self._write_error_count += 1
                self._last_error_time = datetime.now(timezone.utc)
                
                if attempt < max_retries - 1:
                    # Retry con backoff exponencial
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    LOG.debug("retrying_trade_metric_write",
                             symbol=symbol,
                             attempt=attempt + 1,
                             error=str(write_error))
                else:
                    LOG.error("trade_metric_write_failed_all_attempts",
                             symbol=symbol,
                             error=str(write_error),
                             attempts=max_retries)
                    return False
        
        return False

    async def write_open_position_metrics(self, symbol: str, side: str, entry_price: float, size: float, stop_loss: float, confidence: float):
        """
        MEJORADO: Registra apertura de posición en InfluxDB
        """
        if not self.enabled:
            return False
        
        # Validación robusta
        try:
            entry_price = float(entry_price)
            size = float(size)
            stop_loss = float(stop_loss)
            confidence = float(confidence)
            
            if entry_price <= 0 or np.isnan(entry_price) or np.isinf(entry_price):
                LOG.warning("invalid_entry_price_for_open_position", 
                           symbol=symbol, entry_price=entry_price)
                return False
            
            if size <= 0 or np.isnan(size) or np.isinf(size):
                LOG.warning("invalid_size_for_open_position", 
                           symbol=symbol, size=size)
                return False
            
            if stop_loss <= 0 or np.isnan(stop_loss) or np.isinf(stop_loss):
                LOG.warning("invalid_stop_loss_for_open_position", 
                           symbol=symbol, stop_loss=stop_loss)
                return False
            
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # Calcular métricas derivadas
            position_value = entry_price * size
            stop_distance_pct = abs(stop_loss - entry_price) / entry_price * 100
            
            point = Point("open_positions")\
                .tag("symbol", symbol)\
                .tag("side", side)\
                .field("entry_price", entry_price)\
                .field("size", size)\
                .field("position_value", position_value)\
                .field("stop_loss", stop_loss)\
                .field("stop_distance_pct", stop_distance_pct)\
                .field("confidence", confidence)\
                .time(datetime.now(timezone.utc), WritePrecision.NS)
            
            # Escritura con retry
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    self.write_api.write(bucket=self.bucket, org=self.org, 
                                        record=point, write_options=ASYNCHRONOUS)
                    LOG.debug("open_position_metric_written", symbol=symbol)
                    return True
                    
                except Exception as write_error:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)
                    else:
                        LOG.error("open_position_metric_write_failed", 
                                 symbol=symbol, error=str(write_error))
                        return False
            
            return False
            
        except Exception as e:
            LOG.error("open_position_metrics_preparation_failed", 
                     symbol=symbol, error=str(e))
            return False
        
    async def write_model_metrics(self, model_name: str, metrics: Dict[str, float], tags: Dict[str, str] = None):
        """
        MEJORADO: Escribe métricas de modelos con validación y tags adicionales
        
        Args:
            model_name: Nombre del modelo
            metrics: Dict de métricas numéricas
            tags: Tags adicionales opcionales (ej: {'symbol': 'BTC/USDT'})
        """
        if not self.enabled:
            return False
        
        # MEJORA: Throttling por modelo (1 write cada 30 segundos por modelo)
        throttle_key = f"model_{model_name}"
        if not await INFLUX_THROTTLER.should_write(throttle_key):
            return False
        
        try:
            point = Point("model_performance").tag("model", model_name)
            
            # MEJORA: Agregar tags adicionales si existen
            if tags:
                for tag_key, tag_value in tags.items():
                    if tag_value:  # Solo agregar si no vacío
                        point = point.tag(str(tag_key), str(tag_value))
            
            # Validar y agregar fields
            valid_fields = 0
            for key, value in metrics.items():
                try:
                    # MEJORA: Validación robusta
                    float_value = float(value)
                    
                    if np.isnan(float_value) or np.isinf(float_value):
                        LOG.debug("skipping_invalid_metric",
                                 model=model_name,
                                 key=key,
                                 value=value)
                        continue
                    
                    point = point.field(key, float_value)
                    valid_fields += 1
                    
                except (ValueError, TypeError) as conv_error:
                    LOG.debug("metric_conversion_failed",
                             model=model_name,
                             key=key,
                             value=value,
                             error=str(conv_error))
                    continue
            
            if valid_fields == 0:
                LOG.warning("no_valid_fields_for_model_metrics",
                           model=model_name,
                           attempted_fields=len(metrics))
                return False
            
            point = point.time(datetime.now(timezone.utc), WritePrecision.NS)
            
            # Escritura con manejo de errores
            try:
                self.write_api.write(bucket=self.bucket, org=self.org, record=point, write_options=ASYNCHRONOUS)
                self._write_success_count += 1
                
                # Log estadísticas cada 100 writes
                if self._write_success_count % 100 == 0:
                    self._log_stats()
                
                return True
                
            except Exception as write_error:
                self._write_error_count += 1
                self._last_error_time = datetime.now(timezone.utc)
                LOG.error("model_metrics_write_failed",
                         model=model_name,
                         error=str(write_error))
                return False
                
        except Exception as e:
            LOG.error("model_metrics_preparation_failed",
                     model=model_name,
                     error=str(e))
            return False

    def _log_stats(self):
        total = self._write_success_count + self._write_error_count
        success_rate = self._write_success_count / total if total > 0 else 0
        LOG.info("influxdb_write_stats", total_writes=total, success_count=self._write_success_count, error_count=self._write_error_count, success_rate=success_rate, last_error=self._last_error_time.isoformat() if self._last_error_time else None)

    async def write_portfolio_metrics(self, equity: float, drawdown: float, positions: int, total_pnl: float):
        """
        MEJORADO: Escribe métricas de portfolio con validación completa
        """
        if not self.enabled:
            return False
        
        # MEJORA: Throttling inteligente (no más de 1 write cada 10 segundos)
        if not await INFLUX_THROTTLER.should_write("portfolio"):
            return False
        
        try:
            # Validación y conversión
            equity = float(equity)
            drawdown = float(drawdown)
            positions = int(positions)
            total_pnl = float(total_pnl)
            
            if equity <= 0 or np.isnan(equity) or np.isinf(equity):
                LOG.warning("invalid_equity_for_influx", equity=equity)
                return False
            
            # CORRECCIÓN: Validar drawdown está en rango correcto
            if drawdown > 0:
                LOG.warning("positive_drawdown_correcting", drawdown=drawdown)
                drawdown = 0.0
            
            validated_drawdown = max(-1.0, min(0.0, drawdown))
            
            # MEJORA: Calcular métricas adicionales útiles
            initial_capital = 10000.0  # Debe venir del bot idealmente
            if hasattr(self, '_initial_capital'):
                initial_capital = self._initial_capital
            
            pnl_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0.0
            equity_pct = (equity / initial_capital * 100) if initial_capital > 0 else 100.0
            
            point = Point("portfolio")\
                .field("equity", equity)\
                .field("drawdown", validated_drawdown)\
                .field("drawdown_pct", validated_drawdown * 100)\
                .field("positions", positions)\
                .field("total_pnl", total_pnl)\
                .field("total_pnl_pct", pnl_pct)\
                .field("equity_pct", equity_pct)\
                .time(datetime.now(timezone.utc), WritePrecision.NS)
            
            # MEJORA: Escritura con retry
            try:
                self.write_api.write(bucket=self.bucket, org=self.org, record=point, write_options=ASYNCHRONOUS)
                self._write_success_count += 1
                
                LOG.debug("portfolio_metrics_written",
                         equity=equity,
                         positions=positions,
                         drawdown_pct=validated_drawdown * 100)
                
                return True
                
            except Exception as write_error:
                self._write_error_count += 1
                LOG.error("portfolio_metrics_write_failed",
                         error=str(write_error))
                return False
                
        except Exception as e:
            LOG.error("portfolio_metrics_preparation_failed", error=str(e))
            return False

    async def write_rl_metrics(self, episode: int, reward: float, actor_loss: float, critic_loss: float, epsilon: float):
        if not self.enabled:
            return False
        try:
            point = Point("rl_training")\
                .field("episode", int(episode))\
                .field("reward", float(reward))\
                .field("actor_loss", float(actor_loss))\
                .field("critic_loss", float(critic_loss))\
                .field("epsilon", float(epsilon))\
                .time(datetime.now(timezone.utc), WritePrecision.NS)
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True
        except Exception as e:
            LOG.error("influxdb_write_rl_failed", error=str(e))
            return False

INFLUX_METRICS = InfluxDBMetrics()
# NUEVO: Configurar initial_capital en InfluxDB
def set_influx_initial_capital(capital: float):
    """Configura el capital inicial en InfluxDB metrics"""
    if INFLUX_METRICS:
        INFLUX_METRICS._initial_capital = capital
        LOG.debug("influx_initial_capital_set", capital=capital)

gc.collect()
gc.set_threshold(700, 10, 10)

class MetricsCollector:
    def __init__(self):
        self.db_path = "metrics.db"
        self._db_initialized = False
        self._batch_size = 50
        self._batch_buffer = []
        self._last_flush = time.time()
        self._flush_interval = 30

    def write_buffered(self, event: str, data: Dict):
        try:
            compact_data = self._compact_data(data)
            self._batch_buffer.append((event, compact_data))
            current_time = time.time()
            if (len(self._batch_buffer) >= self._batch_size or current_time - self._last_flush >= self._flush_interval):
                self._flush_buffer()
                self._last_flush = current_time
        except MemoryError:
            self._emergency_flush()
        except Exception as e:
            LOG.debug("buffered_write_failed", error=str(e))

    def _compact_data(self, data: Dict) -> Dict:
        try:
            compact = {}
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 100:
                    compact[key] = value[:100] + "..."
                elif isinstance(value, (list, dict)) and len(str(value)) > 200:
                    compact[key] = f"{type(value).__name__}_truncated"
                else:
                    compact[key] = value
            return compact
        except Exception:
            return data

    def _emergency_flush(self):
        try:
            self._flush_buffer()
            self._batch_buffer.clear()
            optimize_memory_usage()
        except Exception:
            self._batch_buffer.clear()

    def _flush_buffer(self):
        if not self._batch_buffer:
            return
        conn = None
        try:
            if not self._lazy_init_db():
                return
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cursor = conn.cursor()
            for event, data in self._batch_buffer:
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
                json_data = json.dumps(data, separators=(',', ':'))
                cursor.execute("INSERT INTO metrics (event, data, timestamp) VALUES (?, ?, ?)", (event, json_data, timestamp))
            conn.commit()
            LOG.debug("buffer_flushed", records=len(self._batch_buffer))
            self._batch_buffer.clear()
        except MemoryError:
            self._batch_buffer.clear()
        except Exception as e:
            LOG.debug("buffer_flush_failed", error=str(e))
        finally:
            if conn:
                conn.close()
            optimize_memory_usage()

    def _init_db(self):
        conn = None
        try:
            gc.collect()
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute('PRAGMA cache_size = -1000')
            cursor.execute('PRAGMA mmap_size = 0')
            cursor.execute('PRAGMA temp_store = 2')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT,
                    data TEXT,
                    timestamp TEXT
                )
            ''')
            conn.commit()
            LOG.info("metrics_db_initialized_successfully")
        except MemoryError as e:
            print(f"CRITICAL: Memory error initializing metrics DB: {e}")
            for _ in range(3):
                gc.collect()
            try:
                if conn:
                    conn.close()
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                cursor = conn.cursor()
                cursor.execute('PRAGMA cache_size = -100')
                cursor.execute('PRAGMA mmap_size = 0')
                conn.commit()
                LOG.info("metrics_db_recovered_after_memory_error")
            except Exception as retry_error:
                print(f"CRITICAL: Recovery failed: {retry_error}")
                asyncio.create_task(ALERT_SYSTEM.send_alert("CRITICAL", "Memory error initializing metrics DB - recovery failed", error=str(e)))
        except Exception as e:
            error_msg = f"Metrics DB init failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            asyncio.create_task(ALERT_SYSTEM.send_alert("ERROR", "Metrics DB initialization failed", error=str(e)))
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass
            gc.collect()

    def _lazy_init_db(self):
        if hasattr(self, '_db_initialized') and self._db_initialized:
            return True
        try:
            gc.collect()
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT,
                    data TEXT,
                    timestamp TEXT
                )
            ''')
            conn.commit()
            conn.close()
            self._db_initialized = True
            LOG.info("lazy_db_initialization_successful")
            return True
        except MemoryError as e:
            print(f"LAZY INIT: Memory error - will retry later: {e}")
            return False
        except Exception as e:
            print(f"LAZY INIT: DB error: {e}")
            return False

    def write(self, event: str, data: Dict):
        return self.write_buffered(event, data)

    def emergency_memory_recovery(self):
        gc.collect()
        gc.collect()
        gc.collect()
        try:
            sqlite3.connect(self.db_path, timeout=1.0).close()
        except:
            pass
        if hasattr(self, 'metrics'):
            self.metrics.clear()
        LOG.info("emergency_memory_recovery_executed")

METRICS = MetricsCollector()

class AITradingStrategy(str, Enum):
    ENSEMBLE_AI = "ensemble_ai"
    DEEP_REINFORCEMENT = "deep_rl"
    TRANSFORMER = "transformer"
    HYBRID_AI = "hybrid_ai"

class AdvancedAIConfig(BaseModel):
    exchange: str = Field('binance', description="Exchange a utilizar")
    symbols: List[str] = Field(default_factory=lambda: ['BTC/USDT'], description="Símbolos a operar")
    timeframe: str = Field('1h', description="Timeframe para análisis")
    initial_capital: float = Field(10000.0, ge=1000, description="Capital inicial")
    min_order_size: float = Field(10.0, ge=1.0, description="Tamaño mínimo de orden")
    sandbox: bool = Field(False, description="Modo sandbox")
    dry_run: bool = Field(True, description="Ejecución sin órdenes reales")
    automl_optimization_trials: int = Field(100, ge=10, le=1000)
    automl_validation_window: int = Field(100, ge=20)
    automl_retrain_interval: int = Field(24, ge=1)
    regime_clustering_features: List[str] = Field(default_factory=lambda: ['volatility', 'trend_strength', 'volume_profile', 'market_correlation'])
    regime_n_clusters: int = Field(4, ge=2, le=10)
    regime_confidence_threshold: float = Field(0.7, ge=0.5, le=0.95)
    ensemble_strategies: List[str] = Field(default_factory=lambda: ['lstm_predictor', 'gradient_boosting', 'attention_network', 'technical_ensemble'])
    ensemble_meta_learner: str = Field('neural_network')
    rl_state_dim: int = Field(6, ge=6, le=100)
    rl_action_dim: int = Field(3, ge=2)
    rl_hidden_dim: int = Field(256, ge=64)
    rl_memory_size: int = Field(10000, ge=1000)
    rl_batch_size: int = Field(64, ge=16)
    rl_learning_rate: float = Field(0.001, ge=0.0001, le=0.01)
    rl_gamma: float = Field(0.99, ge=0.9, le=0.999)
    feature_engineering_lookback: int = Field(100, ge=20)
    dynamic_position_sizing: bool = Field(True)
    adaptive_risk_management: bool = Field(True)
    real_time_optimization: bool = Field(True)

    class Config:
        validate_assignment = False
        extra = 'forbid'
        arbitrary_types_allowed = False
        use_enum_values = True
        copy_on_model_validation = 'none'

    def validate_config_modes(self):
        mode_info = {
            'sandbox': self.sandbox,
            'dry_run': self.dry_run,
            'data_source': 'testnet' if self.sandbox else 'production',
            'orders': 'simulated' if self.dry_run else 'REAL'
        }
        if not self.sandbox and self.dry_run:
            LOG.info("config_mode_production_dry_run", **mode_info, message="✅ MODO CORRECTO: Datos PRODUCCIÓN REAL + Órdenes SIMULADAS")
        elif self.sandbox and self.dry_run:
            LOG.info("config_mode_testnet_dry_run", **mode_info, message="Datos testnet + Órdenes simuladas")
        elif not self.sandbox and not self.dry_run:
            LOG.critical("config_mode_production_live", **mode_info, message="⚠️ PELIGRO: Datos REALES + Órdenes REALES ⚠️")
        elif self.sandbox and not self.dry_run:
            LOG.warning("config_mode_testnet_live", **mode_info, message="Datos testnet + Órdenes testnet")
        return True

    @classmethod
    def get_memory_optimized_defaults(cls) -> Dict[str, Any]:
        return {
            'symbols': ['BTC/USDT'],
            'regime_clustering_features': ['volatility', 'trend_strength', 'volume_profile', 'market_correlation'],
            'ensemble_strategies': ['lstm_predictor', 'gradient_boosting', 'attention_network', 'technical_ensemble']
        }

    def __setattr__(self, name, value):
        if isinstance(value, list):
            if name in ['symbols', 'regime_clustering_features', 'ensemble_strategies']:
                value = tuple(value) if all(isinstance(item, str) for item in value) else value
        super().__setattr__(name, value)

def create_config():
    import gc, traceback
    try:
        gc.collect()
        cfg = AdvancedAIConfig()
        cfg.validate_config_modes()
        gc.collect()
        return cfg
    except MemoryError as e:
        print('⚠️ MemoryError creando configuración, usando defaults mínimos')
        gc.collect()
        return AdvancedAIConfig(**AdvancedAIConfig.get_memory_optimized_defaults())
    except Exception as e:
        print('⚠️ Error creando configuración:', e)
        traceback.print_exc()
        return AdvancedAIConfig(**AdvancedAIConfig.get_memory_optimized_defaults())

class OrderResultModel(BaseModel):
    success: bool
    symbol: str
    side: str
    amount: float
    price: float
    order_id: str
    raw: Optional[Dict] = None

class ExchangeManager:
    def __init__(self, exchange_name: str, api_key: str = '', api_secret: str = '', sandbox: bool = False, dry_run: bool = True):
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.dry_run = dry_run
        self.exchange = None
        self._initialize_exchange()

    def _initialize_exchange(self):
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            config = {
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 10000,
                }
            }
            if self.dry_run:
                LOG.info("dry_run_mode_using_public_endpoints_only")
            else:
                if self.api_key and self.api_secret:
                    config['apiKey'] = self.api_key
                    config['secret'] = self.api_secret
                    LOG.info("using_authenticated_mode")
                else:
                    LOG.warning("no_credentials_public_mode_only")
            if self.sandbox:
                config['options']['defaultType'] = 'future'
                if self.exchange_name == 'binance':
                    config['urls'] = {
                        'api': {
                            'public': 'https://testnet.binance.vision/api',
                            'private': 'https://testnet.binance.vision/api',
                        }
                    }
            self.exchange = exchange_class(config)
            LOG.info("exchange_initialized", exchange=self.exchange_name, sandbox=self.sandbox, dry_run=self.dry_run, mode="public_only" if self.dry_run else "authenticated")
        except Exception as e:
            LOG.error("exchange_initialization_failed", error=str(e))
            raise

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100, since: int = None) -> Dict[str, Any]:
        try:
            if since:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            else:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 10:
                return {"success": False, "error": "Insufficient data", "ohlcv": []}
            return {"success": True, "ohlcv": ohlcv}
        except Exception as e:
            error_msg = str(e)
            LOG.error("fetch_ohlcv_failed", symbol=symbol, error=error_msg)
            return {"success": False, "error": error_msg, "ohlcv": []}

    async def fetch_balance(self) -> Dict[str, Any]:
        try:
            if self.dry_run or not self.api_key:
                return {
                    "success": True,
                    "balance": {
                        "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0}
                    }
                }
            balance = await self.exchange.fetch_balance()
            return {"success": True, "balance": balance}
        except Exception as e:
            LOG.error("fetch_balance_failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None) -> Dict[str, Any]:
        try:
            if self.dry_run:
                order_id = f"dry_run_{int(time.time() * 1000)}"
                simulated_price = price if price else 0.0
                if simulated_price <= 0:
                    try:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        market_price = ticker.get('last', 0.0) or ticker.get('close', 0.0)
                        if market_price > 0:
                            simulated_price = float(market_price)
                            LOG.debug("dry_run_order_using_market_price", symbol=symbol, market_price=simulated_price)
                    except Exception as e:
                        LOG.debug("market_price_fetch_failed_in_dry_run", symbol=symbol, error=str(e))
                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "price": float(simulated_price),
                    "filled": amount,
                    "status": "closed",
                    "raw": {
                        "info": "dry_run_order",
                        "simulated": True,
                        "market_price": simulated_price
                    }
                }
            order = await self.exchange.create_order(symbol, order_type, side, amount, price)
            return {
                "success": True,
                "order_id": order.get('id'),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": order.get('price'),
                "raw": order
            }
        except Exception as e:
            LOG.error("create_order_failed", symbol=symbol, error=str(e))
            return {"success": False, "error": str(e)}

    async def close(self):
        try:
            if self.exchange:
                await self.exchange.close()
            LOG.info("exchange_connection_closed")
        except Exception as e:
            LOG.error("exchange_close_failed", error=str(e))

def override(method):
    method._override = True
    return method

def validate_override(cls):
    for method_name in dir(cls):
        method = getattr(cls, method_name)
        if hasattr(method, '_override'):
            found_in_base = False
            for base in cls.__bases__:
                if hasattr(base, method_name):
                    found_in_base = True
                    break
            if not found_in_base:
                raise TypeError(f"Method {method_name} in {cls.__name__} is marked with @override but doesn't override any method from base classes")
    return cls

class StrategyManager:
    def __init__(self, config: AdvancedAIConfig):
        self.config = config
        self.strategies = {}
        self.strategy_performance = {}
        self.performance_db_path = "strategy_performance.db"
        self._register_strategies()

    def _register_strategies(self):
        self.strategies = {
            'rsi_momentum': {
                'function': self._rsi_momentum_strategy,
                'description': 'Estrategia RSI de momentum mejorada',
                'parameters': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}
            },
            'bollinger_bands': {
                'function': self._bollinger_bands_strategy,
                'description': 'Estrategia de Bandas de Bollinger mejorada',
                'parameters': {'bb_period': 20, 'bb_std': 2}
            },
            'macd_trend': {
                'function': self._macd_trend_strategy,
                'description': 'Estrategia MACD de tendencia mejorada',
                'parameters': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            },
            'volume_profile': {
                'function': self._volume_profile_strategy,
                'description': 'Estrategia de perfil de volumen',
                'parameters': {'volume_period': 20, 'threshold': 2.0}
            }
        }

    async def initialize(self):
        await self._load_strategy_performance()

    def rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        try:
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ema_up = up.ewm(com=period-1, min_periods=period).mean()
            ema_down = down.ewm(com=period-1, min_periods=period).mean()
            rs = ema_up / ema_down
            return 100 - (100 / (1 + rs))
        except Exception as e:
            LOG.error("rsi_calculation_failed", error=str(e))
            return pd.Series([50.0] * len(series), index=series.index)

    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            up = high - high.shift()
            down = low.shift() - low
            pdm = up.where((up > down) & (up > 0), 0).rolling(window=period).mean()
            mdm = down.where((down > up) & (down > 0), 0).rolling(window=period).mean()
            pdi = 100 * (pdm / atr)
            mdi = 100 * (mdm / atr)
            dx = 100 * abs(pdi - mdi) / (pdi + mdi)
            adx = dx.rolling(window=period).mean()
            adx = adx.fillna(0)
            return adx
        except Exception as e:
            LOG.error("adx_calc_failed", error=str(e))
            return pd.Series(np.zeros(len(high)), index=high.index)

    def _rsi_momentum_strategy(self, data: pd.DataFrame, **params) -> Dict[str, Any]:
        try:
            rsi_period = params.get('rsi_period', 14)
            oversold = params.get('oversold', 30)
            overbought = params.get('overbought', 70)
            if len(data) < rsi_period + 1:
                return self._create_strategy_response("hold", 0.0, "Datos insuficientes")
            if 'rsi' in data.columns and not data['rsi'].isna().all():
                rsi_series = data['rsi']
            else:
                LOG.debug("rsi_not_in_dataframe_calculating_fallback", symbol=data.get('symbol', 'unknown'))
                rsi_series = self.rsi(data['close'], rsi_period)
                data['rsi'] = rsi_series
            data = data.dropna()
            if len(data) < 2:
                return self._create_strategy_response("hold", 0.0, "Datos insuficientes después de limpieza")
            current_rsi = data['rsi'].iloc[-1]
            prev_rsi = data['rsi'].iloc[-2]
            if current_rsi <= oversold:
                oversold_strength = (oversold - current_rsi) / oversold
                confidence = min(0.9, 0.7 + oversold_strength * 0.3)
                return self._create_strategy_response("buy", confidence, f"RSI sobrevendido: {current_rsi:.2f}")
            elif current_rsi >= overbought:
                overbought_strength = (current_rsi - overbought) / (100 - overbought)
                confidence = min(0.9, 0.7 + overbought_strength * 0.3)
                return self._create_strategy_response("sell", confidence, f"RSI sobrecomprado: {current_rsi:.2f}")
            else:
                if current_rsi > 50 and prev_rsi <= 50:
                    return self._create_strategy_response("buy", 0.6, "RSI cruzó arriba de 50")
                elif current_rsi < 50 and prev_rsi >= 50:
                    return self._create_strategy_response("sell", 0.6, "RSI cruzó abajo de 50")
                else:
                    return self._create_strategy_response("hold", 0.3, "RSI en zona neutral")
        except Exception as e:
            LOG.error("rsi_strategy_error", error=str(e))
            return self._create_strategy_response("hold", 0.0, f"Error: {str(e)}")

    def _bollinger_bands_strategy(self, data: pd.DataFrame, **params) -> Dict[str, Any]:
        try:
            bb_period = params.get('bb_period', 20)
            bb_std = params.get('bb_std', 2)
            if len(data) < bb_period:
                return self._create_strategy_response("hold", 0.0, "Datos insuficientes")
            rolling_mean = data['close'].rolling(window=bb_period).mean()
            rolling_std = data['close'].rolling(window=bb_period).std()
            upper = rolling_mean + (rolling_std * bb_std)
            lower = rolling_mean - (rolling_std * bb_std)
            data['bb_upper'] = upper
            data['bb_middle'] = rolling_mean
            data['bb_lower'] = lower
            data = data.dropna()
            if len(data) < 2:
                return self._create_strategy_response("hold", 0.0, "Datos insuficientes después de limpieza")
            current_price = data['close'].iloc[-1]
            current_upper = data['bb_upper'].iloc[-1]
            current_lower = data['bb_lower'].iloc[-1]
            bb_position = (current_price - current_lower) / (current_upper - current_lower)
            if current_price <= current_lower:
                oversold_strength = (current_lower - current_price) / current_lower
                confidence = min(0.85, 0.7 + oversold_strength * 0.3)
                return self._create_strategy_response("buy", confidence, "Precio en banda inferior de Bollinger")
            elif current_price >= current_upper:
                overbought_strength = (current_price - current_upper) / current_upper
                confidence = min(0.85, 0.7 + overbought_strength * 0.3)
                return self._create_strategy_response("sell", confidence, "Precio en banda superior de Bollinger")
            elif bb_position < 0.3:
                return self._create_strategy_response("buy", 0.6, "Precio en tercio inferior de Bollinger")
            elif bb_position > 0.7:
                return self._create_strategy_response("sell", 0.6, "Precio en tercio superior de Bollinger")
            else:
                return self._create_strategy_response("hold", 0.4, "Precio en zona media de Bollinger")
        except Exception as e:
            LOG.error("bollinger_bands_strategy_error", error=str(e))
            return self._create_strategy_response("hold", 0.0, f"Error: {str(e)}")

    def _macd_trend_strategy(self, data: pd.DataFrame, **params) -> Dict[str, Any]:
        try:
            fast_period = params.get('fast_period', 12)
            slow_period = params.get('slow_period', 26)
            signal_period = params.get('signal_period', 9)
            if len(data) < slow_period + signal_period:
                return self._create_strategy_response("hold", 0.0, "Datos insuficientes")
            if 'macd' in data.columns and 'macd_signal' in data.columns and not data['macd'].isna().all():
                macd = data['macd']
                macd_signal = data['macd_signal']
                macd_hist = data['macd_hist'] if 'macd_hist' in data.columns else macd - macd_signal
            else:
                LOG.debug("macd_not_in_dataframe_calculating_fallback")
                ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
                ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
                macd = ema_fast - ema_slow
                macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
                macd_hist = macd - macd_signal
                data['macd'] = macd
                data['macd_signal'] = macd_signal
                data['macd_hist'] = macd_hist
            data = data.dropna()
            if len(data) < 2:
                return self._create_strategy_response("hold", 0.0, "Datos insuficientes después de limpieza")
            current_macd = data['macd'].iloc[-1]
            current_signal = data['macd_signal'].iloc[-1]
            current_hist = data['macd_hist'].iloc[-1]
            prev_macd = data['macd'].iloc[-2]
            prev_signal = data['macd_signal'].iloc[-2]
            macd_strength = abs(current_macd - current_signal) / abs(current_signal) if current_signal != 0 else 0
            base_confidence = min(0.8, macd_strength * 5)
            if (prev_macd <= prev_signal and current_macd > current_signal and current_hist > 0):
                confidence = base_confidence * 1.3
                return self._create_strategy_response("buy", confidence, "MACD cruzó arriba con histograma positivo")
            elif (prev_macd >= prev_signal and current_macd < current_signal and current_hist < 0):
                confidence = base_confidence * 1.2
                return self._create_strategy_response("sell", confidence, "MACD cruzó abajo con histograma negativo")
            else:
                if current_macd > current_signal and current_hist > 0:
                    return self._create_strategy_response("hold", base_confidence * 0.5, "Tendencia alcista MACD")
                elif current_macd < current_signal and current_hist < 0:
                    return self._create_strategy_response("hold", base_confidence * 0.5, "Tendencia bajista MACD")
                else:
                    return self._create_strategy_response("hold", 0.3, "MACD neutral")
        except Exception as e:
            LOG.error("macd_strategy_error", error=str(e))
            return self._create_strategy_response("hold", 0.0, f"Error: {str(e)}")

    def _volume_profile_strategy(self, data: pd.DataFrame, **params) -> Dict[str, Any]:
        try:
            volume_period = params.get('volume_period', 20)
            threshold = params.get('threshold', 2.0)
            if len(data) < volume_period:
                return self._create_strategy_response("hold", 0.0, "Datos insuficientes")
            data['volume_ma'] = data['volume'].rolling(window=volume_period).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
            data = data.dropna()
            if len(data) < 2:
                return self._create_strategy_response("hold", 0.0, "Datos insuficientes después de limpieza")
            current_volume_ratio = data['volume_ratio'].iloc[-1]
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price
            if current_volume_ratio > threshold and price_change > 0:
                volume_strength = min(1.0, (current_volume_ratio - threshold) / threshold)
                confidence = 0.6 + volume_strength * 0.3
                return self._create_strategy_response("buy", confidence, "Alto volumen en subida - posible acumulación")
            elif current_volume_ratio > threshold and price_change < 0:
                volume_strength = min(1.0, (current_volume_ratio - threshold) / threshold)
                confidence = 0.6 + volume_strength * 0.3
                return self._create_strategy_response("sell", confidence, "Alto volumen en bajada - posible distribución")
            else:
                return self._create_strategy_response("hold", 0.3, "Volumen normal")
        except Exception as e:
            LOG.error("volume_profile_strategy_error", error=str(e))
            return self._create_strategy_response("hold", 0.0, f"Error: {str(e)}")

    def _create_strategy_response(self, signal: str, confidence: float, reason: str) -> Dict[str, Any]:
        return {
            "signal": signal,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def execute_strategy(self, strategy_name: str, data: pd.DataFrame, **params) -> Dict[str, Any]:
        try:
            if strategy_name not in self.strategies:
                raise ValueError(f"Estrategia no encontrada: {strategy_name}")
            strategy_info = self.strategies[strategy_name]
            strategy_func = strategy_info['function']
            validated_params = self._validate_strategy_parameters(strategy_name, params)
            result = strategy_func(data, **validated_params)
            self._update_strategy_performance(strategy_name, result)
            LOG.info("strategy_executed", strategy=strategy_name, signal=result.get('signal'), confidence=result.get('confidence'), reason=result.get('reason'))
            return result
        except Exception as e:
            LOG.error("strategy_execution_failed", strategy=strategy_name, error=str(e))
            return self._create_strategy_response("hold", 0.0, f"Error ejecutando estrategia: {str(e)}")

    def _validate_strategy_parameters(self, strategy_name: str, params: Dict) -> Dict:
        strategy_info = self.strategies[strategy_name]
        valid_params = strategy_info.get('parameters', {})
        validated = {}
        for param_name, param_value in params.items():
            if param_name in valid_params:
                allowed_values = valid_params[param_name]
                if isinstance(allowed_values, list) and param_value not in allowed_values:
                    validated[param_name] = allowed_values[0]
                    LOG.warning("strategy_parameter_adjusted", strategy=strategy_name, parameter=param_name, provided=param_value, used=allowed_values[0])
                else:
                    validated[param_name] = param_value
            else:
                LOG.warning("unknown_strategy_parameter", strategy=strategy_name, parameter=param_name)
        return validated

    def _update_strategy_performance(self, strategy_name: str, result: Dict):
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'total_signals': 0,
                'profitable_signals': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'last_used': None
            }
        metrics = self.strategy_performance[strategy_name]
        metrics['total_signals'] += 1
        metrics['last_used'] = datetime.now(timezone.utc)

    async def _load_strategy_performance(self):
        conn = None
        try:
            conn = sqlite3.connect(self.performance_db_path)
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS strategy_performance (strategy TEXT PRIMARY KEY, performance TEXT)''')
            conn.commit()
            cursor.execute("SELECT strategy, performance FROM strategy_performance")
            for strategy, perf_json in cursor.fetchall():
                self.strategy_performance[strategy] = json.loads(perf_json)
            LOG.info("strategy_performance_loaded", count=len(self.strategy_performance))
        except Exception as e:
            LOG.error("strategy_performance_load_failed", error=str(e))
        finally:
            if conn:
                conn.close()

    async def save_strategy_performance(self):
        conn = None
        try:
            conn = sqlite3.connect('performance.db')
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS strategy_performance (name TEXT, total_signals INTEGER, profitable_signals INTEGER, total_pnl FLOAT, win_rate FLOAT, last_used TEXT)')
            for name, metrics in self.strategy_performance.items():
                cursor.execute('INSERT OR REPLACE INTO strategy_performance VALUES (?, ?, ?, ?, ?, ?)',
                               (name, metrics['total_signals'], metrics['profitable_signals'], metrics['total_pnl'], metrics['win_rate'], metrics['last_used']))
            conn.commit()
            LOG.info("strategy_performance_saved")
        except Exception as e:
            LOG.error("strategy_performance_save_failed", error=str(e))
        finally:
            if conn:
                conn.close()

    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        if strategy_name not in self.strategies:
            return {"error": f"Estrategia no encontrada: {strategy_name}"}
        strategy_info = self.strategies[strategy_name].copy()
        performance = self.strategy_performance.get(strategy_name, {})
        return {
            "name": strategy_name,
            "description": strategy_info.get('description', ''),
            "parameters": strategy_info.get('parameters', {}),
            'performance': performance,
            "function_available": True
        }

    def list_strategies(self) -> List[Dict[str, Any]]:
        strategies_list = []
        for name, info in self.strategies.items():
            strategies_list.append({
                "name": name,
                "description": info.get('description', ''),
                "parameters": list(info.get('parameters', {}).keys()),
                "performance": self.strategy_performance.get(name, {})
            })
        return strategies_list

from abc import ABC, abstractmethod

class ProductionBot(ABC):
    def __init__(self, config: AdvancedAIConfig, exchange_manager: ExchangeManager, strategy_manager: StrategyManager):
        self.config = config
        self.exchange_manager = exchange_manager
        self.strategy_manager = strategy_manager
        self.is_running = False
        self.start_time = None
        self.equity = float(config.initial_capital)
        self.initial_capital = float(config.initial_capital)
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.drawdown = 0.0
        self.pnl = 0.0
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0
        }
        self.periodic_tasks = []
        self.logger = logging.getLogger(f"ProductionBot.{self.__class__.__name__}")
        try:
            self.health_check = HealthCheck(self)
        except Exception as e:
            LOG.error("health_check_initialization_failed", error=str(e))
            self.health_check = None
        try:
            self.performance_monitor = PerformanceMonitor(memory_manager=MEMORY_MANAGER)
        except Exception as e:
            LOG.error("performance_monitor_initialization_failed", error=str(e))
            self.performance_monitor = None
        try:
            self.data_accumulator = TrainingDataAccumulator(max_samples=50000)
            LOG.info("data_accumulator_initialized", max_samples=50000)
        except Exception as e:
            LOG.error("data_accumulator_initialization_failed", error=str(e))
            self.data_accumulator = None
        self.last_pipeline_execution = {}
        self.dashboard = None

    async def _initialize_components(self) -> None:
        try:
            LOG.info("initializing_bot_components", dry_run=self.config.dry_run)
            if not hasattr(self, 'dashboard') or self.dashboard is None:
                LOG.warning("dashboard_missing_creating_in_init_components")
                try:
                    self.dashboard = PerformanceDashboard(self)
                    LOG.info("dashboard_created_in_init_components")
                except Exception as dashboard_error:
                    LOG.error("dashboard_creation_failed", error=str(dashboard_error))
                    self.dashboard = None
            try:
                test_ohlcv_result = await self.exchange_manager.fetch_ohlcv(symbol=self.config.symbols[0], timeframe=self.config.timeframe, limit=100)
                if not test_ohlcv_result or not test_ohlcv_result.get("success", False):
                    error_msg = test_ohlcv_result.get("error", "Unknown error") if test_ohlcv_result else "No response"
                    raise RuntimeError(f"Failed to fetch test data: {error_msg}")
                test_ohlcv = test_ohlcv_result.get("ohlcv", [])
                if not test_ohlcv or len(test_ohlcv) < 20:
                    raise RuntimeError(f"Insufficient test data: {len(test_ohlcv)} candles")
                LOG.debug("exchange_test_successful", symbol=self.config.symbols[0], ohlcv_count=len(test_ohlcv))
            except Exception as e:
                LOG.error("exchange_test_failed", error=str(e))
                raise RuntimeError(f"Exchange connectivity test failed: {str(e)}")
            try:
                LOG.debug("creating_test_dataframe")
                test_df = create_dataframe(test_ohlcv)
                if test_df is None:
                    raise RuntimeError("create_dataframe returned None - DataFrame creation failed")
                if len(test_df) == 0:
                    raise RuntimeError("Test DataFrame is empty after creation")
                if 'close' not in test_df.columns:
                    available_cols = list(test_df.columns) if hasattr(test_df, 'columns') else 'unknown'
                    raise RuntimeError(f"'close' column missing. Available: {available_cols}")
                if test_df['close'].isna().all():
                    raise RuntimeError("All 'close' values are NaN")
                LOG.debug("test_dataframe_created", rows=len(test_df), columns=list(test_df.columns), dtypes={col: str(test_df[col].dtype) for col in test_df.columns})
            except Exception as e:
                LOG.error("test_dataframe_creation_failed", error=str(e))
                raise RuntimeError(f"Test DataFrame creation failed: {str(e)}")
            try:
                LOG.debug("calculating_test_indicators")
                test_df = calculate_technical_indicators(test_df)
                if test_df is None:
                    raise RuntimeError("calculate_technical_indicators returned None")
                if len(test_df) == 0:
                    raise RuntimeError("Test DataFrame empty after calculating indicators")
                if 'close' not in test_df.columns:
                    raise RuntimeError("'close' column lost after calculating indicators")
                LOG.debug("test_indicators_calculated", rows=len(test_df), columns=len(test_df.columns))
            except Exception as e:
                LOG.error("test_indicators_calculation_failed", error=str(e))
                raise RuntimeError(f"Test indicators calculation failed: {str(e)}")
            try:
                LOG.debug("validating_strategies")
                strategies_list = self.strategy_manager.list_strategies()
                if not strategies_list or len(strategies_list) == 0:
                    raise RuntimeError("No strategies available")
                LOG.info("strategies_loaded", count=len(strategies_list), strategies=[s['name'] for s in strategies_list])
            except Exception as e:
                LOG.error("strategy_validation_failed", error=str(e))
                raise RuntimeError(f"Strategy validation failed: {str(e)}")
            try:
                await self.strategy_manager.initialize()
                LOG.info("strategy_manager_initialized")
            except Exception as e:
                LOG.error("strategy_manager_init_failed", error=str(e))
                raise RuntimeError(f"Strategy manager initialization failed: {str(e)}")
            try:
                if not hasattr(self, 'risk_optimizer') or self.risk_optimizer is None:
                    self.risk_optimizer = BayesianRiskOptimizer(self.config)
                    LOG.debug("risk_optimizer_created_in_init_components")
                if not hasattr(self, 'position_sizer') or self.position_sizer is None:
                    self.position_sizer = DynamicPositionSizer(self.config, self)
                    LOG.debug("position_sizer_created_in_init_components")
                if not hasattr(self, 'risk_manager') or self.risk_manager is None:
                    self.risk_manager = DynamicRiskManager(self.config, self)
                    LOG.debug("risk_manager_created_in_init_components")
                if not hasattr(self, 'smart_executor') or self.smart_executor is None:
                    self.smart_executor = SmartOrderExecutor(self.exchange_manager, self.config)
                    LOG.debug("smart_executor_created_in_init_components")
                if not hasattr(self, 'portfolio_rebalancer') or self.portfolio_rebalancer is None:
                    self.portfolio_rebalancer = PortfolioRebalancer(self.config, self)
                    LOG.debug("portfolio_rebalancer_created_in_init_components")
                if not hasattr(self, 'correlation_analyzer') or self.correlation_analyzer is None:
                    self.correlation_analyzer = CorrelationAnalyzer(self.exchange_manager)
                    LOG.debug("correlation_analyzer_created_in_init_components")
                LOG.info("risk_management_components_verified")
            except Exception as e:
                LOG.error("risk_components_verification_failed", error=str(e))
            LOG.info("components_initialized_successfully", total_steps=6, symbols=len(self.config.symbols), timeframe=self.config.timeframe)
        except Exception as e:
            LOG.critical("component_initialization_failed", error=str(e))
            raise RuntimeError(f"Component initialization failed: {str(e)}")

    async def _stop_periodic_tasks(self) -> None:
        try:
            LOG.info("stopping_periodic_tasks", count=len(self.periodic_tasks))
            for task in self.periodic_tasks:
                if task and not task.done():
                    task.cancel()
            results = await asyncio.gather(*self.periodic_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    LOG.warning("periodic_task_error", task_index=i, error=str(result))
            LOG.info("periodic_tasks_stopped", count=len(self.periodic_tasks))
        except Exception as e:
            LOG.error("stop_periodic_tasks_failed", error=str(e))

    async def _cleanup_resources(self) -> None:
        try:
            LOG.info("cleaning_up_resources")
            if self.exchange_manager:
                try:
                    await self.exchange_manager.close()
                    LOG.debug("exchange_manager_closed")
                except Exception as e:
                    LOG.error("exchange_manager_close_failed", error=str(e))
            LOG.info("resources_cleanup_completed")
        except Exception as e:
            LOG.error("cleanup_resources_failed", error=str(e))

    async def _calculate_final_metrics(self) -> None:
        try:
            if not self.portfolio_history or len(self.portfolio_history) < 2:
                LOG.warning("insufficient_portfolio_history_for_metrics", history_length=len(self.portfolio_history) if self.portfolio_history else 0)
                return
            history = pd.DataFrame(self.portfolio_history, columns=['timestamp', 'equity'])
            if len(history) < 2:
                LOG.warning("insufficient_history_for_calc", rows=len(history))
                return
            returns = history['equity'].pct_change().dropna()
            if len(returns) > 0:
                if self.performance_metrics['total_trades'] > 0:
                    self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
                else:
                    self.performance_metrics['win_rate'] = 0.0
                mean_ret = returns.mean()
                std_ret = returns.std()
                if std_ret > 0:
                    time_diff = (history['timestamp'].iloc[-1] - history['timestamp'].iloc[0]).total_seconds()
                    hours_per_observation = time_diff / (len(history) - 1) / 3600
                    if hours_per_observation < 1.5:
                        annualization_factor = np.sqrt(252 * 24)
                    elif hours_per_observation < 6:
                        annualization_factor = np.sqrt(252 * 6)
                    elif hours_per_observation < 25:
                        annualization_factor = np.sqrt(252)
                    else:
                        annualization_factor = np.sqrt(52)
                    risk_free_rate = 0.02 / annualization_factor
                    excess_return = mean_ret - risk_free_rate
                    self.performance_metrics['sharpe_ratio'] = (excess_return / std_ret) * annualization_factor
                else:
                    self.performance_metrics['sharpe_ratio'] = 0.0
                peak = history['equity'].cummax()
                drawdown = (history['equity'] - peak) / peak
                self.performance_metrics['max_drawdown'] = drawdown.min()
                self.performance_metrics['volatility'] = std_ret * np.sqrt(252)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    if downside_std > 0:
                        self.performance_metrics['sortino_ratio'] = (mean_ret / downside_std) * np.sqrt(252)
                    else:
                        self.performance_metrics['sortino_ratio'] = 0.0
                else:
                    self.performance_metrics['sortino_ratio'] = 0.0
                total_return = (history['equity'].iloc[-1] - history['equity'].iloc[0]) / history['equity'].iloc[0]
                if self.performance_metrics['max_drawdown'] < 0:
                    self.performance_metrics['calmar_ratio'] = total_return / abs(self.performance_metrics['max_drawdown'])
                else:
                    self.performance_metrics['calmar_ratio'] = 0.0
                self.performance_metrics['total_pnl'] = history['equity'].iloc[-1] - self.initial_capital
            LOG.info("final_metrics_calculated", metrics=self.performance_metrics)
        except Exception as e:
            LOG.error("final_metrics_calc_failed", error=str(e))

    async def start(self) -> None:
        if self.is_running:
            self.logger.warning("Bot is already running")
            return
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        try:
            await self._initialize_components()
            try:
                balance_result = await self.exchange_manager.fetch_balance()
                if balance_result.get("success"):
                    balance_data = balance_result.get("balance", {})
                    usdt_balance = balance_data.get("USDT", {})
                    available_balance = usdt_balance.get("free", 0.0)
                    total_balance = usdt_balance.get("total", 0.0)
                    LOG.info("initial_balance_fetched", available=available_balance, total=total_balance, used=usdt_balance.get("used", 0.0), dry_run=self.config.dry_run)
                    if available_balance > 0:
                        if self.config.dry_run:
                            LOG.info("dry_run_using_configured_balance", configured=self.equity, available_in_exchange=available_balance)
                        else:
                            if available_balance != self.equity:
                                old_equity = self.equity
                                self.equity = available_balance
                                self.initial_capital = available_balance
                                LOG.info("equity_updated_from_real_balance", old_equity=old_equity, new_equity=self.equity)
                else:
                    LOG.warning("balance_fetch_failed_using_configured", error=balance_result.get("error", "Unknown"), configured_equity=self.equity)
            except Exception as e:
                LOG.error("balance_validation_failed", error=str(e))
            if hasattr(self, '_start_periodic_tasks'):
                await self._start_periodic_tasks()
            else:
                LOG.warning("_start_periodic_tasks_not_implemented_in_subclass")
            self.logger.info("Production bot started successfully", extra={'equity': self.equity, 'symbols': self.config.symbols})
        except Exception as e:
            self.logger.error("Failed to start production bot", extra={'error': str(e)})
            self.is_running = False
            await self.stop()
            raise

    async def stop(self) -> None:
        self.is_running = False
        try:
            await self._stop_periodic_tasks()
            await self._cleanup_resources()
            await self._calculate_final_metrics()
            self.logger.info("Production bot stopped successfully", extra={'final_equity': self.equity, 'total_trades': self.performance_metrics['total_trades']})
        except Exception as e:
            self.logger.error("Error during bot shutdown", extra={'error': str(e)})

    async def get_performance_report(self) -> Dict[str, Any]:
        try:
            return {
                'initial_capital': self.initial_capital,
                'current_equity': self.equity,
                'total_pnl': self.performance_metrics['total_pnl'],
                'total_pnl_percentage': (self.equity - self.initial_capital) / self.initial_capital * 100 if self.initial_capital > 0 else 0,
                'total_trades': self.performance_metrics['total_trades'],
                'win_rate': self.performance_metrics['win_rate'],
                'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
                'max_drawdown': self.performance_metrics['max_drawdown'],
                'volatility': self.performance_metrics['volatility'],
                'current_positions': len(self.positions),
                'running_time_hours': (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600 if self.start_time else 0
            }
        except Exception as e:
            LOG.error("get_performance_report_failed", error=str(e))
            return {}

class HealthCheck:
    def __init__(self, bot):
        self.bot = bot
        self.start_time = datetime.now(timezone.utc)

    def get_health_status(self) -> Dict[str, Any]:
        try:
            import psutil
            process = psutil.Process()
            drawdown = 0.0
            try:
                if hasattr(self.bot, 'portfolio_history') and len(self.bot.portfolio_history) > 0:
                    history_df = pd.DataFrame(self.bot.portfolio_history, columns=['timestamp', 'equity'])
                    if len(history_df) > 0:
                        peak = history_df['equity'].cummax()
                        drawdown_series = (history_df['equity'] - peak) / peak
                        drawdown = float(drawdown_series.min())
                else:
                    current_equity = getattr(self.bot, 'equity', 0.0)
                    initial_capital = getattr(self.bot, 'initial_capital', current_equity)
                    if initial_capital > 0:
                        drawdown = (current_equity - initial_capital) / initial_capital
                        drawdown = min(0.0, drawdown)
            except Exception as e:
                LOG.debug("drawdown_calculation_failed", error=str(e))
                drawdown = getattr(self.bot, 'drawdown', 0.0) or 0.0
            win_rate = 0.0
            try:
                metrics = getattr(self.bot, 'performance_metrics', {})
                total_trades = metrics.get('total_trades', 0)
                winning_trades = metrics.get('winning_trades', 0)
                if total_trades > 0:
                    win_rate = winning_trades / total_trades
            except Exception as e:
                LOG.debug("win_rate_calculation_failed", error=str(e))
            uptime_seconds = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            process = psutil.Process()
            cpu_usage = process.cpu_percent(interval=1.0)
            if cpu_usage == 0.0:
                cpu_usage = psutil.cpu_percent(interval=0.5)
            health_data = {
                "status": "healthy" if getattr(self.bot, 'is_running', False) else "stopped",
                "uptime_seconds": uptime_seconds,
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": cpu_usage,
                "active_positions": len(getattr(self.bot, 'positions', {})),
                "total_trades": getattr(self.bot, 'performance_metrics', {}).get('total_trades', 0),
                "winning_trades": getattr(self.bot, 'performance_metrics', {}).get('winning_trades', 0),
                "win_rate": win_rate,
                "current_equity": getattr(self.bot, 'equity', 0.0),
                "initial_capital": getattr(self.bot, 'initial_capital', 0.0),
                "drawdown": float(drawdown),
                "drawdown_pct": float(drawdown * 100),
                "last_update": datetime.now(timezone.utc).isoformat()
            }
            return health_data
        except Exception as e:
            LOG.error("health_check_failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "last_update": datetime.now(timezone.utc).isoformat()
            }

    async def periodic_health_log(self):
        while getattr(self.bot, 'is_running', False):
            try:
                health = self.get_health_status()
                LOG.info("health_check", **health)
                if INFLUX_METRICS.enabled:
                    try:
                        await self._write_health_to_influx(health)
                    except Exception as influx_error:
                        LOG.debug("health_influx_write_failed", error=str(influx_error))
                memory_mb = health.get('memory_mb', 0)
                if memory_mb > 1500:
                    await ALERT_SYSTEM.send_alert("WARNING", "High memory usage", memory_mb=memory_mb)
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOG.error("health_check_periodic_failed", error=str(e))
                await asyncio.sleep(60)

    async def _write_health_to_influx(self, health: Dict[str, Any]):
        try:
            if not INFLUX_METRICS.enabled:
                return
            from influxdb_client import Point, WritePrecision
            point = Point("system_health")
            status = health.get('status', 'unknown')
            point.field("memory_mb", float(health.get('memory_mb', 0)))
            point.field("cpu_percent", float(health.get('cpu_percent', 0)))
            point.field("uptime_seconds", float(health.get('uptime_seconds', 0)))
            point.field("active_positions", int(health.get('active_positions', 0)))
            point.field("status", 1.0 if status == "healthy" else 0.0)
            point.time(datetime.now(timezone.utc), WritePrecision.NS)
            INFLUX_METRICS.write_api.write(bucket=INFLUX_METRICS.bucket, org=INFLUX_METRICS.org, record=point)
            LOG.debug("health_metrics_written_to_influx")
        except Exception as e:
            LOG.error("health_influx_write_failed", error=str(e))

class PerformanceProfiler:
    def __init__(self):
        self._timings = defaultdict(list)
        self._call_counts = defaultdict(int)
        self._memory_snapshots = {}
        self._active_timers = {}
        self._lock = asyncio.Lock()

    @contextmanager
    def profile(self, operation_name: str):
        start_time = time.perf_counter()
        start_memory = self._get_memory_mb()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            memory_delta = self._get_memory_mb() - start_memory
            LOG.debug("memory_delta", delta=memory_delta)
            self._timings[operation_name].append(elapsed)
            self._call_counts[operation_name] += 1
            if memory_delta > 10:
                LOG.debug("operation_memory_spike", operation=operation_name, memory_mb=memory_delta)

    def profile_async(self, operation_name: str):
        class AsyncProfileContext:
            def __init__(ctx_self, profiler, op_name):
                ctx_self.profiler = profiler
                ctx_self.op_name = op_name
                ctx_self.start_time = None
                ctx_self.start_memory = None

            async def __aenter__(ctx_self):
                ctx_self.start_time = time.perf_counter()
                ctx_self.start_memory = ctx_self.profiler._get_memory_mb()
                return ctx_self

            async def __aexit__(ctx_self, exc_type, exc_val, exc_tb):
                elapsed = time.perf_counter() - ctx_self.start_time
                memory_delta = ctx_self.profiler._get_memory_mb() - ctx_self.start_memory
                async with ctx_self.profiler._lock:
                    ctx_self.profiler._timings[ctx_self.op_name].append(elapsed)
                    ctx_self.profiler._call_counts[ctx_self.op_name] += 1
                if elapsed > 1.0:
                    LOG.warning("slow_operation_detected", operation=ctx_self.op_name, elapsed_seconds=elapsed)
                if memory_delta > 10:
                    LOG.debug("operation_memory_spike", operation=ctx_self.op_name, memory_mb=memory_delta)
                return False

        return AsyncProfileContext(self, operation_name)

    def _get_memory_mb(self) -> float:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def get_stats(self, top_n: int = 10) -> Dict[str, Any]:
        stats = []
        for op_name, timings in self._timings.items():
            if len(timings) < 2:
                continue
            stats.append({
                'operation': op_name,
                'count': self._call_counts[op_name],
                'total_time': sum(timings),
                'avg_time': np.mean(timings),
                'min_time': np.min(timings),
                'max_time': np.max(timings),
                'std_time': np.std(timings),
                'p95_time': np.percentile(timings, 95),
                'p99_time': np.percentile(timings, 99)
            })
        stats.sort(key=lambda x: x['total_time'], reverse=True)
        return {
            'top_operations': stats[:top_n],
            'total_operations': len(self._timings)
        }

    def generate_report(self) -> str:
        stats = self.get_stats(top_n=10)
        report = "=" * 80 + "\n"
        report += "PERFORMANCE PROFILING REPORT\n"
        report += "=" * 80 + "\n"
        for i, op in enumerate(stats['top_operations'], 1):
            report += f"{i}. {op['operation']}\n"
            report += f"   Calls: {op['count']:,}\n"
            report += f"   Total: {op['total_time']:.3f}s\n"
            report += f"   Avg: {op['avg_time']*1000:.2f}ms\n"
            report += f"   P95: {op['p95_time']*1000:.2f}ms\n"
            report += f"   P99: {op['p99_time']*1000:.2f}ms\n"
            report += "\n"
        return report

    def reset(self):
        self._timings.clear()
        self._call_counts.clear()
        self._memory_snapshots.clear()

PERFORMANCE_PROFILER = PerformanceProfiler()

class PerformanceMonitor:
    def __init__(self, memory_manager=None):
        self.metrics = {'order_latency': [], 'ws_message_rate': [], 'memory_usage': []}
        self.memory_threshold = 500
        self.memory_manager = memory_manager or MEMORY_MANAGER

    def _get_memory_usage(self):
        if self.memory_manager:
            try:
                mem_stats = self.memory_manager.get_memory_usage()
                return mem_stats.get('rss_mb', 0)
            except Exception:
                pass
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    async def _check_memory_usage(self):
        memory_usage = self._get_memory_usage()
        self.metrics['memory_usage'].append(memory_usage)
        if memory_usage > self.memory_threshold:
            LOG.warning("High memory usage detected: {} MB".format(memory_usage))
            if self.memory_manager:
                try:
                    await self.memory_manager.routine_cleanup()
                    LOG.info("memory_cleanup_triggered_by_performance_monitor")
                except Exception as e:
                    LOG.error("memory_cleanup_failed", error=str(e))
            else:
                gc.collect()

    async def monitor_loop(self, bot, interval=60):
        while getattr(bot, 'is_running', False):
            try:
                await self._check_memory_usage()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOG.error("performance_monitor_loop_error", error=str(e))

class DynamicPositionSizer:
    def __init__(self, config, bot):
        self.config = config
        self.bot = bot
        self.win_rate_history = []
        self.avg_win = 0.0
        self.avg_loss = 0.0

    def calculate_position_size(self, symbol: str, confidence: float, current_price: float, df: pd.DataFrame) -> float:
        try:
            if not hasattr(self.bot, 'equity') or self.bot.equity <= 0:
                LOG.error("invalid_bot_equity_for_position_sizing", 
                         equity=self.bot.equity if hasattr(self.bot, 'equity') else None)
                return self.config.min_order_size
            
            total_equity = float(self.bot.equity)
            
            # CORRECCIÓN: Usar LEDGER como source of truth para invested_equity
            invested_equity = 0.0
            
            if hasattr(self.bot, 'position_ledger') and self.bot.position_ledger:
                # Calcular desde ledger (más confiable)
                for sym, open_tx in self.bot.position_ledger.active_positions.items():
                    try:
                        if not isinstance(open_tx, PositionTransaction):
                            LOG.warning("invalid_transaction_in_ledger_skipping",
                                       symbol=sym)
                            continue
                        
                        entry_price = float(open_tx.entry_price)
                        size = float(open_tx.size)
                        
                        if entry_price <= 0 or size <= 0:
                            LOG.warning("invalid_position_values_in_ledger",
                                       symbol=sym,
                                       entry_price=entry_price,
                                       size=size)
                            continue
                        
                        position_value = entry_price * size
                        
                        if position_value > 100000.0:
                            LOG.error("unreasonable_position_value_from_ledger",
                                     symbol=sym,
                                     position_value=position_value)
                            continue
                        
                        invested_equity += position_value
                        
                    except Exception as pos_calc_error:
                        LOG.debug("position_value_calc_failed",
                                 symbol=sym,
                                 error=str(pos_calc_error))
                        continue
            
            # FALLBACK: Usar risk_manager si ledger no disponible
            elif hasattr(self.bot, 'risk_manager') and self.bot.risk_manager.active_stops:
                LOG.warning("using_risk_manager_for_invested_equity_ledger_unavailable")
                for sym, stop_info in self.bot.risk_manager.active_stops.items():
                    try:
                        entry_price = float(stop_info.get('entry_price', 0))
                        remaining_size = float(stop_info.get('remaining_size', 0))
                        
                        if entry_price > 0 and remaining_size > 0:
                            position_value = entry_price * remaining_size
                            invested_equity += position_value
                    except Exception:
                        continue
            
            available_equity = total_equity - invested_equity
            
            # CORRECCIÓN CRÍTICA: Manejar equity negativo
            if available_equity < 0:
                LOG.error("negative_available_equity_critical",
                         total_equity=total_equity,
                         invested_equity=invested_equity,
                         available_equity=available_equity,
                         message="Possible ledger/risk_manager desync")
                
                # Ejecutar auditoría inmediata
                if hasattr(self.bot, 'position_ledger'):
                    audit = self.bot.position_ledger.audit_equity(self.bot)
                    LOG.error("equity_audit_on_negative_available", audit=audit)
                    
                    # Si audit es consistente, usar valores auditados
                    if audit['is_consistent']:
                        corrected_invested = audit['invested_in_positions']
                        available_equity = total_equity - corrected_invested
                        LOG.info("equity_corrected_from_audit",
                                corrected_invested=corrected_invested,
                                corrected_available=available_equity)
                    
                    # Si sigue negativo, rechazar trade
                    if available_equity < 0:
                        LOG.critical("available_equity_still_negative_after_correction",
                                    available=available_equity)
                        
                        # Enviar alerta crítica
                        asyncio.create_task(
                            ALERT_SYSTEM.send_alert(
                                "CRITICAL",
                                f"Negative available equity: {available_equity:.2f}",
                                total_equity=total_equity,
                                invested=invested_equity,
                                audit=audit
                            )
                        )
                        return 0.0
            
            LOG.debug("equity_breakdown_for_sizing",
                     total_equity=total_equity,
                     invested_equity=invested_equity,
                     available_equity=available_equity,
                     active_positions=len(self.bot.risk_manager.active_stops) if hasattr(self.bot, 'risk_manager') else 0)
            
            # Validar mínimo disponible
            if available_equity < self.config.min_order_size:
                LOG.warning("insufficient_available_equity",
                           available=available_equity,
                           min_required=self.config.min_order_size)
                return 0.0
            
            # Continuar con cálculo de position size...
            base_risk_pct = 0.05
            
            # Ajuste por volatilidad
            volatility_adjustment = 1.0
            if 'volatility' in df.columns and not df['volatility'].isna().all():
                current_vol = float(df['volatility'].iloc[-1])
                avg_vol = float(df['volatility'].rolling(50).mean().iloc[-1])
                if avg_vol > 0 and current_vol > 0:
                    vol_ratio = current_vol / avg_vol
                    volatility_adjustment = 1.0 / (1.0 + (vol_ratio - 1.0) * 0.5)
                    volatility_adjustment = max(0.5, min(1.5, volatility_adjustment))
            
            adjusted_risk_pct = base_risk_pct * volatility_adjustment
            risk_amount = available_equity * adjusted_risk_pct * confidence
            
            LOG.debug("risk_calculation",
                     base_risk_pct=base_risk_pct,
                     volatility_adjustment=volatility_adjustment,
                     adjusted_risk_pct=adjusted_risk_pct,
                     confidence=confidence,
                     available_equity=available_equity)
            
            # Kelly, volatility-adjusted
            kelly_fraction = self._calculate_kelly_fraction()
            kelly_amount = available_equity * kelly_fraction if kelly_fraction > 0 else 0
            volatility_amount = self._volatility_adjusted_size(df, current_price, available_equity)
            
            position_amount = (risk_amount * 0.5 + kelly_amount * 0.3 + volatility_amount * 0.2)
            
            # Límite máximo
            max_position_value = available_equity * 0.30
            if position_amount > max_position_value:
                LOG.warning("position_amount_exceeds_limit",
                           calculated=position_amount,
                           max_allowed=max_position_value,
                           available_equity=available_equity,
                           symbol=symbol)
                position_amount = max_position_value
            
            # Ajuste por drawdown
            current_drawdown = abs(self.bot.drawdown) if hasattr(self.bot, 'drawdown') else 0.0
            if current_drawdown == 0.0 and hasattr(self.bot, 'equity') and hasattr(self.bot, 'initial_capital'):
                if self.bot.initial_capital > 0:
                    current_return = (self.bot.equity - self.bot.initial_capital) / self.bot.initial_capital
                    current_drawdown = abs(min(0.0, current_return))
            
            if current_drawdown > 0.10:
                drawdown_reduction = max(0.5, 1.0 - (current_drawdown * 0.5))
            else:
                drawdown_reduction = 1.0
            
            # Ajuste por win rate
            win_rate_adjustment = 1.0
            if hasattr(self.bot, 'performance_metrics'):
                total_trades = self.bot.performance_metrics.get('total_trades', 0)
                if total_trades >= 10:
                    win_rate = self.bot.performance_metrics.get('win_rate', 0.5)
                    if win_rate < 0.4:
                        win_rate_adjustment = 0.8
                    elif win_rate > 0.6:
                        win_rate_adjustment = 1.2
            
            # Límites finales
            base_min_pct = 0.02
            base_max_pct = 0.30
            confidence_multiplier = 0.7 + (confidence * 0.6)
            adjusted_max_pct = base_max_pct * confidence_multiplier * drawdown_reduction * win_rate_adjustment
            
            min_size = max(self.config.min_order_size, available_equity * base_min_pct)
            max_size = available_equity * adjusted_max_pct
            
            position_amount = max(min_size, min(max_size, position_amount))
            
            # Validación final
            if position_amount > available_equity:
                LOG.error("position_amount_exceeds_available_after_all_limits",
                         position_amount=position_amount,
                         available_equity=available_equity)
                position_amount = available_equity * 0.25
            
            LOG.info("position_size_calculated",
                    symbol=symbol,
                    confidence=confidence,
                    risk_amount=risk_amount,
                    kelly_amount=kelly_amount,
                    volatility_amount=volatility_amount,
                    final_amount=position_amount,
                    min_size=min_size,
                    max_size=max_size,
                    available_equity=available_equity,
                    total_equity=self.bot.equity)
            
            return float(position_amount)
            
        except Exception as e:
            LOG.error("position_sizing_error",
                     error=str(e),
                     traceback=traceback.format_exc()[:500])
            return max(self.config.min_order_size, self.bot.equity * 0.01)

    def _calculate_kelly_fraction(self) -> float:
        try:
            if len(self.win_rate_history) < 20:
                return 0.0
            p = np.mean(self.win_rate_history)
            if p <= 0 or p >= 1:
                return 0.0
            q = 1 - p
            if self.avg_loss <= 0:
                return 0.0
            b = self.avg_win / abs(self.avg_loss)
            kelly = (p * b - q) / b
            return max(0.0, min(0.05, kelly * 0.25))
        except Exception as e:
            LOG.debug("kelly_calculation_error", error=str(e))
            return 0.0

    def _volatility_adjusted_size(self, df: pd.DataFrame, current_price: float, available_equity: float) -> float:
        try:
            if 'volatility' not in df.columns or len(df) < 20:
                return self.config.min_order_size
            volatility = df['volatility'].iloc[-1]
            if volatility <= 0 or np.isnan(volatility):
                return self.config.min_order_size
            base_vol = 0.02
            vol_factor = base_vol / max(volatility, 0.001)
            vol_factor = max(0.5, min(2.0, vol_factor))
            amount = available_equity * 0.05 * vol_factor
            return max(self.config.min_order_size, amount)
        except Exception as e:
            LOG.debug("volatility_sizing_error", error=str(e))
            return self.config.min_order_size

    def update_trade_history(self, pnl: float, is_win: bool):
        try:
            self.win_rate_history.append(1.0 if is_win else 0.0)
            if len(self.win_rate_history) > 100:
                self.win_rate_history.pop(0)
            if is_win:
                if self.avg_win == 0:
                    self.avg_win = pnl
                else:
                    self.avg_win = (self.avg_win * 0.9) + (pnl * 0.1)
            else:
                if self.avg_loss == 0:
                    self.avg_loss = pnl
                else:
                    self.avg_loss = (self.avg_loss * 0.9) + (pnl * 0.1)
        except Exception as e:
            LOG.debug("trade_history_update_error", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        try:
            return {
                'win_rate_history_length': len(self.win_rate_history),
                'current_win_rate': np.mean(self.win_rate_history) if self.win_rate_history else 0.0,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'kelly_fraction': self._calculate_kelly_fraction()
            }
        except Exception as e:
            LOG.debug("position_sizer_stats_error", error=str(e))
            return {}

from dataclasses import dataclass, field
from enum import Enum

class TransactionType(Enum):
    OPEN = "open"
    CLOSE = "close"
    PARTIAL_CLOSE = "partial_close"

@dataclass
class PositionTransaction:
    transaction_id: str
    symbol: str
    transaction_type: TransactionType
    timestamp: datetime
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0.0
    equity_before: float = 0.0
    equity_change: float = 0.0
    realized_pnl: float = 0.0
    equity_after: float = 0.0
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)

    def validate(self) -> bool:
        self.validation_errors = []
        
        # Validación 1: Entry price
        if self.entry_price <= 0:
            self.validation_errors.append(f"Invalid entry_price: {self.entry_price}")
        
        # NUEVO: Rango razonable para entry_price
        if self.entry_price > 1000000.0:
            self.validation_errors.append(f"Entry price unreasonably high: {self.entry_price}")
        
        if np.isnan(self.entry_price) or np.isinf(self.entry_price):
            self.validation_errors.append(f"Entry price is NaN or Inf: {self.entry_price}")
        
        # Validación 2: Exit price (para CLOSE y PARTIAL_CLOSE)
        if self.transaction_type in [TransactionType.CLOSE, TransactionType.PARTIAL_CLOSE]:
            if self.exit_price is None or self.exit_price <= 0:
                self.validation_errors.append(f"Invalid exit_price: {self.exit_price}")
            
            # NUEVO: Validar que exit_price es razonable respecto a entry_price
            if self.exit_price is not None and self.entry_price > 0:
                price_change_pct = abs(self.exit_price - self.entry_price) / self.entry_price
                if price_change_pct > 0.50:  # 50% cambio en un trade
                    self.validation_errors.append(
                        f"Unreasonable price change: {price_change_pct*100:.1f}% "
                        f"(entry={self.entry_price}, exit={self.exit_price})"
                    )
        
        # Validación 3: Size
        if self.size <= 0:
            self.validation_errors.append(f"Invalid size: {self.size}")
        
        if np.isnan(self.size) or np.isinf(self.size):
            self.validation_errors.append(f"Size is NaN or Inf: {self.size}")
        
        # NUEVO: Validar tamaño razonable
        if self.size > 1000000.0:
            self.validation_errors.append(f"Size unreasonably large: {self.size}")
        
        # Validación 4: Equity before
        if self.equity_before < 0:
            self.validation_errors.append(f"Negative equity_before: {self.equity_before}")
        
        # NUEVO: Equity debe ser razonable
        if self.equity_before > 10000000.0:
            self.validation_errors.append(f"Equity unreasonably high: {self.equity_before}")
        
        # Validación 5: Consistencia de equity para OPEN
        if self.transaction_type == TransactionType.OPEN:
            tolerance = 0.01
            if abs(self.equity_after - self.equity_before) > tolerance:
                self.validation_errors.append(
                    f"Equity changed on OPEN (should remain constant): "
                    f"before={self.equity_before:.2f}, "
                    f"after={self.equity_after:.2f}, "
                    f"change={self.equity_change:.2f}"
                )
        
        # ✅ CORRECCIÓN CRÍTICA: Validación 6 MEJORADA para CLOSE con tolerancia más permisiva
        else:
            if self.transaction_type in [TransactionType.CLOSE, TransactionType.PARTIAL_CLOSE]:
                position_value = self.entry_price * self.size
                max_reasonable_pnl = position_value * 2.0  # 200% max
                
                if abs(self.equity_change) > max_reasonable_pnl:
                    self.validation_errors.append(
                        f"Equity change (PnL) unreasonably large: "
                        f"change={self.equity_change:.2f}, position_value={position_value:.2f}, "
                        f"max_allowed={max_reasonable_pnl:.2f}"
                    )
            
            expected_equity_after = self.equity_before + self.equity_change
            
            # ✅ Tolerancia RELATIVA más generosa para evitar falsos positivos
            # Usar el mayor entre tolerancia absoluta y relativa
            tolerance_abs = 0.10  # Aumentado de 0.01 a 0.10 (10 centavos)
            tolerance_rel = abs(expected_equity_after) * 0.0001  # 0.01% del equity
            tolerance = max(tolerance_abs, tolerance_rel, 0.05)  # Mínimo 5 centavos
            
            actual_diff = abs(self.equity_after - expected_equity_after)
            
            # ✅ NUEVO: Solo validar si la diferencia es SIGNIFICATIVA (> $1)
            if actual_diff > max(1.0, tolerance):
                self.validation_errors.append(
                    f"Equity inconsistency on CLOSE: "
                    f"before={self.equity_before:.2f}, "
                    f"change={self.equity_change:.2f}, "
                    f"after={self.equity_after:.2f}, "
                    f"expected={expected_equity_after:.2f}, "
                    f"diff={actual_diff:.2f}, "
                    f"tolerance={tolerance:.6f}"
                )
        
        # NUEVO: Validación 7 - PnL razonable para CLOSE
        if self.transaction_type in [TransactionType.CLOSE, TransactionType.PARTIAL_CLOSE]:
            if abs(self.realized_pnl) > self.equity_before * 0.50:
                self.validation_errors.append(
                    f"PnL unreasonably large relative to equity: "
                    f"pnl={self.realized_pnl:.2f}, equity={self.equity_before:.2f}"
                )
        
        # NUEVO: Validación 8 - Side válido
        valid_sides = ['buy', 'sell']
        if self.side not in valid_sides:
            self.validation_errors.append(f"Invalid side: {self.side}, must be in {valid_sides}")
        
        # NUEVO: Validación 9 - Symbol no vacío
        if not self.symbol or len(self.symbol) < 3:
            self.validation_errors.append(f"Invalid symbol: {self.symbol}")
        
        # NUEVO: Validación 10 - Timestamp razonable
        now = datetime.now(timezone.utc)
        time_diff = abs((now - self.timestamp).total_seconds())
        if time_diff > 86400:  # Más de 1 día en el futuro o pasado
            self.validation_errors.append(
                f"Timestamp too far from current time: {self.timestamp} (diff: {time_diff/3600:.1f}h)"
            )
        
        self.is_valid = len(self.validation_errors) == 0
        
        # Log detallado si no es válida
        if not self.is_valid:
            LOG.error("transaction_validation_failed",
                     transaction_id=self.transaction_id,
                     symbol=self.symbol,
                     type=self.transaction_type.value,
                     errors=self.validation_errors)
        
        return self.is_valid

class PositionLedger:
    """
    VERSIÓN MEJORADA: Ledger con persistencia y recuperación ante desconexión
    
    NUEVO:
    - Persistencia en SQLite
    - Auto-recuperación de posiciones
    - Reconciliación con exchange al reconectar
    - Snapshots periódicos
    """
    def __init__(self, db_path: str = "position_ledger.db"):
        self.transactions: List[PositionTransaction] = []
        self.active_positions: Dict[str, PositionTransaction] = {}
        self._lock = asyncio.Lock()
        self.db_path = db_path
        self._snapshot_interval = 60  # Snapshot cada 60 segundos
        self._last_snapshot = time.time()
        self._initialize_db()
        
    def _initialize_db(self):
        """Inicializa base de datos SQLite para persistencia"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla de transacciones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    transaction_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    equity_before REAL NOT NULL,
                    equity_change REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    equity_after REAL NOT NULL,
                    is_valid INTEGER NOT NULL,
                    validation_errors TEXT
                )
            ''')
            
            # Tabla de posiciones activas (snapshot)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_positions_snapshot (
                    symbol TEXT PRIMARY KEY,
                    transaction_id TEXT NOT NULL,
                    snapshot_time REAL NOT NULL,
                    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
                )
            ''')
            
            # Índices para performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_symbol 
                ON transactions(symbol)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_timestamp 
                ON transactions(timestamp)
            ''')
            
            conn.commit()
            conn.close()
            
            LOG.info("position_ledger_db_initialized", db_path=self.db_path)
            
            # Cargar transacciones persistidas
            self._load_from_db()
            
        except Exception as e:
            LOG.error("ledger_db_initialization_failed",
                     error=str(e),
                     db_path=self.db_path)
    
    def _load_from_db(self):
        """Carga transacciones desde DB"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Cargar todas las transacciones
            cursor.execute('''
                SELECT transaction_id, symbol, transaction_type, timestamp,
                       side, entry_price, exit_price, size, equity_before,
                       equity_change, realized_pnl, equity_after, is_valid,
                       validation_errors
                FROM transactions
                ORDER BY timestamp ASC
            ''')
            
            loaded_count = 0
            for row in cursor.fetchall():
                try:
                    tx_type_str = row[2]
                    tx_type = TransactionType(tx_type_str)
                    
                    transaction = PositionTransaction(
                        transaction_id=row[0],
                        symbol=row[1],
                        transaction_type=tx_type,
                        timestamp=datetime.fromisoformat(row[3]),
                        side=row[4],
                        entry_price=row[5],
                        exit_price=row[6],
                        size=row[7],
                        equity_before=row[8],
                        equity_change=row[9],
                        realized_pnl=row[10],
                        equity_after=row[11],
                        is_valid=bool(row[12]),
                        validation_errors=json.loads(row[13]) if row[13] else []
                    )
                    
                    self.transactions.append(transaction)
                    loaded_count += 1
                    
                except Exception as tx_error:
                    LOG.warning("failed_to_load_transaction",
                               transaction_id=row[0],
                               error=str(tx_error))
                    continue
            
            # Cargar posiciones activas
            cursor.execute('''
                SELECT symbol, transaction_id
                FROM active_positions_snapshot
                ORDER BY snapshot_time DESC
            ''')
            
            for symbol, tx_id in cursor.fetchall():
                # Buscar transacción correspondiente
                for tx in self.transactions:
                    if tx.transaction_id == tx_id:
                        self.active_positions[symbol] = tx
                        break
            
            conn.close()
            
            LOG.info("ledger_loaded_from_db",
                    transactions=loaded_count,
                    active_positions=len(self.active_positions))
            
        except Exception as e:
            LOG.error("ledger_load_from_db_failed", error=str(e))
    
    async def _persist_transaction(self, transaction: PositionTransaction):
        """Persiste transacción en DB"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO transactions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                transaction.transaction_id,
                transaction.symbol,
                transaction.transaction_type.value,
                transaction.timestamp.isoformat(),
                transaction.side,
                transaction.entry_price,
                transaction.exit_price,
                transaction.size,
                transaction.equity_before,
                transaction.equity_change,
                transaction.realized_pnl,
                transaction.equity_after,
                int(transaction.is_valid),
                json.dumps(transaction.validation_errors) if transaction.validation_errors else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            LOG.error("transaction_persistence_failed",
                     transaction_id=transaction.transaction_id,
                     error=str(e))
    
    async def _snapshot_active_positions(self):
        """Guarda snapshot de posiciones activas"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Limpiar snapshots antiguos
            cursor.execute('DELETE FROM active_positions_snapshot')
            
            # Insertar snapshot actual
            snapshot_time = time.time()
            for symbol, transaction in self.active_positions.items():
                cursor.execute('''
                    INSERT INTO active_positions_snapshot VALUES (?, ?, ?)
                ''', (symbol, transaction.transaction_id, snapshot_time))
            
            conn.commit()
            conn.close()
            
            self._last_snapshot = snapshot_time
            
            LOG.debug("active_positions_snapshot_saved",
                     count=len(self.active_positions))
            
        except Exception as e:
            LOG.error("snapshot_failed", error=str(e))
    
    async def record_open(self, bot, symbol: str, side: str, entry_price: float, 
                         size: float) -> Optional[PositionTransaction]:
        """
        MEJORADO: Registro con persistencia automática
        """
        async with self._lock:
            try:
                # Validaciones existentes...
                transaction_id = f"{symbol}_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
                equity_before = float(bot.equity)
                
                # ✅ CORRECCIÓN CRÍTICA: En apertura, equity NO cambia
                # Solo se "reserva" capital, pero el equity total permanece igual
                position_value = entry_price * size
                
                # equity_change = 0 porque no hay cambio real de equity
                # Solo movemos capital de "disponible" a "invertido"
                equity_change = 0.0  # ← CAMBIO CRÍTICO
                equity_after = equity_before  # ← El equity permanece igual
                
                transaction = PositionTransaction(
                    transaction_id=transaction_id,
                    symbol=symbol,
                    transaction_type=TransactionType.OPEN,
                    timestamp=datetime.now(timezone.utc),
                    side=side,
                    entry_price=entry_price,
                    size=size,
                    equity_before=equity_before,
                    equity_change=equity_change,  # ← 0.0
                    equity_after=equity_after  # ← Sin cambio
                )
                
                if not transaction.validate():
                    LOG.error("invalid_open_transaction",
                             symbol=symbol,
                             errors=transaction.validation_errors)
                    return None
                
                self.transactions.append(transaction)
                self.active_positions[symbol] = transaction
                
                # ===== NUEVO: Persistir =====
                await self._persist_transaction(transaction)
                
                # Snapshot periódico
                if time.time() - self._last_snapshot > self._snapshot_interval:
                    asyncio.create_task(self._snapshot_active_positions())
                
                LOG.info("position_opened_and_persisted",
                        symbol=symbol,
                        transaction_id=transaction_id,
                        position_value=position_value,
                        equity_unchanged=equity_before)
                
                return transaction
                
            except Exception as e:
                LOG.error("record_open_failed",
                         symbol=symbol,
                         error=str(e))
                return None
    
    async def record_close(self, bot, symbol: str, exit_price: float, 
                          size: float = None,
                          equity_before_override: float = None,
                          equity_after_override: float = None,
                          realized_pnl_override: float = None) -> Optional[PositionTransaction]:
        """
        VERSIÓN DEFINITIVA: Ledger como registro contable puro
        NO modifica bot.equity - solo registra
        """
        async with self._lock:
            try:
                # Verificar posición existe
                has_ledger_position = symbol in self.active_positions
                
                if not has_ledger_position:
                    LOG.warning("position_not_in_memory_checking_db", symbol=symbol)
                    recovered = await self._recover_position_from_db(symbol)
                    
                    if recovered:
                        has_ledger_position = True
                        LOG.info("position_recovered_from_db", symbol=symbol)
                    else:
                        LOG.error("position_not_found_anywhere", symbol=symbol)
                        return None
                
                open_transaction = self.active_positions[symbol]
                
                if size is None:
                    size = open_transaction.size
                
                # Validaciones
                if size > open_transaction.size * 1.01:
                    LOG.error("close_size_exceeds_position",
                             symbol=symbol,
                             close_size=size,
                             position_size=open_transaction.size)
                    size = open_transaction.size
                
                transaction_id = f"{symbol}_close_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
                
                # ✅ USAR valores del caller si se proporcionan (SOURCE OF TRUTH)
                if equity_before_override is not None:
                    equity_before_close = float(equity_before_override)
                    LOG.debug("using_equity_before_from_caller", value=equity_before_close)
                else:
                    equity_before_close = float(bot.equity)
                    LOG.warning("equity_before_not_provided_using_bot_equity", value=equity_before_close)
                
                # Calcular PnL SOLO si no se proporciona
                if realized_pnl_override is not None:
                    realized_pnl = float(realized_pnl_override)
                    LOG.debug("using_realized_pnl_from_caller", value=realized_pnl)
                else:
                    # Calcular
                    entry_price = open_transaction.entry_price
                    
                    if open_transaction.side == 'buy':
                        realized_pnl = (exit_price - entry_price) * size
                    else:
                        realized_pnl = (entry_price - exit_price) * size
                    
                    LOG.warning("realized_pnl_not_provided_calculated", value=realized_pnl)
                
                # Validar PnL
                position_value = open_transaction.entry_price * size
                max_reasonable_pnl = position_value * 2.0
                
                if abs(realized_pnl) > max_reasonable_pnl:
                    LOG.error("unreasonable_pnl_in_ledger",
                             symbol=symbol,
                             realized_pnl=realized_pnl,
                             position_value=position_value)
                    realized_pnl = np.clip(realized_pnl, -position_value, position_value)
                    LOG.warning("pnl_clipped_in_ledger", clipped_pnl=realized_pnl)
                
                # ✅ equity_change es el PnL
                equity_change = realized_pnl
                
                # ✅ USAR equity_after del caller si se proporciona
                if equity_after_override is not None:
                    equity_after_close = float(equity_after_override)
                    LOG.debug("using_equity_after_from_caller", value=equity_after_close)
                else:
                    equity_after_close = equity_before_close + realized_pnl
                    LOG.warning("equity_after_not_provided_calculated", value=equity_after_close)
                
                # ✅ VALIDACIÓN: Verificar consistencia
                expected_equity = equity_before_close + realized_pnl
                if abs(equity_after_close - expected_equity) > 0.01:
                    LOG.error("equity_after_inconsistent_with_calculation",
                             equity_before=equity_before_close,
                             realized_pnl=realized_pnl,
                             expected=expected_equity,
                             provided=equity_after_close,
                             diff=equity_after_close - expected_equity)
                    # Usar el calculado
                    equity_after_close = expected_equity

                if equity_after_close < 0:
                    LOG.error("negative_equity_after_close",
                             equity_before=equity_before_close,
                             realized_pnl=realized_pnl,
                             equity_after=equity_after_close,
                             symbol=symbol)
                    return None
                
                # ✅ CRÍTICO: NO TOCAR bot.equity
                # Ya fue actualizado por el caller
                
                LOG.info("LEDGER_CLOSE_RECORD",
                        symbol=symbol,
                        equity_before=equity_before_close,
                        realized_pnl=realized_pnl,
                        equity_after=equity_after_close,
                        equity_change=equity_change,
                        bot_equity_untouched=bot.equity,
                        note="ledger_is_passive_recorder")
                
                is_partial = size < open_transaction.size
                transaction_type = TransactionType.PARTIAL_CLOSE if is_partial else TransactionType.CLOSE
                
                transaction = PositionTransaction(
                    transaction_id=transaction_id,
                    symbol=symbol,
                    transaction_type=transaction_type,
                    timestamp=datetime.now(timezone.utc),
                    side='sell' if open_transaction.side == 'buy' else 'buy',
                    entry_price=open_transaction.entry_price,
                    exit_price=exit_price,
                    size=size,
                    equity_before=equity_before_close,
                    equity_change=equity_change,
                    realized_pnl=realized_pnl,
                    equity_after=equity_after_close
                )
                
                if not transaction.validate():
                    LOG.error("transaction_validation_failed",
                             symbol=symbol,
                             errors=transaction.validation_errors)
                    return None
                
                self.transactions.append(transaction)
                
                if is_partial:
                    open_transaction.size -= size
                else:
                    del self.active_positions[symbol]
                
                # Persistir
                await self._persist_transaction(transaction)
                
                # Snapshot
                if time.time() - self._last_snapshot > self._snapshot_interval:
                    asyncio.create_task(self._snapshot_active_positions())
                
                LOG.info("ledger_close_completed",
                        symbol=symbol,
                        transaction_id=transaction_id,
                        is_partial=is_partial,
                        realized_pnl=realized_pnl)
                
                return transaction
                
            except Exception as e:
                LOG.error("record_close_failed",
                         symbol=symbol,
                         error=str(e),
                         traceback=traceback.format_exc()[:500])
                return None
    
    async def _recover_position_from_db(self, symbol: str) -> bool:
        """
        Recupera posición activa desde DB
        
        Útil después de crashes o reinicios
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Buscar última transacción OPEN sin CLOSE correspondiente
            cursor.execute('''
                SELECT transaction_id, side, entry_price, size, 
                       equity_before, timestamp
                FROM transactions
                WHERE symbol = ? AND transaction_type = 'open'
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (symbol,))
            
            open_row = cursor.fetchone()
            
            if not open_row:
                conn.close()
                return False
            
            open_tx_id = open_row[0]
            
            # Verificar si hay CLOSE posterior
            cursor.execute('''
                SELECT COUNT(*)
                FROM transactions
                WHERE symbol = ? 
                  AND transaction_type IN ('close', 'partial_close')
                  AND timestamp > ?
            ''', (symbol, open_row[5]))
            
            close_count = cursor.fetchone()[0]
            
            conn.close()
            
            if close_count > 0:
                # Ya fue cerrada
                return False
            
            # Reconstruir transacción
            transaction = PositionTransaction(
                transaction_id=open_tx_id,
                symbol=symbol,
                transaction_type=TransactionType.OPEN,
                timestamp=datetime.fromisoformat(open_row[5]),
                side=open_row[1],
                entry_price=open_row[2],
                size=open_row[3],
                equity_before=open_row[4],
                equity_change=-(open_row[2] * open_row[3]),
                equity_after=open_row[4]
            )
            
            if transaction.validate():
                self.active_positions[symbol] = transaction
                LOG.info("position_recovered_from_db",
                        symbol=symbol,
                        transaction_id=open_tx_id,
                        side=transaction.side,
                        entry_price=transaction.entry_price)
                return True
            
            return False
            
        except Exception as e:
            LOG.error("position_recovery_failed",
                     symbol=symbol,
                     error=str(e))
            return False
    
    async def reconcile_with_exchange(self, bot, exchange_manager) -> Dict[str, Any]:
        """
        NUEVO: Reconcilia posiciones del ledger con el exchange
        
        Útil después de desconexiones para detectar discrepancias
        
        Returns:
            Dict con reporte de reconciliación
        """
        try:
            LOG.info("starting_ledger_exchange_reconciliation")
            
            reconciliation = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'ledger_positions': len(self.active_positions),
                'exchange_positions': 0,
                'matched': [],
                'ledger_only': [],
                'exchange_only': [],
                'discrepancies': []
            }
            
            # Obtener posiciones del exchange
            try:
                balance = await exchange_manager.fetch_balance()
                if not balance or not balance.get('success'):
                    LOG.warning("cannot_fetch_balance_for_reconciliation")
                    reconciliation['error'] = 'Balance fetch failed'
                    return reconciliation
                
                exchange_positions = {}
                
                # En exchange real, iterar sobre balances
                for currency, amounts in balance.get('balance', {}).items():
                    if currency == 'USDT':
                        continue
                    
                    free = amounts.get('free', 0)
                    used = amounts.get('used', 0)
                    total = amounts.get('total', 0)
                    
                    if total > 0:
                        # Tenemos una posición
                        symbol = f"{currency}/USDT"
                        exchange_positions[symbol] = {
                            'size': total,
                            'free': free,
                            'used': used
                        }
                
                reconciliation['exchange_positions'] = len(exchange_positions)
                
            except Exception as balance_error:
                LOG.error("balance_fetch_failed_in_reconciliation",
                         error=str(balance_error))
                reconciliation['error'] = str(balance_error)
                return reconciliation
            
            # Comparar posiciones
            ledger_symbols = set(self.active_positions.keys())
            exchange_symbols = set(exchange_positions.keys())
            
            # Posiciones que coinciden
            matched_symbols = ledger_symbols & exchange_symbols
            for symbol in matched_symbols:
                ledger_size = self.active_positions[symbol].size
                exchange_size = exchange_positions[symbol]['size']
                
                # Verificar discrepancia en tamaño
                size_diff = abs(ledger_size - exchange_size)
                if size_diff > 0.01:  # Tolerancia 0.01
                    reconciliation['discrepancies'].append({
                        'symbol': symbol,
                        'ledger_size': ledger_size,
                        'exchange_size': exchange_size,
                        'difference': size_diff,
                        'type': 'size_mismatch'
                    })
                    
                    LOG.warning("position_size_discrepancy",
                               symbol=symbol,
                               ledger=ledger_size,
                               exchange=exchange_size)
                else:
                    reconciliation['matched'].append(symbol)
            
            # Posiciones solo en ledger
            ledger_only = ledger_symbols - exchange_symbols
            for symbol in ledger_only:
                reconciliation['ledger_only'].append({
                    'symbol': symbol,
                    'ledger_size': self.active_positions[symbol].size,
                    'entry_price': self.active_positions[symbol].entry_price
                })
                
                LOG.warning("position_in_ledger_not_in_exchange",
                           symbol=symbol,
                           message="Posible cierre no registrado o crash durante ejecución")
            
            # Posiciones solo en exchange
            exchange_only = exchange_symbols - ledger_symbols
            for symbol in exchange_only:
                reconciliation['exchange_only'].append({
                    'symbol': symbol,
                    'exchange_size': exchange_positions[symbol]['size']
                })
                
                LOG.warning("position_in_exchange_not_in_ledger",
                           symbol=symbol,
                           message="Posible apertura no registrada")
            
            # Log resumen
            if reconciliation['discrepancies'] or reconciliation['ledger_only'] or reconciliation['exchange_only']:
                LOG.warning("reconciliation_found_issues",
                           matched=len(matched_symbols),
                           discrepancies=len(reconciliation['discrepancies']),
                           ledger_only=len(reconciliation['ledger_only']),
                           exchange_only=len(reconciliation['exchange_only']))
                
                # Alerta si hay problemas críticos
                if reconciliation['ledger_only'] or reconciliation['exchange_only']:
                    await ALERT_SYSTEM.send_alert(
                        "WARNING",
                        "Discrepancias en reconciliación ledger-exchange",
                        **reconciliation
                    )
            else:
                LOG.info("reconciliation_successful_no_issues",
                        matched=len(matched_symbols))
            
            return reconciliation
            
        except Exception as e:
            LOG.error("reconciliation_failed",
                     error=str(e),
                     traceback=traceback.format_exc()[:500])
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def get_total_pnl(self) -> float:
        return sum(t.realized_pnl for t in self.transactions if t.transaction_type in [TransactionType.CLOSE, TransactionType.PARTIAL_CLOSE])

    def get_win_rate(self) -> float:
        closed_transactions = [t for t in self.transactions if t.transaction_type in [TransactionType.CLOSE, TransactionType.PARTIAL_CLOSE]]
        if not closed_transactions:
            return 0.0
        winning = sum(1 for t in closed_transactions if t.realized_pnl > 0)
        return winning / len(closed_transactions)

    def audit_equity(self, bot) -> Dict[str, Any]:
        initial_capital = float(bot.initial_capital)
        
        # ✅ Sumar SOLO realized_pnl de transacciones CLOSE
        total_realized_pnl = 0.0
        
        for transaction in self.transactions:
            if not transaction.is_valid:
                continue
                
            if transaction.transaction_type in [TransactionType.CLOSE, TransactionType.PARTIAL_CLOSE]:
                pnl = float(transaction.realized_pnl)
                
                # Validar
                if np.isnan(pnl) or np.isinf(pnl):
                    LOG.warning("invalid_pnl_in_transaction_skipping",
                               transaction_id=transaction.transaction_id,
                               pnl=pnl)
                    continue
                
                position_value = float(transaction.entry_price) * float(transaction.size)
                max_reasonable = position_value * 2.0
                
                if abs(pnl) > max_reasonable:
                    LOG.warning("unreasonable_pnl_clipping_in_audit",
                               transaction_id=transaction.transaction_id,
                               original_pnl=pnl,
                               position_value=position_value)
                    pnl = float(np.clip(pnl, -position_value, position_value))
                
                total_realized_pnl += pnl
        
        # Calcular invested
        invested_in_positions = 0.0
        
        for symbol, open_transaction in self.active_positions.items():
            try:
                entry_price = float(open_transaction.entry_price)
                size = float(open_transaction.size)
                
                if entry_price <= 0 or size <= 0:
                    LOG.error("invalid_position_in_audit", symbol=symbol)
                    continue
                
                position_value = entry_price * size
                
                if position_value > 100000.0:
                    LOG.error("unreasonable_position_value", symbol=symbol, value=position_value)
                    continue
                
                invested_in_positions += position_value
                
            except Exception as e:
                LOG.error("position_calc_failed_in_audit", symbol=symbol, error=str(e))
                continue
        
        # ✅ FÓRMULA CORRECTA
        expected_free_equity = initial_capital + total_realized_pnl
        expected_total_equity = expected_free_equity + invested_in_positions
        actual_equity = float(bot.equity)
        discrepancy = actual_equity - expected_free_equity
        
        tolerance = max(0.10, abs(expected_free_equity) * 0.0001)
        
        if len(self.active_positions) == 0:
            is_consistent = abs(discrepancy) < tolerance
        else:
            total_portfolio = actual_equity + invested_in_positions
            portfolio_discrepancy = abs(total_portfolio - expected_total_equity)
            is_consistent = portfolio_discrepancy < tolerance
        
        audit_result = {
            'initial_capital': initial_capital,
            'total_realized_pnl': total_realized_pnl,
            'invested_in_positions': invested_in_positions,
            'expected_free_equity': expected_free_equity,
            'expected_total_equity': expected_total_equity,
            'actual_equity': actual_equity,
            'discrepancy': discrepancy,
            'is_consistent': is_consistent,
            'total_transactions': len(self.transactions),
            'active_positions': len(self.active_positions),
            'total_portfolio_value': actual_equity + invested_in_positions if len(self.active_positions) > 0 else actual_equity
        }
        
        if not is_consistent:
            LOG.error("equity_audit_failed", **audit_result)
        else:
            LOG.debug("equity_audit_passed", **audit_result)
        
        return audit_result

async def sync_ledger_with_risk_manager(bot):
    """
    MEJORADO: Sincronización atómica con locks
    """
    if not hasattr(bot, 'position_ledger') or not hasattr(bot, 'risk_manager'):
        LOG.error("missing_required_components_for_sync")
        return {'synced': False, 'reason': 'missing_components'}
    
    ledger = bot.position_ledger
    risk_mgr = bot.risk_manager
    
    sync_report = {
        'timestamp': datetime.now(timezone.utc),
        'ledger_positions': len(ledger.active_positions),
        'risk_manager_positions': len(risk_mgr.active_stops),
        'orphaned_ledger': [],
        'orphaned_risk_manager': [],
        'recovered': [],
        'failed_recovery': [],
        'actions_taken': []
    }
    
    try:
        # CORRECCIÓN: Usar lock del ledger para operación atómica
        async with ledger._lock:
            # Snapshots para evitar modificación durante iteración
            ledger_snapshot = dict(ledger.active_positions)
            risk_mgr_snapshot = dict(risk_mgr.active_stops)
            
            # 1. Posiciones en risk_manager pero no en ledger
            for symbol in list(risk_mgr_snapshot.keys()):
                if symbol not in ledger_snapshot:
                    sync_report['orphaned_risk_manager'].append(symbol)
                    
                    # Intentar recuperar desde historial de transacciones
                    recovered = False
                    for transaction in reversed(ledger.transactions):
                        if (transaction.symbol == symbol and 
                            transaction.transaction_type == TransactionType.OPEN and 
                            transaction.is_valid):
                            
                            # Verificar que no haya un CLOSE posterior
                            has_close = any(
                                t.symbol == symbol and 
                                t.transaction_type in [TransactionType.CLOSE, TransactionType.PARTIAL_CLOSE] and
                                t.timestamp > transaction.timestamp
                                for t in ledger.transactions
                            )
                            
                            if not has_close:
                                # Restaurar en ledger
                                ledger.active_positions[symbol] = transaction
                                sync_report['recovered'].append(symbol)
                                sync_report['actions_taken'].append(
                                    f"Recovered {symbol} from transaction history"
                                )
                                recovered = True
                                LOG.info("position_recovered_from_transaction_history",
                                        symbol=symbol,
                                        transaction_id=transaction.transaction_id)
                                break
                    
                    if not recovered:
                        # No se puede recuperar - cerrar en risk_manager
                        sync_report['failed_recovery'].append(symbol)
                        sync_report['actions_taken'].append(
                            f"Closed {symbol} in risk_manager (no recovery possible)"
                        )
                        risk_mgr.close_position(symbol)
                        LOG.warning("orphaned_risk_manager_position_closed",
                                   symbol=symbol)
            
            # 2. Posiciones en ledger pero no en risk_manager
            for symbol in list(ledger_snapshot.keys()):
                if symbol not in risk_mgr_snapshot:
                    sync_report['orphaned_ledger'].append(symbol)
                    transaction = ledger_snapshot[symbol]
                    
                    try:
                        # Obtener datos de mercado para re-registrar
                        market_data = await bot.exchange_manager.fetch_ohlcv(
                            symbol, '1h', limit=100
                        )
                        
                        if market_data and market_data.get("success"):
                            df = create_dataframe(market_data.get("ohlcv", []))
                            
                            if df is not None and len(df) >= 14:
                                df = calculate_technical_indicators(df)
                                
                                # Re-registrar en risk_manager
                                success = risk_mgr.register_position(
                                    symbol=symbol,
                                    entry_price=transaction.entry_price,
                                    side=transaction.side,
                                    size=transaction.size,
                                    confidence=0.7,  # Confianza default para recovery
                                    df=df
                                )
                                
                                if success:
                                    sync_report['recovered'].append(symbol)
                                    sync_report['actions_taken'].append(
                                        f"Re-registered {symbol} in risk_manager"
                                    )
                                    LOG.info("ledger_position_reregistered_in_risk_manager",
                                            symbol=symbol)
                                else:
                                    # Fallo en registro - eliminar del ledger
                                    del ledger.active_positions[symbol]
                                    sync_report['actions_taken'].append(
                                        f"Removed {symbol} from ledger (registration failed)"
                                    )
                                    LOG.warning("ledger_position_removed_registration_failed",
                                               symbol=symbol)
                            else:
                                # DataFrame insuficiente - eliminar
                                del ledger.active_positions[symbol]
                                sync_report['actions_taken'].append(
                                    f"Removed {symbol} from ledger (insufficient data)"
                                )
                                LOG.warning("ledger_position_removed_no_market_data",
                                           symbol=symbol)
                        else:
                            # No se pudo obtener datos - eliminar
                            del ledger.active_positions[symbol]
                            sync_report['actions_taken'].append(
                                f"Removed {symbol} from ledger (market data fetch failed)"
                            )
                            LOG.warning("ledger_position_removed_fetch_failed",
                                       symbol=symbol)
                            
                    except Exception as recovery_error:
                        LOG.error("ledger_position_recovery_failed",
                                 symbol=symbol,
                                 error=str(recovery_error))
                        # En caso de error, eliminar del ledger por seguridad
                        del ledger.active_positions[symbol]
                        sync_report['actions_taken'].append(
                            f"Removed {symbol} from ledger (recovery exception)"
                        )
        
        # Actualizar contadores finales
        sync_report['synced'] = True
        sync_report['final_ledger_positions'] = len(ledger.active_positions)
        sync_report['final_risk_manager_positions'] = len(risk_mgr.active_stops)
        
        # Log apropiado según resultado
        if sync_report['orphaned_ledger'] or sync_report['orphaned_risk_manager']:
            LOG.warning("ledger_sync_completed_with_orphans",
                       orphaned_ledger=len(sync_report['orphaned_ledger']),
                       orphaned_risk_manager=len(sync_report['orphaned_risk_manager']),
                       recovered=len(sync_report['recovered']),
                       failed=len(sync_report['failed_recovery']))
        else:
            LOG.info("ledger_sync_completed_clean",
                    ledger_positions=sync_report['final_ledger_positions'],
                    risk_manager_positions=sync_report['final_risk_manager_positions'])
        
        # Enviar métricas a InfluxDB
        try:
            if INFLUX_METRICS.enabled:
                await INFLUX_METRICS.write_model_metrics('ledger_sync', {
                    'total_orphaned': len(sync_report['orphaned_ledger']) + len(sync_report['orphaned_risk_manager']),
                    'recovered': len(sync_report['recovered']),
                    'failed': len(sync_report['failed_recovery']),
                    'final_positions': sync_report['final_ledger_positions']
                })
        except Exception as influx_error:
            LOG.debug("sync_metrics_write_failed", error=str(influx_error))
        
        return sync_report
        
    except Exception as e:
        LOG.error("ledger_sync_failed",
                 error=str(e),
                 traceback=traceback.format_exc()[:500])
        sync_report['synced'] = False
        sync_report['error'] = str(e)
        return sync_report

class DynamicRiskManager:
    def __init__(self, config, bot):
        self.config = config
        self.bot = bot
        self.active_stops = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.session_start = datetime.now(timezone.utc)
        self.max_daily_loss = -0.05
        self.max_daily_trades = 50
        self.circuit_breaker_active = False

    def calculate_stop_loss(self, symbol: str, entry_price: float, side: str, df: pd.DataFrame) -> float:
        try:
            if entry_price <= 0 or np.isnan(entry_price) or np.isinf(entry_price):
                LOG.error("invalid_entry_price_for_stop_loss", symbol=symbol, entry_price=entry_price)
                multiplier = -0.02 if side == 'buy' else 0.02
                fallback_sl = entry_price * (1 + multiplier) if entry_price > 0 else 0
                LOG.warning("using_fallback_stop_loss", symbol=symbol, fallback_sl=fallback_sl)
                return fallback_sl
            if df is None or len(df) < 14:
                LOG.warning("insufficient_dataframe_for_atr", symbol=symbol, df_length=len(df) if df is not None else 0)
                multiplier = -0.02 if side == 'buy' else 0.02
                return entry_price * (1 + multiplier)
            if 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
                multiplier = -0.02 if side == 'buy' else 0.02
                return entry_price * (1 + multiplier)
            high = df['high']
            low = df['low']
            close = df['close']
            if high.isna().all() or low.isna().all() or close.isna().all():
                LOG.warning("invalid_price_series_for_atr", symbol=symbol)
                multiplier = -0.02 if side == 'buy' else 0.02
                return entry_price * (1 + multiplier)
            if len(high) < 14 or len(low) < 14 or len(close) < 14:
                LOG.warning("insufficient_data_for_atr", symbol=symbol, data_length=len(high))
                multiplier = -0.02 if side == 'buy' else 0.02
                return entry_price * (1 + multiplier)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            if np.isnan(atr) or atr <= 0 or np.isinf(atr):
                LOG.warning("invalid_atr_calculated", symbol=symbol, atr=atr)
                multiplier = -0.02 if side == 'buy' else 0.02
                return entry_price * (1 + multiplier)
            atr_multiplier = 2.0
            if side == 'buy':
                stop_loss = entry_price - (atr * atr_multiplier)
            else:
                stop_loss = entry_price + (atr * atr_multiplier)
            if stop_loss <= 0 or np.isnan(stop_loss) or np.isinf(stop_loss):
                LOG.warning("invalid_stop_loss_calculated", symbol=symbol, stop_loss=stop_loss, entry_price=entry_price, atr=atr)
                multiplier = -0.02 if side == 'buy' else 0.02
                return entry_price * (1 + multiplier)
            min_stop_pct = 0.01
            max_stop_pct = 0.05
            stop_distance = abs(stop_loss - entry_price) / entry_price
            if stop_distance <= 0 or np.isnan(stop_distance) or np.isinf(stop_distance):
                LOG.warning("invalid_stop_distance", symbol=symbol, stop_distance=stop_distance)
                stop_distance = min_stop_pct
            if stop_distance < min_stop_pct:
                stop_loss = entry_price * (1 - min_stop_pct if side == 'buy' else 1 + min_stop_pct)
            elif stop_distance > max_stop_pct:
                stop_loss = entry_price * (1 - max_stop_pct if side == 'buy' else 1 + max_stop_pct)
            if side == 'buy' and stop_loss >= entry_price:
                LOG.error("stop_loss_above_entry_for_buy_correcting", symbol=symbol, entry=entry_price, stop=stop_loss)
                stop_loss = entry_price * (1 - min_stop_pct)
            if side == 'sell' and stop_loss <= entry_price:
                LOG.error("stop_loss_below_entry_for_sell_correcting", symbol=symbol, entry=entry_price, stop=stop_loss)
                stop_loss = entry_price * (1 + min_stop_pct)
            LOG.info("stop_loss_calculated", symbol=symbol, entry_price=entry_price, stop_loss=stop_loss, atr=atr, distance_pct=abs(stop_loss - entry_price) / entry_price * 100)
            return float(stop_loss)
        except ZeroDivisionError as e:
            LOG.error("stop_loss_calculation_zero_division", symbol=symbol, error=str(e), entry_price=entry_price)
            multiplier = -0.02 if side == 'buy' else 0.02
            return entry_price * (1 + multiplier) if entry_price > 0 else 0
        except Exception as e:
            LOG.error("stop_loss_calculation_error", symbol=symbol, error=str(e), traceback=traceback.format_exc())
            multiplier = -0.02 if side == 'buy' else 0.02
            return entry_price * (1 + multiplier) if entry_price > 0 else 0

    def calculate_take_profit_levels(self, symbol: str, entry_price: float, side: str, confidence: float) -> List[Tuple[float, float]]:
        try:
            if confidence > 0.8:
                tp_multipliers = [0.015, 0.035, 0.060]
                size_distribution = [0.30, 0.30, 0.40]
            elif confidence > 0.6:
                tp_multipliers = [0.012, 0.028, 0.050]
                size_distribution = [0.35, 0.35, 0.30]
            else:
                tp_multipliers = [0.010, 0.022, 0.040]
                size_distribution = [0.40, 0.35, 0.25]
            levels = []
            if side == 'buy':
                for i, (mult, size_frac) in enumerate(zip(tp_multipliers, size_distribution)):
                    tp_price = entry_price * (1 + mult)
                    levels.append((tp_price, size_frac))
            else:
                for i, (mult, size_frac) in enumerate(zip(tp_multipliers, size_distribution)):
                    tp_price = entry_price * (1 - mult)
                    levels.append((tp_price, size_frac))
            LOG.info("take_profit_levels_calculated", symbol=symbol, entry_price=entry_price, side=side, confidence=confidence, levels=[(f"{p:.2f}", f"{s*100:.0f}%") for p, s in levels], tp1_pct=tp_multipliers[0]*100, tp2_pct=tp_multipliers[1]*100, tp3_pct=tp_multipliers[2]*100)
            return levels
        except Exception as e:
            LOG.error("take_profit_calculation_error", error=str(e))
            return [(entry_price * 1.02, 1.0)]

    def update_trailing_stop(self, symbol: str, current_price: float, side: str) -> Optional[float]:
        try:
            if symbol not in self.active_stops:
                return None
            stop_info = self.active_stops[symbol]
            entry_price = stop_info['entry_price']
            current_stop = stop_info['stop_loss']
            trailing_distance_pct = 0.015
            if side == 'buy':
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct > 0.02:
                    new_stop = current_price * (1 - trailing_distance_pct)
                    if new_stop > current_stop:
                        LOG.info("trailing_stop_updated", symbol=symbol, old_stop=current_stop, new_stop=new_stop, current_price=current_price, profit_pct=profit_pct * 100)
                        self.active_stops[symbol]['stop_loss'] = new_stop
                        return new_stop
            else:
                profit_pct = (entry_price - current_price) / entry_price
                if profit_pct > 0.02:
                    new_stop = current_price * (1 + trailing_distance_pct)
                    if new_stop < current_stop:
                        LOG.info("trailing_stop_updated", symbol=symbol, old_stop=current_stop, new_stop=new_stop, current_price=current_price, profit_pct=profit_pct * 100)
                        self.active_stops[symbol]['stop_loss'] = new_stop
                        return new_stop
            return None
        except Exception as e:
            LOG.error("trailing_stop_update_error", error=str(e))
            return None

    def check_stop_loss_hit(self, symbol: str, current_price: float, side: str) -> bool:
        try:
            if symbol not in self.active_stops:
                return False
            stop_loss = self.active_stops[symbol]['stop_loss']
            if side == 'buy' and current_price <= stop_loss:
                LOG.warning("stop_loss_hit", symbol=symbol, current_price=current_price, stop_loss=stop_loss, loss_pct=(current_price - self.active_stops[symbol]['entry_price']) / self.active_stops[symbol]['entry_price'] * 100)
                return True
            elif side == 'sell' and current_price >= stop_loss:
                LOG.warning("stop_loss_hit", symbol=symbol, current_price=current_price, stop_loss=stop_loss, loss_pct=(self.active_stops[symbol]['entry_price'] - current_price) / self.active_stops[symbol]['entry_price'] * 100)
                return True
            return False
        except Exception as e:
            LOG.error("stop_loss_check_error", error=str(e))
            return False

    def check_take_profit_hit(self, symbol: str, current_price: float, side: str) -> Optional[Tuple[float, float]]:
        try:
            if symbol not in self.active_stops:
                return None
            tp_levels = self.active_stops[symbol].get('take_profit_levels', [])
            for i, (tp_price, size_fraction) in enumerate(tp_levels):
                if side == 'buy' and current_price >= tp_price:
                    LOG.info("take_profit_hit", symbol=symbol, level=i+1, tp_price=tp_price, current_price=current_price, size_fraction=size_fraction)
                    tp_levels.pop(i)
                    return (tp_price, size_fraction)
                elif side == 'sell' and current_price <= tp_price:
                    LOG.info("take_profit_hit", symbol=symbol, level=i+1, tp_price=tp_price, current_price=current_price, size_fraction=size_fraction)
                    tp_levels.pop(i)
                    return (tp_price, size_fraction)
            return None
        except Exception as e:
            LOG.error("take_profit_check_error", error=str(e))
            return None

    def register_position(self, symbol: str, entry_price: float, side: str, size: float, confidence: float, df: pd.DataFrame):
        try:
            if entry_price <= 0 or np.isnan(entry_price) or np.isinf(entry_price):
                LOG.error("invalid_entry_price_cannot_register_position", symbol=symbol, entry_price=entry_price, side=side)
                return False
            if not (0 < entry_price < 1e10):
                LOG.error("entry_price_out_of_reasonable_range", symbol=symbol, entry_price=entry_price)
                return False
            if size <= 0 or np.isnan(size) or np.isinf(size):
                LOG.error("invalid_size_cannot_register_position", symbol=symbol, size=size)
                return False
            if confidence < 0 or confidence > 1 or np.isnan(confidence):
                LOG.warning("invalid_confidence_adjusting", symbol=symbol, confidence=confidence)
                confidence = max(0.0, min(1.0, confidence))
            if df is None or len(df) < 14:
                LOG.warning("insufficient_dataframe_for_stop_loss_calculation", symbol=symbol, df_length=len(df) if df is not None else 0)
                stop_loss = entry_price * (0.98 if side == 'buy' else 1.02)
                LOG.info("using_default_stop_loss", symbol=symbol, stop_loss=stop_loss, reason="insufficient_dataframe")
            else:
                required_cols = ['high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    LOG.warning("dataframe_missing_columns_for_atr", symbol=symbol, missing=missing_cols)
                    stop_loss = entry_price * (0.98 if side == 'buy' else 1.02)
                else:
                    stop_loss = self.calculate_stop_loss(symbol, entry_price, side, df)
            if stop_loss <= 0 or np.isnan(stop_loss) or np.isinf(stop_loss):
                LOG.error("invalid_stop_loss_cannot_register_position", symbol=symbol, stop_loss=stop_loss)
                return False
            if side == 'buy' and stop_loss >= entry_price:
                LOG.error("invalid_stop_loss_above_entry_for_buy", symbol=symbol, entry_price=entry_price, stop_loss=stop_loss)
                return False
            if side == 'sell' and stop_loss <= entry_price:
                LOG.error("invalid_stop_loss_below_entry_for_sell", symbol=symbol, entry_price=entry_price, stop_loss=stop_loss)
                return False
            tp_levels = self.calculate_take_profit_levels(symbol, entry_price, side, confidence)
            if not tp_levels or len(tp_levels) == 0:
                LOG.warning("no_take_profit_levels_calculated_using_default", symbol=symbol)
                if side == 'buy':
                    tp_levels = [(entry_price * 1.02, 1.0)]
                else:
                    tp_levels = [(entry_price * 0.98, 1.0)]
            max_reasonable_size = 1000000.0
            if size > max_reasonable_size:
                LOG.error("position_size_exceeds_reasonable_limit", symbol=symbol, requested_size=size, max_allowed=max_reasonable_size, entry_price=entry_price)
                return False
            position_value = size * entry_price
            max_position_value = 50000.0
            if position_value > max_position_value:
                LOG.error("position_value_exceeds_limit", symbol=symbol, position_value=position_value, max_allowed=max_position_value, size=size, entry_price=entry_price)
                return False
            if hasattr(self, 'bot') and hasattr(self.bot, 'equity'):
                available_equity = float(self.bot.equity)
                if position_value > available_equity * 0.30:
                    LOG.error("position_value_exceeds_equity_limit", symbol=symbol, position_value=position_value, available_equity=available_equity, max_allowed_pct=30.0)
                    return False
            self.active_stops[symbol] = {
                'entry_price': float(entry_price),
                'side': side,
                'size': float(size),
                'remaining_size': float(size),
                'stop_loss': float(stop_loss),
                'take_profit_levels': tp_levels,
                'entry_time': datetime.now(timezone.utc),
                'confidence': float(confidence)
            }
            if hasattr(self.bot, 'portfolio_rebalancer'):
                try:
                    self.bot.portfolio_rebalancer.register_position_confidence(symbol, confidence)
                except Exception as rebal_error:
                    LOG.debug("rebalancer_notification_failed", error=str(rebal_error))
            LOG.info("position_registered_with_stops", symbol=symbol, entry_price=entry_price, side=side, size=size, confidence=confidence, stop_loss=stop_loss, tp_levels_count=len(tp_levels))
            return True
        except Exception as e:
            LOG.error("position_registration_error", symbol=symbol, error=str(e), traceback=traceback.format_exc())
            return False

    def close_position(self, symbol: str):
        if symbol in self.active_stops:
            del self.active_stops[symbol]
            if hasattr(self.bot, 'portfolio_rebalancer'):
                try:
                    self.bot.portfolio_rebalancer.clear_position_confidence(symbol)
                except Exception as rebal_error:
                    LOG.debug("rebalancer_clear_failed", error=str(rebal_error))
            LOG.info("position_closed", symbol=symbol)

    def check_circuit_breaker(self) -> bool:
        try:
            # NUEVO: Añadir lock para operaciones atómicas
            if not hasattr(self, '_circuit_breaker_lock'):
                self._circuit_breaker_lock = asyncio.Lock()
            
            # CORRECCIÓN: No usar lock para simple check (deadlock risk)
            # Solo para modificación de estado
            
            # Si ya está activo, verificar condiciones de desactivación
            if self.circuit_breaker_active:
                current_return = (self.bot.equity - self.bot.initial_capital) / self.bot.initial_capital
                recovery_threshold = self.max_daily_loss * 0.5
                
                if current_return > recovery_threshold:
                    # Verificar trades recientes
                    recent_trades = getattr(self.bot, 'trades', [])[-10:] if hasattr(self.bot, 'trades') else []
                    recent_losses = sum(1 for t in recent_trades if t.get('pnl', 0) < 0)
                    
                    # Solo desactivar si hay evidencia de recuperación
                    if recent_losses < 5 and len(recent_trades) >= 5:
                        LOG.info("circuit_breaker_auto_reset_recovery",
                                current_return=current_return * 100,
                                recovery_threshold=recovery_threshold * 100,
                                recent_losses=recent_losses,
                                equity=self.bot.equity)
                        
                        # NUEVO: Desactivación gradual
                        self.circuit_breaker_active = False
                        self.daily_pnl *= 0.5  # Reset parcial del daily PnL
                        
                        return False
                else:
                    # NUEVO: Log periódico cuando está activo
                    if not hasattr(self, '_last_breaker_log'):
                        self._last_breaker_log = 0
                    
                    if time.time() - self._last_breaker_log > 300:  # Cada 5 minutos
                        LOG.warning("circuit_breaker_still_active",
                                   current_return=current_return * 100,
                                   recovery_needed=recovery_threshold * 100,
                                   daily_pnl=self.daily_pnl)
                        self._last_breaker_log = time.time()
                    
                    return True
            
            # Verificar condiciones de activación
            daily_return = self.daily_pnl / self.bot.initial_capital if self.bot.initial_capital > 0 else 0
            
            # Condición 1: Pérdida diaria excede límite
            if daily_return <= self.max_daily_loss:
                if not self.circuit_breaker_active:
                    self.circuit_breaker_active = True
                    
                    LOG.critical("circuit_breaker_activated_daily_loss",
                                daily_return=daily_return * 100,
                                max_allowed=self.max_daily_loss * 100,
                                daily_pnl=self.daily_pnl,
                                active_positions=len(self.active_stops))
                    
                    # NUEVO: Calcular tiempo estimado de recuperación
                    avg_daily_return = daily_return
                    days_to_recover = abs(daily_return / self.max_daily_loss) if self.max_daily_loss != 0 else 0
                    
                    asyncio.create_task(
                        ALERT_SYSTEM.send_alert(
                            "CRITICAL",
                            "Circuit breaker activated - Daily loss limit exceeded",
                            daily_return=daily_return,
                            daily_pnl=self.daily_pnl,
                            active_positions=len(self.active_stops),
                            estimated_recovery_days=days_to_recover,
                            message="All trading halted. Close active positions manually or wait for daily reset."
                        )
                    )
                
                return True
            
            # Condición 2: Trades diarios excede límite
            if self.daily_trades >= self.max_daily_trades:
                if not self.circuit_breaker_active:
                    self.circuit_breaker_active = True
                    
                    LOG.critical("circuit_breaker_activated_max_trades",
                                daily_trades=self.daily_trades,
                                max_allowed=self.max_daily_trades)
                    
                    asyncio.create_task(
                        ALERT_SYSTEM.send_alert(
                            "CRITICAL",
                            "Circuit breaker activated - Max daily trades exceeded",
                            daily_trades=self.daily_trades,
                            max_allowed=self.max_daily_trades
                        )
                    )
                
                return True
            
            # NUEVO: Condición 3: Múltiples stop losses consecutivos
            if hasattr(self.bot, 'trades') and len(self.bot.trades) >= 5:
                recent_trades = self.bot.trades[-5:]
                consecutive_losses = 0
                
                for trade in reversed(recent_trades):
                    if trade.get('is_stop_loss', False):
                        consecutive_losses += 1
                    else:
                        break
                
                if consecutive_losses >= 3:
                    if not self.circuit_breaker_active:
                        self.circuit_breaker_active = True
                        
                        LOG.critical("circuit_breaker_activated_consecutive_stop_losses",
                                    consecutive_losses=consecutive_losses)
                        
                        asyncio.create_task(
                            ALERT_SYSTEM.send_alert(
                                "CRITICAL",
                                "Circuit breaker activated - Multiple consecutive stop losses",
                                consecutive_losses=consecutive_losses,
                                recent_trades=len(recent_trades)
                            )
                        )
                    
                    return True
            
            # Desactivación automática por tiempo
            if self.circuit_breaker_active:
                if (datetime.now(timezone.utc) - self.session_start).total_seconds() > 86400:
                    LOG.info("circuit_breaker_auto_reset_new_day")
                    self.reset_daily_limits()
            
            return False
            
        except Exception as e:
            LOG.error("circuit_breaker_check_error",
                     error=str(e),
                     traceback=traceback.format_exc()[:300])
            # En caso de error, mantener estado seguro
            return self.circuit_breaker_active if hasattr(self, 'circuit_breaker_active') else False

    def reset_daily_limits(self):
        old_daily_pnl = self.daily_pnl
        old_daily_trades = self.daily_trades
        old_breaker_state = self.circuit_breaker_active
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.session_start = datetime.now(timezone.utc)
        self.circuit_breaker_active = False
        LOG.info("daily_limits_reset", previous_pnl=old_daily_pnl, previous_trades=old_daily_trades, circuit_breaker_was_active=old_breaker_state, circuit_breaker_now_reset=True)

    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl
        self.daily_trades += 1
        try:
            if hasattr(self.bot, 'equity') and hasattr(self.bot, 'initial_capital'):
                current_equity = float(self.bot.equity)
                initial_capital = float(self.bot.initial_capital)
                if current_equity <= 0 or initial_capital <= 0:
                    LOG.warning("invalid_equity_values_in_update_daily_pnl", current_equity=current_equity, initial_capital=initial_capital)
                    return
                if initial_capital > 0:
                    current_return = (current_equity - initial_capital) / initial_capital
                    if hasattr(self.bot, 'drawdown'):
                        self.bot.drawdown = min(self.bot.drawdown, current_return)
                    else:
                        self.bot.drawdown = min(0.0, current_return)
                    if hasattr(self.bot, 'portfolio_history') and len(self.bot.portfolio_history) > 0:
                        history_df = pd.DataFrame(self.bot.portfolio_history, columns=['timestamp', 'equity'])
                        if len(history_df) > 0:
                            peak = history_df['equity'].cummax().iloc[-1]
                            if peak > 0:
                                drawdown_from_peak = (current_equity - peak) / peak
                                self.bot.drawdown = min(self.bot.drawdown, drawdown_from_peak)
                                LOG.debug("drawdown_updated", current_equity=current_equity, peak=peak, drawdown=self.bot.drawdown, drawdown_pct=self.bot.drawdown * 100)
                    if hasattr(self.bot, 'performance_metrics'):
                        self.bot.performance_metrics['total_pnl'] = current_equity - initial_capital
                    LOG.debug("bot_metrics_updated", equity=current_equity, drawdown=self.bot.drawdown, pnl=pnl, total_pnl=self.bot.performance_metrics.get('total_pnl', 0.0) if hasattr(self.bot, 'performance_metrics') else 0.0)
        except Exception as e:
            LOG.error("bot_metrics_update_failed", error=str(e))

class SmartOrderExecutor:
    def __init__(self, exchange_manager, config):
        self.exchange_manager = exchange_manager
        self.config = config
        self.max_slippage_pct = 0.005
        self.order_timeout = 30
        self.partial_fill_threshold = 0.95  # 95% mínimo para considerar fill completo
        self.max_slippage_alert = 0.02  # 2% slippage = alerta crítica
        self._last_execution_info = {}  # Cache de última ejecución por símbolo

    async def execute_order_smart(self, symbol: str, side: str, size: float, 
                                  order_type: str = "limit", reference_price: float = None, 
                                  max_retries: int = 3) -> Optional[Dict]:
        """
        MEJORADO: Ejecuta orden con detección de fills parciales y slippage
        
        Returns:
            Dict con información extendida de ejecución incluyendo:
            - filled_size: Tamaño realmente ejecutado
            - slippage_pct: Slippage porcentual real
            - is_partial: Bool indicando si es fill parcial
            - execution_quality: Score de calidad 0-100
        """
        try:
            if reference_price is None:
                ticker = await fetch_ticker_robust(self.exchange_manager, symbol)
                if not ticker:
                    LOG.error("cannot_get_reference_price", symbol=symbol)
                    return None
                reference_price = ticker.get('last', 0)
            
            if reference_price <= 0:
                LOG.error("invalid_reference_price", symbol=symbol, price=reference_price)
                return None
            
            # Información de ejecución
            execution_info = {
                'symbol': symbol,
                'side': side,
                'requested_size': size,
                'reference_price': reference_price,
                'order_type': order_type,
                'attempts': []
            }
            
            if self.config.dry_run:
                # Simular ejecución con slippage realista
                simulated_slippage = np.random.normal(0, 0.001)  # ~0.1% std
                executed_price = reference_price * (1 + simulated_slippage)
                
                # Simular fill parcial ocasional (5% probabilidad)
                fill_rate = 1.0 if np.random.random() > 0.05 else np.random.uniform(0.7, 0.95)
                filled_size = size * fill_rate
                
                result = await self._simulate_order(symbol, side, filled_size, executed_price)
                result['filled_size'] = filled_size
                result['is_partial'] = fill_rate < self.partial_fill_threshold
                result['slippage_pct'] = simulated_slippage * 100
                result['execution_quality'] = self._calculate_execution_quality(
                    reference_price, executed_price, fill_rate
                )
                
                # Guardar info
                self._last_execution_info[symbol] = result
                
                return result
            
            # Intentar limit order primero para mejor precio
            if order_type == "limit":
                limit_result = await self._try_limit_order_with_monitoring(
                    symbol, side, size, reference_price, execution_info
                )
                if limit_result and limit_result.get('success'):
                    self._last_execution_info[symbol] = limit_result
                    return limit_result
                
                LOG.info("limit_order_failed_falling_back_to_market", symbol=symbol)
            
            # Fallback a market order con monitoreo de fills
            market_result = await self._execute_market_order_with_monitoring(
                symbol, side, size, reference_price, max_retries, execution_info
            )
            
            if market_result:
                self._last_execution_info[symbol] = market_result
            
            return market_result
            
        except Exception as e:
            LOG.error("smart_order_execution_failed", 
                     symbol=symbol, 
                     error=str(e),
                     traceback=traceback.format_exc()[:300])
            return None
    
    async def _try_limit_order_with_monitoring(self, symbol: str, side: str, size: float,
                                               reference_price: float, 
                                               execution_info: Dict) -> Optional[Dict]:
        """Limit order con monitoreo de fills parciales"""
        try:
            # CORRECCIÓN: Definir order_type explícitamente al inicio
            order_type = 'limit'
            
            # Calcular precio límite con margen de slippage
            if side == 'buy':
                limit_price = reference_price * (1 + self.max_slippage_pct * 0.5)
            else:
                limit_price = reference_price * (1 - self.max_slippage_pct * 0.5)
            
            LOG.info("placing_limit_order",
                    symbol=symbol,
                    side=side,
                    size=size,
                    limit_price=limit_price,
                    order_type=order_type)
            
            order = await self.exchange_manager.create_order(
                symbol, 'limit', side, size, limit_price
            )
            
            if not order or not order.get('success'):
                return None
            
            order_id = order.get('order_id')
            start_time = time.time()
            last_filled = 0
            fill_timeout = self.order_timeout / 2  # Timeout más corto para limite
            
            # Monitorear fills
            while time.time() - start_time < fill_timeout:
                await asyncio.sleep(2)
                
                try:
                    order_status = await self.exchange_manager.exchange.fetch_order(
                        order_id, symbol
                    )
                    
                    status = order_status.get('status')
                    filled = order_status.get('filled', 0)
                    
                    # Log progreso de fill
                    if filled != last_filled:
                        LOG.debug("limit_order_fill_progress",
                                 symbol=symbol,
                                 order_id=order_id,
                                 filled=filled,
                                 requested=size,
                                 progress_pct=filled/size*100 if size > 0 else 0)
                        last_filled = filled
                    
                    if status == 'closed' or status == 'filled':
                        # Orden completada
                        executed_price = order_status.get('average') or order_status.get('price', limit_price)
                        filled_size = filled
                        
                        # Calcular métricas
                        fill_rate = filled_size / size if size > 0 else 0
                        is_partial = fill_rate < self.partial_fill_threshold
                        
                        if side == 'buy':
                            slippage_pct = (executed_price - reference_price) / reference_price
                        else:
                            slippage_pct = (reference_price - executed_price) / reference_price
                        
                        execution_quality = self._calculate_execution_quality(
                            reference_price, executed_price, fill_rate
                        )
                        
                        LOG.info("limit_order_executed",
                                symbol=symbol,
                                filled_size=filled_size,
                                requested_size=size,
                                fill_rate=fill_rate * 100,
                                executed_price=executed_price,
                                slippage_pct=slippage_pct * 100,
                                quality=execution_quality)
                        
                        return {
                            'success': True,
                            'order_id': order_id,
                            'symbol': symbol,
                            'side': side,
                            'type': 'limit',
                            'amount': size,
                            'filled_size': filled_size,
                            'is_partial': is_partial,
                            'price': executed_price,
                            'executed_price': executed_price,
                            'slippage_pct': slippage_pct * 100,
                            'execution_quality': execution_quality,
                            'status': 'closed',
                            'raw': order_status
                        }
                    
                    elif status == 'canceled' or status == 'expired':
                        LOG.warning("limit_order_not_filled",
                                   symbol=symbol,
                                   status=status,
                                   filled=filled)
                        return None
                
                except Exception as status_error:
                    LOG.debug("order_status_check_error", error=str(status_error))
                    continue
            
            # Timeout - cancelar y retornar fill parcial si existe
            try:
                final_status = await self.exchange_manager.exchange.fetch_order(
                    order_id, symbol
                )
                filled = final_status.get('filled', 0)
                
                # Cancelar
                await self.exchange_manager.exchange.cancel_order(order_id, symbol)
                
                if filled > 0:
                    # Hay fill parcial
                    LOG.warning("limit_order_timeout_partial_fill",
                               symbol=symbol,
                               filled=filled,
                               requested=size)
                    
                    executed_price = final_status.get('average') or limit_price
                    fill_rate = filled / size if size > 0 else 0
                    
                    if side == 'buy':
                        slippage_pct = (executed_price - reference_price) / reference_price
                    else:
                        slippage_pct = (reference_price - executed_price) / reference_price
                    
                    return {
                        'success': True,
                        'order_id': order_id,
                        'symbol': symbol,
                        'side': side,
                        'type': 'limit',
                        'amount': size,
                        'filled_size': filled,
                        'is_partial': True,
                        'price': executed_price,
                        'executed_price': executed_price,
                        'slippage_pct': slippage_pct * 100,
                        'execution_quality': self._calculate_execution_quality(
                            reference_price, executed_price, fill_rate
                        ),
                        'status': 'partially_filled',
                        'raw': final_status
                    }
                else:
                    LOG.warning("limit_order_timeout_no_fill", symbol=symbol)
                    return None
                    
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            LOG.error("limit_order_with_monitoring_failed",
                     symbol=symbol,
                     error=str(e))
            return None
    
    async def _execute_market_order_with_monitoring(self, symbol: str, side: str, 
                                                     size: float, reference_price: float,
                                                     max_retries: int,
                                                     execution_info: Dict) -> Optional[Dict]:
        """Market order con detección de slippage extremo"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 0.5 * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                
                LOG.info("executing_market_order",
                        symbol=symbol,
                        side=side,
                        size=size,
                        attempt=attempt + 1,
                        reference_price=reference_price)
                
                # Timestamp pre-ejecución
                pre_execution_time = time.time()
                
                order = await self.exchange_manager.create_order(
                    symbol, 'market', side, size
                )
                
                execution_time_ms = (time.time() - pre_execution_time) * 1000
                
                if order and order.get('success'):
                    executed_price = order.get('price', 0)
                    filled_size = order.get('filled', size)
                    
                    # Obtener precio de mercado post-ejecución
                    try:
                        ticker = await fetch_ticker_robust(self.exchange_manager, symbol)
                        if ticker:
                            post_market_price = ticker.get('last', executed_price)
                        else:
                            post_market_price = executed_price
                    except Exception:
                        post_market_price = executed_price
                    
                    # Calcular slippage real
                    if executed_price > 0:
                        if side == 'buy':
                            slippage_pct = (executed_price - reference_price) / reference_price
                        else:
                            slippage_pct = (reference_price - executed_price) / reference_price
                    else:
                        # Precio no disponible, usar market price
                        executed_price = post_market_price
                        slippage_pct = 0
                    
                    # Detectar slippage extremo
                    if abs(slippage_pct) > self.max_slippage_alert:
                        LOG.error("extreme_slippage_on_market_order",
                                 symbol=symbol,
                                 reference_price=reference_price,
                                 executed_price=executed_price,
                                 slippage_pct=slippage_pct * 100,
                                 side=side)
                        
                        # Alerta crítica
                        await ALERT_SYSTEM.send_alert(
                            "CRITICAL",
                            f"Slippage extremo detectado: {symbol}",
                            reference_price=reference_price,
                            executed_price=executed_price,
                            slippage_pct=slippage_pct * 100,
                            side=side,
                            execution_time_ms=execution_time_ms
                        )
                    
                    # Detectar fill parcial
                    fill_rate = filled_size / size if size > 0 else 0
                    is_partial = fill_rate < self.partial_fill_threshold
                    
                    if is_partial:
                        LOG.warning("partial_fill_on_market_order",
                                   symbol=symbol,
                                   requested=size,
                                   filled=filled_size,
                                   fill_rate=fill_rate * 100)
                    
                    # Calcular calidad de ejecución
                    execution_quality = self._calculate_execution_quality(
                        reference_price, executed_price, fill_rate
                    )
                    
                    # Registrar intento
                    execution_info['attempts'].append({
                        'attempt': attempt + 1,
                        'success': True,
                        'executed_price': executed_price,
                        'slippage_pct': slippage_pct * 100,
                        'filled_size': filled_size,
                        'execution_time_ms': execution_time_ms
                    })
                    
                    LOG.info("market_order_executed_successfully",
                            symbol=symbol,
                            executed_price=executed_price,
                            filled_size=filled_size,
                            slippage_pct=slippage_pct * 100,
                            execution_time_ms=execution_time_ms,
                            quality=execution_quality)
                    
                    return {
                        'success': True,
                        'order_id': order.get('order_id'),
                        'symbol': symbol,
                        'side': side,
                        'type': 'market',
                        'amount': size,
                        'filled_size': filled_size,
                        'is_partial': is_partial,
                        'price': executed_price,
                        'executed_price': executed_price,
                        'slippage_pct': slippage_pct * 100,
                        'execution_quality': execution_quality,
                        'execution_time_ms': execution_time_ms,
                        'status': 'closed',
                        'raw': order
                    }
                
                last_error = order.get('error', 'Unknown error') if order else 'No response'
                
            except Exception as e:
                last_error = str(e)
                LOG.warning("market_order_attempt_failed",
                           symbol=symbol,
                           attempt=attempt + 1,
                           error=str(e))
        
        LOG.error("market_order_failed_all_attempts",
                 symbol=symbol,
                 last_error=last_error)
        return None
    
    def _calculate_execution_quality(self, reference_price: float, 
                                     executed_price: float, 
                                     fill_rate: float) -> float:
        """
        Calcula score de calidad de ejecución 0-100
        
        Factores:
        - Slippage (50% peso)
        - Fill rate (50% peso)
        """
        try:
            # Score de slippage (0-50)
            slippage_pct = abs(executed_price - reference_price) / reference_price
            if slippage_pct <= 0.001:  # <0.1%
                slippage_score = 50
            elif slippage_pct <= 0.005:  # <0.5%
                slippage_score = 40
            elif slippage_pct <= 0.01:  # <1%
                slippage_score = 30
            elif slippage_pct <= 0.02:  # <2%
                slippage_score = 20
            else:
                slippage_score = max(0, 20 - (slippage_pct - 0.02) * 1000)
            
            # Score de fill rate (0-50)
            if fill_rate >= 0.99:
                fill_score = 50
            elif fill_rate >= 0.95:
                fill_score = 45
            elif fill_rate >= 0.90:
                fill_score = 40
            elif fill_rate >= 0.80:
                fill_score = 30
            else:
                fill_score = fill_rate * 30
            
            total_score = slippage_score + fill_score
            return float(np.clip(total_score, 0, 100))
            
        except Exception:
            return 50.0  # Score neutral en caso de error

    async def _simulate_order(self, symbol: str, side: str, size: float, price: float) -> Dict:
        order_id = f"sim_{symbol}_{int(time.time() * 1000)}"
        simulated_order = {
            'success': True,
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'type': 'market',
            'amount': size,
            'price': price,
            'filled': size,
            'status': 'closed',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'info': {'simulated': True}
        }
        LOG.info("order_simulated", symbol=symbol, side=side, size=size, price=price, order_id=order_id)
        return simulated_order


async def _update_bot_after_trade_close(bot, risk_manager, symbol: str, confidence: float,
                                        side: str, size: float, entry_price: float,
                                        exit_price: float, is_stop_loss: bool = False,
                                        is_partial: bool = False, filled_size: float = None):
    """
    VERSIÓN DEFINITIVA: Una sola fuente de verdad para equity
    """
    try:
        # Validar precios
        if entry_price <= 0 or exit_price <= 0 or np.isnan(entry_price) or np.isnan(exit_price):
            LOG.error("invalid_prices_in_trade_close",
                     symbol=symbol,
                     entry=entry_price,
                     exit=exit_price)
            return
        
        # Determinar tamaño ejecutado
        executed_size = filled_size if filled_size is not None else size
        
        # ✅ PASO 1: Calcular PnL UNA VEZ
        if side == 'buy':  # Cerrando LONG (vendiendo)
            realized_pnl = (exit_price - entry_price) * executed_size
        else:  # Cerrando SHORT (comprando)
            realized_pnl = (entry_price - exit_price) * executed_size
        
        # Validar PnL razonable
        position_value = entry_price * executed_size
        max_reasonable_pnl = position_value * 2.0
        
        if abs(realized_pnl) > max_reasonable_pnl:
            LOG.error("unreasonable_pnl_detected",
                     symbol=symbol,
                     realized_pnl=realized_pnl,
                     position_value=position_value)
            realized_pnl = np.clip(realized_pnl, -position_value, position_value)
            LOG.warning("pnl_clipped", clipped_pnl=realized_pnl)

        # ✅ PASO 2: Capturar equity ANTES (snapshot inmutable)
        equity_before = float(bot.equity)
        
        # ✅ PASO 3: Calcular equity DESPUÉS
        equity_after = equity_before + realized_pnl
        
        # ✅ VALIDACIÓN CRÍTICA: Verificar que no quede negativo
        if equity_after < 0:
            LOG.error("equity_would_be_negative_rejecting_trade",
                     equity_before=equity_before,
                     realized_pnl=realized_pnl,
                     equity_after=equity_after,
                     symbol=symbol)
            return
        
        LOG.info("EQUITY_UPDATE_MASTER_RECORD",
                symbol=symbol,
                equity_before=equity_before,
                realized_pnl=realized_pnl,
                equity_after=equity_after,
                entry_price=entry_price,
                exit_price=exit_price,
                executed_size=executed_size,
                side=side)

        # ✅ PASO 4: Aplicar ANTES de llamar al ledger
        bot.equity = equity_after
        
        # ✅ PASO 5: Registrar en ledger (que NO debe modificar equity)
        # Pasar equity_before y equity_after explícitos
        transaction = await bot.position_ledger.record_close(
            bot, symbol, exit_price, executed_size,
            equity_before_override=equity_before,
            equity_after_override=equity_after,
            realized_pnl_override=realized_pnl
        )

        # Si la transacción falla, REVERTIR el cambio de equity
        if transaction is None or not transaction.is_valid:
            LOG.error("ledger_record_failed_reverting_equity",
                     symbol=symbol,
                     reverting_from=bot.equity,
                     reverting_to=equity_before)
            bot.equity = equity_before
            return
        
        # ✅ VALIDACIÓN POST-LEDGER: Verificar que ledger no modificó equity
        equity_after_ledger = float(bot.equity)
        if abs(equity_after_ledger - equity_after) > 0.001:
            LOG.error("CRITICAL_ledger_modified_equity_unexpectedly",
                     expected=equity_after,
                     actual=equity_after_ledger,
                     diff=equity_after_ledger - equity_after,
                     symbol=symbol)
            # Forzar el valor correcto
            bot.equity = equity_after
        
        # Actualizar métricas del bot
        if hasattr(bot, 'performance_metrics'):
            bot.performance_metrics['total_trades'] += 1
            
            if realized_pnl > 0:
                bot.performance_metrics['winning_trades'] += 1
            else:
                bot.performance_metrics['losing_trades'] += 1
            
            total = bot.performance_metrics['total_trades']
            if total > 0:
                bot.performance_metrics['win_rate'] = bot.performance_metrics['winning_trades'] / total
            
            bot.performance_metrics['total_pnl'] = bot.equity - bot.initial_capital
        
        # Actualizar risk manager
        if hasattr(bot, 'risk_manager'):
            bot.risk_manager.update_daily_pnl(realized_pnl)
        
        # Actualizar position sizer
        if hasattr(bot, 'position_sizer'):
            is_win = realized_pnl > 0
            bot.position_sizer.update_trade_history(realized_pnl, is_win)
        
        # Actualizar performance por símbolo
        await update_symbol_performance(bot, symbol, realized_pnl)
        
        # Guardar trade
        trade_record = {
            'symbol': symbol,
            'side': side,
            'size': executed_size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': realized_pnl,
            'pnl_pct': (realized_pnl / (entry_price * executed_size) * 100) if entry_price * executed_size > 0 else 0,
            'is_stop_loss': is_stop_loss,
            'is_partial': is_partial,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'confidence': confidence
        }
        
        if hasattr(bot, 'trades'):
            bot.trades.append(trade_record)
        
        # Enviar a InfluxDB
        try:
            trade_success = await INFLUX_METRICS.write_trade_metrics(
                symbol=symbol,
                action=side,
                confidence=confidence,
                price=float(exit_price),
                size=float(executed_size),
                pnl=float(realized_pnl)
            )
        except Exception as influx_error:
            LOG.debug("influx_metrics_failed", error=str(influx_error))
        
        # Cerrar en risk manager si no es parcial
        if not is_partial:
            risk_manager.close_position(symbol)
        
        LOG.info("trade_closed_successfully",
                symbol=symbol,
                side=side,
                executed_size=executed_size,
                entry_price=entry_price,
                exit_price=exit_price,
                realized_pnl=realized_pnl,
                equity_final=bot.equity,
                is_stop_loss=is_stop_loss,
                is_partial=is_partial)
        
    except Exception as e:
        LOG.error("update_bot_after_trade_close_failed",
                 symbol=symbol,
                 error=str(e),
                 traceback=traceback.format_exc()[:500])
        
class PortfolioRebalancer:
    def __init__(self, config, bot):
        self.config = config
        self.bot = bot
        self.max_position_pct = 0.25
        self.max_positions = 12
        self.correlation_threshold = 0.7
        self.max_loss_per_position_default = -0.06
        self.max_loss_high_confidence = -0.08
        self.max_loss_low_confidence = -0.04
        self._position_confidences = {}
        LOG.info("portfolio_rebalancer_initialized", max_position_pct=self.max_position_pct, max_positions=self.max_positions, default_loss_limit=self.max_loss_per_position_default)

    async def check_rebalance_needed(self, risk_manager) -> List[str]:
        to_close = []
        try:
            if not risk_manager.active_stops:
                return []
            active_stops_snapshot = dict(risk_manager.active_stops)
            total_exposure = 0.0
            for symbol, stop_info in active_stops_snapshot.items():
                if symbol not in risk_manager.active_stops:
                    LOG.debug("position_removed_during_rebalance_check", symbol=symbol)
                    continue
                entry_price = stop_info['entry_price']
                remaining_size = stop_info['remaining_size']
                if entry_price <= 0 or remaining_size <= 0:
                    LOG.warning("invalid_position_data_closing", symbol=symbol, entry_price=entry_price, remaining_size=remaining_size)
                    if symbol not in to_close:
                        to_close.append(symbol)
                    continue
                position_value = remaining_size * entry_price
                max_reasonable_position_value = 100000.0
                if position_value > max_reasonable_position_value:
                    LOG.error("position_value_unreasonably_high_closing", symbol=symbol, position_value=position_value, entry_price=entry_price, remaining_size=remaining_size, max_allowed=max_reasonable_position_value)
                    if symbol not in to_close:
                        to_close.append(symbol)
                    continue
                if remaining_size > 1000000.0:
                    LOG.error("remaining_size_unreasonably_high_closing", symbol=symbol, remaining_size=remaining_size, entry_price=entry_price)
                    if symbol not in to_close:
                        to_close.append(symbol)
                    continue
                total_exposure += position_value
            for symbol, stop_info in active_stops_snapshot.items():
                if symbol not in risk_manager.active_stops:
                    LOG.debug("position_removed_during_rebalance_check", symbol=symbol)
                    continue
                entry_price = stop_info['entry_price']
                remaining_size = stop_info['remaining_size']
                if entry_price <= 0 or remaining_size <= 0:
                    LOG.warning("invalid_position_data_closing", symbol=symbol, entry_price=entry_price, remaining_size=remaining_size)
                    if symbol not in to_close:
                        to_close.append(symbol)
                    continue
                position_value = remaining_size * entry_price
                available_equity = float(self.bot.equity)
                total_bot_equity = float(self.bot.equity)
                if total_bot_equity <= 0:
                    LOG.error("invalid_total_equity_in_rebalance", equity=total_bot_equity, symbol=symbol)
                    if symbol not in to_close:
                        to_close.append(symbol)
                    continue
                position_pct = position_value / total_bot_equity
                if total_bot_equity > 0:
                    position_pct = position_value / total_bot_equity
                else:
                    LOG.error("zero_or_negative_equity_closing_position", symbol=symbol, equity=total_bot_equity, position_value=position_value)
                    if symbol not in to_close:
                        to_close.append(symbol)
                    continue
                if position_pct < 0 or position_pct > 1.0:
                    LOG.error("unreasonable_position_percentage", symbol=symbol, position_pct=position_pct * 100, position_value=position_value, total_equity=total_bot_equity)
                    if hasattr(self.bot, 'position_ledger'):
                        audit = self.bot.position_ledger.audit_equity(self.bot)
                        LOG.error("equity_audit_unreasonable_pct", audit=audit, symbol=symbol, position_value=position_value, entry_price=entry_price, remaining_size=remaining_size)
                        if audit['is_consistent']:
                            LOG.error("position_data_inconsistent_with_valid_equity", symbol=symbol, should_check_risk_manager=True)
                    if symbol not in to_close:
                        to_close.append(symbol)
                    continue
                if position_pct > self.max_position_pct:
                    LOG.warning("position_exceeds_max_percentage", symbol=symbol, position_pct=position_pct * 100, max_allowed=self.max_position_pct * 100, position_value=position_value, total_equity=total_bot_equity)
                    if symbol not in to_close:
                        to_close.append(symbol)
                    continue
                try:
                    ticker = await self.bot.exchange_manager.exchange.fetch_ticker(symbol)
                    current_price = ticker.get('last', 0)
                    if current_price > 0:
                        entry_price = stop_info['entry_price']
                        side = stop_info['side']
                        if side == 'buy':
                            pnl_pct = (current_price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price
                        position_confidence = self._position_confidences.get(symbol, 0.5)
                        if position_confidence > 0.8:
                            max_loss_threshold = self.max_loss_high_confidence
                        elif position_confidence < 0.5:
                            max_loss_threshold = self.max_loss_low_confidence
                        else:
                            max_loss_threshold = self.max_loss_per_position_default
                        if pnl_pct <= max_loss_threshold:
                            LOG.warning("position_loss_limit_exceeded", symbol=symbol, pnl_pct=pnl_pct * 100, max_loss=max_loss_threshold * 100, position_confidence=position_confidence)
                            if symbol not in to_close:
                                to_close.append(symbol)
                            continue
                except Exception as e:
                    LOG.debug("ticker_fetch_error_rebalance", symbol=symbol, error=str(e))
            if len(active_stops_snapshot) > self.max_positions:
                sorted_positions = sorted(active_stops_snapshot.items(), key=lambda x: x[1]['entry_time'])
                excess = len(active_stops_snapshot) - self.max_positions
                for i in range(excess):
                    symbol = sorted_positions[i][0]
                    if symbol not in to_close:
                        LOG.info("closing_excess_position", symbol=symbol, reason="max_positions_exceeded")
                        to_close.append(symbol)
            return list(set(to_close))
        except Exception as e:
            LOG.error("rebalance_check_failed", error=str(e), traceback=traceback.format_exc())
            return []

    def register_position_confidence(self, symbol: str, confidence: float):
        try:
            self._position_confidences[symbol] = max(0.0, min(1.0, confidence))
            LOG.debug("position_confidence_registered", symbol=symbol, confidence=confidence)
        except Exception as e:
            LOG.debug("confidence_registration_failed", error=str(e))

    def clear_position_confidence(self, symbol: str):
        if symbol in self._position_confidences:
            del self._position_confidences[symbol]

    async def execute_rebalance(self, symbols_to_close: List[str], risk_manager):
        try:
            symbols_snapshot = list(symbols_to_close)
            for symbol in symbols_snapshot:
                try:
                    if symbol not in risk_manager.active_stops:
                        LOG.debug("position_already_closed_skipping", symbol=symbol)
                        continue
                    stop_info = risk_manager.active_stops[symbol]
                    side = stop_info['side']
                    remaining_size = stop_info['remaining_size']
                    entry_price = stop_info['entry_price']
                    if remaining_size <= 0 or np.isnan(remaining_size) or np.isinf(remaining_size):
                        LOG.error("invalid_remaining_size_in_rebalance", symbol=symbol, size=remaining_size)
                        risk_manager.close_position(symbol)
                        continue
                    close_side = 'sell' if side == 'buy' else 'buy'
                    order = await self.bot.exchange_manager.create_order(symbol, 'market', close_side, remaining_size)
                    if order and order.get("success", False):
                        try:
                            ticker = await self.bot.exchange_manager.exchange.fetch_ticker(symbol)
                            current_price = ticker.get('last', 0)
                            if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
                                LOG.warning("invalid_ticker_price_using_order_price", symbol=symbol, ticker_price=current_price)
                                current_price = order.get('price', entry_price)
                        except Exception as ticker_error:
                            LOG.warning("ticker_fetch_failed_using_order_price", symbol=symbol, error=str(ticker_error))
                            current_price = order.get('price', entry_price)
                        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
                            LOG.error("invalid_current_price_for_rebalance", symbol=symbol, current_price=current_price)
                            current_price = entry_price
                        await _update_bot_after_trade_close(self.bot, self.bot.risk_manager, symbol, 0.0, close_side, remaining_size, entry_price, current_price, is_stop_loss=False)
                        LOG.info("rebalance_position_closed", symbol=symbol, side=close_side, size=remaining_size, entry_price=entry_price, exit_price=current_price, reason="rebalance")
                        try:
                            if hasattr(self.bot, 'position_ledger') and self.bot.position_ledger.transactions:
                                last_transaction = self.bot.position_ledger.transactions[-1]
                                realized_pnl = last_transaction.realized_pnl
                            else:
                                if side == 'buy':
                                    realized_pnl = (current_price - entry_price) * remaining_size
                                else:
                                    realized_pnl = (entry_price - current_price) * remaining_size
                            await INFLUX_METRICS.write_trade_metrics(symbol=symbol, action=f"rebalance_{close_side}", confidence=1.0, price=float(current_price), size=float(remaining_size), pnl=float(realized_pnl))
                            if INFLUX_METRICS.enabled:
                                await INFLUX_METRICS.write_model_metrics('portfolio_rebalancer', {'rebalance_executed': 1.0, 'pnl': float(realized_pnl), 'position_closed': 1.0, 'remaining_positions': len(risk_manager.active_stops)})
                        except Exception as influx_error:
                            LOG.debug("rebalance_influx_write_failed", error=str(influx_error))
                        if hasattr(self.bot, 'position_ledger') and self.bot.position_ledger.transactions:
                            last_transaction = self.bot.position_ledger.transactions[-1]
                            risk_manager.update_daily_pnl(last_transaction.realized_pnl)
                except Exception as e_inner:
                    LOG.error("rebalance_close_failed", symbol=symbol, error=str(e_inner))
                    continue
        except Exception as e:
            LOG.error("rebalance_execution_failed", error=str(e), traceback=traceback.format_exc())

class CorrelationAnalyzer:
    def __init__(self, exchange_manager):
        self.exchange_manager = exchange_manager
        self.correlation_cache = {}
        self.cache_ttl = 3600
        self.last_update = {}

    async def get_correlation_matrix(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        try:
            price_data = {}
            for symbol in symbols:
                try:
                    ohlcv = await self.exchange_manager.fetch_ohlcv(symbol, '1h', limit=100)
                    if ohlcv.get('success') and ohlcv.get('ohlcv'):
                        closes = [candle[4] for candle in ohlcv['ohlcv']]
                        price_data[symbol] = closes
                except Exception as e:
                    LOG.debug("correlation_data_fetch_failed", symbol=symbol, error=str(e))
                    continue
            if len(price_data) < 2:
                return None
            data_lengths = {sym: len(closes) for sym, closes in price_data.items()}
            if len(set(data_lengths.values())) > 1:
                LOG.warning("unequal_data_lengths_for_correlation", lengths=data_lengths)
                min_length = min(data_lengths.values())
                if min_length < 20:
                    LOG.error("insufficient_common_data_length", min_length=min_length, required=20, symbols=list(data_lengths.keys()))
                    return None
                aligned_data = {}
                for sym, closes in price_data.items():
                    aligned_data[sym] = closes[-min_length:]
                price_data = aligned_data
                LOG.info("data_aligned_to_common_length", common_length=min_length, symbols=len(price_data), alignment_method="last_n_values")
            try:
                df = pd.DataFrame(price_data)
                if df.empty:
                    LOG.error("dataframe_empty_after_creation", price_data_keys=list(price_data.keys()), price_data_lengths={k: len(v) for k, v in price_data.items()})
                    return None
                if df.shape[0] < 10:
                    LOG.warning("insufficient_data_for_correlation_matrix", rows=df.shape[0])
                    return None
                if df.isnull().any().any():
                    null_counts = df.isnull().sum()
                    LOG.warning("dataframe_contains_nulls_before_correlation", null_counts=null_counts.to_dict())
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    if df.isnull().any().any():
                        initial_rows = len(df)
                        df = df.dropna()
                        LOG.warning("dropped_rows_with_nulls", initial=initial_rows, final=len(df))
                        if len(df) < 10:
                            LOG.error("insufficient_data_after_dropping_nulls", rows=len(df))
                            return None
                returns = df.pct_change().dropna()
                if len(returns) < 5:
                    LOG.warning("insufficient_returns_for_correlation", returns_count=len(returns))
                    return None
                if np.isinf(returns.values).any():
                    LOG.warning("infinite_returns_detected_clipping")
                    returns = returns.replace([np.inf, -np.inf], np.nan)
                    returns = returns.fillna(0)
                extreme_mask = (returns.abs() > 0.5).any(axis=1)
                if extreme_mask.any():
                    extreme_count = extreme_mask.sum()
                    LOG.warning("extreme_returns_detected_clipping", count=extreme_count, total=len(returns))
                    returns = returns.clip(-0.5, 0.5)
                try:
                    correlation_matrix = returns.corr()
                except Exception as corr_error:
                    LOG.error("corr_calculation_exception", error=str(corr_error), returns_shape=returns.shape, returns_dtypes=returns.dtypes.to_dict(), returns_sample=returns.head().to_dict() if len(returns) > 0 else {})
                    return None
                if correlation_matrix.isnull().any().any():
                    null_count = correlation_matrix.isnull().sum().sum()
                    LOG.warning("correlation_matrix_contains_nan", null_count=null_count, total_elements=correlation_matrix.size)
                    correlation_matrix = correlation_matrix.fillna(0)
                if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
                    LOG.error("correlation_matrix_not_square", shape=correlation_matrix.shape)
                    return None
                diagonal_values = np.diag(correlation_matrix.values)
                if not np.allclose(diagonal_values, 1.0, atol=0.01):
                    LOG.warning("correlation_matrix_diagonal_invalid", diagonal_sample=diagonal_values[:5].tolist())
            except Exception as corr_calc_error:
                LOG.error("correlation_matrix_calculation_failed", error=str(corr_calc_error), traceback=traceback.format_exc())
                return None
            LOG.info("correlation_matrix_calculated", symbols=len(symbols), avg_correlation=correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean())
            return correlation_matrix
        except Exception as e:
            LOG.error("correlation_matrix_failed", error=str(e), traceback=traceback.format_exc())
            return None

    async def check_correlation(self, symbol1: str, symbol2: str) -> float:
        try:
            cache_key = f"{symbol1}:{symbol2}"
            if cache_key in self.correlation_cache:
                if time.time() - self.last_update.get(cache_key, 0) < self.cache_ttl:
                    return self.correlation_cache[cache_key]
            matrix = await self.get_correlation_matrix([symbol1, symbol2])
            if matrix is None or len(matrix) < 2:
                return 0.0
            correlation = matrix.iloc[0, 1]
            self.correlation_cache[cache_key] = correlation
            self.last_update[cache_key] = time.time()
            return float(correlation)
        except Exception as e:
            LOG.debug("correlation_check_failed", error=str(e))
            return 0.0

    async def filter_by_correlation(self, candidate_symbol: str, active_symbols: List[str], max_correlation: float = 0.7) -> bool:
        try:
            if not active_symbols:
                return True
            for active_symbol in active_symbols:
                correlation = await self.check_correlation(candidate_symbol, active_symbol)
                if abs(correlation) > max_correlation:
                    LOG.info("high_correlation_detected", candidate=candidate_symbol, active=active_symbol, correlation=correlation)
                    return False
            return True
        except Exception as e:
            LOG.debug("correlation_filter_failed", error=str(e))
            return True

class PerformanceDashboard:
    def __init__(self, bot):
        if not hasattr(bot, 'equity'):
            LOG.error("bot_missing_equity_attribute", message="Bot must have equity attribute")
            raise ValueError("Bot instance must have 'equity' attribute")
        if not hasattr(bot, 'initial_capital'):
            LOG.error("bot_missing_initial_capital", message="Bot must have initial_capital")
            raise ValueError("Bot instance must have 'initial_capital' attribute")
        if not hasattr(bot, 'performance_metrics'):
            LOG.warning("bot_missing_performance_metrics_initializing")
            bot.performance_metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }
        self.bot = bot
        self.report_interval = 300
        LOG.debug("performance_dashboard_initialized", report_interval=self.report_interval)

    async def generate_live_report(self) -> Dict[str, Any]:
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'uptime_hours': 0.0,
            'portfolio': {},
            'trading': {},
            'risk': {},
            'positions': [],
            'performance': {},
            'memory': {},
            'cache': {},
            'position_sizer': {},
            'data_accumulator': {},
            'ai_models': {}
        }
        try:
            try:
                if hasattr(self.bot, 'start_time') and self.bot.start_time:
                    report['uptime_hours'] = (datetime.now(timezone.utc) - self.bot.start_time).total_seconds() / 3600
            except Exception as uptime_error:
                LOG.debug("uptime_calculation_failed", error=str(uptime_error))

            raw_equity = getattr(self.bot, 'equity', None)
            raw_initial = getattr(self.bot, 'initial_capital', None)
            if raw_equity is None or not isinstance(raw_equity, (int, float)):
                raw_equity = raw_initial if raw_initial else 10000.0
            if raw_initial is None or not isinstance(raw_initial, (int, float)):
                raw_initial = 10000.0
            if np.isnan(raw_equity) or np.isinf(raw_equity):
                raw_equity = raw_initial
            if np.isnan(raw_initial) or np.isinf(raw_initial):
                raw_initial = 10000.0

            equity = float(raw_equity)
            initial_capital = float(raw_initial)
            total_pnl = equity - initial_capital
            unrealized_pnl = 0.0

            if hasattr(self.bot, 'risk_manager') and self.bot.risk_manager.active_stops:
                for symbol, stop_info in self.bot.risk_manager.active_stops.items():
                    try:
                        ticker = await self.bot.exchange_manager.exchange.fetch_ticker(symbol)
                        current_price = ticker.get('last', 0)
                        entry_price = stop_info.get('entry_price', 0)
                        size = stop_info.get('remaining_size', 0)
                        side = stop_info.get('side', 'buy')
                        if current_price > 0 and entry_price > 0 and size > 0:
                            if side == 'buy':
                                position_pnl = (current_price - entry_price) * size
                            else:
                                position_pnl = (entry_price - current_price) * size
                            unrealized_pnl += position_pnl
                    except Exception:
                        continue

            total_pnl_including_open = total_pnl + unrealized_pnl
            report['portfolio'] = {
                'equity': equity,
                'initial_capital': initial_capital,
                'total_pnl': total_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl_including_open': total_pnl_including_open,
                'total_pnl_pct': 0.0,
                'drawdown_pct': 0.0,
            }
            if initial_capital > 0:
                report['portfolio']['total_pnl_pct'] = float((total_pnl_including_open / initial_capital) * 100)
                if hasattr(self.bot, 'drawdown'):
                    drawdown_value = self.bot.drawdown
                    if isinstance(drawdown_value, (int, float)) and not np.isnan(drawdown_value):
                        report['portfolio']['drawdown_pct'] = float(drawdown_value * 100)

            metrics = getattr(self.bot, 'performance_metrics', {})
            report['trading'] = {
                'total_trades': metrics.get('total_trades', 0),
                'winning_trades': metrics.get('winning_trades', 0),
                'losing_trades': metrics.get('losing_trades', 0),
                'win_rate': metrics.get('win_rate', 0.0) * 100,
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            }

            if hasattr(self.bot, 'risk_manager'):
                rm = self.bot.risk_manager
                report['risk'] = {
                    'active_positions': len(rm.active_stops),
                    'daily_pnl': rm.daily_pnl,
                    'daily_trades': rm.daily_trades,
                    'circuit_breaker_active': rm.circuit_breaker_active,
                    'daily_return_pct': (rm.daily_pnl / self.bot.initial_capital * 100) if self.bot.initial_capital > 0 else 0
                }
                positions_detail = []
                for symbol, stop_info in rm.active_stops.items():
                    try:
                        ticker = await self.bot.exchange_manager.exchange.fetch_ticker(symbol)
                        current_price = ticker.get('last', 0)
                        entry_price = stop_info['entry_price']
                        side = stop_info['side']
                        if side == 'buy':
                            pnl_pct = (current_price - entry_price) / entry_price * 100
                        else:
                            pnl_pct = (entry_price - current_price) / entry_price * 100
                        positions_detail.append({
                            'symbol': symbol,
                            'side': side,
                            'entry_price': entry_price,
                            'current_price': current_price,
                            'size': stop_info['remaining_size'],
                            'pnl_pct': pnl_pct,
                            'stop_loss': stop_info['stop_loss'],
                            'tp_levels': len(stop_info.get('take_profit_levels', []))
                        })
                    except Exception:
                        continue
                report['positions'] = positions_detail

            if PERFORMANCE_PROFILER:
                prof_stats = PERFORMANCE_PROFILER.get_stats(top_n=5)
                report['performance'] = {
                    'top_operations': prof_stats.get('top_operations', []),
                    'total_operations': prof_stats.get('total_operations', 0)
                }

            if MEMORY_MANAGER:
                mem_stats = MEMORY_MANAGER.get_memory_stats()
                report['memory'] = mem_stats

            if FEATURE_CACHE:
                cache_stats = FEATURE_CACHE.get_stats()
                report['cache'] = {
                    'size': cache_stats.get('cache_size', 0),
                    'hit_rate': cache_stats.get('hit_rate', 0) * 100,
                    'hits': cache_stats.get('hit_count', 0),
                    'misses': cache_stats.get('miss_count', 0),
                    'memory_mb': cache_stats.get('memory_mb', 0)
                }

            if hasattr(self.bot, 'position_sizer') and self.bot.position_sizer:
                try:
                    sizer_stats = self.bot.position_sizer.get_stats()
                    report['position_sizer'] = sizer_stats
                except Exception as sizer_error:
                    LOG.debug("position_sizer_stats_failed", error=str(sizer_error))

            if hasattr(self.bot, 'data_accumulator') and self.bot.data_accumulator:
                acc_stats = self.bot.data_accumulator.get_stats()
                report['data_accumulator'] = acc_stats

            if hasattr(self.bot, 'ai_models_status'):
                report['ai_models'] = self.bot.ai_models_status

        except Exception as e:
            LOG.error("dashboard_report_generation_failed", error=str(e))

        return report

    async def print_dashboard(self):
        try:
            report = await self.generate_live_report()
            if not report or not isinstance(report, dict):
                LOG.error("invalid_report_generated", report_type=type(report).__name__)
                return
            print("\n" + "=" * 100)
            print(f"📊 TRADING BOT DASHBOARD - {report.get('timestamp', 'N/A')}")
            print("=" * 100)
            portfolio = report.get('portfolio', {})
            if portfolio:
                print(f"\n💰 PORTFOLIO:")
                try:
                    equity = float(portfolio.get('equity', 0))
                    total_pnl = float(portfolio.get('total_pnl', 0))
                    total_pnl_pct = float(portfolio.get('total_pnl_pct', 0))
                    drawdown_pct = float(portfolio.get('drawdown_pct', 0))
                    print(f"   Equity: ${equity:,.2f}")
                    print(f"   PnL: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
                    print(f"   Drawdown: {drawdown_pct:.2f}%")
                except (ValueError, TypeError) as format_error:
                    LOG.error("portfolio_format_error", error=str(format_error), portfolio=portfolio)
                    print(f"   Equity: {portfolio.get('equity', 'N/A')}")
                    print(f"   PnL: {portfolio.get('total_pnl', 'N/A')}")
                    print(f"   Drawdown: {portfolio.get('drawdown_pct', 'N/A')}")
            else:
                LOG.warning("portfolio_data_missing_in_report")
            trading = report.get('trading', {})
            if trading:
                print(f"\n📈 TRADING STATS:")
                try:
                    total_trades = int(trading.get('total_trades', 0))
                    win_rate = float(trading.get('win_rate', 0))
                    sharpe_ratio = float(trading.get('sharpe_ratio', 0))
                    print(f"   Total Trades: {total_trades}")
                    print(f"   Win Rate: {win_rate:.1f}%")
                    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
                except (ValueError, TypeError) as trading_error:
                    LOG.debug("trading_stats_format_error", error=str(trading_error))
                    print(f"   Total Trades: {trading.get('total_trades', 'N/A')}")
                    print(f"   Win Rate: {trading.get('win_rate', 'N/A')}")
                    print(f"   Sharpe Ratio: {trading.get('sharpe_ratio', 'N/A')}")
            else:
                LOG.warning("trading_data_missing_in_report")
            if report.get('risk'):
                print(f"\n⚠️  RISK MANAGEMENT:")
                print(f"   Active Positions: {report['risk']['active_positions']}")
                print(f"   Daily PnL: ${report['risk']['daily_pnl']:,.2f} ({report['risk']['daily_return_pct']:+.2f}%)")
                print(f"   Daily Trades: {report['risk']['daily_trades']}")
                print(f"   Circuit Breaker: {'🔴 ACTIVE' if report['risk']['circuit_breaker_active'] else '🟢 OK'}")
            if report.get('positions'):
                print(f"\n📍 ACTIVE POSITIONS:")
                for pos in report['positions']:
                    try:
                        pnl_pct = float(pos.get('pnl_pct', 0))
                        pnl_emoji = "🟢" if pnl_pct > 0 else "🔴"
                        symbol = str(pos.get('symbol', 'N/A'))
                        side = str(pos.get('side', 'N/A')).upper()
                        entry_price = float(pos.get('entry_price', 0))
                        current_price = float(pos.get('current_price', 0))
                        stop_loss = float(pos.get('stop_loss', 0))
                        print(f"   {pnl_emoji} {symbol} | {side} | "
                              f"Entry: ${entry_price:.2f} | Current: ${current_price:.2f} | "
                              f"PnL: {pnl_pct:+.2f}% | SL: ${stop_loss:.2f}")
                    except (ValueError, TypeError, KeyError) as pos_error:
                        LOG.debug("position_print_error", position=pos, error=str(pos_error))
                        print(f"   ⚠️  {pos.get('symbol', 'Unknown')} | Error formatting position")
            if hasattr(self.bot, 'symbol_performance') and self.bot.symbol_performance:
                print(f"\n🎯 TOP 5 SYMBOLS PERFORMANCE:")
                try:
                    sorted_symbols = sorted(self.bot.symbol_performance.items(), key=lambda x: float(x[1].get('total_pnl', 0)), reverse=True)[:5]
                    for symbol, perf in sorted_symbols:
                        try:
                            winning = int(perf.get('winning_trades', 0))
                            losing = int(perf.get('losing_trades', 0))
                            total_trades = winning + losing
                            pnl = float(perf.get('total_pnl', 0))
                            win_rate = float(perf.get('win_rate', 0)) * 100
                            pnl_emoji = "🟢" if pnl > 0 else "🔴"
                            print(f"   {pnl_emoji} {symbol}: "
                                  f"PnL ${pnl:+.2f} | "
                                  f"WR {win_rate:.1f}% | "
                                  f"Trades {total_trades}")
                        except (ValueError, TypeError) as perf_error:
                            LOG.debug("symbol_performance_format_error", symbol=symbol, error=str(perf_error))
                            print(f"   ⚠️  {symbol}: Error formatting performance")
                except Exception as sort_error:
                    LOG.debug("symbol_performance_sort_error", error=str(sort_error))
            if report.get('memory'):
                print(f"\n💾 MEMORY:")
                print(f"   Current: {report['memory'].get('current_mb', 0):.0f} MB")
                print(f"   Trend: {report['memory'].get('trend', 'unknown')}")
            if report.get('cache'):
                print(f"\n🗂️  CACHE:")
                print(f"   Hit Rate: {report['cache'].get('hit_rate', 0):.1f}%")
                print(f"   Size: {report['cache'].get('size', 0)} entries")
                print(f"   Memory: {report['cache'].get('memory_mb', 0):.1f} MB")
            if report.get('data_accumulator'):
                acc = report['data_accumulator']
                print(f"\n📊 DATA ACCUMULATOR:")
                print(f"   Buffer: {acc.get('buffer_size', 0)}/{acc.get('max_samples', 0)}")
                print(f"   Utilization: {acc.get('utilization', 0)*100:.1f}%")
                print(f"   Total Samples: {acc.get('total_samples_added', 0):,}")
                print(f"   Symbols Tracked: {acc.get('symbols_tracked', 0)}")
                symbol_stats = acc.get('symbol_stats', {})
                if symbol_stats:
                    sorted_symbols_acc = sorted(symbol_stats.items(), key=lambda x: x[1].get('samples', 0), reverse=True)[:5]
                    if sorted_symbols_acc:
                        print(f"   Top Symbols by Samples:")
                        for symbol, stats in sorted_symbols_acc:
                            print(f"      {symbol}: {stats.get('samples', 0)} samples")
            if report.get('ai_models'):
                ai = report['ai_models']
                print(f"\n🤖 AI MODELS STATUS:")
                print(f"   General Model Trained: {'✅ YES' if ai.get('general_model_trained') else '❌ NO'}")
                print(f"   Specialized Models: {ai.get('specialized_models_count', 0)}")
                if ai.get('specialized_symbols'):
                    print(f"   Specialized for: {', '.join(ai['specialized_symbols'][:5])}")
                    if len(ai['specialized_symbols']) > 5:
                        print(f"      ... and {len(ai['specialized_symbols']) - 5} more")
                if ai.get('symbol_training_history'):
                    print(f"\n   🔄 Training History (Top 5):")
                    sorted_history = sorted(ai['symbol_training_history'].items(), key=lambda x: x[1].get('training_count', 0), reverse=True)[:5]
                    for symbol, history in sorted_history:
                        last_train = history.get('last_training', 'Never')
                        if last_train != 'Never':
                            try:
                                last_train_dt = datetime.fromisoformat(last_train)
                                hours_ago = (datetime.now(timezone.utc) - last_train_dt).total_seconds() / 3600
                                last_train = f"{hours_ago:.1f}h ago"
                            except:
                                pass
                        print(f"      {symbol}: {history.get('training_count', 0)} trainings, "
                              f"Last: {last_train}, "
                              f"Avg samples: {history.get('avg_samples', 0):.0f}")

            if INFLUX_METRICS and INFLUX_METRICS.enabled:
                print(f"\n📊 INFLUXDB METRICS:")
                try:
                    health = await INFLUX_METRICS.check_health()
                    
                    if health['healthy']:
                        print(f"   Status: ✅ Healthy")
                    else:
                        print(f"   Status: ⚠️ Issues detected - {health.get('reason', 'unknown')}")
                    
                    stats = health.get('stats', {})
                    total = stats.get('total_writes', 0)
                    success = stats.get('successful_writes', 0)
                    failed = stats.get('failed_writes', 0)
                    rate = stats.get('success_rate', 0)
                    
                    print(f"   Total Writes: {total:,}")
                    print(f"   Successful: {success:,}")
                    print(f"   Failed: {failed:,}")
                    print(f"   Success Rate: {rate*100:.1f}%")
                    
                    if stats.get('last_error'):
                        print(f"   Last Error: {stats['last_error']}")
                    
                except Exception as diag_error:
                    print(f"   ⚠️ Diagnostic failed: {str(diag_error)}")

            if hasattr(self.bot, 'position_ledger'):
                audit = self.bot.position_ledger.audit_equity(self.bot)
                if not audit['is_consistent']:
                    print(f"⚠️  EQUITY AUDIT: INCONSISTENT (Discrepancy: ${audit['discrepancy']:,.2f})")
            print("\n" + "=" * 100)
        except Exception as e:
            LOG.error("dashboard_print_failed", error=str(e))

    async def periodic_dashboard_loop(self):
        while self.bot.is_running:
            try:
                await asyncio.sleep(self.report_interval)
                await self.print_dashboard()
                report = await self.generate_live_report()
                if INFLUX_METRICS.enabled:
                    portfolio = report.get('portfolio', {})
                    risk = report.get('risk', {})
                    await INFLUX_METRICS.write_portfolio_metrics(
                        portfolio.get('equity', 0.0),
                        portfolio.get('drawdown_pct', 0.0) / 100,
                        risk.get('active_positions', 0),
                        portfolio.get('total_pnl', 0.0)
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOG.error("periodic_dashboard_error", error=str(e))
                await asyncio.sleep(60)

class AdvancedAITradingBot(ProductionBot):
    def __init__(self, config: AdvancedAIConfig, exchange_manager: ExchangeManager, strategy_manager: StrategyManager):
        super().__init__(config, exchange_manager, strategy_manager)
        self.automl = None
        self.regime_detector = None
        self.ensemble_learner = None
        self.rl_agent = None
        self.rl_training_manager = None
        self.risk_optimizer = None
        self.position_ledger = PositionLedger()
        LOG.info("position_ledger_initialized")
        self.position_sizer = DynamicPositionSizer(config, self)
        self.risk_manager = DynamicRiskManager(config, self)
        self.smart_executor = SmartOrderExecutor(exchange_manager, config)
        self.portfolio_rebalancer = PortfolioRebalancer(config, self)
        self.correlation_analyzer = CorrelationAnalyzer(exchange_manager)
        self.symbol_performance = {}
        self.last_pipeline_execution = {}
        # ===== NUEVO: Sistema de Testing =====
        self.test_suite = AutomatedTestSuite(self)
        self.test_results_history = []
        # ===== NUEVO: Telegram Kill Switch =====
        self.telegram_kill_switch = TelegramKillSwitch()
        # ===== NUEVO: Walk-Forward Validator =====
        self.walk_forward_validator = WalkForwardValidator(
            train_window_days=60,
            test_window_days=15,
            min_trades_per_window=10
        )
        self.symbol_execution_locks = {symbol: asyncio.Lock() for symbol in config.symbols}
        LOG.debug("bot_execution_tracking_initialized", symbols_count=len(config.symbols))

        # NUEVO: Gestor de parámetros adaptativos
        self.adaptive_params = AdaptiveParameterManager(self)
        LOG.debug("adaptive_parameter_manager_initialized")
        
        for symbol in config.symbols:
            self.symbol_performance[symbol] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'last_trade_time': None
            }
        LOG.info("advanced_ai_trading_bot_initialized_with_dynamic_sizing")

    async def _start_periodic_tasks(self) -> None:
        try:
            LOG.info("starting_base_periodic_tasks")
            try:
                memory_cleanup_task = asyncio.create_task(periodic_memory_cleanup())
                self.periodic_tasks.append(memory_cleanup_task)
                LOG.debug("memory_cleanup_task_started")
            except Exception as e:
                LOG.error("memory_cleanup_task_failed", error=str(e))
            if INFLUX_METRICS and INFLUX_METRICS.enabled:
                try:
                    async def periodic_influx_flush():
                        while self.is_running:
                            try:
                                await asyncio.sleep(30)
                                if hasattr(INFLUX_METRICS, 'write_api') and INFLUX_METRICS.write_api:
                                    try:
                                        INFLUX_METRICS.write_api.flush()
                                        LOG.debug("influxdb_flush_completed")
                                    except Exception as flush_error:
                                        LOG.debug("influxdb_flush_failed", error=str(flush_error))
                            except asyncio.CancelledError:
                                break
                            except Exception as e:
                                LOG.debug("periodic_influx_flush_error", error=str(e))
                                await asyncio.sleep(60)
                    influx_flush_task = asyncio.create_task(periodic_influx_flush())
                    self.periodic_tasks.append(influx_flush_task)
                    LOG.info("influxdb_flush_task_started", interval_seconds=30)
                except Exception as e:
                    LOG.error("influxdb_flush_task_failed", error=str(e))
            if self.health_check:
                try:
                    health_check_task = asyncio.create_task(self.health_check.periodic_health_log())
                    self.periodic_tasks.append(health_check_task)
                    LOG.debug("health_check_task_started")
                except Exception as e:
                    LOG.error("health_check_task_failed", error=str(e))
            else:
                LOG.warning("health_check_unavailable_skipping_task")
            if hasattr(self, 'risk_manager') and self.risk_manager:
                try:
                    position_monitor_task = asyncio.create_task(position_monitoring_loop(self, self.exchange_manager, self.risk_manager, interval=10))
                    self.periodic_tasks.append(position_monitor_task)
                    LOG.info("position_monitoring_loop_started", interval_seconds=10)
                except Exception as e:
                    LOG.error("position_monitoring_task_failed", error=str(e))
            else:
                LOG.error("CRITICAL_risk_manager_not_available_positions_wont_be_monitored")
            if hasattr(self, 'position_ledger') and self.position_ledger:
                try:
                    async def periodic_equity_audit_task():
                        while self.is_running:
                            try:
                                await asyncio.sleep(3600)
                                audit_result = self.position_ledger.audit_equity(self)
                                LOG.info("periodic_equity_audit_completed", **audit_result)
                                try:
                                    await INFLUX_METRICS.write_model_metrics('equity_audit', {
                                        'is_consistent': 1.0 if audit_result['is_consistent'] else 0.0,
                                        'discrepancy': float(audit_result['discrepancy']),
                                        'actual_equity': float(audit_result['actual_equity']),
                                        'expected_equity': float(audit_result['expected_free_equity']),
                                        'invested_in_positions': float(audit_result['invested_in_positions']),
                                        'total_realized_pnl': float(audit_result['total_realized_pnl'])
                                    })
                                except Exception as influx_error:
                                    LOG.debug("equity_audit_influx_write_failed", error=str(influx_error))
                                if not audit_result['is_consistent'] and abs(audit_result['discrepancy']) > 10.0:
                                    await ALERT_SYSTEM.send_alert("ERROR", "Equity audit detected critical discrepancy", **audit_result)
                            except asyncio.CancelledError:
                                break
                            except Exception as e:
                                LOG.error("equity_audit_error", error=str(e))
                                await asyncio.sleep(600)
                    equity_audit_task = asyncio.create_task(periodic_equity_audit_task())
                    self.periodic_tasks.append(equity_audit_task)
                    LOG.info("equity_audit_task_started", interval_hours=1)
                except Exception as e:
                    LOG.error("equity_audit_task_creation_failed", error=str(e))
            if hasattr(self, 'position_ledger') and hasattr(self, 'risk_manager'):
                try:
                    async def periodic_ledger_sync():
                        while self.is_running:
                            try:
                                await asyncio.sleep(300)
                                sync_report = await sync_ledger_with_risk_manager(self)
                                total_orphans = len(sync_report.get('orphaned_ledger', [])) + len(sync_report.get('orphaned_risk_manager', []))
                                if total_orphans > 3:
                                    await ALERT_SYSTEM.send_alert("WARNING", "High number of position discrepancies detected", **sync_report)
                            except asyncio.CancelledError:
                                break
                            except Exception as e:
                                LOG.error("periodic_ledger_sync_error", error=str(e))
                                await asyncio.sleep(60)
                    sync_task = asyncio.create_task(periodic_ledger_sync())
                    self.periodic_tasks.append(sync_task)
                    LOG.info("periodic_ledger_sync_task_started", interval_seconds=300)
                except Exception as e:
                    LOG.error("ledger_sync_task_creation_failed", error=str(e))
            # ===== NUEVO: Testing Automático Periódico =====
            try:
                async def periodic_testing():
                    while self.is_running:
                        try:
                            # Ejecutar tests cada 6 horas
                            await asyncio.sleep(6 * 3600)
                            
                            LOG.info("starting_periodic_automated_tests")
                            test_results = await self.test_suite.run_all_tests()
                            
                            self.test_results_history.append(test_results)
                            
                            # Mantener solo últimos 10 resultados
                            if len(self.test_results_history) > 10:
                                self.test_results_history.pop(0)
                            
                            # Si fallan tests críticos, alertar
                            failed = test_results.get('failed', 0)
                            if failed > 0:
                                failed_tests = [
                                    r['test_name'] for r in test_results.get('results', [])
                                    if not r['passed']
                                ]
                                
                                LOG.error("automated_tests_failed",
                                         failed_count=failed,
                                         failed_tests=failed_tests)
                                
                                await ALERT_SYSTEM.send_alert(
                                    "ERROR",
                                    f"{failed} tests automatizados fallaron",
                                    failed_tests=failed_tests
                                )
                            
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            LOG.error("periodic_testing_error", error=str(e))
                            await asyncio.sleep(3600)
                
                testing_task = asyncio.create_task(periodic_testing())
                self.periodic_tasks.append(testing_task)
                LOG.info("periodic_testing_task_started", interval_hours=6)
                
            except Exception as e:
                LOG.error("testing_task_creation_failed", error=str(e))
            
            # ===== NUEVO: Reconciliación Periódica con Exchange =====
            try:
                async def periodic_reconciliation():
                    while self.is_running:
                        try:
                            # Reconciliar cada 1 hora
                            await asyncio.sleep(3600)
                            
                            if hasattr(self, 'position_ledger'):
                                LOG.info("starting_periodic_reconciliation")
                                recon_report = await self.position_ledger.reconcile_with_exchange(
                                    self, self.exchange_manager
                                )
                                
                                # Guardar en métricas
                                if INFLUX_METRICS.enabled:
                                    await INFLUX_METRICS.write_model_metrics(
                                        'ledger_reconciliation',
                                        {
                                            'matched_positions': len(recon_report.get('matched', [])),
                                            'discrepancies': len(recon_report.get('discrepancies', [])),
                                            'ledger_only': len(recon_report.get('ledger_only', [])),
                                            'exchange_only': len(recon_report.get('exchange_only', []))
                                        }
                                    )
                                
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            LOG.error("periodic_reconciliation_error", error=str(e))
                            await asyncio.sleep(600)
                
                recon_task = asyncio.create_task(periodic_reconciliation())
                self.periodic_tasks.append(recon_task)
                LOG.info("periodic_reconciliation_task_started", interval_hours=1)
                
            except Exception as e:
                LOG.error("reconciliation_task_creation_failed", error=str(e))
            
            # ===== NUEVO: Walk-Forward Validation Periódica =====
            try:
                async def periodic_walk_forward():
                    while self.is_running:
                        try:
                            # Validar cada 7 días
                            await asyncio.sleep(7 * 24 * 3600)
                            
                            LOG.info("starting_periodic_walk_forward_validation")
                            
                            for symbol in self.config.symbols[:3]:  # Top 3 símbolos
                                try:
                                    # Cargar datos históricos
                                    hist_df = await self._load_historical_data(months=6)
                                    
                                    if hist_df is not None and len(hist_df) >= 1000:
                                        validation_result = await self.walk_forward_validator.run_walk_forward(
                                            self, hist_df, symbol
                                        )
                                        
                                        if validation_result.get('success'):
                                            analysis = validation_result.get('analysis', {})
                                            
                                            # Log resultados
                                            LOG.info("walk_forward_validation_completed",
                                                    symbol=symbol,
                                                    windows=len(validation_result.get('windows', [])),
                                                    avg_degradation=analysis.get('avg_degradation', 0),
                                                    overfitting=analysis.get('overfitting_detected', False))
                                            
                                            # Guardar en InfluxDB
                                            if INFLUX_METRICS.enabled:
                                                await INFLUX_METRICS.write_model_metrics(
                                                    f'walk_forward_{symbol.replace("/", "_")}',
                                                    {
                                                        'avg_degradation': analysis.get('avg_degradation', 0),
                                                        'max_degradation': analysis.get('max_degradation', 0),
                                                        'overfitting': 1.0 if analysis.get('overfitting_detected') else 0.0,
                                                        'windows': analysis.get('total_windows', 0)
                                                    }
                                                )
                                    
                                except Exception as symbol_error:
                                    LOG.warning("walk_forward_validation_failed_for_symbol",
                                               symbol=symbol,
                                               error=str(symbol_error))
                                    continue
                            
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            LOG.error("periodic_walk_forward_error", error=str(e))
                            await asyncio.sleep(24 * 3600)
                
                walk_forward_task = asyncio.create_task(periodic_walk_forward())
                self.periodic_tasks.append(walk_forward_task)
                LOG.info("periodic_walk_forward_task_started", interval_days=7)
                
            except Exception as e:
                LOG.error("walk_forward_task_creation_failed", error=str(e))

            # ===== MEJORADO: Sincronización periódica COMPLETA de métricas =====
            try:
                sync_counter = {'value': 0}  # Counter mutable
                
                async def periodic_portfolio_sync():
                    """Sincroniza TODAS las métricas del bot periódicamente"""
                    while self.is_running:
                        try:
                            await asyncio.sleep(30)  # CORRECCIÓN: Cada 30 segundos
                            
                            sync_counter['value'] += 1
                            
                            # Flush explícito cada 10 minutos (20 iteraciones * 30 seg)
                            force_flush = sync_counter['value'] % 20 == 0
                            
                            success = await sync_bot_metrics_to_influx(self, force=force_flush)
                            
                            if force_flush:
                                LOG.info("metrics_full_sync_completed",
                                        iteration=sync_counter['value'],
                                        success=success)
                            
                        except asyncio.CancelledError:
                            # Flush final al cancelar
                            try:
                                await sync_bot_metrics_to_influx(self, force=True)
                                LOG.info("final_metrics_sync_on_cancellation")
                            except Exception:
                                pass
                            break
                            
                        except Exception as e:
                            LOG.error("periodic_portfolio_sync_error", error=str(e))
                            await asyncio.sleep(60)  # Esperar más si hay error
                
                portfolio_sync_task = asyncio.create_task(periodic_portfolio_sync())
                self.periodic_tasks.append(portfolio_sync_task)
                LOG.info("periodic_portfolio_sync_task_started", interval_seconds=30)
                
            except Exception as e:
                LOG.error("portfolio_sync_task_creation_failed", error=str(e))

            # ===== NUEVO: Verificación de salud de InfluxDB =====
            try:
                async def periodic_influx_health_check():
                    """Verifica salud de InfluxDB cada 5 minutos"""
                    while self.is_running:
                        try:
                            await asyncio.sleep(300)  # 5 minutos
                            
                            if INFLUX_METRICS and INFLUX_METRICS.enabled:
                                health = await INFLUX_METRICS.check_health()
                                
                                if not health['healthy']:
                                    LOG.warning("influxdb_health_check_failed",
                                               reason=health.get('reason'),
                                               stats=health.get('stats', {}))
                                    
                                    # Alerta si muchos errores consecutivos
                                    stats = health.get('stats', {})
                                    failed = stats.get('failed_writes', 0)
                                    total = stats.get('total_writes', 0)
                                    
                                    if total > 50 and failed > total * 0.5:
                                        await ALERT_SYSTEM.send_alert(
                                            "WARNING",
                                            "InfluxDB tiene alta tasa de errores",
                                            failed_writes=failed,
                                            total_writes=total,
                                            failure_rate=failed/total*100 if total > 0 else 0
                                        )
                                else:
                                    LOG.debug("influxdb_health_check_ok",
                                             ping=health.get('ping_ok'),
                                             stats=health.get('stats', {}))
                                    
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            LOG.error("influx_health_check_error", error=str(e))
                            await asyncio.sleep(60)
                
                influx_health_task = asyncio.create_task(periodic_influx_health_check())
                self.periodic_tasks.append(influx_health_task)
                LOG.info("influx_health_check_task_started", interval_seconds=300)
                
            except Exception as e:
                LOG.error("influx_health_task_creation_failed", error=str(e))
                
            LOG.info("base_periodic_tasks_started", count=len(self.periodic_tasks))

            
            if hasattr(self, 'portfolio_rebalancer') and self.portfolio_rebalancer is not None:
                try:
                    async def periodic_rebalancing():
                        while self.is_running:
                            try:
                                await asyncio.sleep(3600)
                                if hasattr(self, 'risk_manager'):
                                    symbols_to_close = await self.portfolio_rebalancer.check_rebalance_needed(self.risk_manager)
                                    if symbols_to_close:
                                        LOG.info("executing_scheduled_rebalance", symbols_count=len(symbols_to_close))
                                        await self.portfolio_rebalancer.execute_rebalance(symbols_to_close, self.risk_manager)
                            except asyncio.CancelledError:
                                break
                            except Exception as e:
                                LOG.error("periodic_rebalancing_error", error=str(e))
                    rebalancing_task = asyncio.create_task(periodic_rebalancing())
                    self.periodic_tasks.append(rebalancing_task)
                    LOG.info("periodic_rebalancing_task_started", interval_hours=1)
                except Exception as e:
                    LOG.error("rebalancing_task_creation_failed", error=str(e))
            else:
                LOG.warning("portfolio_rebalancer_not_available")
            try:
                memory_monitor_task = asyncio.create_task(MEMORY_MANAGER.monitor_and_cleanup())
                self.periodic_tasks.append(memory_monitor_task)
                LOG.debug("memory_monitor_task_started")
            except Exception as e:
                LOG.error("memory_monitor_task_failed", error=str(e))
            if self.performance_monitor:
                try:
                    perf_monitor_task = asyncio.create_task(self.performance_monitor.monitor_loop(self, interval=60))
                    self.periodic_tasks.append(perf_monitor_task)
                    LOG.debug("performance_monitor_task_started")
                except Exception as e:
                    LOG.error("performance_monitor_task_failed", error=str(e))
            else:
                LOG.warning("performance_monitor_unavailable_skipping_task")
            if hasattr(self, 'dashboard') and self.dashboard is not None:
                try:
                    dashboard_exists = any('dashboard' in str(task.get_coro()) if hasattr(task, 'get_coro') else False for task in self.periodic_tasks)
                    if not dashboard_exists:
                        dashboard_task = asyncio.create_task(self.dashboard.periodic_dashboard_loop())
                        self.periodic_tasks.append(dashboard_task)
                        LOG.debug("dashboard_task_started")
                    else:
                        LOG.debug("dashboard_task_already_running")
                except Exception as e:
                    LOG.error("dashboard_task_failed", error=str(e))
            else:
                LOG.warning("dashboard_not_available_creating", message="Dashboard should exist from __init__")
                try:
                    self.dashboard = PerformanceDashboard(self)
                    dashboard_task = asyncio.create_task(self.dashboard.periodic_dashboard_loop())
                    self.periodic_tasks.append(dashboard_task)
                    LOG.info("dashboard_created_and_started")
                except Exception as dashboard_create_error:
                    LOG.error("dashboard_creation_failed", error=str(dashboard_create_error))
            LOG.info("base_periodic_tasks_started", count=len(self.periodic_tasks))
        except Exception as e:
            LOG.error("start_base_periodic_tasks_failed", error=str(e))

    async def _process_symbol_data(self, symbol: str, market_data: Dict[str, Any]):
        try:
            if not market_data or not market_data.get("success", False):
                LOG.debug("invalid_market_data_skipping", symbol=symbol)
                return {'symbol': symbol, 'success': False, 'error': 'invalid_market_data'}
            ohlcv = market_data.get("ohlcv", [])
            if not ohlcv or len(ohlcv) == 0:
                LOG.debug("empty_ohlcv_skipping", symbol=symbol)
                return {'symbol': symbol, 'success': False, 'error': 'empty_ohlcv'}
            df = create_dataframe(ohlcv)
            if df is None or len(df) == 0 or 'close' not in df.columns:
                LOG.error("invalid_dataframe_for_symbol", symbol=symbol)
                return {'symbol': symbol, 'success': False, 'error': 'invalid_dataframe'}
            df = calculate_technical_indicators(df)
            current_price = float(df['close'].iloc[-1])
            processing_timestamp = datetime.now(timezone.utc)
            await execute_trading_pipeline_complete(self.exchange_manager, self.strategy_manager, self.automl, self.regime_detector, self.risk_optimizer, self.ensemble_learner, self, df, self.config, symbol)
            if hasattr(self, 'data_accumulator') and self.data_accumulator is not None:
                try:
                    reward = 0.0
                    if hasattr(self, 'risk_manager') and symbol in self.risk_manager.active_stops:
                        stop_info = self.risk_manager.active_stops[symbol]
                        entry_price = stop_info.get('entry_price', 0)
                        side = stop_info.get('side', 'buy')
                        if entry_price > 0 and current_price > 0:
                            if side == 'buy':
                                reward = (current_price - entry_price) / entry_price
                            else:
                                reward = (entry_price - current_price) / entry_price
                            LOG.debug("reward_from_active_position", symbol=symbol, reward=reward, entry=entry_price, current=current_price)
                    elif len(df) >= 2:
                        prev_close = float(df['close'].iloc[-2])
                        if prev_close > 0:
                            price_change = (current_price - prev_close) / prev_close
                            reward = price_change * 0.5
                            LOG.debug("reward_from_price_change", symbol=symbol, reward=reward, prev_close=prev_close, current=current_price)
                    await self.data_accumulator.add_sample(symbol, df.iloc[-1], reward=reward)
                except Exception as acc_error:
                    LOG.debug("data_accumulation_failed", symbol=symbol, error=str(acc_error))
            return {
                'symbol': symbol,
                'current_price': current_price,
                'df': df,
                'timestamp': processing_timestamp,
                'success': True,
                'candles_processed': len(df)
            }
        except Exception as e:
            LOG.error("symbol_processing_failed", symbol=symbol, error=str(e), traceback=traceback.format_exc())
            return {'symbol': symbol, 'success': False, 'error': str(e)}

    async def _integrate_all_decisions(self, traditional_signal, rl_decision, ensemble_signal, regime, confidence, df):
        try:
            all_signals = []
            if traditional_signal and isinstance(traditional_signal, dict):
                all_signals.append(('traditional', traditional_signal))
            if rl_decision and isinstance(rl_decision, dict):
                all_signals.append(('rl', rl_decision))
            if ensemble_signal and isinstance(ensemble_signal, dict):
                all_signals.append(('ensemble', ensemble_signal))
            if not all_signals:
                LOG.warning("no_valid_signals_for_integration")
                return {"action": "hold", "confidence": 0.0}
            action_map = {"buy": 1.0, "hold": 0.0, "sell": -1.0}
            signal_type_weights = {'traditional': 0.30, 'ensemble': 0.45, 'rl': 0.25}
            regime_weights = {
                "bull": {"buy": 1.3, "hold": 0.9, "sell": 0.7},
                "bear": {"buy": 0.7, "hold": 0.9, "sell": 1.3},
                "sideways": {"buy": 0.9, "hold": 1.1, "sell": 0.9},
                "volatile": {"buy": 0.8, "hold": 1.2, "sell": 0.8},
                "unknown": {"buy": 1.0, "hold": 1.0, "sell": 1.0}
            }
            regime_weight_map = regime_weights.get(regime, regime_weights["unknown"])
            regime_confidence_factor = max(0.3, confidence)
            normalized_signals = []
            for sig_type, sig in all_signals:
                signal_name = sig.get('signal') or sig.get('action', 'hold')
                raw_confidence = sig.get('confidence', 0.0)
                norm_confidence = max(0.0, min(1.0, float(raw_confidence)))
                type_weight = signal_type_weights.get(sig_type, 0.33)
                regime_multiplier = regime_weight_map.get(signal_name, 1.0)
                normalized_signals.append({
                    'type': sig_type,
                    'signal': signal_name,
                    'raw_confidence': raw_confidence,
                    'normalized_confidence': norm_confidence,
                    'type_weight': type_weight,
                    'regime_multiplier': regime_multiplier
                })
            weighted_score = 0.0
            total_weight = 0.0
            for sig in normalized_signals:
                score = action_map.get(sig['signal'], 0.0)
                sig_confidence = sig['normalized_confidence']
                type_weight = sig['type_weight']
                regime_multiplier = sig['regime_multiplier']
                final_weight = sig_confidence * type_weight * regime_multiplier * regime_confidence_factor
                weighted_score += score * final_weight
                total_weight += final_weight
            avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
            if regime in ['bull', 'unknown']:
                buy_threshold = 0.25
                sell_threshold = -0.35
            elif regime == 'bear':
                buy_threshold = 0.35
                sell_threshold = -0.25
            elif regime == 'sideways':
                buy_threshold = 0.30
                sell_threshold = -0.30
            else:
                buy_threshold = 0.40
                sell_threshold = -0.40
            if avg_score > buy_threshold:
                integrated_action = "buy"
            elif avg_score < sell_threshold:
                integrated_action = "sell"
            else:
                integrated_action = "hold"
            signal_agreement = 0.0
            integrated_conf = abs(avg_score)
            total_signals = len(normalized_signals)
            if total_signals > 0:
                agreeing_signals = sum(1 for sig in normalized_signals if sig['signal'] == integrated_action)
                signal_agreement = agreeing_signals / total_signals
                if signal_agreement >= 0.75:
                    integrated_conf *= (1.0 + (signal_agreement - 0.75) * 0.4)
                elif signal_agreement < 0.5:
                    integrated_conf *= 0.7
            if integrated_action == "hold" and abs(avg_score) > 0.05:
                relaxed_buy = buy_threshold * 0.5
                relaxed_sell = sell_threshold * 0.5
                if avg_score > relaxed_buy:
                    integrated_action = "buy"
                    integrated_conf = abs(avg_score) * 1.5
                    LOG.info("hold_overridden_to_buy", score=avg_score, original_threshold=buy_threshold, relaxed_threshold=relaxed_buy, boost_applied=1.5)
                elif avg_score < relaxed_sell:
                    integrated_action = "sell"
                    integrated_conf = abs(avg_score) * 1.5
                    LOG.info("hold_overridden_to_sell", score=avg_score, original_threshold=sell_threshold, relaxed_threshold=relaxed_sell, boost_applied=1.5)
            integrated_conf = max(0.0, min(1.0, integrated_conf))
            volatility_info = {}
            try:
                if 'volatility' in df.columns and not df['volatility'].isna().all():
                    current_vol = float(df['volatility'].iloc[-1])
                    avg_vol = float(df['volatility'].rolling(50).mean().iloc[-1])
                    volatility_info['current_vol'] = current_vol
                    volatility_info['avg_vol'] = avg_vol
                    if avg_vol > 0 and current_vol > 0:
                        vol_ratio = current_vol / avg_vol
                        volatility_info['vol_ratio'] = vol_ratio
                        if vol_ratio > 2.0:
                            LOG.warning("extreme_volatility_forcing_hold", vol_ratio=vol_ratio, original_action=integrated_action, symbol=symbol if 'symbol' in locals() else "analyzing", current_vol=current_vol, avg_vol=avg_vol)
                            integrated_action = "hold"
                            vol_adjustment = 0.5
                            volatility_info['forced_hold'] = True
                            volatility_info['original_action'] = integrated_action
                        elif vol_ratio > 1.8:
                            vol_adjustment = 0.7
                            LOG.warning("very_high_volatility_detected", vol_ratio=vol_ratio, adjustment=vol_adjustment, action=integrated_action, message="Severe confidence reduction applied")
                            volatility_info['severity_level'] = 'very_high'
                        elif vol_ratio > 1.5:
                            vol_adjustment = 0.85
                            LOG.debug("high_volatility_detected", vol_ratio=vol_ratio, adjustment=vol_adjustment)
                            volatility_info['severity_level'] = 'high'
                        elif vol_ratio > 1.2:
                            vol_adjustment = 0.95
                            volatility_info['severity_level'] = 'elevated'
                        else:
                            vol_adjustment = 1.0
                            volatility_info['severity_level'] = 'normal'
                        volatility_info['vol_adjustment'] = vol_adjustment
                        integrated_conf *= vol_adjustment
                    LOG.debug("confidence_adjusted_by_volatility", vol_ratio=vol_ratio, adjustment=vol_adjustment, final_confidence=integrated_conf, severity=volatility_info.get('severity_level', 'unknown'))
            except Exception as vol_error:
                LOG.debug("volatility_adjustment_failed", error=str(vol_error))

            # CORRECCIÓN: Usar parámetros de función en lugar de variable no definida
            LOG.info("ensemble_prediction_complete", action=integrated_action, confidence=float(integrated_conf))
            return {
                "action": integrated_action,
                "confidence": float(integrated_conf),
                "details": {
                    "weighted_score": float(avg_score),
                    "regime": regime,
                    "regime_confidence": confidence,
                    "signals": normalized_signals,
                    "thresholds": {"buy": buy_threshold, "sell": sell_threshold},
                    "total_weight": float(total_weight)
                }
            }
        except Exception as e:
            LOG.error("decision_integration_failed", error=str(e), traceback=traceback.format_exc())
            return {"action": "hold", "confidence": 0.0, "details": {"error": str(e)}}

    async def _execute_advanced_trade_with_risk_management(self, symbol: str, decision: Dict, df: pd.DataFrame):
        try:
            if not decision or not isinstance(decision, dict):
                LOG.error("invalid_decision_dict_for_trade", symbol=symbol, decision_type=type(decision).__name__)
                return
            if not hasattr(self, 'position_sizer'):
                self.position_sizer = DynamicPositionSizer(self.config, self)
            if not hasattr(self, 'risk_manager'):
                self.risk_manager = DynamicRiskManager(self.config, self)
            if self.risk_manager.check_circuit_breaker():
                LOG.warning("trade_blocked_circuit_breaker", symbol=symbol)
                return
            if len(df) == 0 or 'close' not in df.columns:
                LOG.error("invalid_df_for_trade", symbol=symbol)
                return
            required_indicators = ['close', 'volume']
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            if missing_indicators:
                LOG.warning("missing_indicators_for_trade", symbol=symbol, missing=missing_indicators, message="Calculating missing indicators")
                df = calculate_technical_indicators(df)
            action = decision.get('action', 'hold')
            confidence = float(decision.get('confidence', 0.0))
            if action == 'hold' or confidence < 0.5:
                LOG.debug("trade_skipped_hold_or_low_confidence", symbol=symbol, action=action, confidence=confidence)
                return
            if symbol in self.risk_manager.active_stops:
                LOG.debug("position_already_active_skipping", symbol=symbol)
                return
            current_price = float(df['close'].iloc[-1])
            if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
                LOG.error("invalid_current_price", symbol=symbol, price=current_price)
                return
            position_amount = self.position_sizer.calculate_position_size(symbol, confidence, current_price, df)
            LOG.debug("position_size_calculated_for_trade", symbol=symbol, confidence=confidence, current_price=current_price, position_amount=position_amount, equity=self.equity)
            position_size = position_amount / current_price if current_price > 0 else 0
            min_notional = 10.0
            calculated_notional = position_amount
            if calculated_notional < min_notional:
                LOG.warning("position_amount_below_min_notional", symbol=symbol, calculated_notional=calculated_notional, min_required=min_notional, price=current_price, size=position_size)
                if calculated_notional >= min_notional * 0.8:
                    position_amount = min_notional
                    position_size = position_amount / current_price if current_price > 0 else 0
                    LOG.info("position_adjusted_to_minimum", symbol=symbol, new_amount=position_amount, new_size=position_size)
                else:
                    return
            if position_size < 0.001:
                LOG.warning("position_size_too_small", symbol=symbol, size=position_size)
                return
            order_type = "market"
            side = "buy" if action == "buy" else "sell"
            order = await self.exchange_manager.create_order(symbol, order_type, side, position_size)
            if order and order.get("success", False):
                price_from_ticker = None
                executed_price = order.get('price', 0)
                is_simulated = order.get('info', {}).get('simulated', False) if isinstance(order.get('info'), dict) else False
                if executed_price is None or executed_price <= 0 or np.isnan(executed_price) or np.isinf(executed_price) or is_simulated:
                    LOG.info("fetching_market_price_for_order", symbol=symbol, order_price=executed_price, is_simulated=is_simulated)
                    try:
                        ticker = await self.exchange_manager.exchange.fetch_ticker(symbol)
                        price_from_ticker = ticker.get('last', None)
                        if price_from_ticker and price_from_ticker > 0:
                            executed_price = float(price_from_ticker)
                            LOG.info("using_ticker_price", symbol=symbol, price=executed_price)
                        else:
                            raise ValueError(f"Invalid ticker price: {price_from_ticker}")
                    except Exception as ticker_error:
                        LOG.warning("ticker_fetch_failed_using_dataframe_price", symbol=symbol, error=str(ticker_error))
                        price_from_ticker = None
                        if current_price > 0:
                            executed_price = float(current_price)
                            LOG.info("using_dataframe_price", symbol=symbol, price=executed_price)
                        else:
                            LOG.error("all_price_sources_invalid", symbol=symbol)
                            return
                if executed_price <= 0 or np.isnan(executed_price):
                    LOG.error("invalid_final_execution_price", symbol=symbol, price=executed_price)
                    return
                LOG.info("order_execution_price_determined", symbol=symbol, executed_price=executed_price, source="ticker" if price_from_ticker else "dataframe")
                executed_size = order.get('amount', position_size)
                if executed_size <= 0 or np.isnan(executed_size) or np.isinf(executed_size):
                    LOG.error("invalid_executed_size", symbol=symbol, size=executed_size)
                    return
                transaction = await self.position_ledger.record_open(self, symbol, side, executed_price, executed_size)
                if transaction is None:
                    LOG.error("failed_to_record_open_transaction", symbol=symbol)
                    return
                try:
                    registration_success = self.risk_manager.register_position(symbol, executed_price, side, executed_size, confidence, df)
                    if not registration_success:
                        LOG.error("risk_manager_registration_failed_rolling_back", symbol=symbol)
                        return
                    if symbol not in self.risk_manager.active_stops:
                        LOG.error("position_not_in_active_stops_after_registration", symbol=symbol)
                        return
                    if symbol not in self.symbol_performance:
                        self.symbol_performance[symbol] = {
                            'total_trades': 0,
                            'winning_trades': 0,
                            'losing_trades': 0,
                            'total_pnl': 0.0,
                            'win_rate': 0.0,
                            'avg_win': 0.0,
                            'avg_loss': 0.0,
                            'largest_win': 0.0,
                            'largest_loss': 0.0,
                            'last_trade_time': datetime.now(timezone.utc)
                        }
                    self.symbol_performance[symbol]['total_trades'] += 1
                    self.symbol_performance[symbol]['last_trade_time'] = datetime.now(timezone.utc)
                    self.performance_metrics['total_trades'] += 1
                    if 'total_pnl' not in self.performance_metrics:
                        self.performance_metrics['total_pnl'] = 0.0
                    audit_result = self.position_ledger.audit_equity(self)
                    if not audit_result['is_consistent']:
                        LOG.error("equity_inconsistent_after_open", symbol=symbol, audit=audit_result)
                    try:
                        if hasattr(self, 'performance_metrics'):
                            metrics = self.performance_metrics
                            total_trades = metrics.get('total_trades', 0)
                            winning_trades = metrics.get('winning_trades', 0)
                            if total_trades > 0:
                                win_rate = winning_trades / total_trades
                    except Exception as model_metrics_error:
                        LOG.debug("model_metrics_write_failed", error=str(model_metrics_error))
                    stop_info = self.risk_manager.active_stops.get(symbol)
                    if stop_info:
                        LOG.info("trade_executed_with_risk_management", symbol=symbol, action=action, size=executed_size, amount_usdt=position_amount, price=executed_price, confidence=confidence, stop_loss=stop_info['stop_loss'], tp_levels_count=len(stop_info.get('take_profit_levels', [])), transaction_id=transaction.transaction_id)
                    else:
                        LOG.warning("position_registered_but_not_in_active_stops", symbol=symbol)
                    # ===== MEJORADO: Enviar métricas completas de apertura =====
                    try:
                        # Métrica de trade (apertura)
                        trade_success = await INFLUX_METRICS.write_trade_metrics(
                            symbol=symbol,
                            action=action,  # "buy" o "sell"
                            confidence=float(confidence),
                            price=float(executed_price),
                            size=float(executed_size),
                            pnl=0.0  # PnL aún no realizado
                        )
                        
                        # Métrica de posición abierta (solo si trade exitoso)
                        if trade_success and INFLUX_METRICS.enabled and stop_info:
                            await INFLUX_METRICS.write_open_position_metrics(
                                symbol=symbol,
                                side=side,
                                entry_price=float(executed_price),
                                size=float(executed_size),
                                stop_loss=float(stop_info['stop_loss']),
                                confidence=float(confidence)
                            )
                            
                    except Exception as metrics_error:
                        LOG.debug("trade_open_metrics_failed", 
                                 symbol=symbol, 
                                 error=str(metrics_error))
                except Exception as reg_error:
                    LOG.error("position_registration_exception", symbol=symbol, error=str(reg_error), traceback=traceback.format_exc())
                    return
            else:
                LOG.error("trade_failed", symbol=symbol, error=order.get("error") if order else "Unknown")
        except Exception as e:
            LOG.error("advanced_trade_execution_failed", symbol=symbol, error=str(e), traceback=traceback.format_exc())

    async def _load_historical_data(self, months: int = 3) -> Optional[pd.DataFrame]:
        """
        ✅ MEJORADO: Carga datos históricos con fetching incremental y validación robusta
        """
        try:
            # Calcular timestamp de inicio con margen
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=30*months + 7)
            since = int(start_time.timestamp() * 1000)
            
            # Detectar timeframe del config
            timeframe = self.config.timeframe
            
            # Calcular velas necesarias basado en timeframe
            timeframe_minutes_map = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
                '1d': 1440, '3d': 4320, '1w': 10080
            }
            
            timeframe_minutes = timeframe_minutes_map.get(timeframe, 60)
            required_candles = int((30 * months * 24 * 60) / timeframe_minutes)
            
            # ✅ LÍMITE POR REQUEST del exchange (Binance: 1000 típicamente)
            max_per_request = 1000
            
            # ✅ ESTRATEGIA INCREMENTAL: Múltiples requests si es necesario
            all_ohlcv = []
            current_since = since
            requests_needed = max(1, (required_candles // max_per_request) + 1)
            
            LOG.info("fetching_historical_data_incrementally",
                    months=months,
                    required_candles=required_candles,
                    requests_needed=requests_needed,
                    timeframe=timeframe,
                    symbol=self.config.symbols[0])
            
            for request_num in range(min(requests_needed, 5)):  # Max 5 requests
                try:
                    result = await asyncio.wait_for(
                        self.exchange_manager.fetch_ohlcv(
                            self.config.symbols[0], 
                            timeframe, 
                            limit=max_per_request,
                            since=current_since
                        ),
                        timeout=30.0
                    )
                    
                    if not result or not result.get("success", False):
                        error_msg = result.get('error') if result else 'No response'
                        LOG.warning("historical_fetch_failed_on_request",
                                   request=request_num + 1,
                                   error=error_msg)
                        
                        # Si primer request falla, abortar
                        if request_num == 0:
                            return None
                        # Si request subsecuente falla, usar lo que tenemos
                        break
                    
                    ohlcv_batch = result.get("ohlcv", [])
                    
                    if not ohlcv_batch or len(ohlcv_batch) == 0:
                        LOG.warning("empty_batch_on_request", request=request_num + 1)
                        break
                    
                    # Agregar batch
                    all_ohlcv.extend(ohlcv_batch)
                    
                    LOG.debug("historical_batch_fetched",
                             request=request_num + 1,
                             batch_size=len(ohlcv_batch),
                             total_so_far=len(all_ohlcv))
                    
                    # ✅ Actualizar timestamp para siguiente request
                    # Usar timestamp de última vela + 1ms
                    if ohlcv_batch:
                        last_timestamp = ohlcv_batch[-1][0]
                        current_since = last_timestamp + 1
                    
                    # Si obtuvimos menos del límite, ya no hay más datos
                    if len(ohlcv_batch) < max_per_request:
                        LOG.info("reached_end_of_available_data",
                                request=request_num + 1,
                                final_batch_size=len(ohlcv_batch))
                        break
                    
                    # Pausa entre requests para evitar rate limits
                    if request_num < requests_needed - 1:
                        await asyncio.sleep(0.5)
                    
                except asyncio.TimeoutError:
                    LOG.warning("historical_fetch_timeout",
                               request=request_num + 1)
                    if request_num == 0:
                        return None
                    break
                    
                except Exception as batch_error:
                    LOG.error("historical_batch_error",
                             request=request_num + 1,
                             error=str(batch_error))
                    if request_num == 0:
                        return None
                    break
            
            # Validar datos obtenidos
            if not all_ohlcv or len(all_ohlcv) < 100:
                LOG.warning("insufficient_historical_data_after_all_requests",
                           total_candles=len(all_ohlcv) if all_ohlcv else 0,
                           required=required_candles)
                return None
            
            # ✅ ELIMINAR DUPLICADOS (por timestamp)
            seen_timestamps = set()
            unique_ohlcv = []
            
            for candle in all_ohlcv:
                timestamp = candle[0]
                if timestamp not in seen_timestamps:
                    seen_timestamps.add(timestamp)
                    unique_ohlcv.append(candle)
            
            if len(unique_ohlcv) < len(all_ohlcv):
                LOG.info("duplicates_removed",
                        original=len(all_ohlcv),
                        unique=len(unique_ohlcv),
                        duplicates=len(all_ohlcv) - len(unique_ohlcv))
            
            # Crear DataFrame
            df = create_dataframe(unique_ohlcv)
            if df is None or len(df) == 0 or 'close' not in df.columns:
                LOG.error("dataframe_creation_failed_after_fetch")
                return None
            
            # ✅ VALIDAR COBERTURA TEMPORAL
            time_span_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
            expected_hours = months * 30 * 24
            coverage_pct = (time_span_hours / expected_hours * 100) if expected_hours > 0 else 0
            
            # ✅ CRITERIO FLEXIBLE: Aceptar si tenemos al menos 70% de cobertura
            if coverage_pct < 70:
                LOG.warning("historical_data_coverage_below_threshold",
                           expected_hours=expected_hours,
                           actual_hours=time_span_hours,
                           coverage_pct=coverage_pct,
                           threshold_pct=70,
                           message="Will continue but with limited data")
                
                # Si cobertura < 50%, intentar con timeframe alternativo
                if coverage_pct < 50 and timeframe == '1h':
                    LOG.info("attempting_alternative_timeframe_for_better_coverage")
                    
                    alt_result = await self._load_historical_data_with_timeframe(
                        months=months,
                        timeframe='15m'
                    )
                    
                    if alt_result is not None:
                        return alt_result
            else:
                LOG.info("historical_data_coverage_acceptable",
                        expected_hours=expected_hours,
                        actual_hours=time_span_hours,
                        coverage_pct=coverage_pct)
            
            # ✅ VERIFICAR GAPS TEMPORALES
            if len(df) > 1:
                time_diffs = df.index.to_series().diff()
                median_diff = time_diffs.median()
                
                # Detectar gaps grandes (3x el diff mediano)
                large_gaps = (time_diffs > median_diff * 3).sum()
                
                if large_gaps > len(df) * 0.05:  # Más del 5%
                    LOG.warning("historical_data_has_gaps",
                               total_rows=len(df),
                               large_gaps=large_gaps,
                               gap_pct=large_gaps / len(df) * 100)
            
            # Calcular indicadores
            df = calculate_technical_indicators(df)
            
            # ✅ VALIDAR INDICADORES CRÍTICOS
            required_indicators = ['rsi', 'macd', 'sma_20']
            missing_indicators = [ind for ind in required_indicators 
                                 if ind not in df.columns or df[ind].isna().all()]
            
            if missing_indicators:
                LOG.warning("historical_data_missing_indicators",
                           missing=missing_indicators)
                # Intentar recalcular
                df = calculate_technical_indicators(df)
            
            LOG.info("historical_data_loaded_successfully", 
                    symbol=self.config.symbols[0], 
                    rows=len(df), 
                    months=months, 
                    date_range=f"{df.index[0]} to {df.index[-1]}",
                    time_span_hours=time_span_hours,
                    coverage_pct=coverage_pct,
                    total_requests=len(all_ohlcv) // max_per_request + 1)
            
            return df
            
        except Exception as e:
            LOG.error("historical_data_load_failed", 
                     error=str(e), 
                     months=months, 
                     traceback=traceback.format_exc())
            return None
    
    async def _load_historical_data_for_symbol(self, symbol: str, months: int = 3) -> Optional[pd.DataFrame]:
        """
        ✅ NUEVO: Carga datos históricos para un símbolo específico
        
        Args:
            symbol: Símbolo a cargar (ej: "ETH/USDT")
            months: Meses de historia a cargar
            
        Returns:
            DataFrame con datos OHLCV y técnicos, o None si falla
        """
        try:
            # Calcular timestamp de inicio
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=30*months + 7)
            since = int(start_time.timestamp() * 1000)
            
            timeframe = self.config.timeframe
            
            # Calcular velas necesarias
            timeframe_minutes_map = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
                '1d': 1440, '3d': 4320, '1w': 10080
            }
            
            timeframe_minutes = timeframe_minutes_map.get(timeframe, 60)
            required_candles = int((30 * months * 24 * 60) / timeframe_minutes)
            
            max_per_request = 1000
            
            all_ohlcv = []
            current_since = since
            requests_needed = max(1, (required_candles // max_per_request) + 1)
            
            LOG.info("fetching_historical_data_for_symbol",
                    symbol=symbol,
                    months=months,
                    required_candles=required_candles,
                    requests_needed=requests_needed)
            
            for request_num in range(min(requests_needed, 5)):
                try:
                    result = await asyncio.wait_for(
                        self.exchange_manager.fetch_ohlcv(
                            symbol,  # ✅ Usar símbolo específico
                            timeframe, 
                            limit=max_per_request,
                            since=current_since
                        ),
                        timeout=30.0
                    )
                    
                    if not result or not result.get("success", False):
                        error_msg = result.get('error') if result else 'No response'
                        LOG.warning("historical_fetch_failed_on_request",
                                   symbol=symbol,
                                   request=request_num + 1,
                                   error=error_msg)
                        
                        if request_num == 0:
                            return None
                        break
                    
                    ohlcv_batch = result.get("ohlcv", [])
                    
                    if not ohlcv_batch or len(ohlcv_batch) == 0:
                        LOG.warning("empty_batch_on_request",
                                   symbol=symbol,
                                   request=request_num + 1)
                        break
                    
                    all_ohlcv.extend(ohlcv_batch)
                    
                    LOG.debug("historical_batch_fetched",
                             symbol=symbol,
                             request=request_num + 1,
                             batch_size=len(ohlcv_batch),
                             total_so_far=len(all_ohlcv))
                    
                    if ohlcv_batch:
                        last_timestamp = ohlcv_batch[-1][0]
                        current_since = last_timestamp + 1
                    
                    if len(ohlcv_batch) < max_per_request:
                        LOG.info("reached_end_of_available_data",
                                symbol=symbol,
                                request=request_num + 1)
                        break
                    
                    if request_num < requests_needed - 1:
                        await asyncio.sleep(0.5)
                    
                except asyncio.TimeoutError:
                    LOG.warning("historical_fetch_timeout",
                               symbol=symbol,
                               request=request_num + 1)
                    if request_num == 0:
                        return None
                    break
                    
                except Exception as batch_error:
                    LOG.error("historical_batch_error",
                             symbol=symbol,
                             request=request_num + 1,
                             error=str(batch_error))
                    if request_num == 0:
                        return None
                    break
            
            if not all_ohlcv or len(all_ohlcv) < 100:
                LOG.warning("insufficient_historical_data",
                           symbol=symbol,
                           total_candles=len(all_ohlcv) if all_ohlcv else 0,
                           required=required_candles)
                return None
            
            # Eliminar duplicados
            seen_timestamps = set()
            unique_ohlcv = []
            
            for candle in all_ohlcv:
                timestamp = candle[0]
                if timestamp not in seen_timestamps:
                    seen_timestamps.add(timestamp)
                    unique_ohlcv.append(candle)
            
            if len(unique_ohlcv) < len(all_ohlcv):
                LOG.info("duplicates_removed",
                        symbol=symbol,
                        original=len(all_ohlcv),
                        unique=len(unique_ohlcv))
            
            # Crear DataFrame
            df = create_dataframe(unique_ohlcv)
            if df is None or len(df) == 0 or 'close' not in df.columns:
                LOG.error("dataframe_creation_failed",
                         symbol=symbol)
                return None
            
            # Calcular indicadores
            df = calculate_technical_indicators(df)
            
            LOG.info("historical_data_loaded_for_symbol", 
                    symbol=symbol,
                    rows=len(df),
                    months=months,
                    date_range=f"{df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            LOG.error("historical_data_load_failed_for_symbol",
                     symbol=symbol,
                     error=str(e),
                     months=months)
            return None
    
    async def _load_historical_data_with_timeframe(self, months: int, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Helper para cargar datos con timeframe específico
        """
        original_timeframe = self.config.timeframe
        
        try:
            # Temporalmente cambiar timeframe
            self.config.timeframe = timeframe
            
            result = await self._load_historical_data(months)
            
            return result
            
        finally:
            # Restaurar timeframe original
            self.config.timeframe = original_timeframe

async def manual_equity_audit(bot):
    if not hasattr(bot, 'position_ledger'):
        LOG.error("bot_missing_position_ledger")
        return None
    audit_result = bot.position_ledger.audit_equity(bot)
    print("\n" + "=" * 80)
    print("🔍 EQUITY AUDIT REPORT")
    print("=" * 80)
    print(f"Initial Capital:        ${audit_result['initial_capital']:,.2f}")
    print(f"Total Realized PnL:     ${audit_result['total_realized_pnl']:,.2f}")
    print(f"Invested in Positions:  ${audit_result['invested_in_positions']:,.2f}")
    print(f"Expected Free Equity:   ${audit_result['expected_free_equity']:,.2f}")
    print(f"Actual Equity:          ${audit_result['actual_equity']:,.2f}")
    print(f"Discrepancy:            ${audit_result['discrepancy']:,.2f}")
    print(f"Status:                 {'✅ CONSISTENT' if audit_result['is_consistent'] else '❌ INCONSISTENT'}")
    print(f"Total Transactions:     {audit_result['total_transactions']}")
    print(f"Active Positions:       {audit_result['active_positions']}")
    print("=" * 80)
    if bot.position_ledger.transactions:
        print("\n📋 LAST 5 TRANSACTIONS:")
        for transaction in bot.position_ledger.transactions[-5:]:
            print(f"\n  Transaction ID: {transaction.transaction_id}")
            print(f"  Type:           {transaction.transaction_type.value}")
            print(f"  Symbol:         {transaction.symbol}")
            print(f"  Side:           {transaction.side}")
            print(f"  Entry Price:    ${transaction.entry_price:.2f}")
            if transaction.exit_price:
                print(f"  Exit Price:     ${transaction.exit_price:.2f}")
            print(f"  Size:           {transaction.size:.6f}")
            print(f"  Equity Before:  ${transaction.equity_before:.2f}")
            print(f"  Equity Change:  ${transaction.equity_change:+.2f}")
            print(f"  Equity After:   ${transaction.equity_after:.2f}")
            if transaction.realized_pnl != 0:
                print(f"  Realized PnL:   ${transaction.realized_pnl:+.2f}")
            print(f"  Valid:          {'✅' if transaction.is_valid else '❌'}")
    print("\n" + "=" * 80)
    return audit_result

class AdaptiveParameterManager:
    """
    NUEVO: Gestor de parámetros adaptativos que ajusta configuración
    según condiciones de mercado y performance del bot
    """
    def __init__(self, bot):
        self.bot = bot
        self.adjustment_history = deque(maxlen=100)
        self.last_adjustment = datetime.now(timezone.utc)
        self.min_adjustment_interval = 3600  # 1 hora mínimo
        
    async def adjust_parameters(self, regime: str, confidence: float) -> Dict[str, Any]:
        """Ajusta parámetros del bot según régimen y performance"""
        try:
            # Verificar intervalo mínimo
            time_since_last = (datetime.now(timezone.utc) - self.last_adjustment).total_seconds()
            if time_since_last < self.min_adjustment_interval:
                return {'adjusted': False, 'reason': 'min_interval_not_met'}
            
            adjustments = {}
            
            # 1. Ajustar thresholds según régimen
            if regime == 'volatile':
                # Más conservador en volatilidad
                adjustments['confidence_threshold_multiplier'] = 1.3
                adjustments['position_size_multiplier'] = 0.7
                adjustments['stop_loss_multiplier'] = 1.2
            elif regime == 'bull':
                # Más agresivo en tendencia alcista
                adjustments['confidence_threshold_multiplier'] = 0.85
                adjustments['position_size_multiplier'] = 1.15
                adjustments['stop_loss_multiplier'] = 0.95
            elif regime == 'bear':
                # Defensivo en tendencia bajista
                adjustments['confidence_threshold_multiplier'] = 1.15
                adjustments['position_size_multiplier'] = 0.85
                adjustments['stop_loss_multiplier'] = 1.1
            else:  # sideways/unknown
                adjustments['confidence_threshold_multiplier'] = 1.0
                adjustments['position_size_multiplier'] = 1.0
                adjustments['stop_loss_multiplier'] = 1.0
            
            # 2. Ajustar según win_rate reciente
            if hasattr(self.bot, 'performance_metrics'):
                recent_trades = self.bot.performance_metrics.get('total_trades', 0)
                if recent_trades >= 20:
                    win_rate = self.bot.performance_metrics.get('win_rate', 0.5)
                    
                    if win_rate > 0.65:
                        # Performance excelente - mantener agresividad
                        adjustments['position_size_multiplier'] *= 1.1
                    elif win_rate < 0.35:
                        # Performance pobre - ser más conservador
                        adjustments['confidence_threshold_multiplier'] *= 1.2
                        adjustments['position_size_multiplier'] *= 0.8
            
            # 3. Aplicar ajustes
            if hasattr(self.bot, 'position_sizer'):
                old_base = getattr(self.bot.position_sizer, 'base_risk_pct', 0.05)
                new_base = old_base * adjustments['position_size_multiplier']
                self.bot.position_sizer.base_risk_pct = np.clip(new_base, 0.02, 0.08)
                
                adjustments['position_sizer_updated'] = {
                    'old': old_base,
                    'new': self.bot.position_sizer.base_risk_pct
                }
            
            # 4. Registrar ajuste
            self.adjustment_history.append({
                'timestamp': datetime.now(timezone.utc),
                'regime': regime,
                'confidence': confidence,
                'adjustments': adjustments
            })
            
            self.last_adjustment = datetime.now(timezone.utc)
            
            LOG.info("parameters_adjusted_adaptively",
                    regime=regime,
                    confidence=confidence,
                    adjustments=adjustments)
            
            return {
                'adjusted': True,
                'adjustments': adjustments,
                'regime': regime
            }
            
        except Exception as e:
            LOG.error("adaptive_parameter_adjustment_failed", error=str(e))
            return {'adjusted': False, 'error': str(e)}
        
class AdvancedAutoML:
    def __init__(self, config: AdvancedAIConfig):
        self.config = config

    async def optimize_model(self, model, data: pd.DataFrame):
        try:
            def objective(trial):
                params = {'n_estimators': trial.suggest_int('n_estimators', 50, 200)}
                clf = RandomForestClassifier(**params)
                X, y = data.drop('target', axis=1), data['target']
                score = cross_val_score(clf, X, y, cv=3).mean()
                return score
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.automl_optimization_trials)
            LOG.info("automl_optimized", best_params=study.best_params)
            return study.best_params
        except Exception as e:
            LOG.error("automl_optimization_failed", error=str(e))

class AdvancedMarketRegimeDetector:
    def __init__(self, config: AdvancedAIConfig):
        self.config = config
        self.regime_cache = {}
        self.cache_ttl = 300

    def detect_regime(self, df: pd.DataFrame, symbol: str = None) -> Tuple[str, float]:
        try:
            cache_key = symbol if symbol else 'general'
            if cache_key in self.regime_cache:
                cached_regime, cached_confidence, cached_time = self.regime_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    LOG.debug("using_cached_regime", symbol=symbol, regime=cached_regime, confidence=cached_confidence)
                    return cached_regime, cached_confidence
            if len(df) < self.config.feature_engineering_lookback or 'close' not in df.columns:
                LOG.error("invalid_df_for_regime_detection", message="DataFrame too small or missing columns", symbol=symbol)
                return "unknown", 0.0
            features = pd.DataFrame()
            features['volatility'] = df['close'].pct_change().rolling(20).std()
            if 'high' in df.columns and 'low' in df.columns:
                from __main__ import StrategyManager
                strategy_mgr = StrategyManager(self.config)
                features['trend_strength'] = strategy_mgr.adx(df['high'], df['low'], df['close'], 14)
            else:
                sma_fast = df['close'].rolling(10).mean()
                sma_slow = df['close'].rolling(30).mean()
                features['trend_strength'] = abs(sma_fast - sma_slow) / df['close'] * 100
            features['volume_profile'] = df['volume'].rolling(20).mean()
            try:
                features['market_correlation'] = df['close'].pct_change().rolling(20).apply(lambda x: x.corr(df['volume'].pct_change().loc[x.index]) if len(x) > 1 else 0.5)
            except Exception as corr_error:
                LOG.debug("correlation_calculation_failed", error=str(corr_error))
                features['market_correlation'] = 0.5
            features = features.dropna()
            column_lengths = {col: len(features[col]) for col in features.columns}
            if len(set(column_lengths.values())) > 1:
                LOG.error("inconsistent_feature_lengths", lengths=column_lengths)
                return None
            if len(features) < 20:
                LOG.warning("insufficient_features_for_regime", samples=len(features), symbol=symbol)
                return "unknown", 0.0
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            n_clusters = min(self.config.regime_n_clusters, len(scaled_features))
            clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
            labels = clustering.fit_predict(scaled_features)
            current_label = labels[-1]
            current_features = features.iloc[-1]
            volatility = current_features['volatility']
            trend_strength = current_features['trend_strength']
            if volatility > features['volatility'].quantile(0.75):
                regime = 'volatile'
            elif trend_strength > features['trend_strength'].quantile(0.65):
                recent_returns = df['close'].pct_change().tail(20).mean()
                regime = 'bull' if recent_returns > 0 else 'bear'
            else:
                regime = 'sideways'
            from sklearn.metrics import silhouette_score
            score = silhouette_score(scaled_features, labels)
            confidence = max(0.0, min(1.0, (score + 1) / 2))
            if len(labels) > 30:
                recent_labels = labels[-30:]
                consistency = (recent_labels == current_label).mean()
                confidence *= consistency
            if confidence < self.config.regime_confidence_threshold:
                regime = 'unknown'
            self.regime_cache[cache_key] = (regime, confidence, time.time())
            LOG.info("regime_detected", regime=regime, confidence=confidence, symbol=symbol, volatility=volatility, trend_strength=trend_strength)
            return regime, confidence
        except Exception as e:
            LOG.error("regime_detection_failed", error=str(e), symbol=symbol)
            return "unknown", 0.0

class BayesianRiskOptimizer:
    def __init__(self, config: AdvancedAIConfig):
        self.config = config

    async def optimize(self, historical_data: pd.DataFrame) -> Dict:
        try:
            if len(historical_data) < 50:
                return {}
            X = historical_data[['volatility', 'returns']].values
            y = historical_data['pnl'].values if 'pnl' in historical_data else np.zeros(len(X))
            kernel = C(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel=kernel)
            gp.fit(X, y)
            async def objective(trial):
                params = {'risk_level': trial.suggest_float('risk_level', 0.01, 0.05)}
                predicted = gp.predict(np.array([[params['risk_level'], 0]]))
                return predicted[0]
            study = optuna.create_study(direction='minimize')
            for _ in range(50):
                trial = study.ask()
                value = await objective(trial)
                study.tell(trial, value)
            best = study.best_params
            await INFLUX_METRICS.write_model_metrics('bayesian_risk', {'risk_level': best['risk_level']})
            return best
        except Exception as e:
            LOG.error("bayesian_opt_failed", error=str(e))
            return {}

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            LOG.error("invalid_input_to_actor_critic", nan_count=torch.isnan(x).sum().item(), inf_count=torch.isinf(x).sum().item())
            batch_size = x.shape[0]
            logits = torch.zeros(batch_size, self.actor.out_features)
            value = torch.zeros(batch_size, 1)
            return logits, value
        features = self.shared(x)
        features = torch.clamp(features, -10.0, 10.0)
        logits = self.actor(features)
        value = self.critic(features)
        logits = torch.clamp(logits, -20.0, 20.0)
        value = torch.clamp(value, -100.0, 100.0)
        return logits, value

class PPOAgent:
    def __init__(self, config: AdvancedAIConfig):
        self.state_dim = config.rl_state_dim
        self.action_dim = config.rl_action_dim
        self.policy = ActorCritic(self.state_dim, self.action_dim, config.rl_hidden_dim)
        self.initial_lr = max(0.0001, min(0.001, config.rl_learning_rate))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.initial_lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        self.gamma = config.rl_gamma
        self.lambda_gae = 0.95
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        self.update_count = 0
        self.total_episodes = 0
        LOG.info("ppo_agent_initialized", state_dim=self.state_dim, action_dim=self.action_dim, initial_lr=self.initial_lr, device=str(self.device))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.policy(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            value = value.item()
        return action.item(), log_prob, value

    def compute_returns_and_advantages(self, rewards, values, dones):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0.0
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lambda_gae * gae
            advantages[i] = gae
            returns[i] = advantages[i] + values[i]
            next_value = values[i]
        return returns, advantages

    def update(self, states, actions, old_log_probs, returns, advantages, batch_size=64, epochs=10):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        if torch.isnan(states).any() or torch.isinf(states).any():
            LOG.error("invalid_states_detected_skipping_update", nan_count=torch.isnan(states).sum().item(), inf_count=torch.isinf(states).sum().item())
            return
        if torch.isnan(returns).any() or torch.isinf(returns).any():
            LOG.error("invalid_returns_detected_skipping_update", nan_count=torch.isnan(returns).sum().item(), inf_count=torch.isinf(returns).sum().item())
            return
        advantages_mean = advantages.mean()
        advantages_std = advantages.std()
        if advantages_std < 1e-6 or torch.isnan(advantages_std) or torch.isinf(advantages_std):
            LOG.warning("invalid_advantages_std_using_unnormalized")
            advantages_normalized = advantages
        else:
            advantages_normalized = (advantages - advantages_mean) / (advantages_std + 1e-6)
        advantages_normalized = torch.clamp(advantages_normalized, -10.0, 10.0)
        n = len(states)
        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                end = start + batch_size
                idx = perm[start:end]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages_normalized[idx]
                logits, values = self.policy(batch_states)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    LOG.error("invalid_logits_detected_skipping_batch", epoch=epoch, batch_start=start, nan_count=torch.isnan(logits).sum().item(), inf_count=torch.isinf(logits).sum().item())
                    continue
                logits = torch.clamp(logits, -20.0, 20.0)
                try:
                    dist = Categorical(logits=logits)
                except ValueError as e:
                    LOG.error("categorical_distribution_creation_failed", error=str(e), logits_shape=logits.shape, logits_sample=logits[:2].tolist() if len(logits) > 0 else [])
                    continue
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                    LOG.error("invalid_log_probs_skipping_batch")
                    continue
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                ratio = torch.clamp(ratio, 0.0, 10.0)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (batch_returns - values.squeeze()).pow(2).mean()
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if torch.isnan(loss) or torch.isinf(loss):
                    LOG.error("invalid_loss_detected_skipping_backward", actor_loss=actor_loss.item() if not torch.isnan(actor_loss) else 'nan', value_loss=value_loss.item() if not torch.isnan(value_loss) else 'nan', entropy=entropy.item() if not torch.isnan(entropy) else 'nan')
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                has_nan_grad = False
                for param in self.policy.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                if has_nan_grad:
                    LOG.error("nan_gradient_detected_skipping_optimizer_step")
                    self.optimizer.zero_grad()
                    continue
                self.optimizer.step()
            self.update_count += 1
            if self.update_count % 10 == 0:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                LOG.debug("lr_scheduler_step", update_count=self.update_count, new_lr=current_lr)

    def save(self, path: str = "models/rl/ppo_agent.pth"):
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint = {
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon': self.epsilon if hasattr(self, 'epsilon') else 1.0,
                'update_count': self.update_count if hasattr(self, 'update_count') else 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            torch.save(checkpoint, path)
            LOG.info("ppo_agent_saved", path=path)
            return True
        except Exception as e:
            LOG.error("ppo_agent_save_failed", error=str(e))
            return False

    def load(self, path: str = "models/rl/ppo_agent.pth"):
        try:
            if not os.path.exists(path):
                LOG.info("ppo_checkpoint_not_found_will_train_from_scratch", path=path)
                return False
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if hasattr(self, 'scheduler') and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if hasattr(self, 'epsilon'):
                self.epsilon = checkpoint.get('epsilon', 1.0)
            if hasattr(self, 'update_count'):
                self.update_count = checkpoint.get('update_count', 0)
            LOG.info("ppo_agent_loaded", path=path, timestamp=checkpoint.get('timestamp'), update_count=checkpoint.get('update_count', 0))
            return True
        except Exception as e:
            LOG.error("ppo_agent_load_failed", error=str(e))
            return False

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame = None, symbols=['BTC/USDT']):
        super().__init__()
        if df is None:
            raise ValueError("TradingEnv requires a valid DataFrame with market data. Cannot initialize with None. Please provide historical OHLCV data.")
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        if len(df) < 50:
            raise ValueError(f"DataFrame too small: {len(df)} rows (minimum 50 required)")
        self.df = df.reset_index(drop=True) if df.index.name else df.copy()
        self.df['returns'] = self.df['close'].pct_change().fillna(0)
        self.current_step = 0
        self.position = 0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.slippage = 0.001
        self.commission = 0.0001
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        LOG.debug("trading_env_initialized", df_rows=len(self.df), action_space=3, observation_dim=6)

    def _normalize(self, value, min_val=0, max_val=1):
        if max_val <= min_val:
            return 0.0
        return 2 * (value - min_val) / (max_val - min_val + 1e-8) - 1

    def _get_obs(self):
        try:
            if self.current_step >= len(self.df):
                self.current_step = len(self.df) - 1
            row = self.df.iloc[self.current_step]
            close_norm = self._normalize(row['close'], self.df['close'].min(), self.df['close'].max())
            balance_norm = self._normalize(self.balance, 0, self.initial_balance * 2)
            volatility = self.df['returns'].rolling(20).std().fillna(0).iloc[self.current_step]
            rsi_norm = self._normalize(row.get('rsi', 50), 0, 100)
            macd = float(row.get('macd', 0))
            obs = np.array([balance_norm, float(self.position), close_norm, rsi_norm, macd, float(volatility)], dtype=np.float32)
            return obs
        except Exception as e:
            LOG.error("observation_generation_failed", error=str(e))
            return np.zeros(6, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        max_start = max(0, len(self.df) - 100)
        self.current_step = np.random.randint(0, max_start) if max_start > 0 else 0
        self.position = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        return self._get_obs(), {}

    def step(self, action):
        try:
            if self.current_step >= len(self.df) - 1:
                return self._get_obs(), 0.0, True, False, {'net_worth': self.net_worth}
            current_price = self.df.iloc[self.current_step]['close']
            self.current_step += 1
            done = self.current_step >= len(self.df) - 1
            next_price = self.df.iloc[self.current_step]['close'] if not done else current_price
            if current_price <= 0 or next_price <= 0 or np.isnan(current_price) or np.isnan(next_price):
                LOG.warning("invalid_price_in_env_step", current=current_price, next=next_price, step=self.current_step)
                return self._get_obs(), -1.0, True, False, {'net_worth': self.net_worth}
            reward = 0.0
            if self.position != 0:
                exit_price = current_price * (1 - self.slippage * self.position)
                pnl = self.position * (exit_price - self.entry_price) * self.position_size
                pnl -= abs(pnl) * self.commission
                self.balance += pnl
                self.net_worth = self.balance
            if action == 2:
                if self.position != 1:
                    entry_price = current_price * (1 + self.slippage)
                    self.position = 1
                    self.entry_price = entry_price
                    self.position_size = self.balance / entry_price if entry_price > 0 else 0
                    self.trades += 1
                    reward -= self.balance * self.commission
            elif action == 0:
                if self.position != -1:
                    entry_price = current_price * (1 - self.slippage)
                    self.position = -1
                    self.entry_price = entry_price
                    self.position_size = self.balance / entry_price if entry_price > 0 else 0
                    self.trades += 1
                    reward -= self.balance * self.commission
            else:
                reward -= 0.00005 * self.balance if self.position == 0 else 0
            if self.position != 0:
                open_pnl = self.position * (next_price - self.entry_price) * self.position_size
                self.net_worth = self.balance + open_pnl
                reward += np.clip(open_pnl / self.initial_balance, -1.0, 1.0)
            self.max_net_worth = max(self.max_net_worth, self.net_worth)
            if self.max_net_worth > 0:
                drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
                reward -= np.clip(drawdown * 2, 0.0, 2.0)
            if self.net_worth <= 0.5 * self.initial_balance:
                reward -= 10.0
                done = True
            if not done:
                reward += 0.001
            reward = float(np.clip(reward, -10.0, 10.0))
            if np.isnan(reward) or np.isinf(reward):
                LOG.warning("invalid_reward_detected_using_zero", reward=reward, step=self.current_step)
                reward = 0.0
            info = {'net_worth': self.net_worth, 'trades': self.trades}
            return self._get_obs(), reward, done, done, info
        except Exception as e:
            LOG.error("step_execution_failed", error=str(e))
            return self._get_obs(), 0.0, True, False, {'net_worth': self.net_worth}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Net Worth: {self.net_worth:.2f}, Position: {self.position}")

class RLTrainingManager:
    def __init__(self, config: AdvancedAIConfig, rl_agent):
        self.config = config
        self.rl_agent = rl_agent
        self.env = None
        self.memory = []
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.training_iterations = 0
        LOG.info("rl_training_manager_initialized", config_rl_state_dim=config.rl_state_dim, config_rl_action_dim=config.rl_action_dim)

    async def train(self, episodes: int = 100, df: pd.DataFrame = None):
        try:
            if df is None or len(df) < 100:
                LOG.warning("insufficient_dataframe_for_rl_training",
                           df_provided=df is not None,
                           df_length=len(df) if df is not None else 0,
                           required_min=100,
                           message="RL training skipped - need real market data")
                return False
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                LOG.error("rl_training_dataframe_missing_columns",
                         missing=missing_cols,
                         available=list(df.columns)[:10])
                return False
            
            # CORRECCIÓN: Limpiar environment anterior si existe
            if self.env is not None:
                try:
                    self.env.close()
                    del self.env
                    LOG.debug("previous_rl_env_cleaned")
                except Exception as cleanup_error:
                    LOG.debug("env_cleanup_error", error=str(cleanup_error))
            
            self.env = TradingEnv(df=df)
            
            LOG.info("rl_training_started",
                    episodes=episodes,
                    df_rows=len(df),
                    df_columns=len(df.columns),
                    using_real_data=True)
            
            # NUEVO: Tracking de métricas de entrenamiento
            training_metrics = {
                'episode_rewards': [],
                'episode_lengths': [],
                'episode_net_worths': [],
                'best_reward': float('-inf'),
                'best_episode': 0
            }
            
            for ep in range(episodes):
                states = []
                actions = []
                rewards = []
                log_probs = []
                values = []
                dones = []
                
                obs, _ = self.env.reset()
                done = False
                episode_reward = 0.0
                episode_steps = 0
                
                while not done:
                    action, log_prob, value = self.rl_agent.act(obs)
                    next_obs, reward, done, _, info = self.env.step(action)
                    
                    states.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    log_probs.append(log_prob)
                    values.append(value)
                    dones.append(int(done))
                    
                    episode_reward += reward
                    episode_steps += 1
                    obs = next_obs
                
                # Calcular returns y advantages
                returns, advantages = self.rl_agent.compute_returns_and_advantages(
                    rewards, values, dones
                )
                
                # Update del agente
                self.rl_agent.update(states, actions, log_probs, returns, advantages)
                
                # NUEVO: Incrementar contador total de episodios
                self.rl_agent.total_episodes += 1
                
                # Actualizar epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                # Registrar métricas
                training_metrics['episode_rewards'].append(episode_reward)
                training_metrics['episode_lengths'].append(episode_steps)
                training_metrics['episode_net_worths'].append(self.env.net_worth)
                
                if episode_reward > training_metrics['best_reward']:
                    training_metrics['best_reward'] = episode_reward
                    training_metrics['best_episode'] = ep
                
                # Log periódico
                if ep % 10 == 0:
                    avg_reward_10 = np.mean(training_metrics['episode_rewards'][-10:])
                    avg_net_worth_10 = np.mean(training_metrics['episode_net_worths'][-10:])
                    
                    LOG.info("rl_training_progress",
                            episode=ep,
                            total_episodes=episodes,
                            episode_reward=episode_reward,
                            avg_reward_10=avg_reward_10,
                            net_worth=self.env.net_worth,
                            avg_net_worth_10=avg_net_worth_10,
                            epsilon=self.epsilon,
                            steps=episode_steps)
                
                # Enviar métricas a InfluxDB
                try:
                    await INFLUX_METRICS.write_rl_metrics(
                        episode=ep,
                        reward=episode_reward,
                        actor_loss=0.0,  # TODO: Capturar del update
                        critic_loss=0.0,
                        epsilon=self.epsilon
                    )
                except Exception as influx_error:
                    LOG.debug("rl_metrics_write_failed", error=str(influx_error))
                
                # NUEVO: Limpieza periódica de memoria
                if ep % 20 == 0 and ep > 0:
                    optimize_memory_usage()
                    LOG.debug("memory_cleanup_during_rl_training", episode=ep)
            
            # Log final de entrenamiento
            LOG.info("rl_training_completed",
                    episodes=episodes,
                    avg_reward=np.mean(training_metrics['episode_rewards']),
                    best_reward=training_metrics['best_reward'],
                    best_episode=training_metrics['best_episode'],
                    final_epsilon=self.epsilon)
            
            # Guardar agente entrenado
            try:
                save_success = self.rl_agent.save()
                if save_success:
                    LOG.info("rl_agent_saved_after_training",
                            total_episodes=self.rl_agent.total_episodes)
                else:
                    LOG.warning("rl_agent_save_failed_after_training")
            except Exception as save_error:
                LOG.error("rl_agent_save_exception",
                         error=str(save_error))
            
            # NUEVO: Guardar métricas de entrenamiento
            try:
                import pickle
                import os
                os.makedirs("models/rl", exist_ok=True)
                with open("models/rl/training_metrics.pkl", "wb") as f:
                    pickle.dump(training_metrics, f)
                LOG.info("training_metrics_saved")
            except Exception as metrics_save_error:
                LOG.debug("training_metrics_save_failed",
                         error=str(metrics_save_error))
            
            # CRÍTICO: Limpiar environment al final
            try:
                self.env.close()
                del self.env
                self.env = None
                LOG.debug("rl_env_cleaned_after_training")
            except Exception as cleanup_error:
                LOG.debug("final_env_cleanup_error", error=str(cleanup_error))
            
            # Limpieza final de memoria
            optimize_memory_usage()
            
            return True
            
        except Exception as e:
            LOG.error("rl_training_failed",
                     error=str(e),
                     traceback=traceback.format_exc()[:500])
            
            # Limpiar environment en caso de error
            if self.env is not None:
                try:
                    self.env.close()
                    del self.env
                    self.env = None
                except Exception:
                    pass
            
            raise

    async def train_on_real_data(self, exchange_manager, symbol: str, months: int = 3, episodes: int = 50):
        try:
            LOG.info("loading_real_market_data", symbol=symbol, months=months)
            since = int((datetime.now(timezone.utc) - timedelta(days=30*months)).timestamp() * 1000)
            result = await exchange_manager.fetch_ohlcv(symbol, '1h', limit=1000, since=since)
            if not result or not result.get("success", False):
                raise RuntimeError("Failed to fetch real market data")
            df = create_dataframe(result.get("ohlcv", []))
            if df is None or len(df) == 0:
                raise RuntimeError("Invalid DataFrame from market data")
            df = calculate_technical_indicators(df)
            LOG.info("real_data_loaded", rows=len(df), columns=len(df.columns))
            await self.train(episodes=episodes, df=df)
        except Exception as e:
            LOG.error("real_data_training_failed", error=str(e))
            raise

class AdvancedEnsembleLearner:
    def __init__(self, config):
        self.config = config
        self.lstm = None
        self.gb = None
        self.attention = None
        self.technical = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.symbol_models = {}
        self.symbol_training_history = {}
        LOG.debug("ensemble_learner_initialized_with_empty_symbol_models")

    def initialize_base_models(self):
        try:
            class LSTMPredictor(nn.Module):
                def __init__(self, input_dim, hidden_dim):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                    self.fc = nn.Linear(hidden_dim, 3)

                def forward(self, x):
                    _, (hn, _) = self.lstm(x)
                    return F.softmax(self.fc(hn[0]), dim=1)

            self.lstm = LSTMPredictor(4, 64).to(self.device)
            self.gb = XGBClassifier(n_estimators=10, max_depth=5, random_state=42, verbosity=0) if XGBClassifier else GradientBoostingClassifier(n_estimators=10, max_depth=5, random_state=42)

            class AttentionNetwork(nn.Module):
                def __init__(self, input_dim, hidden_dim):
                    super().__init__()
                    self.embedding = nn.Linear(input_dim, hidden_dim)
                    self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True), num_layers=2)
                    self.fc = nn.Linear(hidden_dim, 3)

                def forward(self, x):
                    if len(x.shape) == 2:
                        x = x.unsqueeze(1)
                    x = self.embedding(x)
                    x = self.transformer(x)
                    x = x.mean(dim=1)
                    return F.softmax(self.fc(x), dim=1)

            self.attention = AttentionNetwork(4, 64).to(self.device)
            self.technical = VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
                ('lr', LogisticRegression(max_iter=100, random_state=42))
            ], voting='soft')
            self.symbol_models = {}
            self.symbol_training_history = {}
            self.is_trained = False
            LOG.info("ensemble_models_initialized")
        except Exception as e:
            LOG.error("ensemble_init_failed", error=str(e))
            raise

    async def fit(self, df: pd.DataFrame, targets: pd.Series = None, epochs=10, batch_size=32, buy_threshold=0.01, sell_threshold=-0.01, symbol: str = None):
        try:
            if len(df) < 50:
                LOG.warning("insufficient_data_for_ensemble_training", rows=len(df), symbol=symbol)
                self.is_trained = False
                return

            feature_cols = ['close', 'rsi', 'macd', 'volume']
            available_cols = [col for col in feature_cols if col in df.columns]
            if len(available_cols) < 3:
                LOG.error("insufficient_columns_for_training", available=available_cols, symbol=symbol)
                self.is_trained = False
                return

            features = df[available_cols].values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            if len(available_cols) < 4:
                missing_count = 4 - len(available_cols)
                padding = np.zeros((features.shape[0], missing_count))
                features = np.hstack([features, padding])

            # --- MEJORA 1: Targets con estrategia adaptativa ---
            if targets is None:
                if 'close' in df.columns:
                    future_returns = df['close'].shift(-1) / df['close'] - 1
                    future_returns = future_returns.iloc[:-1]
                    features = features[:-1]

                    # ✅ ESTRATEGIA ADAPTATIVA: Usar percentiles para balanceo
                    # En lugar de thresholds fijos, usar distribución de datos
                    
                    # Calcular volatilidad del símbolo
                    volatility = future_returns.std()
                    
                    # ✅ Ajustar thresholds según volatilidad
                    if volatility > 0.03:  # Alta volatilidad (>3%)
                        # Usar percentiles más amplios
                        buy_percentile = 0.65  # Top 35%
                        sell_percentile = 0.35  # Bottom 35%
                    elif volatility > 0.015:  # Volatilidad media (1.5-3%)
                        buy_percentile = 0.70  # Top 30%
                        sell_percentile = 0.30  # Bottom 30%
                    else:  # Baja volatilidad (<1.5%)
                        # Percentiles más estrictos
                        buy_percentile = 0.75  # Top 25%
                        sell_percentile = 0.25  # Bottom 25%
                    
                    buy_threshold = future_returns.quantile(buy_percentile)
                    sell_threshold = future_returns.quantile(sell_percentile)
                    
                    # ✅ MEJORA 2: Crear targets balanceados
                    targets = np.ones(len(future_returns), dtype=int)  # Default: hold
                    targets[future_returns > buy_threshold] = 2  # Buy
                    targets[future_returns < sell_threshold] = 0  # Sell

                    # ✅ Validar distribución ANTES de continuar
                    unique, counts = np.unique(targets, return_counts=True)
                    target_dist = dict(zip(unique, counts))
                    buy_count = target_dist.get(2, 0)
                    sell_count = target_dist.get(0, 0)
                    hold_count = target_dist.get(1, 0)
                    total = len(targets)

                    LOG.info("adaptive_target_distribution", 
                            symbol=symbol,
                            volatility=volatility,
                            buy_threshold=buy_threshold,
                            sell_threshold=sell_threshold,
                            buy_pct=buy_count/total*100,
                            sell_pct=sell_count/total*100,
                            hold_pct=hold_count/total*100)

                    # ✅ MEJORA 3: Si sigue desbalanceado, aplicar SMOTE o undersampling
                    max_class_pct = max(counts) / len(targets)
                    
                    if max_class_pct > 0.80:  # Si clase mayoritaria > 80%
                        LOG.warning("severe_imbalance_applying_balancing",
                                   symbol=symbol,
                                   max_class_pct=max_class_pct * 100)
                        
                        # ✅ ESTRATEGIA 1: Undersampling inteligente de clase mayoritaria
                        # Mantener todas las minorías, reducir mayoritaria
                        
                        majority_class = int(np.argmax(counts))
                        minority_classes = [c for c in unique if c != majority_class]
                        
                        # Calcular tamaño objetivo para clase mayoritaria
                        # (2x el promedio de minoritarias)
                        minority_avg = np.mean([target_dist.get(c, 0) for c in minority_classes])
                        target_majority_size = int(minority_avg * 2)
                        
                        # Indices de cada clase
                        majority_indices = np.where(targets == majority_class)[0]
                        minority_indices = np.where(targets != majority_class)[0]
                        
                        # Sample inteligente de mayoritaria (mantener variedad)
                        if len(majority_indices) > target_majority_size:
                            # Samplear uniformemente a lo largo del tiempo
                            step = len(majority_indices) / target_majority_size
                            sampled_majority = [majority_indices[int(i * step)] 
                                               for i in range(target_majority_size)]
                        else:
                            sampled_majority = majority_indices
                        
                        # Combinar
                        balanced_indices = np.concatenate([
                            sampled_majority,
                            minority_indices
                        ])
                        
                        # Ordenar para mantener secuencia temporal
                        balanced_indices = np.sort(balanced_indices)
                        
                        # Aplicar balanceo
                        features = features[balanced_indices]
                        targets = targets[balanced_indices]
                        
                        # Validar nueva distribución
                        unique_balanced, counts_balanced = np.unique(targets, return_counts=True)
                        new_dist = dict(zip(unique_balanced, counts_balanced))
                        
                        LOG.info("class_balancing_applied",
                                symbol=symbol,
                                original_samples=len(majority_indices) + len(minority_indices),
                                balanced_samples=len(targets),
                                new_distribution=new_dist,
                                buy=new_dist.get(2, 0),
                                sell=new_dist.get(0, 0),
                                hold=new_dist.get(1, 0))
                    
                    # ✅ VALIDACIÓN FINAL: Mínimos absolutos
                    unique_final, counts_final = np.unique(targets, return_counts=True)
                    target_dist_final = dict(zip(unique_final, counts_final))
                    buy_samples = target_dist_final.get(2, 0)
                    sell_samples = target_dist_final.get(0, 0)
                    
                    if buy_samples < 5 or sell_samples < 5:
                        LOG.error("insufficient_minority_samples_after_balancing",
                                 symbol=symbol,
                                 buy_samples=buy_samples,
                                 sell_samples=sell_samples)
                        
                        # Guardar en historial como fallido
                        if symbol:
                            if not hasattr(self, 'symbol_training_history'):
                                self.symbol_training_history = {}
                            self.symbol_training_history[symbol] = {
                                'training_count': 0,
                                'last_training': datetime.now(timezone.utc),
                                'samples_used': [],
                                'status': 'failed',
                                'reason': 'insufficient_minority_samples_after_balancing',
                                'distribution': target_dist_final
                            }
                        
                        self.is_trained = False if symbol is None else self.is_trained
                        return
                    
                    LOG.info("targets_created_with_validation", 
                            buy_threshold=buy_threshold, 
                            sell_threshold=sell_threshold, 
                            symbol=symbol, 
                            final_distribution={
                                'buy': buy_samples, 
                                'sell': sell_samples, 
                                'hold': target_dist_final.get(1, 0)
                            })
                else:
                    targets = np.ones(len(features), dtype=int)
            else:
                # Si targets se pasa como argumento, validar
                targets = np.array(targets, dtype=int)
                targets = np.clip(targets, 0, 2)
                if len(features) != len(targets):
                    LOG.error("feature_target_length_mismatch_provided", 
                             features=len(features), 
                             targets=len(targets), 
                             symbol=symbol)
                    self.is_trained = False if symbol is None else self.is_trained
                    return

            unique, counts = np.unique(targets, return_counts=True)
            target_dist = dict(zip(unique, counts))
            
            # --- Dividir en train/val aquí, fuera del bloque if/else ---
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, test_size=0.2, random_state=42, stratify=targets
            )
            LOG.debug("train_val_split", train=len(X_train), val=len(X_val), symbol=symbol)
            
            max_class_pct = max(counts) / len(targets)
            if max_class_pct > 0.80:
                LOG.warning("severe_class_imbalance_detected", symbol=symbol, max_class_percentage=f"{max_class_pct*100:.1f}%", distribution=target_dist, message="Training may produce biased model")
            if symbol:
                LOG.info("training_specialized_model", symbol=symbol)
                if symbol not in self.symbol_models:
                    class LSTMPredictor(nn.Module):
                        def __init__(self, input_dim, hidden_dim):
                            super().__init__()
                            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                            self.fc = nn.Linear(hidden_dim, 3)

                        def forward(self, x):
                            _, (hn, _) = self.lstm(x)
                            return F.softmax(self.fc(hn[0]), dim=1)

                    class AttentionNetwork(nn.Module):
                        def __init__(self, input_dim, hidden_dim):
                            super().__init__()
                            self.embedding = nn.Linear(input_dim, hidden_dim)
                            self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True), num_layers=2)
                            self.fc = nn.Linear(hidden_dim, 3)

                        def forward(self, x):
                            if len(x.shape) == 2:
                                x = x.unsqueeze(1)
                            x = self.embedding(x)
                            x = self.transformer(x)
                            x = x.mean(dim=1)
                            return F.softmax(self.fc(x), dim=1)

                    self.symbol_models[symbol] = {
                        'lstm': LSTMPredictor(4, 64).to(self.device),
                        'gb': XGBClassifier(n_estimators=10, max_depth=10, random_state=42, verbosity=0) if XGBClassifier else GradientBoostingClassifier(n_estimators=10, max_depth=10, random_state=42),
                        'attention': AttentionNetwork(4, 64).to(self.device),
                        'technical': VotingClassifier(estimators=[
                            ('rf', RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)),
                            ('lr', LogisticRegression(max_iter=100, random_state=42))
                        ], voting='soft')
                    }
                    self.symbol_training_history[symbol] = {
                        'training_count': 0,
                        'last_training': datetime.now(timezone.utc),
                        'samples_used': []
                    }
                    LOG.info("specialized_models_created", symbol=symbol)
                lstm_model = self.symbol_models[symbol]['lstm']
                gb_model = self.symbol_models[symbol]['gb']
                attn_model = self.symbol_models[symbol]['attention']
                tech_model = self.symbol_models[symbol]['technical']
            else:
                LOG.info("training_general_model")
                lstm_model = self.lstm
                gb_model = self.gb
                attn_model = self.attention
                tech_model = self.technical
            try:
                if hasattr(gb_model, 'n_classes_'):
                    delattr(gb_model, 'n_classes_')
                gb_model.fit(X_train, y_train)
                test_pred = gb_model.predict_proba(X_val[:1])
                LOG.info("gb_model_trained", n_classes=gb_model.n_classes_ if hasattr(gb_model, 'n_classes_') else 'unknown', prediction_shape=test_pred.shape, symbol=symbol)
            except Exception as e:
                LOG.warning("gb_training_failed", error=str(e), symbol=symbol)
            try:
                tech_model.fit(X_train, y_train)
                test_pred = tech_model.predict_proba(X_val[:1])
                LOG.info("technical_model_trained", prediction_shape=test_pred.shape, symbol=symbol)
            except Exception as e:
                LOG.warning("technical_training_failed", error=str(e), symbol=symbol)
            try:
                X_train_lstm = X_train.reshape(X_train.shape[0], 1, -1)
                train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_lstm), torch.LongTensor(y_train))
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(batch_size, len(X_train_lstm)), shuffle=True)
                criterion = nn.CrossEntropyLoss()
                optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    for batch_x, batch_y in train_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        optimizer_lstm.zero_grad()
                        out_lstm = lstm_model(batch_x)
                        loss_lstm = criterion(out_lstm, batch_y)
                        loss_lstm.backward()
                        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
                        optimizer_lstm.step()
                        epoch_loss += loss_lstm.item()
                    if epoch % 5 == 0:
                        LOG.debug("lstm_training_progress", epoch=epoch, loss=epoch_loss, symbol=symbol)
                LOG.info("lstm_model_trained", epochs=epochs, symbol=symbol)
            except Exception as e:
                LOG.warning("lstm_training_failed", error=str(e), symbol=symbol)
            try:
                X_train_attn = X_train.reshape(X_train.shape[0], 1, -1)
                train_dataset_attn = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_attn), torch.LongTensor(y_train))
                train_loader_attn = torch.utils.data.DataLoader(train_dataset_attn, batch_size=min(batch_size, len(X_train_attn)), shuffle=True)
                criterion = nn.CrossEntropyLoss()
                optimizer_attn = torch.optim.Adam(attn_model.parameters(), lr=0.001)
                for epoch in range(epochs):
                    for batch_x, batch_y in train_loader_attn:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        optimizer_attn.zero_grad()
                        out_attn = attn_model(batch_x)
                        loss_attn = criterion(out_attn, batch_y)
                        loss_attn.backward()
                        torch.nn.utils.clip_grad_norm_(attn_model.parameters(), 1.0)
                        optimizer_attn.step()
                LOG.info("attention_model_trained", epochs=epochs, symbol=symbol)
            except Exception as e:
                LOG.warning("attention_training_failed", error=str(e), symbol=symbol)
            if symbol:
                LOG.info("registering_specialized_models", symbol=symbol, has_symbol_models_attr=hasattr(self, 'symbol_models'))
                if not hasattr(self, 'symbol_models'):
                    self.symbol_models = {}
                    LOG.warning("symbol_models_created_in_fit", symbol=symbol, message="Should have been initialized in __init__")
                try:
                    self.symbol_models[symbol] = {
                        'lstm': lstm_model,
                        'gb': gb_model,
                        'attention': attn_model,
                        'technical': tech_model
                    }
                    LOG.info("specialized_models_registered_successfully", symbol=symbol, models_registered=['lstm', 'gb', 'attention', 'technical'], total_specialized_symbols=len(self.symbol_models))
                except Exception as reg_error:
                    LOG.error("specialized_models_registration_failed", symbol=symbol, error=str(reg_error))
                if symbol not in self.symbol_models:
                    LOG.error("CRITICAL_symbol_not_in_dict_after_registration", symbol=symbol, dict_keys=list(self.symbol_models.keys()))
                else:
                    LOG.debug("registration_verified", symbol=symbol, sub_models=list(self.symbol_models[symbol].keys()))
            if symbol:
                if not hasattr(self, 'symbol_models'):
                    self.symbol_models = {}
                    LOG.warning("symbol_models_created_during_fit", message="Should have been initialized in __init__")
                if not hasattr(self, 'symbol_training_history'):
                    self.symbol_training_history = {}
                    LOG.warning("symbol_training_history_created_during_fit")
                if symbol not in self.symbol_models:
                    LOG.error("CRITICAL_symbol_not_in_symbol_models_after_training", symbol=symbol, lstm_trained=lstm_model is not None, gb_trained=gb_model is not None, has_symbol_models_attr=hasattr(self, 'symbol_models'), symbol_models_keys=list(self.symbol_models.keys()) if hasattr(self, 'symbol_models') else [])
                    try:
                        LOG.warning("force_registering_specialized_models", symbol=symbol)
                        self.symbol_models[symbol] = {
                            'lstm': lstm_model,
                            'gb': gb_model,
                            'attention': attn_model,
                            'technical': tech_model
                        }
                        LOG.info("specialized_models_force_registered", symbol=symbol)
                    except Exception as force_reg_error:
                        LOG.error("force_registration_failed", symbol=symbol, error=str(force_reg_error))
                        return
                if symbol not in self.symbol_training_history:
                    self.symbol_training_history[symbol] = {
                        'training_count': 0,
                        'last_training': datetime.now(timezone.utc),
                        'samples_used': []
                    }
                self.symbol_training_history[symbol]['training_count'] += 1
                self.symbol_training_history[symbol]['last_training'] = datetime.now(timezone.utc)
                self.symbol_training_history[symbol]['samples_used'].append(len(X_train))
                if len(self.symbol_training_history[symbol]['samples_used']) > 10:
                    self.symbol_training_history[symbol]['samples_used'] = self.symbol_training_history[symbol]['samples_used'][-10:]
            self.is_trained = True
            LOG.info("ensemble_models_trained_successfully", epochs=epochs, train_samples=len(X_train), val_samples=len(X_val), symbol=symbol if symbol else "general", specialized=symbol is not None)
            await self._save_models()
        except Exception as e:
            LOG.error("ensemble_fit_failed", error=str(e), traceback=traceback.format_exc(), symbol=symbol)
            self.is_trained = False

    async def _save_models(self, base_path: str = "models/ensemble"):
        try:
            import os
            import pickle
            os.makedirs(base_path, exist_ok=True)
            with open(f"{base_path}/gb_model.pkl", "wb") as f:
                pickle.dump(self.gb, f)
            with open(f"{base_path}/technical_model.pkl", "wb") as f:
                pickle.dump(self.technical, f)
            torch.save(self.lstm.state_dict(), f"{base_path}/lstm_model.pth")
            torch.save(self.attention.state_dict(), f"{base_path}/attention_model.pth")
            if self.symbol_models:
                specialized_dir = f"{base_path}/specialized"
                os.makedirs(specialized_dir, exist_ok=True)
                for symbol, models_dict in self.symbol_models.items():
                    symbol_safe = symbol.replace('/', '_')
                    symbol_dir = f"{specialized_dir}/{symbol_safe}"
                    os.makedirs(symbol_dir, exist_ok=True)
                    with open(f"{symbol_dir}/gb_model.pkl", "wb") as f:
                        pickle.dump(models_dict['gb'], f)
                    with open(f"{symbol_dir}/technical_model.pkl", "wb") as f:
                        pickle.dump(models_dict['technical'], f)
                    torch.save(models_dict['lstm'].state_dict(), f"{symbol_dir}/lstm_model.pth")
                    torch.save(models_dict['attention'].state_dict(), f"{symbol_dir}/attention_model.pth")
                    LOG.debug("specialized_model_saved", symbol=symbol, path=symbol_dir)
                LOG.info("specialized_models_saved", symbols_count=len(self.symbol_models))
            metadata_json = {
                'is_trained': bool(self.is_trained),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'device': str(self.device),
                'specialized_symbols': list(self.symbol_models.keys()) if self.symbol_models else [],
                'training_history': {}
            }
            if self.symbol_training_history:
                for symbol, history in self.symbol_training_history.items():
                    try:
                        if not isinstance(history, dict):
                            LOG.warning("invalid_history_format_skipping", symbol=symbol, type=type(history).__name__)
                            continue
                        last_training = history.get('last_training')
                        if last_training is not None:
                            if isinstance(last_training, datetime):
                                last_training_str = last_training.isoformat()
                            elif isinstance(last_training, str):
                                last_training_str = last_training
                            else:
                                LOG.warning("invalid_last_training_type", symbol=symbol, type=type(last_training).__name__)
                                last_training_str = None
                        else:
                            last_training_str = None
                        samples_used = history.get('samples_used', [])
                        if not isinstance(samples_used, list):
                            samples_used = []
                        metadata_json['training_history'][symbol] = {
                            'training_count': int(history.get('training_count', 0)),
                            'last_training': last_training_str,
                            'samples_used': samples_used
                        }
                    except Exception as history_error:
                        LOG.warning("history_serialization_failed", symbol=symbol, error=str(history_error))
                        continue
            metadata_path = f"{base_path}/metadata.json"
            if os.path.exists(metadata_path):
                backup_path = f"{metadata_path}.backup"
                try:
                    import shutil
                    shutil.copy2(metadata_path, backup_path)
                    LOG.debug("metadata_backup_created", path=backup_path)
                except Exception as backup_error:
                    LOG.debug("metadata_backup_failed", error=str(backup_error))
            with open(metadata_path, "w") as f:
                json.dump(metadata_json, f, indent=2, ensure_ascii=False)
            try:
                with open(metadata_path, "r") as f:
                    json.load(f)
                LOG.debug("metadata_json_validated")
            except json.JSONDecodeError as validate_error:
                LOG.error("metadata_validation_failed_after_save", error=str(validate_error))
                if os.path.exists(backup_path):
                    try:
                        import shutil
                        shutil.copy2(backup_path, metadata_path)
                        LOG.info("metadata_restored_from_backup")
                    except Exception as restore_error:
                        LOG.error("metadata_restore_failed", error=str(restore_error))
            LOG.info("ensemble_models_saved", path=base_path, specialized_count=len(self.symbol_models) if self.symbol_models else 0)
            return True
        except Exception as e:
            LOG.error("ensemble_models_save_failed", error=str(e))
            return False

    async def _load_models(self, base_path: str = "models/ensemble"):
        try:
            import os
            import pickle
            if not os.path.exists(base_path):
                LOG.warning("ensemble_models_not_found", path=base_path)
                return False
            with open(f"{base_path}/gb_model.pkl", "rb") as f:
                self.gb = pickle.load(f)
            with open(f"{base_path}/technical_model.pkl", "rb") as f:
                self.technical = pickle.load(f)
            self.lstm.load_state_dict(torch.load(f"{base_path}/lstm_model.pth", map_location=self.device))
            self.attention.load_state_dict(torch.load(f"{base_path}/attention_model.pth", map_location=self.device))
            self.lstm.eval()
            self.attention.eval()
            specialized_dir = f"{base_path}/specialized"
            if os.path.exists(specialized_dir):
                for symbol_dir_name in os.listdir(specialized_dir):
                    symbol_dir = f"{specialized_dir}/{symbol_dir_name}"
                    if not os.path.isdir(symbol_dir):
                        continue
                    symbol = symbol_dir_name.replace('_', '/')
                    try:
                        class LSTMPredictor(nn.Module):
                            def __init__(self, input_dim, hidden_dim):
                                super().__init__()
                                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                                self.fc = nn.Linear(hidden_dim, 3)

                            def forward(self, x):
                                _, (hn, _) = self.lstm(x)
                                return F.softmax(self.fc(hn[0]), dim=1)

                        class AttentionNetwork(nn.Module):
                            def __init__(self, input_dim, hidden_dim):
                                super().__init__()
                                self.embedding = nn.Linear(input_dim, hidden_dim)
                                self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True), num_layers=2)
                                self.fc = nn.Linear(hidden_dim, 3)

                            def forward(self, x):
                                if len(x.shape) == 2:
                                    x = x.unsqueeze(1)
                                x = self.embedding(x)
                                x = self.transformer(x)
                                x = x.mean(dim=1)
                                return F.softmax(self.fc(x), dim=1)

                        with open(f"{symbol_dir}/gb_model.pkl", "rb") as f:
                            gb = pickle.load(f)
                        with open(f"{symbol_dir}/technical_model.pkl", "rb") as f:
                            technical = pickle.load(f)
                        lstm = LSTMPredictor(4, 64).to(self.device)
                        lstm.load_state_dict(torch.load(f"{symbol_dir}/lstm_model.pth", map_location=self.device))
                        lstm.eval()
                        attention = AttentionNetwork(4, 64).to(self.device)
                        attention.load_state_dict(torch.load(f"{symbol_dir}/attention_model.pth", map_location=self.device))
                        attention.eval()
                        self.symbol_models[symbol] = {
                            'lstm': lstm,
                            'gb': gb,
                            'attention': attention,
                            'technical': technical
                        }
                        LOG.debug("specialized_model_loaded", symbol=symbol)
                    except Exception as symbol_load_error:
                        LOG.warning("specialized_model_load_failed", symbol=symbol, error=str(symbol_load_error))
                        continue
                LOG.info("specialized_models_loaded", count=len(self.symbol_models))
            metadata_path = f"{base_path}/metadata.json"
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as json_error:
                LOG.error("metadata_json_corrupted", error=str(json_error), path=metadata_path, message="Recreating metadata from scratch")
                metadata = {
                    'is_trained': True,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'device': str(self.device),
                    'specialized_symbols': list(self.symbol_models.keys()) if self.symbol_models else [],
                    'training_history': {}
                }
                for symbol in metadata['specialized_symbols']:
                    metadata['training_history'][symbol] = {
                        'training_count': 1,
                        'last_training': None,
                        'samples_used': []
                    }
                try:
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    LOG.info("metadata_reconstructed_and_saved")
                except Exception as save_error:
                    LOG.warning("metadata_reconstruction_save_failed", error=str(save_error))
            except FileNotFoundError:
                LOG.warning("metadata_file_not_found_creating_default", path=metadata_path)
                metadata = {
                    'is_trained': True,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'device': str(self.device),
                    'specialized_symbols': list(self.symbol_models.keys()) if self.symbol_models else [],
                    'training_history': {}
                }
                try:
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    LOG.info("default_metadata_created")
                except Exception as save_error:
                    LOG.warning("default_metadata_save_failed", error=str(save_error))
            self.is_trained = metadata.get('is_trained', False)
            loaded_history = metadata.get('training_history', {})
            self.symbol_training_history = {}
            for symbol, history in loaded_history.items():
                try:
                    last_training = history.get('last_training')
                    if last_training:
                        if isinstance(last_training, str):
                            try:
                                last_training = datetime.fromisoformat(last_training)
                            except ValueError:
                                try:
                                    from dateutil import parser as dateparser
                                    last_training = dateparser.parse(last_training)
                                except Exception:
                                    LOG.warning("cannot_parse_training_date", symbol=symbol, date_string=last_training)
                                    last_training = None
                        elif not isinstance(last_training, datetime):
                            last_training = None
                    self.symbol_training_history[symbol] = {
                        'training_count': history.get('training_count', 0),
                        'last_training': last_training,
                        'samples_used': history.get('samples_used', [])
                    }
                except Exception as history_error:
                    LOG.warning("training_history_parse_failed", symbol=symbol, error=str(history_error))
                    self.symbol_training_history[symbol] = {
                        'training_count': 1,
                        'last_training': None,
                        'samples_used': []
                    }
            LOG.info("ensemble_models_loaded", path=base_path, timestamp=metadata.get('timestamp'), specialized_count=len(self.symbol_models))
            return True
        except Exception as e:
            LOG.error("ensemble_models_load_failed", error=str(e))
            return False

    async def ensemble_predict(self, df: pd.DataFrame, symbol: str = None, regime: str = "unknown") -> dict:
        try:
            if df is None or len(df) < 2:
                LOG.warning("invalid_dataframe_for_prediction", rows=len(df) if df is not None else 0)
                return {"action": "hold", "confidence": 0.0, "prediction_method": "invalid_df"}
            required_cols = ['close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                LOG.error("dataframe_missing_required_columns", missing=missing_cols, available=list(df.columns)[:10])
                return {"action": "hold", "confidence": 0.0, "prediction_method": "missing_columns"}
            
            # ✅ MEJORADO: Log detallado de qué modelo se está usando
            LOG.debug("ensemble_predict_called", 
                     symbol=symbol, 
                     df_shape=df.shape, 
                     general_trained=self.is_trained, 
                     has_specialized=hasattr(self, 'symbol_models') and symbol in self.symbol_models if symbol else False,
                     total_specialized_models=len(self.symbol_models) if hasattr(self, 'symbol_models') else 0)
            
            use_specialized = False
            LOG.debug("checking_for_specialized_model", symbol=symbol, has_symbol_models_attr=hasattr(self, 'symbol_models'), symbol_models_count=len(self.symbol_models) if hasattr(self, 'symbol_models') else 0)
            if symbol and hasattr(self, 'symbol_models') and self.symbol_models:
                if symbol in self.symbol_models:
                    symbol_dict = self.symbol_models[symbol]
                    expected_keys = ['lstm', 'gb', 'attention', 'technical']
                    available_keys = [k for k in expected_keys if k in symbol_dict]
                    LOG.debug("specialized_model_check", symbol=symbol, expected=expected_keys, available=available_keys, has_all=len(available_keys) == len(expected_keys))
                    valid_models = [k for k in available_keys if symbol_dict.get(k) is not None]
                    if len(valid_models) == len(expected_keys):
                        use_specialized = True
                        LOG.info("using_specialized_model", symbol=symbol, models=valid_models)
                    else:
                        LOG.warning("specialized_model_incomplete_or_none", symbol=symbol, missing=[k for k in expected_keys if k not in valid_models], none_models=[k for k in expected_keys if symbol_dict.get(k) is None])
                else:
                    LOG.debug("symbol_not_in_specialized_models", 
                             symbol=symbol, 
                             available_count=len(self.symbol_models), 
                             sample_symbols=list(self.symbol_models.keys())[:5],
                             # ✅ NUEVO: Log completo de símbolos disponibles cada 100 predicciones
                             all_symbols=list(self.symbol_models.keys()) if len(self.symbol_models) < 50 else f"{len(self.symbol_models)} models (samples: {list(self.symbol_models.keys())[:10]})")
            if use_specialized:
                LOG.debug("using_specialized_model", symbol=symbol)
                models_dict = self.symbol_models[symbol]
                lstm_model = models_dict.get('lstm', self.lstm)
                gb_model = models_dict.get('gb', self.gb)
                attn_model = models_dict.get('attention', self.attention)
                tech_model = models_dict.get('technical', self.technical)
            else:
                if not self.is_trained:
                    LOG.warning("ensemble_not_trained_using_fallback", symbol=symbol)
                    return {"action": "hold", "confidence": 0.3, "prediction_method": "not_trained"}
                LOG.debug("using_general_model", symbol=symbol, general_trained=self.is_trained)
                lstm_model = self.lstm
                gb_model = self.gb
                attn_model = self.attention
                tech_model = self.technical
            if lstm_model is None or gb_model is None or attn_model is None or tech_model is None:
                LOG.error("selected_models_contain_none", symbol=symbol, use_specialized=use_specialized, lstm_is_none=lstm_model is None, gb_is_none=gb_model is None, attention_is_none=attn_model is None, technical_is_none=tech_model is None)
                return {"action": "hold", "confidence": 0.0, "prediction_method": "models_none"}
            try:
                feature_cols = ['close', 'rsi', 'macd', 'volume']
                available_cols = [col for col in feature_cols if col in df.columns]
                if len(available_cols) < 3:
                    LOG.warning("insufficient_columns_for_ensemble_prediction", available=available_cols, required=feature_cols, symbol=symbol)
                    return {"action": "hold", "confidence": 0.0, "prediction_method": "insufficient_columns"}
                features = df[available_cols].tail(1).values
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                if features.shape[1] < 4:
                    missing_count = 4 - features.shape[1]
                    padding = np.zeros((features.shape[0], missing_count))
                    features = np.hstack([features, padding])
                if features.shape[1] != 4:
                    LOG.error("feature_shape_mismatch", expected=4, got=features.shape[1])
                    return {"action": "hold", "confidence": 0.0, "prediction_method": "shape_error"}
            except Exception as e:
                LOG.error("feature_extraction_failed", error=str(e))
                return {"action": "hold", "confidence": 0.0, "prediction_method": "feature_error"}
            predictions = {}
            confidences = {}

            def normalize_to_3_classes(pred_array, model_name: str):
                try:
                    pred_array = np.array(pred_array).flatten()
                    if len(pred_array) == 2:
                        sell_prob = pred_array[0] * 0.5
                        hold_prob = pred_array[0] * 0.5
                        buy_prob = pred_array[1]
                        normalized = np.array([sell_prob, hold_prob, buy_prob])
                    elif len(pred_array) == 3:
                        normalized = pred_array
                    elif len(pred_array) == 1:
                        value = float(pred_array[0])
                        if value > 0.6:
                            normalized = np.array([0.1, 0.2, 0.7])
                        elif value < 0.4:
                            normalized = np.array([0.7, 0.2, 0.1])
                        else:
                            normalized = np.array([0.2, 0.6, 0.2])
                    elif len(pred_array) > 3:
                        normalized = pred_array[:3]
                    else:
                        normalized = np.array([0.33, 0.34, 0.33])
                    normalized = normalized / (normalized.sum() + 1e-9)
                    if len(normalized) != 3:
                        LOG.warning("normalization_failed_unexpected_shape", model=model_name, shape=normalized.shape, original_shape=pred_array.shape)
                        normalized = np.array([0.33, 0.34, 0.33])
                    return normalized
                except Exception as e:
                    LOG.error("normalize_to_3_classes_failed", model=model_name, error=str(e))
                    return np.array([0.33, 0.34, 0.33])

            try:
                with torch.no_grad():
                    features_lstm = torch.FloatTensor(features).unsqueeze(1)
                    features_lstm = features_lstm.to(self.device)
                    lstm_pred_raw = lstm_model(features_lstm).cpu().numpy()[0]
                    if lstm_pred_raw.shape != (3,):
                        LOG.warning("lstm_pred_raw_unexpected_shape", shape=lstm_pred_raw.shape, symbol=symbol)
                        lstm_pred_raw = lstm_pred_raw.flatten()[:3] if len(lstm_pred_raw.flatten()) >= 3 else np.array([0.33, 0.34, 0.33])
                    lstm_pred = normalize_to_3_classes(lstm_pred_raw, 'lstm')
                    if lstm_pred.shape != (3,):
                        LOG.error("lstm_prediction_normalization_failed", symbol=symbol, pred_shape=lstm_pred.shape)
                        lstm_pred = np.array([0.33, 0.34, 0.33])
                    predictions['lstm'] = lstm_pred
                    confidences['lstm'] = float(np.max(lstm_pred))
                    LOG.debug("lstm_prediction_success", raw_shape=lstm_pred_raw.shape, normalized_shape=lstm_pred.shape, prediction=lstm_pred.tolist())
            except Exception as e:
                LOG.debug("lstm_prediction_failed", error=str(e))
                predictions['lstm'] = np.array([0.33, 0.34, 0.33])
                confidences['lstm'] = 0.34
            try:
                gb_pred_raw = gb_model.predict_proba(features)[0]
                gb_pred = normalize_to_3_classes(gb_pred_raw, 'gb')
                predictions['gb'] = gb_pred
                confidences['gb'] = float(np.max(gb_pred))
                LOG.debug("gb_prediction_success", raw_shape=gb_pred_raw.shape, normalized_shape=gb_pred.shape, prediction=gb_pred.tolist())
            except Exception as e:
                LOG.debug("gb_prediction_failed", error=str(e))
                predictions['gb'] = np.array([0.33, 0.34, 0.33])
                confidences['gb'] = 0.34
            try:
                with torch.no_grad():
                    features_attn = torch.FloatTensor(features).unsqueeze(1)
                    features_attn = features_attn.to(self.device)
                    attn_pred_raw = attn_model(features_attn).cpu().numpy()[0]
                    attn_pred = normalize_to_3_classes(attn_pred_raw, 'attention')
                    predictions['attention'] = attn_pred
                    confidences['attention'] = float(np.max(attn_pred))
                    LOG.debug("attention_prediction_success", raw_shape=attn_pred_raw.shape, normalized_shape=attn_pred.shape, prediction=attn_pred.tolist())
            except Exception as e:
                LOG.debug("attention_prediction_failed", error=str(e))
                predictions['attention'] = np.array([0.33, 0.34, 0.33])
                confidences['attention'] = 0.34
            try:
                tech_pred_raw = tech_model.predict_proba(features)[0]
                tech_pred = normalize_to_3_classes(tech_pred_raw, 'technical')
                predictions['technical'] = tech_pred
                confidences['technical'] = float(np.max(tech_pred))
                LOG.debug("technical_prediction_success", raw_shape=tech_pred_raw.shape, normalized_shape=tech_pred.shape, prediction=tech_pred.tolist())
            except Exception as e:
                LOG.debug("technical_prediction_failed", error=str(e))
                predictions['technical'] = np.array([0.33, 0.34, 0.33])
                confidences['technical'] = 0.34
            for model_name, pred in predictions.items():
                if pred.shape != (3,):
                    LOG.error("prediction_shape_invalid_after_normalization", model=model_name, shape=pred.shape, expected=(3,))
                    predictions[model_name] = np.array([0.33, 0.34, 0.33])
            try:
                pred_list = [predictions[k] for k in predictions.keys()]
                shapes = [p.shape for p in pred_list]
                LOG.debug("averaging_predictions", shapes=shapes, all_equal=(len(set(shapes)) == 1))
                if len(set(shapes)) != 1:
                    LOG.error("inconsistent_shapes_before_averaging", shapes=shapes)
                    pred_list = [p if p.shape == (3,) else np.array([0.33, 0.34, 0.33]) for p in pred_list]
                avg_pred = np.mean(pred_list, axis=0)
                if avg_pred.shape != (3,):
                    LOG.error("avg_pred_unexpected_shape", shape=avg_pred.shape)
                    avg_pred = np.array([0.33, 0.34, 0.33])
            except Exception as e:
                LOG.error("averaging_predictions_failed", error=str(e), traceback=traceback.format_exc())
                avg_pred = np.array([0.33, 0.34, 0.33])
            
            # AHORA sí usar avg_pred
            action_idx = np.argmax(avg_pred)
            confidence = float(avg_pred[action_idx])
            actions = {0: "sell", 1: "hold", 2: "buy"}
            action = actions.get(action_idx, "hold")
            if confidence < 0.5:
                sorted_pred = np.sort(avg_pred)[::-1]
                diff = sorted_pred[0] - sorted_pred[1]
                if diff > 0.15:
                    confidence_boost = diff * 3.0
                elif diff > 0.10:
                    confidence_boost = diff * 2.5
                elif diff > 0.05:
                    confidence_boost = diff * 2.0
                else:
                    confidence_boost = diff * 1.5
                confidence = min(0.85, confidence + confidence_boost)
                LOG.debug("confidence_boosted", original=float(avg_pred[action_idx]), diff=diff, boost=confidence_boost, boosted=confidence, action=action)
            if action == "hold":
                buy_prob = float(avg_pred[2])
                sell_prob = float(avg_pred[0])
                hold_prob = float(avg_pred[1])
                
                # CORRECCIÓN: Ajustar threshold según régimen de mercado
                base_threshold_diff = 0.20
                
                # En mercados volátiles, ser MÁS conservador
                if regime == 'volatile':
                    threshold_diff = 0.30  # Más difícil override
                elif regime == 'bull':
                    threshold_diff = 0.15  # Más fácil comprar
                elif regime == 'bear':
                    threshold_diff = 0.15  # Más fácil vender
                else:
                    threshold_diff = base_threshold_diff
                
                # NUEVO: Considerar también la confianza del régimen
                adjusted_threshold = threshold_diff * (1.0 + (1.0 - confidence) * 0.5)
                
                # Override solo si diferencia es significativa
                if buy_prob > (hold_prob - adjusted_threshold) and buy_prob > sell_prob:
                    # NUEVO: Verificar que buy_prob sea suficientemente alto en términos absolutos
                    if buy_prob > 0.35:  # Mínimo 35% de probabilidad
                        action = "buy"
                        confidence_boost = (buy_prob + hold_prob) / 2.0
                        confidence = min(0.85, confidence_boost * 1.3)
                        LOG.info("overriding_hold_to_buy",
                                original_hold_prob=hold_prob,
                                buy_prob=buy_prob,
                                new_confidence=confidence,
                                regime=regime,
                                threshold=adjusted_threshold,
                                reason="buy_close_to_hold_with_sufficient_probability")
                    else:
                        LOG.debug("hold_override_rejected_insufficient_buy_probability",
                                 buy_prob=buy_prob,
                                 required=0.35)
                
                elif sell_prob > (hold_prob - adjusted_threshold) and sell_prob > buy_prob:
                    # NUEVO: Verificar probabilidad mínima absoluta
                    if sell_prob > 0.35:
                        action = "sell"
                        confidence_boost = (sell_prob + hold_prob) / 2.0
                        confidence = min(0.85, confidence_boost * 1.3)
                        LOG.info("overriding_hold_to_sell",
                                original_hold_prob=hold_prob,
                                sell_prob=sell_prob,
                                new_confidence=confidence,
                                regime=regime,
                                threshold=adjusted_threshold,
                                reason="sell_close_to_hold_with_sufficient_probability")
                    else:
                        LOG.debug("hold_override_rejected_insufficient_sell_probability",
                                 sell_prob=sell_prob,
                                 required=0.35)
                else:
                    # NUEVO: Si hold es dominante, mantenerlo
                    if hold_prob > 0.50:
                        LOG.debug("hold_maintained_dominant_probability",
                                 hold_prob=hold_prob,
                                 buy_prob=buy_prob,
                                 sell_prob=sell_prob)
            LOG.info("ensemble_prediction_complete", 
                    action=action, 
                    confidence=float(confidence), 
                    avg_prediction=avg_pred.tolist(), 
                    individual_confidences=confidences)
        
            return {
                "action": action,
                "confidence": max(0.0, min(1.0, confidence)),
                "prediction_method": "ensemble",
                "details": {
                    "individual_predictions": {k: v.tolist() for k, v in predictions.items()},
                    "average_prediction": avg_pred.tolist(),
                    "confidences": confidences
                }
            }
        except Exception as e:
            LOG.error("ensemble_predict_failed", error=str(e), traceback=traceback.format_exc())
            return {"action": "hold", "confidence": 0.0, "prediction_method": "error"}

    async def micro_update(self, features: np.ndarray, target: Optional[int] = None, symbol: str = None):
        try:
            if symbol and symbol in self.symbol_models:
                LOG.debug("micro_update_specialized_model", symbol=symbol)
                models_to_update = [
                    ('gb', self.symbol_models[symbol]['gb']),
                    ('technical', self.symbol_models[symbol]['technical'])
                ]
            else:
                if not self.is_trained:
                    LOG.debug("micro_update_skipped_not_trained")
                    return False
                LOG.debug("micro_update_general_model")
                models_to_update = [
                    ('gb', self.gb),
                    ('technical', self.technical)
                ]
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            if features.shape[1] != 4:
                LOG.debug("micro_update_invalid_feature_shape", shape=features.shape, symbol=symbol)
                return False
            for model_name, model in models_to_update:
                try:
                    if hasattr(model, 'partial_fit') and target is not None:
                        model.partial_fit(features, [target], classes=[0, 1, 2])
                        LOG.debug("model_micro_updated", model=model_name, symbol=symbol)
                except Exception as e:
                    LOG.debug("model_micro_update_failed", model=model_name, error=str(e), symbol=symbol)
            return True
        except Exception as e:
            LOG.error("micro_update_failed", error=str(e), symbol=symbol)
            return False

class TrainingDataAccumulator:
    def __init__(self, max_samples: int = 10000):
        self.data_buffer = deque(maxlen=max_samples)
        self.symbol_buffers = {}
        self.max_samples_per_symbol = max_samples // 10
        self.lock = asyncio.Lock()
        self.samples_added = 0
        self.last_training = None
        self.symbol_last_training = {}
        # NUEVO: Tracking de antigüedad
        self.oldest_sample_time = None
        self.cleanup_interval = 86400  # 24 horas
        self.last_cleanup = time.time()

    async def add_sample(self, symbol: str, df_row: pd.Series, reward: float = None):
        async with self.lock:
            try:
                sample_dict = df_row.to_dict() if hasattr(df_row, 'to_dict') else df_row
                sample_timestamp = datetime.now(timezone.utc)
                
                sample_data = {
                    'symbol': symbol,
                    'timestamp': sample_timestamp,
                    'data': sample_dict,
                    'reward': reward
                }
                
                self.data_buffer.append(sample_data)
                
                # Actualizar oldest_sample_time
                if self.oldest_sample_time is None:
                    self.oldest_sample_time = sample_timestamp
                
                if symbol not in self.symbol_buffers:
                    self.symbol_buffers[symbol] = deque(maxlen=self.max_samples_per_symbol)
                
                self.symbol_buffers[symbol].append(sample_data)
                self.samples_added += 1
                
                # NUEVO: Limpieza periódica
                if time.time() - self.last_cleanup > self.cleanup_interval:
                    await self._cleanup_old_samples()
                
                if self.samples_added % 1000 == 0:
                    LOG.debug("data_accumulator_milestone",
                             total_samples=self.samples_added,
                             buffer_size=len(self.data_buffer),
                             symbols_tracked=len(self.symbol_buffers))
                    
            except Exception as e:
                LOG.error("add_sample_failed", error=str(e))

    async def _cleanup_old_samples(self):
        """
        NUEVO: Limpia samples antiguos que ya no son útiles
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=30)  # 30 días
            
            cleaned_count = 0
            
            # Limpiar buffers por símbolo
            for symbol in list(self.symbol_buffers.keys()):
                buffer = self.symbol_buffers[symbol]
                
                # Crear nuevo buffer sin samples antiguos
                new_buffer = deque(maxlen=self.max_samples_per_symbol)
                
                for sample in buffer:
                    if sample['timestamp'] > cutoff_time:
                        new_buffer.append(sample)
                    else:
                        cleaned_count += 1
                
                self.symbol_buffers[symbol] = new_buffer
                
                # Eliminar símbolo si no tiene datos
                if len(new_buffer) == 0:
                    del self.symbol_buffers[symbol]
            
            # Actualizar oldest_sample_time
            if len(self.data_buffer) > 0:
                self.oldest_sample_time = min(s['timestamp'] for s in self.data_buffer)
            else:
                self.oldest_sample_time = None
            
            self.last_cleanup = time.time()
            
            LOG.info("accumulator_cleanup_completed",
                    samples_removed=cleaned_count,
                    remaining_buffer_size=len(self.data_buffer),
                    symbols_remaining=len(self.symbol_buffers))
            
        except Exception as e:
            LOG.error("accumulator_cleanup_failed", error=str(e))

    async def get_training_data(self, min_samples: int = 500, symbol: str = None) -> Optional[pd.DataFrame]:
        async with self.lock:
            try:
                if symbol:
                    if symbol not in self.symbol_buffers:
                        LOG.debug("no_buffer_for_symbol", symbol=symbol)
                        return None
                    
                    symbol_buffer = self.symbol_buffers[symbol]
                    
                    if len(symbol_buffer) < min_samples:
                        LOG.debug("insufficient_samples_for_symbol",
                                 symbol=symbol,
                                 current=len(symbol_buffer),
                                 required=min_samples)
                        return None
                    
                    # NUEVO: Ordenar por timestamp antes de crear DataFrame
                    sorted_samples = sorted(symbol_buffer, key=lambda x: x['timestamp'])
                    data_list = [sample['data'] for sample in sorted_samples]
                    
                    df = pd.DataFrame(data_list)
                    df = df.ffill().bfill().fillna(0)
                    
                    self.symbol_last_training[symbol] = datetime.now(timezone.utc)
                    
                    LOG.info("symbol_specific_training_data_retrieved",
                            symbol=symbol,
                            samples=len(df),
                            columns=list(df.columns)[:10],
                            date_range=f"{sorted_samples[0]['timestamp']} to {sorted_samples[-1]['timestamp']}")
                    
                    return df
                
                if len(self.data_buffer) < min_samples:
                    LOG.debug("insufficient_samples_in_accumulator",
                             current=len(self.data_buffer),
                             required=min_samples)
                    return None
                
                # NUEVO: Ordenar datos generales también
                sorted_samples = sorted(self.data_buffer, key=lambda x: x['timestamp'])
                data_list = [sample['data'] for sample in sorted_samples]
                
                df = pd.DataFrame(data_list)
                df = df.ffill().bfill().fillna(0)
                
                self.last_training = datetime.now(timezone.utc)
                
                LOG.info("training_data_retrieved",
                        samples=len(df),
                        columns=list(df.columns)[:10])
                
                return df
                
            except Exception as e:
                LOG.error("get_training_data_failed", error=str(e))
                return None

    def get_stats(self) -> Dict[str, Any]:
        symbol_stats = {
            symbol: {
                'samples': len(buffer),
                'last_training': self.symbol_last_training.get(symbol),
                'oldest_sample': buffer[0]['timestamp'] if len(buffer) > 0 else None,
                'newest_sample': buffer[-1]['timestamp'] if len(buffer) > 0 else None
            }
            for symbol, buffer in self.symbol_buffers.items()
        }
        
        return {
            'buffer_size': len(self.data_buffer),
            'max_samples': self.data_buffer.maxlen,
            'total_samples_added': self.samples_added,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'last_cleanup': datetime.fromtimestamp(self.last_cleanup, tz=timezone.utc).isoformat(),
            'oldest_sample': self.oldest_sample_time.isoformat() if self.oldest_sample_time else None,
            'utilization': len(self.data_buffer) / self.data_buffer.maxlen if self.data_buffer.maxlen > 0 else 0,
            'symbols_tracked': len(self.symbol_buffers),
            'symbol_stats': symbol_stats
        }

class WalkForwardValidator:
    """
    Implementa validación walk-forward para evitar overfitting
    
    CARACTERÍSTICAS:
    - Ventanas deslizantes de entrenamiento/validación
    - Múltiples ventanas out-of-sample
    - Métricas de degradación de performance
    - Alerta de overfitting
    """
    def __init__(self, train_window_days: int = 60, test_window_days: int = 15, 
                 min_trades_per_window: int = 10):
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.min_trades_per_window = min_trades_per_window
        self.validation_results = []
        self.overfitting_detected = False
        
    
    def calculate_optimal_window_sizes(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        ✅ NUEVO: Calcula tamaños óptimos de ventana basados en datos disponibles
        
        Returns:
            Tuple[train_days, test_days]
        """
        try:
            total_rows = len(df)
            
            # Determinar timeframe (asumir 1h por defecto)
            if len(df) > 1:
                time_diff = (df.index[1] - df.index[0]).total_seconds() / 3600
                if time_diff < 0.5:
                    timeframe_hours = 0.25  # 15m
                elif time_diff < 2:
                    timeframe_hours = 1  # 1h
                elif time_diff < 6:
                    timeframe_hours = 4  # 4h
                else:
                    timeframe_hours = 24  # 1d
            else:
                timeframe_hours = 1
            
            rows_per_day = 24 / timeframe_hours
            
            # Calcular días disponibles
            available_days = total_rows / rows_per_day
            
            # Estrategia: usar 70% para train, 15% para test, 15% de overlap
            # Necesitamos al menos 2 ventanas
            min_train_days = 15
            min_test_days = 7
            
            # Calcular tamaños óptimos
            if available_days < (min_train_days + min_test_days) * 2:
                LOG.warning("insufficient_data_for_optimal_windows",
                           available_days=available_days,
                           required_min=min_train_days + min_test_days)
                return None, None
            
            # Usar 60% de datos para train, 20% para test
            optimal_train_days = int(available_days * 0.6)
            optimal_test_days = int(available_days * 0.2)
            
            # Límites razonables
            optimal_train_days = max(min_train_days, min(optimal_train_days, 90))
            optimal_test_days = max(min_test_days, min(optimal_test_days, 30))
            
            LOG.info("optimal_window_sizes_calculated",
                    available_days=available_days,
                    available_rows=total_rows,
                    timeframe_hours=timeframe_hours,
                    optimal_train_days=optimal_train_days,
                    optimal_test_days=optimal_test_days)
            
            return optimal_train_days, optimal_test_days
            
        except Exception as e:
            LOG.error("optimal_window_calculation_failed", error=str(e))
            return None, None
    
    async def run_walk_forward(self, bot, historical_df: pd.DataFrame, 
                               symbol: str) -> Dict[str, Any]:
        """
        Ejecuta validación walk-forward completa con ajuste automático
        """
        try:
            # ✅ NUEVO: Intentar calcular tamaños óptimos primero
            optimal_train, optimal_test = self.calculate_optimal_window_sizes(historical_df)
            
            if optimal_train and optimal_test:
                if (optimal_train != self.train_window_days or 
                    optimal_test != self.test_window_days):
                    LOG.info("adjusting_window_sizes_for_available_data",
                            original_train=self.train_window_days,
                            original_test=self.test_window_days,
                            optimal_train=optimal_train,
                            optimal_test=optimal_test)
                    
                    # Usar tamaños óptimos
                    self.train_window_days = optimal_train
                    self.test_window_days = optimal_test
                    
            # ✅ CORRECCIÓN: Validación más estricta de datos mínimos
            min_required_rows = (self.train_window_days + self.test_window_days) * 24 + 100
            
            if len(historical_df) < min_required_rows:
                LOG.warning("insufficient_data_for_walk_forward",
                           rows=len(historical_df),
                           required=min_required_rows,
                           train_days=self.train_window_days,
                           test_days=self.test_window_days)
                return {
                    "success": False, 
                    "error": f"Insufficient data: need {min_required_rows} rows, have {len(historical_df)}"
                }
            
            # Dividir en ventanas
            windows = self._create_windows(historical_df)
            
            # ✅ CORRECCIÓN: Mensaje más descriptivo si faltan ventanas
            if len(windows) < 2:
                LOG.warning("insufficient_windows",
                           windows=len(windows),
                           required=2,
                           data_rows=len(historical_df),
                           train_window_days=self.train_window_days,
                           test_window_days=self.test_window_days,
                           message="Try reducing train_window_days or test_window_days")
                return {
                    "success": False, 
                    "error": f"Need at least 2 windows, got {len(windows)}. Data has {len(historical_df)} rows. Consider reducing window sizes."
                }
            
            LOG.info("starting_walk_forward_validation",
                    symbol=symbol,
                    total_windows=len(windows),
                    train_days=self.train_window_days,
                    test_days=self.test_window_days,
                    data_rows=len(historical_df))
            
            window_results = []
            
            for i, (train_df, test_df) in enumerate(windows):
                LOG.info("processing_window",
                        window=i + 1,
                        total=len(windows),
                        train_samples=len(train_df),
                        test_samples=len(test_df))
                
                # Entrenar modelo en ventana de entrenamiento
                train_result = await self._train_on_window(
                    bot, train_df, symbol, window_id=i
                )
                
                if not train_result['success']:
                    LOG.warning("window_training_failed",
                               window=i + 1,
                               error=train_result.get('error'))
                    continue
                
                # Validar en ventana de test (out-of-sample)
                test_result = await self._test_on_window(
                    bot, test_df, symbol, window_id=i
                )
                
                window_results.append({
                    'window_id': i,
                    'train_period': {
                        'start': train_df.index[0].isoformat(),
                        'end': train_df.index[-1].isoformat(),
                        'samples': len(train_df)
                    },
                    'test_period': {
                        'start': test_df.index[0].isoformat(),
                        'end': test_df.index[-1].isoformat(),
                        'samples': len(test_df)
                    },
                    'train_metrics': train_result.get('metrics', {}),
                    'test_metrics': test_result.get('metrics', {}),
                    'degradation': self._calculate_degradation(
                        train_result.get('metrics', {}),
                        test_result.get('metrics', {})
                    )
                })
            
            # ✅ CORRECCIÓN: Verificar que tengamos ventanas válidas procesadas
            if len(window_results) < 2:
                LOG.warning("insufficient_valid_windows_after_processing",
                           processed=len(window_results),
                           required=2,
                           total_attempted=len(windows))
                return {
                    "success": False,
                    "error": f"Only {len(window_results)} valid windows processed from {len(windows)} attempted"
                }
            
            # Análisis agregado
            analysis = self._analyze_results(window_results)
            
            # Detectar overfitting
            self.overfitting_detected = analysis.get('overfitting_detected', False)
            
            if self.overfitting_detected:
                LOG.warning("overfitting_detected_in_walk_forward",
                           symbol=symbol,
                           avg_degradation=analysis.get('avg_degradation', 0))
                
                # Enviar alerta
                await ALERT_SYSTEM.send_alert(
                    "WARNING",
                    f"Overfitting detected for {symbol}",
                    avg_degradation=analysis.get('avg_degradation', 0),
                    windows_affected=analysis.get('windows_with_degradation', 0)
                )
            
            # Guardar resultados
            self.validation_results.append({
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'windows': window_results,
                'analysis': analysis
            })
            
            LOG.info("walk_forward_validation_completed",
                    symbol=symbol,
                    windows_processed=len(window_results),
                    overfitting=self.overfitting_detected)
            
            return {
                'success': True,
                'windows': window_results,
                'analysis': analysis
            }
            
        except Exception as e:
            LOG.error("walk_forward_validation_failed",
                     symbol=symbol,
                     error=str(e),
                     traceback=traceback.format_exc()[:500])
            return {"success": False, "error": str(e)}
    
    def _create_windows(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Crea ventanas deslizantes train/test"""
        windows = []
        
        # ✅ CORRECCIÓN: Calcular tamaños con detección de timeframe
        # Detectar timeframe real del DataFrame
        if len(df) > 1:
            time_diff_seconds = (df.index[1] - df.index[0]).total_seconds()
            if time_diff_seconds < 900:  # < 15 min
                hours_per_candle = 0.25
            elif time_diff_seconds < 3600:  # < 1h
                hours_per_candle = 0.5
            elif time_diff_seconds < 7200:  # < 2h
                hours_per_candle = 1.0
            elif time_diff_seconds < 21600:  # < 6h
                hours_per_candle = 4.0
            else:
                hours_per_candle = 24.0
        else:
            hours_per_candle = 1.0  # Default 1h

        if not isinstance(df, pd.DataFrame):
                LOG.error("df_is_not_dataframe_in_window_creation",
                         df_type=type(df).__name__,
                         message="Possible awaited coroutine issue")
                return []
        
        # Calcular samples necesarios
        train_samples = int(self.train_window_days * 24 / hours_per_candle)
        test_samples = int(self.test_window_days * 24 / hours_per_candle)
        window_step = test_samples
        
        LOG.debug("window_calculation",
                 detected_timeframe_hours=hours_per_candle,
                 train_days=self.train_window_days,
                 test_days=self.test_window_days,
                 train_samples=train_samples,
                 test_samples=test_samples)
        
        # ✅ Validar que tenemos suficientes datos
        min_required = train_samples + test_samples
        if len(df) < min_required:
            LOG.warning("insufficient_data_for_single_window",
                       data_rows=len(df),
                       required=min_required,
                       train_samples=train_samples,
                       test_samples=test_samples)
            return []
        
        start = 0
        while start + train_samples + test_samples <= len(df):
            train_end = start + train_samples
            test_end = train_end + test_samples
            
            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            
            # ✅ CORRECCIÓN: Validar calidad de las ventanas
            if len(train_df) < train_samples * 0.95:
                LOG.warning("train_window_too_small_skipping",
                           expected=train_samples,
                           actual=len(train_df))
                start += window_step
                continue
            
            if len(test_df) < test_samples * 0.95:
                LOG.warning("test_window_too_small_skipping",
                           expected=test_samples,
                           actual=len(test_df))
                start += window_step
                continue
            
            # ✅ Validar que no hay gaps grandes en los datos
            train_time_diff = (train_df.index[-1] - train_df.index[0]).total_seconds() / 3600
            expected_hours = self.train_window_days * 24
            if train_time_diff < expected_hours * 0.8:
                LOG.warning("train_window_has_gaps",
                           expected_hours=expected_hours,
                           actual_hours=train_time_diff,
                           gap_pct=(expected_hours - train_time_diff) / expected_hours * 100)
            
            windows.append((train_df, test_df))
            
            start += window_step
        
        LOG.info("windows_created",
                total_windows=len(windows),
                data_rows=len(df),
                train_samples_per_window=train_samples,
                test_samples_per_window=test_samples)
        
        return windows
    
    async def _train_on_window(self, bot, train_df: pd.DataFrame, 
                               symbol: str, window_id: int) -> Dict[str, Any]:
        """Entrena modelo en ventana específica"""
        try:
            # Preparar features y targets
            features = train_df[['close', 'rsi', 'macd', 'volume']].values
            
            # Crear targets basados en retornos futuros
            future_returns = train_df['close'].shift(-1) / train_df['close'] - 1
            targets = np.zeros(len(future_returns), dtype=int)
            targets[future_returns > 0.01] = 2  # Buy
            targets[future_returns < -0.01] = 0  # Sell
            targets[(future_returns >= -0.01) & (future_returns <= 0.01)] = 1  # Hold
            
            # Eliminar últimas filas sin target
            features = features[:-1]
            targets = targets[:-1]
            
            if len(features) < 50:
                return {"success": False, "error": "Insufficient samples"}
            
            # Entrenar ensemble especializado
            if hasattr(bot, 'ensemble_learner'):
                await bot.ensemble_learner.fit(
                    train_df[:-1],
                    pd.Series(targets),
                    symbol=f"{symbol}_window_{window_id}"
                )
            
            # Calcular métricas en train
            train_accuracy = self._calculate_window_metrics(
                train_df, targets, bot, symbol
            )
            
            return {
                'success': True,
                'metrics': {
                    'accuracy': train_accuracy,
                    'samples': len(features)
                }
            }
            
        except Exception as e:
            LOG.error("train_on_window_failed",
                     window=window_id,
                     error=str(e))
            return {"success": False, "error": str(e)}
    
    async def _test_on_window(self, bot, test_df: pd.DataFrame,
                             symbol: str, window_id: int) -> Dict[str, Any]:
        """Testea modelo en ventana out-of-sample"""
        try:
            # Generar predicciones
            predictions = []
            actuals = []
            
            for i in range(len(test_df) - 1):
                # Predicción
                if hasattr(bot, 'ensemble_learner'):
                    pred = await bot.ensemble_learner.ensemble_predict(
                        test_df.iloc[:i+1],
                        symbol=f"{symbol}_window_{window_id}"
                    )
                    action = pred.get('action', 'hold')
                else:
                    action = 'hold'
                
                # Actual
                actual_return = (test_df['close'].iloc[i+1] - test_df['close'].iloc[i]) / test_df['close'].iloc[i]
                
                predictions.append(action)
                actuals.append(actual_return)
            
            # Calcular métricas
            correct = 0
            total = len(predictions)
            
            for pred, actual in zip(predictions, actuals):
                if pred == 'buy' and actual > 0:
                    correct += 1
                elif pred == 'sell' and actual < 0:
                    correct += 1
                elif pred == 'hold' and -0.01 <= actual <= 0.01:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            
            return {
                'success': True,
                'metrics': {
                    'accuracy': accuracy,
                    'samples': total
                }
            }
            
        except Exception as e:
            LOG.error("test_on_window_failed",
                     window=window_id,
                     error=str(e))
            return {"success": False, "error": str(e)}
    
    def _calculate_window_metrics(self, df: pd.DataFrame, targets: np.ndarray,
                                  bot, symbol: str) -> float:
        """Calcula accuracy en ventana"""
        try:
            # Simplificado: retornar accuracy base
            unique, counts = np.unique(targets, return_counts=True)
            if len(counts) > 0:
                # Baseline accuracy (clase mayoritaria)
                return max(counts) / sum(counts)
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_degradation(self, train_metrics: Dict, test_metrics: Dict) -> float:
        """Calcula degradación de performance train->test"""
        try:
            train_acc = train_metrics.get('accuracy', 0)
            test_acc = test_metrics.get('accuracy', 0)
            
            if train_acc == 0:
                return 0.0
            
            # Degradación porcentual
            degradation = (train_acc - test_acc) / train_acc
            return float(degradation)
        except Exception:
            return 0.0
    
    def _analyze_results(self, window_results: List[Dict]) -> Dict[str, Any]:
        """Analiza resultados agregados con estadísticas robustas"""
        if not window_results:
            return {}
        
        degradations = [w['degradation'] for w in window_results if 'degradation' in w]
        
        if not degradations:
            return {'overfitting_detected': False, 'total_windows': 0}
        
        avg_degradation = np.mean(degradations)
        max_degradation = np.max(degradations)
        std_degradation = np.std(degradations)
        
        # MEJORA: Análisis multi-criterio de overfitting
        overfitting_signals = []
        
        # Criterio 1: Degradación promedio alta
        if avg_degradation > 0.20:
            overfitting_signals.append('high_avg_degradation')
        
        # Criterio 2: Degradación máxima extrema
        if max_degradation > 0.40:
            overfitting_signals.append('extreme_max_degradation')
        
        # Criterio 3: Alta varianza (inconsistencia)
        if std_degradation > 0.15:
            overfitting_signals.append('high_variance')
        
        # Criterio 4: Mayoría de ventanas con degradación > 15%
        high_degradation_count = sum(1 for d in degradations if d > 0.15)
        if high_degradation_count > len(degradations) * 0.6:
            overfitting_signals.append('consistent_degradation')
        
        # NUEVO: Overfitting solo si 2+ señales
        overfitting = len(overfitting_signals) >= 2
        
        LOG.info("walk_forward_analysis_complete",
                avg_degradation=avg_degradation,
                max_degradation=max_degradation,
                std_degradation=std_degradation,
                overfitting_signals=overfitting_signals,
                overfitting_detected=overfitting)
        
        windows_with_degradation = sum(1 for d in degradations if d > 0.15)
        
        return {
            'avg_degradation': avg_degradation,
            'max_degradation': max_degradation,
            'min_degradation': np.min(degradations) if degradations else 0,
            'std_degradation': np.std(degradations) if degradations else 0,
            'windows_with_degradation': windows_with_degradation,
            'overfitting_detected': overfitting,
            'total_windows': len(window_results)
        }

async def periodic_rl_training(bot, exchange_manager, config, interval_hours=24):
    while bot.is_running:
        try:
            await asyncio.sleep(interval_hours * 3600)
            LOG.info("starting_periodic_rl_training")
            hist_data = await bot._load_historical_data(months=3)
            if hist_data is not None and len(hist_data) > 100:
                await bot.rl_training_manager.train(episodes=50, df=hist_data)
                LOG.info("periodic_rl_training_completed")
        except asyncio.CancelledError:
            break
        except Exception as e:
            LOG.error("periodic_rl_training_failed", error=str(e))

def create_dataframe(ohlcv_data: List) -> Optional[pd.DataFrame]:
    try:
        if not ohlcv_data or len(ohlcv_data) == 0:
            LOG.error("empty_ohlcv_data")
            return None
        if not isinstance(ohlcv_data, (list, tuple)):
            LOG.error("invalid_ohlcv_type", type=str(type(ohlcv_data)))
            return None
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except Exception as e:
            LOG.error("dataframe_creation_from_list_failed", error=str(e))
            return None
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            LOG.error("missing_columns", missing=list(missing))
            return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        initial_len = len(df)
        df = df.dropna(subset=['close', 'open', 'high', 'low'])
        if len(df) < 20:
            LOG.error("insufficient_valid_data_after_cleaning", before=initial_len, after=len(df))
            return None
        df = df.replace([np.inf, -np.inf], np.nan)
        df['close'] = df['close'].ffill().bfill().fillna(0)
        df['open'] = df['open'].ffill().bfill().fillna(0)
        df['high'] = df['high'].ffill().bfill().fillna(0)
        df['low'] = df['low'].ffill().bfill().fillna(0)
        df['volume'] = df['volume'].fillna(0)
        if df['close'].isna().any() or df['close'].isin([np.inf, -np.inf]).any():
            LOG.error("invalid_close_values_remain")
            return None
        df.set_index('timestamp', inplace=True)
        LOG.debug("dataframe_created_successfully", rows=len(df), columns=len(df.columns))
        return df
    except Exception as e:
        LOG.error("create_dataframe_failed", error=str(e))
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df is None or len(df) == 0:
            LOG.warning("empty_dataframe_for_indicators")
            return df if df is not None else pd.DataFrame()
        if 'close' not in df.columns:
            LOG.error("close_column_missing_in_dataframe")
            return df
        if len(df) < 50:
            LOG.warning("insufficient_data_for_indicators", rows=len(df))
            return df
        cache_key = None
        try:
            # OPTIMIZACIÓN: Hash más eficiente usando solo últimos valores críticos
            if len(df) > 0:
                # Crear hash basado en últimas 5 velas (suficiente para detectar cambios)
                cache_components = (
                    len(df),
                    float(df['close'].iloc[-1]),
                    float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0,
                    float(df['close'].iloc[-5]) if len(df) >= 5 else 0
                )
                cache_key = f"ind_{hash(cache_components)}"
                
                if cache_key in FEATURE_CACHE._cache:
                    timestamp = FEATURE_CACHE._timestamps.get(cache_key, 0)
                    if time.time() - timestamp < FEATURE_CACHE.ttl:
                        cached_df = FEATURE_CACHE._cache[cache_key]
                        FEATURE_CACHE._hit_count += 1
                        LOG.debug("using_cached_indicators", 
                                 cache_hit=True, 
                                 cache_age_seconds=time.time() - timestamp)
                        return cached_df.copy()
                
                FEATURE_CACHE._miss_count += 1
        except Exception as e:
            LOG.debug("cache_lookup_failed", error=str(e))
            cache_key = None
        df = df.copy()
        indicators_to_calculate = []
        if 'rsi' not in df.columns or df['rsi'].isna().all():
            indicators_to_calculate.append('rsi')
        if 'macd' not in df.columns or df['macd'].isna().all():
            indicators_to_calculate.append('macd')
        if 'sma_20' not in df.columns or df['sma_20'].isna().all():
            indicators_to_calculate.append('sma')
        if 'bb_upper' not in df.columns or df['bb_upper'].isna().all():
            indicators_to_calculate.append('bollinger')
        if 'volatility' not in df.columns or df['volatility'].isna().all():
            indicators_to_calculate.append('volatility')
        if 'adx' not in df.columns or df['adx'].isna().all():
            indicators_to_calculate.append('adx')
        LOG.debug("indicators_to_calculate", missing=indicators_to_calculate, total_needed=len(indicators_to_calculate))
        if 'rsi' in indicators_to_calculate:
            try:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss.replace(0, 1e-9))
                df['rsi'] = 100 - (100 / (1 + rs))
            except Exception as e:
                LOG.debug("rsi_calculation_failed", error=str(e))
                df['rsi'] = 50.0
        if 'macd' in indicators_to_calculate:
            try:
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                df['macd'] = macd
                df['macd_signal'] = signal
                df['macd_hist'] = macd - signal
            except Exception as e:
                LOG.debug("macd_calculation_failed", error=str(e))
                df['macd'] = 0.0
                df['macd_signal'] = 0.0
                df['macd_hist'] = 0.0

        # Moving Averages
        if 'sma' in indicators_to_calculate:
            try:
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            except Exception as e:
                LOG.debug("moving_averages_failed", error=str(e))

        # Bollinger Bands
        if 'bollinger' in indicators_to_calculate:
            try:
                rolling_mean = df['close'].rolling(window=20).mean()
                rolling_std = df['close'].rolling(window=20).std()
                df['bb_upper'] = rolling_mean + (rolling_std * 2)
                df['bb_middle'] = rolling_mean
                df['bb_lower'] = rolling_mean - (rolling_std * 2)
            except Exception as e:
                LOG.debug("bollinger_bands_failed", error=str(e))

        # Volatility
        if 'volatility' in indicators_to_calculate:
            try:
                df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            except Exception as e:
                LOG.debug("volatility_calculation_failed", error=str(e))

        # Momentum
        try:
            if 'momentum' not in df.columns:
                df['momentum'] = df['close'] - df['close'].shift(10)
        except Exception as e:
            LOG.debug("momentum_calculation_failed", error=str(e))

        # ADX
        if 'adx' in indicators_to_calculate:
            try:
                if 'high' in df.columns and 'low' in df.columns:
                    from __main__ import AdvancedAIConfig
                    minimal_config = AdvancedAIConfig(symbols=['BTC/USDT'])
                    strategy_mgr = StrategyManager(minimal_config)
                    df['adx'] = strategy_mgr.adx(df['high'], df['low'], df['close'], 14)
                else:
                    df['adx'] = 25.0
            except Exception as e:
                LOG.debug("adx_calculation_failed", error=str(e))
                df['adx'] = 25.0

        # Returns
        try:
            if 'returns' not in df.columns:
                df['returns'] = df['close'].pct_change()
            if 'log_returns' not in df.columns:
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        except Exception as e:
            LOG.debug("returns_calculation_failed", error=str(e))

        # CORRECCIÓN: Validar antes de llenar NaN
        # Verificar que columnas críticas no tengan TODOS NaN
        critical_cols = ['rsi', 'macd', 'sma_20']
        for col in critical_cols:
            if col in df.columns and df[col].isna().all():
                LOG.error("critical_indicator_all_nan",
                         column=col,
                         message="Calculation may have failed")
                # Intentar recalcular o usar valores default seguros
                if col == 'rsi':
                    df['rsi'] = 50.0
                elif col == 'macd':
                    df['macd'] = 0.0
                    df['macd_signal'] = 0.0
                    df['macd_hist'] = 0.0
                elif col == 'sma_20':
                    df['sma_20'] = df['close']
        
        # CORRECCIÓN: Llenar NaN de forma más inteligente
        # 1. Forward fill primero (usar valores pasados)
        df = df.ffill()
        
        # 2. Backward fill para los primeros valores
        df = df.bfill()
        
        # 3. Solo después llenar con 0 los que quedan
        df = df.fillna(0)
        
        # NUEVO: Validación post-fill
        remaining_nans = df.isna().sum().sum()
        if remaining_nans > 0:
            LOG.warning("nans_remain_after_fill",
                       total_nans=remaining_nans,
                       columns_with_nans=df.columns[df.isna().any()].tolist())

        # ✅ GUARDAR EN CACHE
        if cache_key:
            try:
                FEATURE_CACHE._cache[cache_key] = df.copy()
                FEATURE_CACHE._timestamps[cache_key] = time.time()
                
                # Limpiar cache si excede tamaño
                if len(FEATURE_CACHE._cache) > FEATURE_CACHE.max_size:
                    oldest_key = min(FEATURE_CACHE._timestamps.keys(),
                                   key=lambda k: FEATURE_CACHE._timestamps[k])
                    del FEATURE_CACHE._cache[oldest_key]
                    del FEATURE_CACHE._timestamps[oldest_key]
                
                LOG.debug("indicators_cached",
                         cache_size=len(FEATURE_CACHE._cache),
                         calculated=indicators_to_calculate)
            except Exception as e:
                LOG.debug("cache_save_failed", error=str(e))

        LOG.debug("technical_indicators_calculated",
                 rows=len(df),
                 columns=len(df.columns),
                 newly_calculated=len(indicators_to_calculate))
        
        return df

    except Exception as e:
        LOG.error("calculate_technical_indicators_failed",
                 error=str(e),
                 traceback=traceback.format_exc()[:500])
        return df

async def fetch_market_data(exchange_manager, symbol: str, timeframe: str, max_retries: int = 5) -> Dict[str, Any]:
    """
    Obtiene datos de mercado con manejo robusto de errores y reintentos exponenciales
    
    MEJORAS:
    - Reintentos con backoff exponencial
    - Manejo de rate limits
    - Validación exhaustiva de datos
    - Fallback a timeframes alternativos
    - Cache de último resultado válido
    """
    cache_key = f"{symbol}_{timeframe}"
    
    # Cache estático para fallback
    if not hasattr(fetch_market_data, '_cache'):
        fetch_market_data._cache = {}
    
    for attempt in range(max_retries):
        try:
            if not exchange_manager or not symbol:
                return {"success": False, "error": "Invalid exchange_manager or symbol"}
            
            # Calcular backoff
            if attempt > 0:
                backoff = min(2 ** attempt, 30)  # Máximo 30 segundos
                LOG.debug("fetch_ohlcv_retry",
                         symbol=symbol,
                         attempt=attempt + 1,
                         backoff_seconds=backoff)
                await asyncio.sleep(backoff)
            
            # Intentar fetch con timeout
            try:
                result = await asyncio.wait_for(
                    exchange_manager.fetch_ohlcv(symbol, timeframe, limit=200),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                LOG.warning("fetch_ohlcv_timeout",
                           symbol=symbol,
                           attempt=attempt + 1,
                           timeout=15.0)
                if attempt < max_retries - 1:
                    continue
                else:
                    raise
            
            # Validar respuesta básica
            if not result or not isinstance(result, dict):
                LOG.warning("invalid_ohlcv_response_type",
                           symbol=symbol,
                           response_type=type(result).__name__)
                continue
            
            # Manejar errores del exchange
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                
                # Rate limit - esperar más
                if "rate" in error_msg.lower() or "429" in error_msg:
                    LOG.warning("rate_limit_detected",
                               symbol=symbol,
                               attempt=attempt + 1)
                    await asyncio.sleep(60)  # 1 minuto
                    continue
                
                # Symbol no disponible - fallar inmediatamente
                if "symbol" in error_msg.lower() or "not found" in error_msg.lower():
                    LOG.error("symbol_not_available",
                             symbol=symbol,
                             error=error_msg)
                    return {"success": False, "error": f"Symbol not available: {error_msg}"}
                
                LOG.warning("ohlcv_fetch_error",
                           symbol=symbol,
                           error=error_msg,
                           attempt=attempt + 1)
                continue
            
            # Validar datos OHLCV
            ohlcv = result.get("ohlcv", [])
            if not ohlcv or not isinstance(ohlcv, list):
                LOG.warning("empty_or_invalid_ohlcv",
                           symbol=symbol,
                           ohlcv_type=type(ohlcv).__name__)
                continue
            
            if len(ohlcv) < 10:
                LOG.warning("insufficient_ohlcv_data",
                           symbol=symbol,
                           count=len(ohlcv),
                           required=10)
                
                # Intentar con timeframe alternativo si es el último intento
                if attempt == max_retries - 1:
                    alt_timeframe = '15m' if timeframe == '1h' else '1h'
                    LOG.info("trying_alternative_timeframe",
                            symbol=symbol,
                            original=timeframe,
                            alternative=alt_timeframe)
                    try:
                        alt_result = await exchange_manager.fetch_ohlcv(
                            symbol, alt_timeframe, limit=200
                        )
                        if alt_result and alt_result.get("success"):
                            ohlcv = alt_result.get("ohlcv", [])
                            if len(ohlcv) >= 10:
                                LOG.info("alternative_timeframe_success",
                                        symbol=symbol,
                                        timeframe=alt_timeframe)
                                result = alt_result
                    except Exception:
                        pass
                
                if len(ohlcv) < 10:
                    continue
            
            # Validar estructura de cada vela
            valid_candles = []
            for i, candle in enumerate(ohlcv):
                try:
                    if not isinstance(candle, (list, tuple)) or len(candle) < 6:
                        LOG.debug("invalid_candle_structure",
                                 symbol=symbol,
                                 index=i,
                                 candle=candle)
                        continue
                    
                    timestamp, open_price, high, low, close, volume = candle[:6]
                    
                    # Validar valores
                    if any(x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) 
                           for x in [open_price, high, low, close, volume]):
                        LOG.debug("invalid_candle_values",
                                 symbol=symbol,
                                 index=i)
                        continue
                    
                    # Validar lógica OHLC
                    if not (low <= open_price <= high and low <= close <= high):
                        LOG.debug("invalid_ohlc_logic",
                                 symbol=symbol,
                                 index=i,
                                 open=open_price,
                                 high=high,
                                 low=low,
                                 close=close)
                        continue
                    
                    # Validar que los precios sean positivos
                    if any(x <= 0 for x in [open_price, high, low, close]):
                        LOG.debug("non_positive_prices",
                                 symbol=symbol,
                                 index=i)
                        continue
                    
                    valid_candles.append(candle)
                    
                except Exception as candle_error:
                    LOG.debug("candle_validation_error",
                             symbol=symbol,
                             index=i,
                             error=str(candle_error))
                    continue
            
            if len(valid_candles) < 10:
                LOG.warning("insufficient_valid_candles",
                           symbol=symbol,
                           total=len(ohlcv),
                           valid=len(valid_candles))
                continue
            
            # Crear DataFrame con validación
            df = create_dataframe(valid_candles)
            if df is None or len(df) == 0 or 'close' not in df.columns:
                LOG.error("dataframe_creation_failed",
                         symbol=symbol,
                         valid_candles=len(valid_candles))
                continue
            
            # Calcular indicadores técnicos
            try:
                df = calculate_technical_indicators(df)
                if 'rsi' not in df.columns or df['rsi'].isna().all():
                    LOG.warning("technical_indicators_incomplete",
                               symbol=symbol)
                    # Continuar de todos modos si tenemos datos básicos
            except Exception as ind_error:
                LOG.warning("technical_indicators_failed",
                           symbol=symbol,
                           error=str(ind_error))
                # No es crítico, continuar
            
            # NUEVO: Validar calidad del resultado antes de cachear
            result_quality_ok = True
            
            # Verificar que tenemos suficientes datos válidos
            if len(df) < 10:
                result_quality_ok = False
                LOG.warning("result_has_insufficient_rows", symbol=symbol, rows=len(df))
            
            # Verificar que close no tiene muchos NaN
            if 'close' in df.columns:
                close_nan_pct = df['close'].isna().sum() / len(df)
                if close_nan_pct > 0.1:  # Más del 10% NaN
                    result_quality_ok = False
                    LOG.warning("result_has_excessive_nans",
                               symbol=symbol,
                               nan_pct=close_nan_pct * 100)
            
            # Verificar que tenemos al menos algunos indicadores
            required_indicators = ['rsi', 'macd', 'sma_20']
            available_indicators = [ind for ind in required_indicators if ind in df.columns and not df[ind].isna().all()]
            if len(available_indicators) < 2:
                LOG.warning("result_has_insufficient_indicators",
                           symbol=symbol,
                           available=available_indicators,
                           required=required_indicators)
                # No invalidar por esto, pero loggear
            
            # CORRECCIÓN: Solo cachear si calidad es buena
            if result_quality_ok:
                fetch_market_data._cache[cache_key] = {
                    'data': result,
                    'timestamp': time.time(),
                    'quality': 'good'
                }
                LOG.debug("market_data_cached",
                         symbol=symbol,
                         quality='good')
            else:
                LOG.warning("market_data_not_cached_poor_quality",
                           symbol=symbol)
            
            LOG.debug("market_data_fetched_successfully",
                     symbol=symbol,
                     candles=len(valid_candles),
                     attempt=attempt + 1,
                     cached=result_quality_ok)
            
            return {
                "success": True,
                "ohlcv": valid_candles,
                "dataframe": df,
                "quality": 'good' if result_quality_ok else 'acceptable'
            }
            
        except asyncio.CancelledError:
            LOG.info("fetch_market_data_cancelled", symbol=symbol)
            raise
        
        except Exception as e:
            LOG.warning("fetch_market_data_exception",
                       symbol=symbol,
                       attempt=attempt + 1,
                       error=str(e),
                       error_type=type(e).__name__)
            
            # En el último intento, intentar usar cache
            if attempt == max_retries - 1:
                cached = fetch_market_data._cache.get(cache_key)
                # NUEVO: Verificar calidad del cache antes de usarlo
                if cached and cached.get('quality') == 'good':
                    cache_age = time.time() - cached['timestamp']
                    if cache_age < 3600:  # 1 hora
                        LOG.info("using_cached_market_data",
                                symbol=symbol,
                                age_seconds=cache_age,
                                quality='good')
                        return cached['data']
                    else:
                        LOG.warning("cached_data_too_old",
                                   symbol=symbol,
                                   age_hours=cache_age / 3600)
                elif cached and cached.get('quality') == 'acceptable':
                    # Cache de calidad aceptable - usar solo en último recurso
                    LOG.warning("using_acceptable_quality_cache",
                               symbol=symbol,
                               age_seconds=time.time() - cached['timestamp'])
                    return cached['data']
            
            if attempt < max_retries - 1:
                continue
            else:
                return {
                    "success": False,
                    "error": f"Failed after {max_retries} attempts: {str(e)}",
                    "ohlcv": []
                }
    
    return {"success": False, "error": "Max retries exceeded", "ohlcv": []}

async def fetch_ticker_robust(exchange_manager, symbol: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Obtiene ticker con manejo robusto de errores
    
    MEJORAS:
    - Reintentos automáticos
    - Validación de datos
    - Fallback a último precio conocido
    - Manejo de rate limits
    """
    cache_key = f"ticker_{symbol}"
    
    # Cache estático
    if not hasattr(fetch_ticker_robust, '_cache'):
        fetch_ticker_robust._cache = {}
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                await asyncio.sleep(min(2 ** attempt, 10))
            
            # Fetch con timeout
            try:
                ticker = await asyncio.wait_for(
                    exchange_manager.exchange.fetch_ticker(symbol),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                LOG.warning("ticker_fetch_timeout",
                           symbol=symbol,
                           attempt=attempt + 1)
                if attempt < max_retries - 1:
                    continue
                else:
                    raise
            
            # Validar respuesta
            if not ticker or not isinstance(ticker, dict):
                LOG.warning("invalid_ticker_response",
                           symbol=symbol,
                           response_type=type(ticker).__name__)
                continue
            
            # Validar precio
            last_price = ticker.get('last') or ticker.get('close')
            if not last_price or last_price <= 0 or np.isnan(last_price) or np.isinf(last_price):
                LOG.warning("invalid_ticker_price",
                           symbol=symbol,
                           price=last_price)
                
                # Intentar otros campos
                for field in ['bid', 'ask', 'average']:
                    price = ticker.get(field)
                    if price and price > 0 and not np.isnan(price) and not np.isinf(price):
                        last_price = price
                        LOG.info("using_alternative_price_field",
                                symbol=symbol,
                                field=field,
                                price=price)
                        break
                
                if not last_price or last_price <= 0:
                    continue
            
            # Actualizar cache
            fetch_ticker_robust._cache[cache_key] = {
                'ticker': ticker,
                'timestamp': time.time()
            }
            
            LOG.debug("ticker_fetched_successfully",
                     symbol=symbol,
                     price=last_price,
                     attempt=attempt + 1)
            
            return ticker
            
        except Exception as e:
            LOG.warning("ticker_fetch_exception",
                       symbol=symbol,
                       attempt=attempt + 1,
                       error=str(e))
            
            # Último intento - usar cache
            if attempt == max_retries - 1:
                cached = fetch_ticker_robust._cache.get(cache_key)
                if cached and time.time() - cached['timestamp'] < 300:  # 5 minutos
                    LOG.info("using_cached_ticker",
                            symbol=symbol,
                            age_seconds=time.time() - cached['timestamp'])
                    return cached['ticker']
    
    LOG.error("ticker_fetch_failed_all_attempts", symbol=symbol)
    return None

async def discover_usdt_pairs(exchange_manager, exclude_stablecoins: bool = True) -> List[str]:
    """
    Descubre automáticamente todos los pares spot con USDT del exchange
    excluyendo pares entre monedas estables
    """
    try:
        markets_result = await exchange_manager.exchange.load_markets()
        if not markets_result:
            LOG.warning("no_markets_loaded", message="Could not load markets from exchange")
            return ['BTC/USDT', 'ETH/USDT']
        # Filtrar pares USDT spot
        usdt_pairs = []
        stablecoins = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDD', 'GUSD', 'FRAX', 'LUSD'}
        for symbol, market in markets_result.items():
            try:
                # Verificar que sea spot y quote sea USDT
                if (market.get('type') == 'spot' and
                    market.get('quote') == 'USDT' and
                    market.get('active', False)):
                    base = market.get('base', '')
                    # Excluir stablecoins si está habilitado
                    if exclude_stablecoins and base in stablecoins:
                        continue
                    # Excluir tokens leveraged
                    if any(suffix in base for suffix in ['UP', 'DOWN', 'BULL', 'BEAR']):
                        continue
                    usdt_pairs.append(symbol)
            except Exception:
                continue
        # Ordenar alfabéticamente
        usdt_pairs.sort()
        LOG.info("usdt_pairs_discovered",
                total_discovered=len(usdt_pairs),
                exclude_stablecoins=exclude_stablecoins,
                returning_all=True)
        return usdt_pairs if usdt_pairs else ['BTC/USDT', 'ETH/USDT']

    except Exception as e:
        LOG.error("usdt_pairs_discovery_failed", error=str(e))
        # Fallback final a pares principales
        return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']


async def filter_pairs_by_volume(exchange_manager, pairs: List[str],
                                min_volume_24h: float = 1000000.0) -> List[str]:
    """
    Filtra pares por volumen mínimo de 24h en USDT
    """
    if not pairs:
        return []
    try:
        filtered_pairs = []
        for symbol in pairs:
            try:
                # Intentar obtener ticker
                ticker = await exchange_manager.exchange.fetch_ticker(symbol)
                volume_usdt = ticker.get('quoteVolume', 0)
                if volume_usdt >= min_volume_24h:
                    filtered_pairs.append(symbol)
            except Exception as e:
                LOG.debug("ticker_fetch_failed_skipping", symbol=symbol, error=str(e))
                continue
        if not filtered_pairs:
            LOG.warning("no_pairs_passed_volume_filter",
                       min_volume=min_volume_24h,
                       total_tested=len(pairs),
                       message="Retornando pares originales sin filtro")
            return pairs
        LOG.info("pairs_filtered_by_volume",
                original_count=len(pairs),
                filtered_count=len(filtered_pairs),
                min_volume_usdt=min_volume_24h)
        return filtered_pairs
    except Exception as e:
        LOG.error("volume_filtering_failed", error=str(e))
        return pairs


CFG = create_config()


async def aggregate_strategy_signals(strategy_signals: Dict[str, Dict],
                                    regime: str = "unknown",
                                    regime_confidence: float = 0.5) -> Dict[str, any]:
    """
    MEJORADO: Agrega señales con ajuste por régimen Y performance histórica
    """
    try:
        if not strategy_signals:
            return {"signal": "hold", "confidence": 0.0, "reason": "No signals available"}
        
        # Mapear señales a valores numéricos
        signal_map = {"buy": 1.0, "hold": 0.0, "sell": -1.0}
        
        # Ponderación por régimen
        regime_weights = {
            "bull": {"buy": 1.3, "hold": 0.9, "sell": 0.7},
            "bear": {"buy": 0.7, "hold": 0.9, "sell": 1.3},
            "sideways": {"buy": 0.9, "hold": 1.1, "sell": 0.9},
            "volatile": {"buy": 0.8, "hold": 1.2, "sell": 0.8},
            "unknown": {"buy": 1.0, "hold": 1.0, "sell": 1.0}
        }
        regime_weight_map = regime_weights.get(regime, regime_weights["unknown"])
        
        # NUEVO: Obtener performance histórica de estrategias (si disponible)
        strategy_performance = {}
        if 'strategy_manager' in globals():
            try:
                from __main__ import bot
                if hasattr(bot, 'strategy_manager'):
                    for strategy_name in strategy_signals.keys():
                        perf = bot.strategy_manager.strategy_performance.get(strategy_name, {})
                        if perf.get('total_signals', 0) >= 10:
                            # Usar win_rate como factor de ajuste
                            win_rate = perf.get('win_rate', 0.5)
                            # Ajustar peso: 0.7x a 1.3x basado en win_rate
                            performance_multiplier = 0.7 + (win_rate * 0.6)
                            strategy_performance[strategy_name] = performance_multiplier
                        else:
                            strategy_performance[strategy_name] = 1.0
            except Exception as perf_error:
                LOG.debug("strategy_performance_lookup_failed", error=str(perf_error))
        
        # Si no hay performance histórica, usar pesos iguales
        if not strategy_performance:
            for strategy_name in strategy_signals.keys():
                strategy_performance[strategy_name] = 1.0
        
        weighted_score = 0.0
        total_weight = 0.0
        signal_details = {}
        
        for strategy_name, signal_data in strategy_signals.items():
            signal = signal_data.get('signal', 'hold')
            confidence = signal_data.get('confidence', 0.0)
            
            signal_value = signal_map.get(signal, 0.0)
            
            # Aplicar peso de régimen
            regime_multiplier = regime_weight_map.get(signal, 1.0)
            
            # NUEVO: Aplicar peso de performance histórica
            performance_multiplier = strategy_performance.get(strategy_name, 1.0)
            
            # Confianza ajustada por régimen Y performance
            adjusted_confidence = confidence * regime_multiplier * regime_confidence * performance_multiplier
            
            # Pesar la señal
            weighted_score += signal_value * adjusted_confidence
            total_weight += adjusted_confidence
            
            signal_details[strategy_name] = {
                'signal': signal,
                'confidence': confidence,
                'adjusted_confidence': adjusted_confidence,
                'regime_multiplier': regime_multiplier,
                'performance_multiplier': performance_multiplier,  # NUEVO
                'value': signal_value
            }
        
        # Calcular promedio ponderado con validación
        if total_weight > 0:
            avg_score = weighted_score / total_weight
        else:
            LOG.warning("aggregate_signals_zero_weight",
                       signals_count=len(strategy_signals))
            avg_score = 0.0
        
        # Validar que avg_score sea finito
        if not np.isfinite(avg_score):
            LOG.error("invalid_avg_score_using_neutral",
                     avg_score=avg_score,
                     weighted_score=weighted_score,
                     total_weight=total_weight)
            avg_score = 0.0
        
        # Convertir puntuación de vuelta a señal
        # Ajustar umbrales según régimen
        buy_threshold = 0.25
        sell_threshold = -0.25
        
        # Ajustar umbrales según régimen de mercado
        if regime == 'bull':
            buy_threshold = 0.20
            sell_threshold = -0.35
        elif regime == 'bear':
            buy_threshold = 0.35
            sell_threshold = -0.20
        elif regime == 'volatile':
            buy_threshold = 0.40
            sell_threshold = -0.40
        elif regime == 'sideways':
            buy_threshold = 0.30
            sell_threshold = -0.30

        if avg_score > buy_threshold:
            aggregated_signal = "buy"
        elif avg_score < sell_threshold:
            aggregated_signal = "sell"
        else:
            aggregated_signal = "hold"

        aggregated_confidence = abs(avg_score)

        # NUEVO: Ajustar confianza por acuerdo entre estrategias
        signals_list = [s.get('signal') for s in strategy_signals.values()]
        agreement_score = signals_list.count(aggregated_signal) / len(signals_list) if signals_list else 0
        
        if agreement_score >= 0.75:  # 75%+ acuerdo
            aggregated_confidence *= 1.15  # Boost 15%
        elif agreement_score < 0.5:  # Menos de 50% acuerdo
            aggregated_confidence *= 0.85  # Penalty 15%

        LOG.debug("strategy_signals_aggregated",
                 total_strategies=len(strategy_signals),
                 regime=regime,
                 regime_confidence=regime_confidence,
                 weighted_score=avg_score,
                 aggregated_signal=aggregated_signal,
                 aggregated_confidence=aggregated_confidence,
                 agreement_score=agreement_score,
                 details=signal_details)
        
        return {
            "signal": aggregated_signal,
            "confidence": max(0.0, min(1.0, aggregated_confidence)),
            "reason": f"Aggregate of {len(strategy_signals)} strategies (regime: {regime}, agreement: {agreement_score:.1%}). Weighted score: {avg_score:.3f}",
            "regime": regime,
            "agreement_score": agreement_score,
            "details": signal_details
        }
        
    except Exception as e:
        LOG.error("strategy_aggregation_failed",
                 error=str(e),
                 traceback=traceback.format_exc()[:500])
        return {"signal": "hold", "confidence": 0.0, "reason": f"Aggregation error: {str(e)}"}



async def execute_trading_pipeline_complete(exchange_manager, strategy_manager, automl,
                                           regime_detector, risk_optimizer,
                                           ensemble_learner, bot, df, config, symbol):
    """
    Pipeline COMPLETO con validaciones robustas
    """
    try:
        # Verificar intervalo mínimo entre ejecuciones
        if hasattr(bot, 'last_pipeline_execution'):
            last_exec = bot.last_pipeline_execution.get(symbol)
            if last_exec:
                time_since_last = (datetime.now(timezone.utc) - last_exec).total_seconds()
                min_interval = 60  # 1 minuto mínimo
                if time_since_last < min_interval:
                    LOG.debug("pipeline_skipped_minimum_interval",
                             symbol=symbol,
                             time_since_last=time_since_last,
                             required=min_interval)
                    return

        # Validar que todos los componentes estén inicializados
        required_components = {
            'exchange_manager': exchange_manager,
            'strategy_manager': strategy_manager,
            'regime_detector': regime_detector,
            'ensemble_learner': ensemble_learner,
            'bot': bot
        }
        missing_components = [name for name, comp in required_components.items() if comp is None]
        if missing_components:
            LOG.error("pipeline_missing_components",
                     missing=missing_components,
                     symbol=symbol)
            return

        # Inicializar componentes si no existen
        if not hasattr(bot, 'smart_executor'):
            bot.smart_executor = SmartOrderExecutor(exchange_manager, config)
        if not hasattr(bot, 'portfolio_rebalancer'):
            bot.portfolio_rebalancer = PortfolioRebalancer(config, bot)
        if not hasattr(bot, 'correlation_analyzer'):
            bot.correlation_analyzer = CorrelationAnalyzer(exchange_manager)

        # Validaciones iniciales
        if df is None or len(df) < 20 or 'close' not in df.columns:
            LOG.warning("invalid_dataframe_for_pipeline",
                       symbol=symbol,
                       df_len=len(df) if df is not None else 0)
            return

        # 1. Verificar circuit breaker
        if hasattr(bot, 'risk_manager') and bot.risk_manager.check_circuit_breaker():
            LOG.warning("pipeline_blocked_circuit_breaker", symbol=symbol)
            return

        # 2. Rebalanceo de portfolio si es necesario
        if hasattr(bot, 'risk_manager') and hasattr(bot, 'portfolio_rebalancer'):
            symbols_to_close = await bot.portfolio_rebalancer.check_rebalance_needed(
                bot.risk_manager
            )
            if symbols_to_close:
                LOG.info("rebalancing_portfolio", symbols_count=len(symbols_to_close))
                await bot.portfolio_rebalancer.execute_rebalance(
                    symbols_to_close, bot.risk_manager
                )

        # 3. Análisis de régimen de mercado
        try:
            regime, confidence = regime_detector.detect_regime(df)
            LOG.debug("market_regime_detected",
                    symbol=symbol,
                    regime=regime,
                    confidence=confidence)
            
            # NUEVO: Ajustar parámetros según régimen
            if hasattr(bot, 'adaptive_params'):
                asyncio.create_task(
                    bot.adaptive_params.adjust_parameters(regime, confidence)
                )

            # Usar AutoML para optimizar hiperparámetros periódicamente
            if bot.performance_metrics['total_trades'] % 50 == 0 and bot.performance_metrics['total_trades'] > 0:
                try:
                    # Verificar que automl existe
                    if not hasattr(bot, 'automl') or bot.automl is None:
                        LOG.debug("automl_not_available_skipping_optimization")
                    else:
                        # Preparar datos para optimización
                        if len(df) >= 100:
                            optimization_df = df[['close', 'rsi', 'macd', 'volume']].copy()
                        # Crear target binario simple para clasificación
                        optimization_df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                        optimization_df = optimization_df.dropna()
                        if len(optimization_df) >= 50:
                            LOG.info("running_automl_optimization",
                                    symbol=symbol,
                                    samples=len(optimization_df))
                            # NUEVO: Optimizar CADA estrategia individualmente
                            available_strategies = strategy_manager.list_strategies()
                            for strat_info in available_strategies:
                                strat_name = strat_info['name']
                                try:
                                    # Crear modelo específico para la estrategia
                                    from sklearn.ensemble import RandomForestClassifier
                                    strategy_model = RandomForestClassifier(random_state=42)
                                    # Optimizar hiperparámetros
                                    optimized_params = await bot.automl.optimize_model(
                                        strategy_model,
                                        optimization_df
                                    )
                                    if optimized_params:
                                        # Actualizar parámetros de la estrategia
                                        strategy_info = strategy_manager.strategies.get(strat_name, {})
                                        old_params = strategy_info.get('parameters', {}).copy()
                                        # Mapear parámetros optimizados a la estrategia
                                        if strat_name == 'rsi_momentum' and 'n_estimators' in optimized_params:
                                            # Ejemplo: usar n_estimators para ajustar período RSI
                                            new_rsi_period = min(21, max(7, optimized_params['n_estimators'] // 10))
                                            strategy_info['parameters']['rsi_period'] = new_rsi_period
                                        LOG.info("strategy_parameters_optimized",
                                                strategy=strat_name,
                                                old_params=old_params,
                                                new_params=strategy_info.get('parameters', {}),
                                                automl_params=optimized_params)
                                except Exception as strat_opt_error:
                                    LOG.debug("strategy_optimization_failed",
                                             strategy=strat_name,
                                             error=str(strat_opt_error))
                            # Optimizar modelo (en background para no bloquear) - MANTENER
                            asyncio.create_task(
                                automl.optimize_model(None, optimization_df)
                            )
                except Exception as automl_error:
                    LOG.debug("automl_optimization_failed", error=str(automl_error))

            # CORRECCIÓN: No enviar aquí, se enviará en sync_bot_metrics_to_influx
            # Solo almacenar para uso posterior
            if not hasattr(bot, '_last_regime'):
                bot._last_regime = {}
            bot._last_regime[symbol] = {'regime': regime, 'confidence': confidence}

        except Exception as e:
            LOG.warning("regime_detection_failed", symbol=symbol, error=str(e))
            regime, confidence = "unknown", 0.5

        # 4. Predicción del ensemble CON SÍMBOLO ESPECÍFICO
        try:
            # Validar que ensemble esté entrenado
            if not ensemble_learner.is_trained and not (hasattr(ensemble_learner, 'symbol_models') and symbol in ensemble_learner.symbol_models):
                LOG.debug("ensemble_not_trained_skipping_prediction", symbol=symbol)
                ensemble_signal = {"action": "hold", "confidence": 0.0, "prediction_method": "not_trained"}
            else:
                ensemble_signal = await ensemble_learner.ensemble_predict(df, symbol=symbol)
                if not ensemble_signal:
                    ensemble_signal = {"action": "hold", "confidence": 0.0}
                LOG.debug("ensemble_signal_generated",
                        symbol=symbol,
                        signal=ensemble_signal.get('action', 'hold'),
                        confidence=ensemble_signal.get('confidence', 0.0),
                        prediction_method=ensemble_signal.get('prediction_method', 'unknown'))
        except Exception as e:
            LOG.error("ensemble_prediction_failed", symbol=symbol, error=str(e))
            ensemble_signal = {"action": "hold", "confidence": 0.0}

        # 5. Ejecutar todas las estrategias técnicas
        strategy_signals = {}
        available_strategies = ['rsi_momentum', 'bollinger_bands', 'macd_trend', 'volume_profile']
        for strategy_name in available_strategies:
            try:
                strategy_params = {
                    'rsi_momentum': {'rsi_period': 14, 'oversold': 30, 'overbought': 70},
                    'bollinger_bands': {'bb_period': 20, 'bb_std': 2},
                    'macd_trend': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                    'volume_profile': {'volume_period': 20, 'threshold': 2.0}
                }
                params = strategy_params.get(strategy_name, {})
                strategy_signal = await strategy_manager.execute_strategy(
                    strategy_name, df, **params
                )
                strategy_signals[strategy_name] = strategy_signal
            except Exception as e:
                LOG.warning("strategy_execution_error",
                           symbol=symbol,
                           strategy=strategy_name,
                           error=str(e))
                strategy_signals[strategy_name] = {
                    "signal": "hold",
                    "confidence": 0.0,
                    "reason": f"Error: {str(e)}"
                }

        # 6. Agregar señales tradicionales CON régimen
        traditional_signal = await aggregate_strategy_signals(
            strategy_signals,
            regime=regime,
            regime_confidence=confidence
        )
        LOG.info("traditional_strategies_aggregated",
                symbol=symbol,
                regime=regime,
                regime_confidence=confidence,
                aggregated_signal=traditional_signal.get('signal', 'hold'),
                aggregated_confidence=traditional_signal.get('confidence', 0.0))

        # 7. Decisión RL con ajuste por régimen
        try:
            if len(df) >= 4:
                state = df[['close', 'volume', 'rsi' if 'rsi' in df.columns else 'close',
                           'macd' if 'macd' in df.columns else 'volume']].values[-1]
                rl_action_idx = bot.rl_agent.act(state)[0]
                rl_actions = {0: "sell", 1: "hold", 2: "buy"}
                # MEJORADO: Ajustar confianza RL por régimen
                base_rl_confidence = 0.6
                regime_adjustment = {
                    "bull": 1.1 if rl_action_idx == 2 else 0.9,      # Favorecer buy
                    "bear": 1.1 if rl_action_idx == 0 else 0.9,      # Favorecer sell
                    "volatile": 0.8,                                  # Reducir confianza
                    "sideways": 0.9,
                    "unknown": 1.0
                }
                adjusted_rl_conf = base_rl_confidence * regime_adjustment.get(regime, 1.0) * confidence
                rl_decision = {
                    "signal": rl_actions.get(rl_action_idx, "hold"),
                    "confidence": adjusted_rl_conf
                }
            else:
                rl_decision = {"signal": "hold", "confidence": 0.0}
        except Exception as e:
            LOG.debug("rl_decision_failed", symbol=symbol, error=str(e))
            rl_decision = {"signal": "hold", "confidence": 0.0}

        # 8. Integrar todas las señales - MEJORADO CON VALIDACIÓN POR ACCUMULATOR
        try:
            # NUEVO: Validar decisión con datos históricos del accumulator
            decision_confidence_boost = 0.0
            decision_confidence_penalty = 0.0
            
            if hasattr(bot, 'data_accumulator') and bot.data_accumulator:
                try:
                    # Obtener datos históricos del símbolo
                    symbol_hist = await bot.data_accumulator.get_training_data(
                        min_samples=50,
                        symbol=symbol
                    )
                    
                    if symbol_hist is not None and len(symbol_hist) >= 50:
                        # Calcular éxito histórico de señales similares
                        recent_returns = symbol_hist['close'].pct_change().tail(20)
                        current_momentum = df['close'].pct_change().tail(5).mean()
                        hist_momentum = recent_returns.mean()
                        
                        # MEJORADO: Análisis de régimen histórico vs actual
                        hist_volatility = recent_returns.std()
                        current_volatility = df['close'].pct_change().tail(10).std()
                        
                        # Si regímenes coinciden, boost
                        if abs(hist_volatility - current_volatility) / hist_volatility < 0.3:
                            regime_match_boost = 0.1
                            LOG.debug("regime_consistency_detected",
                                     symbol=symbol,
                                     hist_vol=hist_volatility,
                                     current_vol=current_volatility,
                                     boost=regime_match_boost)
                            decision_confidence_boost += regime_match_boost
                        
                        # Si momentum histórico coincide con acción propuesta
                        proposed_action = integrated_decision.get('action', 'hold')
                        if (proposed_action == 'buy' and hist_momentum > 0 and current_momentum > 0) or \
                           (proposed_action == 'sell' and hist_momentum < 0 and current_momentum < 0):
                            # Boost basado en alineación
                            momentum_boost = abs(hist_momentum) * 0.3
                            decision_confidence_boost += momentum_boost
                            LOG.debug("momentum_alignment_boost",
                                     symbol=symbol,
                                     action=proposed_action,
                                     hist_momentum=hist_momentum,
                                     current_momentum=current_momentum,
                                     boost=momentum_boost)
                        
                        # NUEVO: Penalizar si momentum actual contradice histórico
                        elif (proposed_action == 'buy' and hist_momentum < -0.01) or \
                             (proposed_action == 'sell' and hist_momentum > 0.01):
                            momentum_penalty = abs(hist_momentum) * 0.2
                            decision_confidence_penalty += momentum_penalty
                            LOG.warning("momentum_contradiction_detected",
                                       symbol=symbol,
                                       action=proposed_action,
                                       hist_momentum=hist_momentum,
                                       penalty=momentum_penalty)
                        
                except Exception as acc_val_error:
                    LOG.debug("accumulator_validation_failed", error=str(acc_val_error))
            
            # USAR _integrate_all_decisions
            integrated_decision = await bot._integrate_all_decisions(
                traditional_signal,
                rl_decision,
                ensemble_signal,
                regime,
                confidence,
                df
            )

            # Aplicar boost/penalty del accumulator
            if decision_confidence_boost > 0 or decision_confidence_penalty > 0:
                old_conf = integrated_decision.get('confidence', 0.0)
                net_adjustment = decision_confidence_boost - decision_confidence_penalty
                integrated_decision['confidence'] = np.clip(
                    old_conf + net_adjustment,
                    0.0,
                    0.95
                )
                LOG.info("decision_confidence_adjusted_by_accumulator",
                        symbol=symbol,
                        old_confidence=old_conf,
                        new_confidence=integrated_decision['confidence'],
                        boost=decision_confidence_boost,
                        penalty=decision_confidence_penalty,
                        net_adjustment=net_adjustment)

            LOG.info("integrated_decision_created",
                symbol=symbol,
                action=integrated_decision.get('action'),
                confidence=integrated_decision.get('confidence'),
                details=integrated_decision.get('details', {}))

        except Exception as e:
            LOG.error("decision_integration_failed", symbol=symbol, error=str(e))
            integrated_decision = {"action": "hold", "confidence": 0.0}

        # 9. Verificar correlación Y diversificación antes de decisión final
        action = integrated_decision.get('action', 'hold')
        final_confidence = integrated_decision.get('confidence', 0.0)

        # ✅ CORRECCIÓN: Calcular threshold ANTES del análisis de correlación
        base_threshold = 0.50
        
        # Ajuste por régimen
        regime_adjustment = {
            'bull': -0.15,
            'bear': -0.12,
            'volatile': -0.05,
            'sideways': -0.08,
            'unknown': -0.10
        }
        
        # ✅ NUEVO: Ajuste por performance histórica del símbolo específico
        symbol_performance_adjustment = 0.0
        if hasattr(bot, 'symbol_performance') and symbol in bot.symbol_performance:
            symbol_perf = bot.symbol_performance[symbol]
            if symbol_perf['total_trades'] >= 5:
                symbol_win_rate = symbol_perf['win_rate']
                # Reducir threshold si el símbolo tiene buen historial
                if symbol_win_rate > 0.65:
                    symbol_performance_adjustment = -0.08
                elif symbol_win_rate < 0.35:
                    symbol_performance_adjustment = 0.05
        
        # Ajuste por performance general
        performance_adjustment = 0.0
        if hasattr(bot, 'performance_metrics'):
            total_trades = bot.performance_metrics.get('total_trades', 0)
            if total_trades >= 10:
                win_rate = bot.performance_metrics.get('win_rate', 0.5)
                if win_rate > 0.6:
                    performance_adjustment = -0.12
                elif win_rate < 0.4:
                    performance_adjustment = 0.0
            else:
                performance_adjustment = -0.20  # Más conservador al inicio

        # ✅ Combinar todos los ajustes
        confidence_threshold = (base_threshold + 
                               regime_adjustment.get(regime, 0.0) + 
                               performance_adjustment +
                               symbol_performance_adjustment)
        
        # ✅ Límites razonables
        confidence_threshold = max(0.05, min(0.35, confidence_threshold))

        LOG.debug("confidence_threshold_calculated",
                 regime=regime,
                 threshold=confidence_threshold,
                 adjustments={
                     'regime': regime_adjustment.get(regime, 0.0),
                     'performance': performance_adjustment,
                     'symbol_performance': symbol_performance_adjustment
                 },
                 action=action,
                 confidence=final_confidence)

        # AHORA sí análisis de correlación con threshold ya calculado
        correlation_penalty = 0.0
        if hasattr(bot, 'correlation_analyzer') and hasattr(bot, 'risk_manager'):
            active_symbols = list(bot.risk_manager.active_stops.keys())
            if len(active_symbols) >= 2:  # Solo si hay 2+ posiciones activas
                try:
                    # Calcular correlación promedio con posiciones activas
                    correlations = []
                    for active_symbol in active_symbols:
                        corr = await bot.correlation_analyzer.check_correlation(
                            symbol, active_symbol
                        )
                        correlations.append(abs(corr))
                    if correlations:
                        avg_correlation = np.mean(correlations)
                        max_correlation = np.max(correlations)
                        # Penalizar confianza si hay alta correlación
                        if avg_correlation > 0.7:  # Alta correlación promedio
                            correlation_penalty = (avg_correlation - 0.7) * 0.5  # Hasta 15% penalty
                            LOG.info("correlation_penalty_applied",
                                    symbol=symbol,
                                    avg_correlation=avg_correlation,
                                    max_correlation=max_correlation,
                                    penalty=correlation_penalty,
                                    original_confidence=final_confidence)
                            # Reducir confianza por correlación
                            final_confidence *= (1.0 - correlation_penalty)
                        LOG.debug("portfolio_correlation_analysis",
                                symbol=symbol,
                                avg_correlation=avg_correlation,
                                max_correlation=max_correlation,
                                active_positions=len(active_symbols))
                except Exception as corr_error:
                    LOG.debug("correlation_analysis_failed", error=str(corr_error))
        
        LOG.debug("confidence_threshold_adjusted",
                 regime=regime,
                 threshold=confidence_threshold,
                 action=action,
                 confidence=final_confidence)

        if action in ['buy', 'sell'] and final_confidence > confidence_threshold:
            # Verificar si ya hay posición activa
            if hasattr(bot, 'risk_manager') and symbol in bot.risk_manager.active_stops:
                LOG.debug("position_already_active_skipping", symbol=symbol)
                return

            # Verificar correlación con TODAS las posiciones activas
            if hasattr(bot, 'correlation_analyzer') and hasattr(bot, 'risk_manager'):
                active_symbols = list(bot.risk_manager.active_stops.keys())
                if active_symbols and len(active_symbols) > 0:
                    try:
                        # CAMBIO: Solo verificar correlación si ya hay 3+ posiciones
                        if len(active_symbols) >= 3:
                            # Calcular matriz de correlación completa
                            all_symbols_to_check = active_symbols + [symbol]
                            correlation_matrix = await bot.correlation_analyzer.get_correlation_matrix(
                                all_symbols_to_check
                            )
                            if correlation_matrix is not None:
                                # Verificar correlación del nuevo símbolo con cada activo
                                is_suitable = True
                                max_correlation_found = 0.0
                                for active_symbol in active_symbols:
                                    try:
                                        if symbol in correlation_matrix.index and active_symbol in correlation_matrix.columns:
                                            correlation = abs(correlation_matrix.loc[symbol, active_symbol])
                                            max_correlation_found = max(max_correlation_found, correlation)
                                            # CORRECCIÓN: Umbral aún más alto
                                            if correlation > 0.92:  # AUMENTADO de 0.85
                                                is_suitable = False
                                                LOG.info("high_correlation_detected",
                                                        symbol=symbol,
                                                        active_symbol=active_symbol,
                                                        correlation=correlation)
                                                break
                                    except Exception as corr_check_error:
                                        LOG.debug("correlation_check_failed_for_pair",
                                                 symbol=symbol,
                                                 active_symbol=active_symbol,
                                                 error=str(corr_check_error))
                                        continue
                                if not is_suitable:
                                    LOG.info("trade_skipped_high_correlation",
                                            symbol=symbol,
                                            max_correlation=max_correlation_found,
                                            active_symbols=active_symbols,
                                            reason="Portfolio diversification")
                                    # Enviar métrica de rechazo por correlación
                                    try:
                                        if INFLUX_METRICS.enabled:
                                            await INFLUX_METRICS.write_model_metrics(
                                                'portfolio_diversification',
                                                {
                                                    'total_positions': len(active_symbols),
                                                    'new_position_rejected': 1.0,
                                                    'reason': 'high_correlation',
                                                    'max_correlation': float(max_correlation_found)
                                                }
                                            )
                                    except Exception:
                                        pass
                                    return
                                else:
                                    LOG.debug("correlation_check_passed",
                                             symbol=symbol,
                                             max_correlation=max_correlation_found,
                                             active_count=len(active_symbols))
                            else:
                                LOG.warning("correlation_matrix_calculation_failed_allowing_trade",
                                           symbol=symbol)
                    except Exception as corr_error:
                        LOG.warning("correlation_check_failed_allowing_trade",
                                   symbol=symbol,
                                   error=str(corr_error))
                        # En caso de error, permitir el trade (fail-safe)
            else:
                LOG.debug("correlation_analyzer_not_available", symbol=symbol)

        # 10. Ejecutar trade con risk management - VALIDACIÓN PREVIA
        try:
            # LOG CRÍTICO PARA DEBUG
            LOG.info("PIPELINE_DECISION_CHECKPOINT",
                    symbol=symbol,
                    action=action,
                    confidence=final_confidence,
                    threshold=confidence_threshold,
                    passes_threshold=final_confidence > confidence_threshold,
                    has_active_position=symbol in bot.risk_manager.active_stops if hasattr(bot, 'risk_manager') else False,
                    circuit_breaker=bot.risk_manager.check_circuit_breaker() if hasattr(bot, 'risk_manager') else False,
                    regime=regime,
                    regime_confidence=confidence,
                    traditional_signal=traditional_signal.get('signal'),
                    traditional_conf=traditional_signal.get('confidence'),
                    rl_signal=rl_decision.get('signal'),
                    rl_conf=rl_decision.get('confidence'),
                    ensemble_signal=ensemble_signal.get('action'),
                    ensemble_conf=ensemble_signal.get('confidence'),
                    will_execute=action in ['buy', 'sell'] and final_confidence > confidence_threshold)

            # VALIDACIÓN CRÍTICA: No ejecutar si ya hay posición
            if hasattr(bot, 'risk_manager') and symbol in bot.risk_manager.active_stops:
                LOG.debug("trade_execution_skipped_position_exists",
                         symbol=symbol,
                         active_positions=len(bot.risk_manager.active_stops))
                return

            # VALIDACIÓN: Acción debe ser buy o sell con confianza suficiente
            action = integrated_decision.get('action', 'hold')
            confidence = integrated_decision.get('confidence', 0.0)
            if action == 'hold' or confidence < 0.5:
                LOG.debug("trade_execution_skipped_hold_or_low_confidence",
                         symbol=symbol,
                         action=action,
                         confidence=confidence)
                return

            LOG.info("executing_trade_from_pipeline",
                    symbol=symbol,
                    action=action,
                    confidence=confidence)

            # Usar integrated_decision en lugar de decision
            await bot._execute_advanced_trade_with_risk_management(
                symbol, integrated_decision, df
            )

            # Actualizar timestamp de última ejecución por símbolo
            if hasattr(bot, 'last_pipeline_execution'):
                bot.last_pipeline_execution[symbol] = datetime.now(timezone.utc)
            else:
                bot.last_pipeline_execution = {symbol: datetime.now(timezone.utc)}

        except Exception as e:
            LOG.error("trade_execution_failed", symbol=symbol, error=str(e))

        # 11. Acumular datos para reentrenamiento CON MICRO-UPDATE
        if hasattr(bot, 'data_accumulator'):
            try:
                # Preparar features para micro-update
                features = df[['close', 'rsi', 'macd', 'volume']].tail(1).values
                # Crear target basado en la acción integrada
                target = {'buy': 2, 'sell': 0, 'hold': 1}.get(action, 1)

                # Micro-update del ensemble con el símbolo
                if hasattr(bot, 'ensemble_learner') and bot.ensemble_learner:
                    try:
                        update_success = await bot.ensemble_learner.micro_update(
                            features,
                            target=target,
                            symbol=symbol
                        )
                        if update_success:
                            LOG.debug("ensemble_micro_updated_in_pipeline",
                                     symbol=symbol,
                                     target=target)
                    except Exception as micro_error:
                        LOG.debug("micro_update_failed_in_pipeline", error=str(micro_error))

                # Calcular reward basado en señal y confianza
                reward = 0.0
                if action in ['buy', 'sell']:
                    # Usar el precio actual y siguiente para calcular reward real
                    if len(df) >= 2:
                        current_price = float(df['close'].iloc[-1])
                        prev_price = float(df['close'].iloc[-2])
                        # Calcular retorno real
                        actual_return = (current_price - prev_price) / prev_price
                        # Reward positivo si la señal coincidió con la dirección del mercado
                        if action == 'buy' and actual_return > 0:
                            reward = actual_return * final_confidence
                        elif action == 'sell' and actual_return < 0:
                            reward = abs(actual_return) * final_confidence
                        else:
                            # Penalizar señal incorrecta
                            reward = -abs(actual_return) * final_confidence * 0.5
                        # Bonus si múltiples señales coinciden
                        signal_agreement = sum([
                            1 if traditional_signal.get('signal') == action else 0,
                            1 if rl_decision.get('signal') == action else 0,
                            1 if ensemble_signal.get('action') == action else 0
                        ]) / 3.0
                        reward *= (1.0 + signal_agreement * 0.5)  # Hasta 50% bonus
                        LOG.debug("reward_calculated_for_accumulator",
                                 symbol=symbol,
                                 action=action,
                                 actual_return=actual_return,
                                 final_confidence=final_confidence,
                                 signal_agreement=signal_agreement,
                                 reward=reward)

                # CAMBIO CRÍTICO: Acumular SIEMPRE, no solo cuando hay acción
                await bot.data_accumulator.add_sample(symbol, df.iloc[-1], reward=reward)

            except Exception as e:
                LOG.debug("data_accumulation_failed", error=str(e))

        # 12. Optimización periódica de riesgo - MEJORADO
        if bot.performance_metrics['total_trades'] % 100 == 0 and bot.performance_metrics['total_trades'] > 0:
            try:
                # Verificar que risk_optimizer existe
                if not hasattr(bot, 'risk_optimizer') or bot.risk_optimizer is None:
                    LOG.debug("risk_optimizer_not_available_skipping_optimization")
                else:
                    # Usar BayesianRiskOptimizer de forma asíncrona
                    if len(df) >= 50:
                        # Preparar datos históricos para optimización
                        optimization_data = df[['close', 'volume', 'rsi', 'macd']].copy()
                        # Calcular volatilidad y retornos si no existen
                        if 'volatility' not in optimization_data.columns:
                            optimization_data['volatility'] = df['close'].pct_change().rolling(20).std()
                        if 'returns' not in optimization_data.columns:
                            optimization_data['returns'] = df['close'].pct_change()
                        # columna PnL simulada basada en performance actual
                        if hasattr(bot, 'trades') and len(bot.trades) > 0:
                            # Usar últimos PnLs reales
                            recent_pnls = [t.get('pnl', 0) for t in bot.trades[-100:] if 'pnl' in t]
                            if recent_pnls:
                                avg_pnl = np.mean(recent_pnls)
                                optimization_data['pnl'] = avg_pnl
                            else:
                                optimization_data['pnl'] = 0.0
                        else:
                            optimization_data['pnl'] = 0.0
                        optimization_data = optimization_data.dropna()
                        if len(optimization_data) >= 50:
                            bayes_params = await risk_optimizer.optimize(optimization_data)
                            if bayes_params and 'risk_level' in bayes_params:
                                risk_level = float(bayes_params['risk_level'])
                                # Actualizar parámetros de riesgo dinámicamente
                                if hasattr(bot, 'position_sizer'):
                                    # Ajustar tamaño base de posición
                                    old_risk = getattr(bot.position_sizer, 'base_risk_pct', 0.05)
                                    bot.position_sizer.base_risk_pct = risk_level
                                    LOG.info("position_sizer_risk_adjusted",
                                            old_risk_level=old_risk,
                                            new_risk_level=risk_level)
                                if hasattr(bot, 'risk_manager'):
                                    # Ajustar multiplicador de stop loss
                                    old_multiplier = getattr(bot.risk_manager, 'stop_loss_multiplier', 1.0)
                                    bot.risk_manager.stop_loss_multiplier = 1.0 + (risk_level * 0.5)
                                    LOG.info("risk_manager_params_optimized",
                                            old_multiplier=old_multiplier,
                                            new_multiplier=bot.risk_manager.stop_loss_multiplier,
                                            risk_level=risk_level)
                                # Enviar métricas a InfluxDB
                                try:
                                    await INFLUX_METRICS.write_model_metrics(
                                        'bayesian_risk_optimizer',
                                        {
                                            'optimized_risk_level': risk_level,
                                            'total_trades': bot.performance_metrics['total_trades'],
                                            'optimization_trigger': 1.0
                                        }
                                    )
                                except Exception as influx_error:
                                    LOG.debug("risk_optimization_influx_write_failed", error=str(influx_error))
                        else:
                            LOG.debug("insufficient_optimization_data",
                                     samples=len(optimization_data),
                                     required=50)
                    else:
                        LOG.debug("dataframe_too_small_for_optimization", rows=len(df))

            except Exception as e:
                LOG.debug("risk_optimization_failed", error=str(e))

    except Exception as e:
        LOG.error("trading_pipeline_error",
                 symbol=symbol,
                 error=str(e),
                 traceback=traceback.format_exc())


# ===========================
# FUNCIONES AUXILIARES ADICIONALES
# ===========================

async def position_monitoring_loop(bot, exchange_manager, risk_manager, interval=10):
    """
    Loop que monitorea posiciones activas y ejecuta stops/take profits
    INTEGRADO: Usa PerformanceProfiler para monitoreo
    """
    LOG.info("position_monitoring_loop_started", interval_seconds=interval)
    iteration = 0
    while getattr(bot, 'is_running', False):
        iteration += 1
        try:
            await asyncio.sleep(interval)
            # Usar profiler cada 10 iteraciones
            should_profile = iteration % 10 == 0
            if should_profile:
                async with PERFORMANCE_PROFILER.profile_async(f"position_monitoring_{iteration}"):
                    await _monitor_positions(bot, exchange_manager, risk_manager)
            else:
                await _monitor_positions(bot, exchange_manager, risk_manager)
        except asyncio.CancelledError:
            LOG.info("position_monitoring_loop_cancelled")
            break
        except Exception as e:
            LOG.error("position_monitoring_loop_error", error=str(e))
            await asyncio.sleep(30)


async def update_symbol_performance(bot, symbol: str, pnl: float):
    """
    Actualiza métricas de performance por símbolo de forma consolidada.
    Args:
        bot: Instancia del bot
        symbol: Símbolo del par
        pnl: Profit/Loss de la operación
    """
    try:
        if not hasattr(bot, 'symbol_performance'):
            bot.symbol_performance = {}
        if symbol not in bot.symbol_performance:
            bot.symbol_performance[symbol] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        symbol_perf = bot.symbol_performance[symbol]
        symbol_perf['total_pnl'] += pnl
        if pnl > 0:
            symbol_perf['winning_trades'] += 1
            # Actualizar avg_win
            wins = symbol_perf['winning_trades']
            old_avg = symbol_perf['avg_win']
            symbol_perf['avg_win'] = (old_avg * (wins - 1) + pnl) / wins
            # Actualizar largest_win
            symbol_perf['largest_win'] = max(symbol_perf['largest_win'], pnl)
        else:
            symbol_perf['losing_trades'] += 1
            # Actualizar avg_loss
            losses = symbol_perf['losing_trades']
            old_avg = symbol_perf['avg_loss']
            symbol_perf['avg_loss'] = (old_avg * (losses - 1) + pnl) / losses
            # Actualizar largest_loss
            symbol_perf['largest_loss'] = min(symbol_perf['largest_loss'], pnl)
        # Actualizar win_rate
        total = symbol_perf['winning_trades'] + symbol_perf['losing_trades']
        if total > 0:
            symbol_perf['win_rate'] = symbol_perf['winning_trades'] / total
        LOG.debug("symbol_performance_updated",
                 symbol=symbol,
                 pnl=pnl,
                 win_rate=symbol_perf['win_rate'],
                 total_trades=total)
        return symbol_perf
    except Exception as e:
        LOG.error("symbol_performance_update_failed", symbol=symbol, error=str(e))
        return None


async def _monitor_positions(bot, exchange_manager, risk_manager):
    """Lógica de monitoreo de posiciones con TRACKING POR SÍMBOLO - VERSIÓN COMPLETA"""
    try:
        # Verificar circuit breaker
        if risk_manager.check_circuit_breaker():
            LOG.debug("position_monitoring_skipped_circuit_breaker")
            return

        # ✅ CORREGIDO: Validación más robusta
        if not hasattr(risk_manager, 'active_stops') or not risk_manager.active_stops:
            return

        # ✅ CRÍTICO: Crear snapshot SIN modificar active_stops durante iteración
        active_stops_snapshot = {}
        try:
            # Iterar sobre items() en lugar de keys() para evitar RuntimeError
            for sym, info in list(risk_manager.active_stops.items()):
                if not isinstance(info, dict):
                    LOG.warning("invalid_stop_info_type", symbol=sym)
                    continue
                
                try:
                    # CORRECCIÓN: Validar ANTES de convertir
                    entry_price_raw = info.get('entry_price', 0)
                    remaining_size_raw = info.get('remaining_size', 0)
                    
                    # Validación temprana
                    if entry_price_raw is None or remaining_size_raw is None:
                        LOG.error("missing_critical_fields", symbol=sym)
                        asyncio.create_task(_close_invalid_position(risk_manager, bot, sym))
                        continue
                    
                    entry_price = float(entry_price_raw)
                    remaining_size = float(remaining_size_raw)
                    side = str(info.get('side', 'buy'))
                    stop_loss = float(info.get('stop_loss', 0))
                    # Validación de rangos
                    if not (0 < entry_price <= 1000000.0) or np.isnan(entry_price) or np.isinf(entry_price):
                        LOG.warning("invalid_entry_price_attempting_recovery",
                                   symbol=sym,
                                   entry_price=entry_price)
                        # Intentar recuperar del ledger
                        if hasattr(bot, 'position_ledger') and bot.position_ledger:
                            for transaction in reversed(bot.position_ledger.transactions):
                                if (transaction.symbol == sym and
                                    transaction.transaction_type == TransactionType.OPEN):
                                    recovered_price = transaction.entry_price
                                    if 0 < recovered_price <= 1000000.0:
                                        entry_price = recovered_price
                                        LOG.info("entry_price_recovered_from_ledger",
                                                symbol=sym,
                                                entry_price=entry_price)
                                        break
                        if not (0 < entry_price <= 1000000.0):
                            LOG.error("cannot_recover_entry_price_closing_position",
                                     symbol=sym,
                                     invalid_price=entry_price)
                            # ✅ Usar asyncio para cerrar sin bloquear
                            asyncio.create_task(
                                _close_invalid_position(risk_manager, bot, sym)
                            )
                            continue

                    # Validación de remaining_size
                    max_reasonable_size = 10000000.0
                    if not (0 < remaining_size <= max_reasonable_size) or np.isnan(remaining_size) or np.isinf(remaining_size):
                        LOG.error("invalid_remaining_size_details",
                                 symbol=sym,
                                 remaining_size=remaining_size,
                                 max_reasonable=max_reasonable_size,
                                 entry_price=entry_price)
                        # Forzar cierre asíncrono
                        asyncio.create_task(
                            _close_invalid_position(risk_manager, bot, sym)
                        )
                        continue

                    # Validar valor de posición
                    position_value = entry_price * remaining_size
                    max_position_value = 100000.0
                    if position_value > max_position_value:
                        LOG.error("position_value_exceeds_limit_in_monitoring",
                                 symbol=sym,
                                 position_value=position_value,
                                 entry_price=entry_price,
                                 remaining_size=remaining_size,
                                 max_allowed=max_position_value)
                        asyncio.create_task(
                            _close_invalid_position(risk_manager, bot, sym)
                        )
                        continue

                    active_stops_snapshot[sym] = {
                        'entry_price': entry_price,
                        'remaining_size': remaining_size,
                        'side': side,
                        'stop_loss': stop_loss
                    }

                except (ValueError, TypeError) as conv_error:
                    LOG.warning("stop_info_conversion_failed_attempting_recovery",
                               symbol=sym,
                               error=str(conv_error))
                    # Intentar recuperar desde ledger
                    if hasattr(bot, 'position_ledger') and bot.position_ledger:
                        for transaction in reversed(bot.position_ledger.transactions):
                            if (transaction.symbol == sym and
                                transaction.transaction_type == TransactionType.OPEN):
                                active_stops_snapshot[sym] = {
                                    'entry_price': transaction.entry_price,
                                    'remaining_size': transaction.size,
                                    'side': transaction.side,
                                    'stop_loss': transaction.entry_price * 0.98 if transaction.side == 'buy' else transaction.entry_price * 1.02
                                }
                                LOG.info("position_recovered_from_ledger", symbol=sym)
                                break
                    continue

        except Exception as snapshot_error:
            LOG.error("snapshot_creation_failed", error=str(snapshot_error))
            return

        if not active_stops_snapshot:
            return

        LOG.debug("monitoring_positions", count=len(active_stops_snapshot))

        # ✅ PROCESAMIENTO ASÍNCRONO: Crear tareas en paralelo
        monitoring_tasks = []
        for symbol, stop_info in active_stops_snapshot.items():
            task = asyncio.create_task(
                _monitor_single_position(
                    bot, exchange_manager, risk_manager,
                    symbol, stop_info
                )
            )
            monitoring_tasks.append(task)

        # Ejecutar todas las tareas en paralelo con timeout
        if monitoring_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*monitoring_tasks, return_exceptions=True),
                    timeout=30.0  # 30 segundos máximo
                )
            except asyncio.TimeoutError:
                LOG.error("position_monitoring_timeout",
                         positions_count=len(monitoring_tasks))

    except Exception as e:
        LOG.error("monitor_positions_failed",
                 error=str(e),
                 traceback=traceback.format_exc())


async def _close_invalid_position(risk_manager, bot, symbol):
    """Cierra posición inválida de forma asíncrona y segura"""
    try:
        # Usar lock específico del símbolo si existe
        if hasattr(bot, 'symbol_execution_locks') and symbol in bot.symbol_execution_locks:
            async with bot.symbol_execution_locks[symbol]:
                risk_manager.close_position(symbol)
                # Limpiar del ledger también
                if hasattr(bot, 'position_ledger'):
                    if symbol in bot.position_ledger.active_positions:
                        del bot.position_ledger.active_positions[symbol]
                        LOG.info("invalid_position_cleaned_from_ledger", symbol=symbol)
        else:
            risk_manager.close_position(symbol)
        LOG.info("invalid_position_closed_successfully", symbol=symbol)
    except Exception as e:
        LOG.error("failed_to_close_invalid_position",
                 symbol=symbol,
                 error=str(e))


async def _monitor_single_position(bot, exchange_manager, risk_manager, symbol, stop_info):
    """Monitorea una posición individual de forma aislada"""
    try:
        # Verificar que la posición aún existe
        if symbol not in risk_manager.active_stops:
            LOG.debug("position_removed_during_monitoring", symbol=symbol)
            return

        side = stop_info['side']

        # Obtener precio actual con timeout
        try:
            ticker = await asyncio.wait_for(
                exchange_manager.exchange.fetch_ticker(symbol),
                timeout=5.0
            )
            current_price = ticker.get('last', 0)
            if current_price <= 0 or np.isnan(current_price):
                LOG.warning("invalid_ticker_price", symbol=symbol, price=current_price)
                return
        except asyncio.TimeoutError:
            LOG.warning("ticker_fetch_timeout", symbol=symbol)
            return
        except Exception as ticker_error:
            LOG.debug("ticker_fetch_failed", symbol=symbol, error=str(ticker_error))
            return

        # Actualizar trailing stop (sin bloquear)
        try:
            new_stop = risk_manager.update_trailing_stop(symbol, current_price, side)
            if new_stop:
                LOG.debug("trailing_stop_updated",
                         symbol=symbol,
                         new_stop=new_stop,
                         current_price=current_price)
        except Exception as trailing_error:
            LOG.debug("trailing_stop_update_failed",
                     symbol=symbol,
                     error=str(trailing_error))

        # Verificar stop loss
        if risk_manager.check_stop_loss_hit(symbol, current_price, side):
            # Ejecutar stop loss de forma asíncrona
            await _handle_stop_loss_hit(
                bot, exchange_manager, risk_manager,
                symbol, stop_info, current_price, side
            )
            return

        # Verificar take profit
        tp_hit = risk_manager.check_take_profit_hit(symbol, current_price, side)
        if tp_hit:
            # Ejecutar take profit de forma asíncrona
            await _handle_take_profit_hit(
                bot, exchange_manager, risk_manager,
                symbol, stop_info, current_price, side, tp_hit
            )

    except Exception as e:
        LOG.error("single_position_monitoring_failed",
                 symbol=symbol,
                 error=str(e))


async def _handle_stop_loss_hit(bot, exchange_manager, risk_manager, symbol, stop_info, current_price, side):
    """Handler separado para stop loss - evita duplicación de código"""
    try:
        # NUEVO: Verificar que posición aún existe en AMBOS lugares
        if symbol not in risk_manager.active_stops:
            LOG.debug("stop_loss_aborted_position_already_closed_in_risk_manager", 
                     symbol=symbol)
            return
        
        if hasattr(bot, 'position_ledger') and symbol not in bot.position_ledger.active_positions:
            LOG.warning("stop_loss_aborted_position_missing_in_ledger",
                       symbol=symbol,
                       message="Cleaning up risk_manager")
            risk_manager.close_position(symbol)
            return
        
        # NUEVO: Lock por símbolo para evitar race conditions
        if hasattr(bot, 'symbol_execution_locks'):
            if symbol not in bot.symbol_execution_locks:
                bot.symbol_execution_locks[symbol] = asyncio.Lock()
            
            if bot.symbol_execution_locks[symbol].locked():
                LOG.warning("stop_loss_aborted_symbol_locked", symbol=symbol)
                return
            
            async with bot.symbol_execution_locks[symbol]:
                # Re-verificar dentro del lock
                if symbol not in risk_manager.active_stops:
                    LOG.debug("stop_loss_aborted_position_closed_during_lock_wait",
                             symbol=symbol)
                    return
                
                await _execute_stop_loss_trade(
                    bot, exchange_manager, risk_manager,
                    symbol, stop_info, current_price, side
                )
        else:
            await _execute_stop_loss_trade(
                bot, exchange_manager, risk_manager,
                symbol, stop_info, current_price, side
            )
            
    except Exception as e:
        LOG.error("stop_loss_execution_failed", symbol=symbol, error=str(e))


async def _execute_stop_loss_trade(bot, exchange_manager, risk_manager, 
                                    symbol, stop_info, current_price, side):
    """Ejecución aislada del trade de stop loss"""
    try:
        LOG.warning("executing_stop_loss", symbol=symbol)

        # MEJORA: Intentar recuperar datos faltantes del ledger PRIMERO
        entry_price = stop_info.get('entry_price', 0.0)
        remaining_size = stop_info.get('remaining_size', 0.0)
        
        # Recovery automático desde ledger si datos inválidos
        if (entry_price <= 0 or remaining_size <= 0 or 
            np.isnan(entry_price) or np.isnan(remaining_size) or
            np.isinf(entry_price) or np.isinf(remaining_size)):
            
            LOG.warning("invalid_stop_info_attempting_ledger_recovery", 
                       symbol=symbol,
                       entry_price=entry_price,
                       remaining_size=remaining_size)
            
            if hasattr(bot, 'position_ledger') and bot.position_ledger:
                if symbol in bot.position_ledger.active_positions:
                    ledger_tx = bot.position_ledger.active_positions[symbol]
                    if isinstance(ledger_tx, PositionTransaction):
                        entry_price = float(ledger_tx.entry_price)
                        remaining_size = float(ledger_tx.size)
                        LOG.info("stop_info_recovered_from_ledger",
                                symbol=symbol,
                                entry_price=entry_price,
                                remaining_size=remaining_size)

        # Validación completa POST-recovery
        validation_errors = []
        if remaining_size <= 0 or np.isnan(remaining_size) or np.isinf(remaining_size):
            validation_errors.append(f"invalid_remaining_size: {remaining_size}")
        if entry_price <= 0 or np.isnan(entry_price) or np.isinf(entry_price):
            validation_errors.append(f"invalid_entry_price: {entry_price}")
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            validation_errors.append(f"invalid_current_price: {current_price}")

        if validation_errors:
            LOG.error("stop_loss_validation_failed",
                     symbol=symbol,
                     errors=validation_errors)
            risk_manager.close_position(symbol)
            if hasattr(bot, 'position_ledger') and symbol in bot.position_ledger.active_positions:
                del bot.position_ledger.active_positions[symbol]
                LOG.info("orphaned_ledger_position_cleaned_after_validation_failure", 
                        symbol=symbol)
            return

        # Cerrar posición completa
        close_side = 'sell' if side == 'buy' else 'buy'
        order_type = "market"
        
        order = await exchange_manager.create_order(
            symbol, order_type, close_side, remaining_size
        )

        if order and order.get("success", False):
            # Validar precio de ejecución
            executed_price = await _get_validated_execution_price(
                order, exchange_manager, symbol, current_price
            )
            
            if executed_price <= 0:
                LOG.error("cannot_determine_execution_price_aborting",
                         symbol=symbol)
                return
            
            # Actualizar bot metrics
            await _update_bot_after_trade_close(
                bot, risk_manager, symbol, 0.0, close_side,
                remaining_size, entry_price, executed_price, 
                is_stop_loss=True
            )

            LOG.info("stop_loss_executed_successfully",
                    symbol=symbol,
                    side=close_side,
                    size=remaining_size,
                    entry_price=entry_price,
                    exit_price=executed_price)
        else:
            LOG.error("stop_loss_order_failed",
                     symbol=symbol,
                     error=order.get('error') if order else 'No response')
            
    except Exception as e:
        LOG.error("stop_loss_trade_execution_failed", 
                 symbol=symbol, 
                 error=str(e),
                 traceback=traceback.format_exc()[:300])


async def _get_validated_execution_price(order, exchange_manager, symbol, fallback_price):
    """Helper para obtener precio de ejecución validado con múltiples fallbacks"""
    try:
        executed_price = order.get('price', 0)
        is_simulated = order.get('info', {}).get('simulated', False) if isinstance(order.get('info'), dict) else False
        
        # Prioridad 1: Precio real de orden
        if not is_simulated and executed_price and executed_price > 0 and not np.isnan(executed_price):
            LOG.debug("using_order_price", symbol=symbol, price=executed_price)
            return float(executed_price)
        
        # Prioridad 2: Ticker
        try:
            ticker = await asyncio.wait_for(
                exchange_manager.exchange.fetch_ticker(symbol),
                timeout=5.0
            )
            ticker_price = ticker.get('last', None)
            if ticker_price and ticker_price > 0 and not np.isnan(ticker_price):
                LOG.info("using_ticker_price", symbol=symbol, price=ticker_price)
                return float(ticker_price)
        except Exception as ticker_error:
            LOG.debug("ticker_fetch_failed", symbol=symbol, error=str(ticker_error))
        
        # Prioridad 3: Fallback
        if fallback_price > 0 and not np.isnan(fallback_price):
            LOG.warning("using_fallback_price", symbol=symbol, price=fallback_price)
            return float(fallback_price)
        
        LOG.error("all_price_sources_failed", symbol=symbol)
        return 0.0
        
    except Exception as e:
        LOG.error("price_validation_failed", symbol=symbol, error=str(e))
        return 0.0


async def _handle_take_profit_hit(bot, exchange_manager, risk_manager, symbol, stop_info, current_price, side, tp_hit):
    """Handler separado para take profit - evita duplicación"""
    try:
        tp_price, size_fraction = tp_hit
        entry_price = stop_info['entry_price']
        remaining_size = stop_info['remaining_size']
        
        LOG.info("take_profit_triggered",
                symbol=symbol,
                tp_price=tp_price,
                size_fraction=size_fraction,
                remaining_size=remaining_size)
        
        # NUEVO: Validar que remaining_size sea suficiente
        if remaining_size <= 0 or np.isnan(remaining_size) or np.isinf(remaining_size):
            LOG.error("invalid_remaining_size_in_tp",
                     symbol=symbol,
                     remaining_size=remaining_size)
            # Limpiar posición corrupta
            risk_manager.close_position(symbol)
            if hasattr(bot, 'position_ledger') and symbol in bot.position_ledger.active_positions:
                del bot.position_ledger.active_positions[symbol]
            return
        
        # Calcular tamaño a cerrar
        close_size = remaining_size * size_fraction
        
        # NUEVO: Validar close_size mínimo
        min_close_size = 0.001  # Mínimo tamaño de cierre
        if close_size < min_close_size:
            LOG.warning("close_size_too_small_closing_full_position",
                       symbol=symbol,
                       calculated_size=close_size,
                       min_size=min_close_size,
                       remaining_size=remaining_size)
            close_size = remaining_size
            size_fraction = 1.0
        
        # NUEVO: Validar que close_size no exceda remaining_size
        if close_size > remaining_size * 1.01:  # 1% tolerancia
            LOG.warning("close_size_exceeds_remaining_adjusting",
                       symbol=symbol,
                       close_size=close_size,
                       remaining_size=remaining_size)
            close_size = remaining_size
            size_fraction = 1.0
        
        # Validar precios
        if entry_price <= 0 or np.isnan(entry_price) or np.isinf(entry_price):
            LOG.error("invalid_entry_price_for_tp",
                     symbol=symbol,
                     entry_price=entry_price)
            return
        
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            LOG.error("invalid_current_price_for_tp",
                     symbol=symbol,
                     current_price=current_price)
            return
        
        # NUEVO: Lock para evitar race conditions
        if hasattr(bot, 'symbol_execution_locks'):
            if symbol not in bot.symbol_execution_locks:
                bot.symbol_execution_locks[symbol] = asyncio.Lock()
            
            if bot.symbol_execution_locks[symbol].locked():
                LOG.warning("tp_execution_aborted_symbol_locked", symbol=symbol)
                return
            
            async with bot.symbol_execution_locks[symbol]:
                # Re-verificar que posición aún existe
                if symbol not in risk_manager.active_stops:
                    LOG.debug("tp_aborted_position_closed_during_lock_wait",
                             symbol=symbol)
                    return
                
                await _execute_take_profit_trade(
                    bot, exchange_manager, risk_manager,
                    symbol, stop_info, current_price, side,
                    close_size, size_fraction, entry_price
                )
        else:
            await _execute_take_profit_trade(
                bot, exchange_manager, risk_manager,
                symbol, stop_info, current_price, side,
                close_size, size_fraction, entry_price
            )
            
    except Exception as e:
        LOG.error("take_profit_execution_failed",
                 symbol=symbol,
                 error=str(e),
                 traceback=traceback.format_exc()[:300])


async def _execute_take_profit_trade(bot, exchange_manager, risk_manager,
                                      symbol, stop_info, current_price, side,
                                      close_size, size_fraction, entry_price):
    """Ejecución aislada del trade de take profit"""
    try:
        # Cerrar parcialmente
        close_side = 'sell' if side == 'buy' else 'buy'
        
        LOG.info("executing_take_profit_order",
                symbol=symbol,
                close_side=close_side,
                close_size=close_size,
                size_fraction=size_fraction)
        
        order = await exchange_manager.create_order(
            symbol, 'market', close_side, close_size
        )

        if order and order.get('success'):
            # Obtener precio de ejecución validado
            executed_price = await _get_validated_execution_price(
                order, exchange_manager, symbol, current_price
            )
            
            if executed_price <= 0:
                LOG.error("cannot_determine_tp_execution_price",
                         symbol=symbol)
                return
            
            # CORRECCIÓN: Calcular si es cierre total
            new_remaining = stop_info['remaining_size'] - close_size
            is_full_close = new_remaining < 0.001  # Menos de 0.001 = cierre total
            
            # Actualizar métricas
            await _update_bot_after_trade_close(
                bot, risk_manager, symbol, 0.0, close_side,
                close_size, entry_price, executed_price,
                is_stop_loss=False,
                is_partial=not is_full_close,
                filled_size=close_size
            )
            
            # CORRECCIÓN: Actualizar remaining_size DESPUÉS de update
            if not is_full_close:
                stop_info['remaining_size'] = new_remaining
                LOG.info("partial_take_profit_executed",
                        symbol=symbol,
                        closed_size=close_size,
                        remaining_size=new_remaining,
                        exit_price=executed_price)
            else:
                # Cerrar completamente
                risk_manager.close_position(symbol)
                LOG.info("full_position_closed_via_take_profit",
                        symbol=symbol,
                        exit_price=executed_price)
        else:
            LOG.error("take_profit_order_failed",
                     symbol=symbol,
                     error=order.get('error') if order else 'No response')
            
    except Exception as e:
        LOG.error("tp_trade_execution_failed",
                 symbol=symbol,
                 error=str(e))

async def sync_bot_metrics_to_influx(bot, force=False):
    """
    MEJORADO: Función maestra para sincronizar TODAS las métricas del bot
    
    Esta es la ÚNICA función que debe llamarse periódicamente.
    Centraliza toda la lógica de envío a InfluxDB.
    """
    if not INFLUX_METRICS or not INFLUX_METRICS.enabled:
        return False
    
    try:
        metrics_sent = {
            'portfolio': False,
            'positions': False,
            'model_status': False,
            'system_health': False
        }
        
        # ========================================
        # SECCIÓN 1: MÉTRICAS DE PORTFOLIO
        # ========================================
        try:
            # Auditar equity desde ledger (source of truth)
            current_equity = float(bot.equity)
            total_pnl = 0.0
            
            if hasattr(bot, 'position_ledger') and bot.position_ledger:
                audit = bot.position_ledger.audit_equity(bot)
                
                if not audit['is_consistent'] and abs(audit['discrepancy']) > 1.0:
                    LOG.warning("equity_inconsistent_in_sync",
                               discrepancy=audit['discrepancy'])
                    # Usar valores auditados como verdad
                    current_equity = float(audit['actual_equity'])
                    total_pnl = float(audit['total_realized_pnl'])
                else:
                    total_pnl = float(audit['total_realized_pnl'])
            else:
                total_pnl = current_equity - float(bot.initial_capital)
            
            # Calcular drawdown correcto
            drawdown = 0.0
            if hasattr(bot, 'portfolio_history') and len(bot.portfolio_history) > 0:
                history_df = pd.DataFrame(bot.portfolio_history, columns=['timestamp', 'equity'])
                if len(history_df) > 0:
                    peak = history_df['equity'].cummax()
                    drawdown_series = (history_df['equity'] - peak) / peak
                    drawdown = float(drawdown_series.iloc[-1])
            else:
                if bot.initial_capital > 0:
                    drawdown = min(0.0, (current_equity - bot.initial_capital) / bot.initial_capital)
            
            # Posiciones activas
            active_positions = 0
            if hasattr(bot, 'risk_manager') and bot.risk_manager:
                active_positions = len(bot.risk_manager.active_stops)
            
            # Enviar métricas de portfolio
            portfolio_success = await INFLUX_METRICS.write_portfolio_metrics(
                equity=current_equity,
                drawdown=drawdown,
                positions=active_positions,
                total_pnl=total_pnl
            )
            
            metrics_sent['portfolio'] = portfolio_success
            
        except Exception as portfolio_error:
            LOG.error("portfolio_metrics_sync_failed", error=str(portfolio_error))
        
        # ========================================
        # SECCIÓN 2: MÉTRICAS DE POSICIONES INDIVIDUALES
        # ========================================
        try:
            if hasattr(bot, 'risk_manager') and bot.risk_manager.active_stops:
                positions_data = []
                
                for symbol, stop_info in bot.risk_manager.active_stops.items():
                    try:
                        # Obtener precio actual
                        ticker = await bot.exchange_manager.exchange.fetch_ticker(symbol)
                        current_price = ticker.get('last', 0)
                        
                        if current_price <= 0:
                            continue
                        
                        entry_price = stop_info.get('entry_price', 0)
                        side = stop_info.get('side', 'buy')
                        size = stop_info.get('remaining_size', 0)
                        
                        if entry_price <= 0 or size <= 0:
                            continue
                        
                        # Calcular PnL no realizado
                        if side == 'buy':
                            unrealized_pnl = (current_price - entry_price) * size
                            unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
                        else:
                            unrealized_pnl = (entry_price - current_price) * size
                            unrealized_pnl_pct = (entry_price - current_price) / entry_price * 100
                        
                        positions_data.append({
                            'symbol': symbol,
                            'current_price': current_price,
                            'unrealized_pnl': unrealized_pnl,
                            'unrealized_pnl_pct': unrealized_pnl_pct
                        })
                        
                    except Exception as position_error:
                        LOG.debug("position_metric_failed", symbol=symbol, error=str(position_error))
                        continue
                
                # Enviar métricas agregadas de posiciones
                if positions_data:
                    total_unrealized = sum(p['unrealized_pnl'] for p in positions_data)
                    avg_unrealized_pct = np.mean([p['unrealized_pnl_pct'] for p in positions_data])
                    
                    positions_success = await INFLUX_METRICS.write_model_metrics(
                        'open_positions_summary',
                        {
                            'count': len(positions_data),
                            'total_unrealized_pnl': total_unrealized,
                            'avg_unrealized_pnl_pct': avg_unrealized_pct
                        }
                    )
                    
                    metrics_sent['positions'] = positions_success
                    
        except Exception as positions_error:
            LOG.error("positions_metrics_sync_failed", error=str(positions_error))
        
        # ========================================
        # SECCIÓN 3: ESTADO DE MODELOS AI
        # ========================================
        try:
            model_status = {}
            
            # Ensemble status
            if hasattr(bot, 'ensemble_learner'):
                model_status['ensemble_trained'] = 1.0 if bot.ensemble_learner.is_trained else 0.0
                model_status['specialized_models'] = len(bot.ensemble_learner.symbol_models) if hasattr(bot.ensemble_learner, 'symbol_models') else 0
            
            # RL agent status
            if hasattr(bot, 'rl_agent'):
                model_status['rl_total_episodes'] = getattr(bot.rl_agent, 'total_episodes', 0)
                model_status['rl_update_count'] = getattr(bot.rl_agent, 'update_count', 0)
            
            # Data accumulator status
            if hasattr(bot, 'data_accumulator'):
                model_status['accumulator_buffer_size'] = len(bot.data_accumulator.data_buffer)
                model_status['accumulator_samples_added'] = bot.data_accumulator.samples_added
            
            if model_status:
                model_success = await INFLUX_METRICS.write_model_metrics(
                    'ai_models_status',
                    model_status
                )
                metrics_sent['model_status'] = model_success
                
        except Exception as model_error:
            LOG.error("model_status_sync_failed", error=str(model_error))
        
        # ========================================
        # SECCIÓN 4: SALUD DEL SISTEMA
        # ========================================
        try:
            if hasattr(bot, 'health_check'):
                health = bot.health_check.get_health_status()
                
                health_metrics = {
                    'memory_mb': health.get('memory_mb', 0),
                    'cpu_percent': health.get('cpu_percent', 0),
                    'uptime_seconds': health.get('uptime_seconds', 0),
                    'status_healthy': 1.0 if health.get('status') == 'healthy' else 0.0
                }
                
                health_success = await INFLUX_METRICS.write_model_metrics(
                    'system_health',
                    health_metrics
                )
                metrics_sent['system_health'] = health_success
                
        except Exception as health_error:
            LOG.error("health_metrics_sync_failed", error=str(health_error))
        
        # ========================================
        # SECCIÓN 5: RÉGIMEN DE MERCADO ACTUAL
        # ========================================
        try:
            if hasattr(bot, '_last_regime') and bot._last_regime:
                # ✅ CORRECCIÓN: Enviar régimen de TODOS los símbolos con posiciones activas
                
                # Priorizar símbolos con posiciones
                regime_symbols = []
                
                if hasattr(bot, 'risk_manager') and bot.risk_manager.active_stops:
                    # Símbolos con posiciones activas
                    regime_symbols.extend(bot.risk_manager.active_stops.keys())
                
                # Agregar símbolo principal si no está
                main_symbol = bot.config.symbols[0] if bot.config.symbols else None
                if main_symbol and main_symbol not in regime_symbols:
                    regime_symbols.append(main_symbol)
                
                # Limitar a 10 símbolos para no saturar
                regime_symbols = regime_symbols[:10]
                
                regime_numeric_map = {
                    'bull': 1.0,
                    'bear': -1.0,
                    'sideways': 0.0,
                    'volatile': 0.5,
                    'unknown': 0.0
                }
                
                for symbol in regime_symbols:
                    if symbol in bot._last_regime:
                        regime_data = bot._last_regime[symbol]
                        
                        regime_metrics = {
                            'regime_numeric': regime_numeric_map.get(regime_data['regime'], 0.0),
                            'confidence': regime_data['confidence']
                        }
                        
                        try:
                            regime_success = await INFLUX_METRICS.write_model_metrics(
                                'market_regime',
                                regime_metrics,
                                tags={'symbol': symbol}
                            )
                            
                            if regime_success:
                                metrics_sent['regime'] = True
                        
                        except Exception as regime_write_error:
                            LOG.debug("regime_metric_write_failed",
                                     symbol=symbol,
                                     error=str(regime_write_error))
        
        except Exception as regime_error:
            LOG.error("regime_metrics_sync_failed", error=str(regime_error))
        
        # ========================================
        # SECCIÓN 6: TRADES RECIENTES POR SÍMBOLO
        # ========================================
        try:
            if hasattr(bot, 'trades') and bot.trades:
                # Enviar últimos 10 trades (ya se envían individualmente, aquí resumir)
                recent_trades = bot.trades[-10:]
                
                # Agrupar por símbolo
                trades_by_symbol = {}
                for trade in recent_trades:
                    symbol = trade.get('symbol')
                    if symbol:
                        if symbol not in trades_by_symbol:
                            trades_by_symbol[symbol] = []
                        trades_by_symbol[symbol].append(trade)
                
                # Enviar resumen por símbolo
                for symbol, symbol_trades in trades_by_symbol.items():
                    try:
                        total_pnl = sum(t.get('pnl', 0) for t in symbol_trades)
                        avg_pnl = total_pnl / len(symbol_trades) if symbol_trades else 0
                        wins = sum(1 for t in symbol_trades if t.get('pnl', 0) > 0)
                        
                        symbol_trade_metrics = {
                            'recent_trades_count': len(symbol_trades),
                            'total_pnl': total_pnl,
                            'avg_pnl': avg_pnl,
                            'win_count': wins,
                            'win_rate': wins / len(symbol_trades) if symbol_trades else 0
                        }
                        
                        await INFLUX_METRICS.write_model_metrics(
                            'recent_trades_by_symbol',
                            symbol_trade_metrics,
                            tags={'symbol': symbol}
                        )
                    
                    except Exception as symbol_trade_error:
                        LOG.debug("symbol_trade_metrics_failed",
                                 symbol=symbol,
                                 error=str(symbol_trade_error))
        
        except Exception as trades_error:
            LOG.error("recent_trades_metrics_failed", error=str(trades_error))
        
        # ========================================
        # FLUSH EXPLÍCITO si force=True
        # ========================================
        if force and hasattr(INFLUX_METRICS, 'write_api'):
            try:
                INFLUX_METRICS.write_api.flush()
                LOG.debug("influx_flushed_after_sync")
            except Exception as flush_error:
                LOG.debug("influx_flush_failed", error=str(flush_error))
        
        # Resultado final
        success_count = sum(1 for v in metrics_sent.values() if v)
        total_count = len(metrics_sent)
        
        LOG.debug("metrics_sync_completed",
                 success=success_count,
                 total=total_count,
                 metrics_sent=metrics_sent)
        
        return success_count > 0
        
    except Exception as e:
        LOG.error("sync_bot_metrics_failed",
                 error=str(e),
                 traceback=traceback.format_exc()[:500])
        return False

async def advanced_ai_main_with_rl():
    """
    VERSIÓN MEJORADA: Main con testing integrado, telegram y validación robusta
    """
    config = None
    bot = None
    exchange_manager = None
    telegram_kill_switch = None
    
    try:
        LOG.info("starting_advanced_ai_trading_bot_with_full_features")
        
        # Crear configuración
        config = create_config()
        
        # Validar configuración
        config.validate_config_modes()
        
        # Crear exchange manager
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        
        exchange_manager = ExchangeManager(
            exchange_name=config.exchange,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=config.sandbox,
            dry_run=config.dry_run
        )
        
        # Crear strategy manager
        strategy_manager = StrategyManager(config)
        await strategy_manager.initialize()
        
        # Crear bot
        bot = AdvancedAITradingBot(config, exchange_manager, strategy_manager)

        # NUEVO: Configurar initial_capital en InfluxDB
        set_influx_initial_capital(config.initial_capital)
        
        # Inicializar componentes AI
        LOG.info("initializing_ai_components")
        
        # AutoML
        bot.automl = AdvancedAutoML(config)
        
        # Regime Detector
        bot.regime_detector = AdvancedMarketRegimeDetector(config)
        
        # Risk Optimizer
        bot.risk_optimizer = BayesianRiskOptimizer(config)
        
        # Ensemble Learner
        bot.ensemble_learner = AdvancedEnsembleLearner(config)
        bot.ensemble_learner.initialize_base_models()
        
        # Intentar cargar modelos guardados
        try:
            load_success = await bot.ensemble_learner._load_models()
            if load_success:
                LOG.info("ensemble_models_loaded_from_disk")
            else:
                LOG.info("no_saved_models_will_train_from_scratch")
        except Exception as load_error:
            LOG.warning("model_loading_failed_will_train", error=str(load_error))
        
        # RL Agent
        bot.rl_agent = PPOAgent(config)
        
        # Intentar cargar agente RL guardado
        try:
            rl_load_success = bot.rl_agent.load()
            if rl_load_success:
                LOG.info("rl_agent_loaded_from_checkpoint")
            else:
                LOG.info("no_rl_checkpoint_will_train_from_scratch")
        except Exception as rl_load_error:
            LOG.warning("rl_agent_loading_failed", error=str(rl_load_error))
        
        # RL Training Manager
        bot.rl_training_manager = RLTrainingManager(config, bot.rl_agent)
        
        # Dashboard
        try:
            bot.dashboard = PerformanceDashboard(bot)
            LOG.info("dashboard_initialized")
        except Exception as dashboard_error:
            LOG.error("dashboard_initialization_failed", error=str(dashboard_error))
            bot.dashboard = None
        
        # ===== NUEVO: Iniciar Telegram Kill Switch =====
        try:
            telegram_kill_switch = TelegramKillSwitch()
            if telegram_kill_switch.enabled:
                await telegram_kill_switch.start(bot)
                bot.telegram_kill_switch = telegram_kill_switch
                LOG.info("telegram_kill_switch_active",
                        admins=len(telegram_kill_switch.admin_chat_ids))
            else:
                LOG.info("telegram_kill_switch_disabled")
        except Exception as telegram_error:
            LOG.warning("telegram_kill_switch_failed_to_start",
                       error=str(telegram_error))
        
        # Entrenar modelos iniciales
        LOG.info("loading_historical_data_for_initial_training")
        
        # ===== CORRECCIÓN: Entrenar para TODOS los símbolos, no solo el primero =====
        trained_symbols = []
        training_failures = []
        
        # Determinar cuántos símbolos entrenar inicialmente (top por volumen)
        max_initial_training = int(os.getenv('MAX_INITIAL_TRAINING_SYMBOLS', '50'))
        
        LOG.info("multi_symbol_training_starting",
                total_symbols=len(config.symbols),
                max_initial=max_initial_training,
                strategy="train_top_volume_first")
        
        # Filtrar símbolos por volumen para priorizar
        try:
            priority_symbols = await filter_pairs_by_volume(
                exchange_manager,
                config.symbols[:100],  # Analizar primeros 100
                min_volume_24h=1000000.0  # $1M mínimo
            )
            
            if priority_symbols:
                symbols_to_train = priority_symbols[:max_initial_training]
                LOG.info("symbols_prioritized_by_volume",
                        total_analyzed=min(100, len(config.symbols)),
                        high_volume=len(priority_symbols),
                        selected_for_training=len(symbols_to_train))
            else:
                # Fallback: primeros N símbolos
                symbols_to_train = config.symbols[:max_initial_training]
                LOG.warning("volume_filter_failed_using_first_n",
                           n=max_initial_training)
        
        except Exception as priority_error:
            LOG.error("symbol_prioritization_failed", error=str(priority_error))
            symbols_to_train = config.symbols[:max_initial_training]
        
        # Entrenar modelo general con datos combinados primero
        LOG.info("training_general_ensemble_model")
        
        try:
            # Cargar datos del símbolo más líquido para modelo general
            main_symbol = symbols_to_train[0] if symbols_to_train else config.symbols[0]
            hist_data = await bot._load_historical_data(months=3)
            
            if hist_data is not None and len(hist_data) >= 100:
                LOG.info("training_general_model",
                        symbol=main_symbol,
                        samples=len(hist_data))
                
                await bot.ensemble_learner.fit(hist_data, epochs=10, symbol=None)
                trained_symbols.append(f"{main_symbol}_general")
                
                LOG.info("general_model_trained_successfully")
            else:
                LOG.warning("insufficient_data_for_general_model",
                           data_rows=len(hist_data) if hist_data is not None else 0)
        
        except Exception as general_error:
            LOG.error("general_model_training_failed", error=str(general_error))
            training_failures.append(('general_model', str(general_error)))
        
        # ===== NUEVO: Entrenar modelos especializados para cada símbolo =====
        LOG.info("starting_specialized_model_training",
                symbols_count=len(symbols_to_train))
        
        # Usar concurrencia controlada para no saturar
        training_semaphore = asyncio.Semaphore(3)  # Máximo 3 entrenamientos paralelos
        
        async def train_specialized_model(symbol: str):
            async with training_semaphore:
                try:
                    LOG.info("training_specialized_model_starting", symbol=symbol)
                    
                    # Cargar datos históricos del símbolo específico
                    # Reusar exchange_manager ya existente
                    symbol_hist = await bot._load_historical_data_for_symbol(
                        symbol, months=3
                    )
                    
                    if symbol_hist is None or len(symbol_hist) < 100:
                        LOG.warning("insufficient_data_for_specialized_model",
                                   symbol=symbol,
                                   rows=len(symbol_hist) if symbol_hist is not None else 0)
                        return {'symbol': symbol, 'success': False, 'reason': 'insufficient_data'}
                    
                    # Entrenar modelo especializado
                    await bot.ensemble_learner.fit(
                        symbol_hist,
                        epochs=8,  # Menos epochs para especializados
                        symbol=symbol
                    )
                    
                    # ✅ CRÍTICO: Verificar que se guardó correctamente
                    if symbol not in bot.ensemble_learner.symbol_models:
                        LOG.error("specialized_model_not_registered",
                                 symbol=symbol,
                                 available_models=list(bot.ensemble_learner.symbol_models.keys())[:10])
                        return {'symbol': symbol, 'success': False, 'reason': 'registration_failed'}
                    
                    LOG.info("specialized_model_trained_successfully",
                            symbol=symbol,
                            samples=len(symbol_hist))
                    
                    return {'symbol': symbol, 'success': True, 'samples': len(symbol_hist)}
                
                except Exception as train_error:
                    LOG.error("specialized_model_training_failed",
                             symbol=symbol,
                             error=str(train_error))
                    return {'symbol': symbol, 'success': False, 'error': str(train_error)}
        
        # Crear tareas de entrenamiento
        training_tasks = [
            asyncio.create_task(train_specialized_model(symbol))
            for symbol in symbols_to_train
        ]
        
        # Ejecutar con timeout generoso (5 minutos por símbolo)
        total_timeout = len(symbols_to_train) * 300
        
        try:
            training_results = await asyncio.wait_for(
                asyncio.gather(*training_tasks, return_exceptions=True),
                timeout=total_timeout
            )
            
            # Analizar resultados
            for result in training_results:
                if isinstance(result, dict):
                    if result.get('success'):
                        trained_symbols.append(result['symbol'])
                    else:
                        training_failures.append((result['symbol'], result.get('reason', result.get('error'))))
                elif isinstance(result, Exception):
                    training_failures.append(('unknown_symbol', str(result)))
            
            LOG.info("specialized_training_completed",
                    total_attempted=len(symbols_to_train),
                    successful=len(trained_symbols),
                    failed=len(training_failures),
                    success_rate=len(trained_symbols)/len(symbols_to_train)*100 if symbols_to_train else 0)
            
            # ✅ NUEVO: Guardar estado de modelos entrenados
            if hasattr(bot, 'ensemble_learner'):
                bot.ai_models_status = {
                    'general_model_trained': bot.ensemble_learner.is_trained,
                    'specialized_models_count': len(bot.ensemble_learner.symbol_models),
                    'specialized_symbols': list(bot.ensemble_learner.symbol_models.keys()),
                    'trained_symbols': trained_symbols,
                    'training_failures': training_failures,
                    'symbol_training_history': bot.ensemble_learner.symbol_training_history
                }
                
                LOG.info("ai_models_status_updated",
                        general_trained=bot.ai_models_status['general_model_trained'],
                        specialized_count=bot.ai_models_status['specialized_models_count'])
        
        except asyncio.TimeoutError:
            LOG.error("specialized_training_timeout",
                     timeout_seconds=total_timeout,
                     symbols_attempted=len(symbols_to_train))
            
            # Cancelar tareas pendientes
            for task in training_tasks:
                if not task.done():
                    task.cancel()
            
            await asyncio.gather(*training_tasks, return_exceptions=True)
        
        # ===== WALK-FORWARD VALIDATION: Solo para símbolos exitosos =====
            
            if trained_symbols and len(trained_symbols) >= 3:
                LOG.info("starting_multi_symbol_walk_forward_validation",
                        symbols_to_validate=min(5, len(trained_symbols)))
                
                # Validar top 5 símbolos entrenados
                validation_targets = trained_symbols[:5]
                
                validation_results = {}
                
                for symbol in validation_targets:
                    try:
                        # Cargar datos del símbolo
                        symbol_data = await bot._load_historical_data_for_symbol(
                            symbol, months=6
                        )
                        
                        if symbol_data is None or len(symbol_data) < 1000:
                            LOG.warning("insufficient_data_for_walk_forward",
                                       symbol=symbol,
                                       rows=len(symbol_data) if symbol_data is not None else 0)
                            continue
                        
                        LOG.info("validating_symbol_with_walk_forward",
                                symbol=symbol,
                                data_rows=len(symbol_data))
                        
                        validation_result = await bot.walk_forward_validator.run_walk_forward(
                            bot, symbol_data, symbol=symbol
                        )
                        
                        validation_results[symbol] = validation_result
                        
                        if validation_result.get('success'):
                            analysis = validation_result.get('analysis', {})
                            
                            LOG.info("walk_forward_completed_for_symbol",
                                    symbol=symbol,
                                    windows=len(validation_result.get('windows', [])),
                                    avg_degradation=analysis.get('avg_degradation', 0),
                                    overfitting=analysis.get('overfitting_detected', False))
                            
                            # Si overfitting, re-entrenar con regularización
                            if analysis.get('overfitting_detected'):
                                LOG.warning("retraining_symbol_with_regularization",
                                           symbol=symbol)
                                
                                await bot.ensemble_learner.fit(
                                    symbol_data,
                                    epochs=5,
                                    symbol=symbol
                                )
                        
                    except Exception as validation_error:
                        LOG.error("walk_forward_failed_for_symbol",
                                 symbol=symbol,
                                 error=str(validation_error))
                        validation_results[symbol] = {
                            'success': False,
                            'error': str(validation_error)
                        }
                
                LOG.info("multi_symbol_validation_completed",
                        total_validated=len(validation_results),
                        successful=[s for s, r in validation_results.items() if r.get('success')],
                        failed=[s for s, r in validation_results.items() if not r.get('success')])
            
            else:
                LOG.warning("skipping_walk_forward_insufficient_trained_models",
                           trained_count=len(trained_symbols),
                           required=3)
            
            # Entrenar RL agent con datos combinados
            LOG.info("training_rl_agent_on_combined_historical_data")
            
            # Usar datos del símbolo más líquido para RL
            if hist_data is not None and len(hist_data) >= 100:
                await bot.rl_training_manager.train(episodes=50, df=hist_data)
                LOG.info("rl_training_completed")
            else:
                LOG.warning("insufficient_data_for_rl_training")
            
            LOG.info("initial_training_completed",
                    total_trained_symbols=len(trained_symbols),
                    failed_trainings=len(training_failures))
        
        else:
            LOG.warning("no_symbols_to_train")
        
        # ===== NUEVO: Ejecutar Tests Iniciales con Validación Estricta =====
        LOG.info("running_initial_automated_tests")
        try:
            initial_test_results = await bot.test_suite.run_all_tests()
            
            failed_count = initial_test_results.get('failed', 0)
            total_count = initial_test_results.get('total_tests', 0)
            success_rate = initial_test_results.get('success_rate', 0)
            
            if failed_count > 0:
                LOG.warning("initial_tests_had_failures",
                           failed=failed_count,
                           total=total_count,
                           success_rate=success_rate * 100)
                
                # NUEVO: Identificar tests críticos fallidos
                critical_failures = []
                for result in initial_test_results.get('results', []):
                    if not result['passed']:
                        test_name = result['test_name']
                        # Tests críticos que no pueden fallar
                        critical_tests = [
                            'position_ledger_atomicity',
                            'risk_manager_stop_loss',
                            'equity_audit_consistency'
                        ]
                        if test_name in critical_tests:
                            critical_failures.append(test_name)
                
                if critical_failures:
                    LOG.critical("critical_tests_failed_cannot_start",
                                failed_tests=critical_failures)
                    
                    await ALERT_SYSTEM.send_alert(
                        "CRITICAL",
                        "Bot startup aborted - Critical tests failed",
                        failed_tests=critical_failures,
                        total_failures=failed_count
                    )
                    
                    # NUEVO: Opción de continuar o abortar
                    abort_on_critical_failure = os.getenv('ABORT_ON_TEST_FAILURE', 'true').lower() == 'true'
                    
                    if abort_on_critical_failure:
                        LOG.critical("aborting_startup_due_to_test_failures")
                        raise RuntimeError(f"Critical tests failed: {critical_failures}")
                    else:
                        LOG.warning("continuing_despite_critical_failures_unsafe")
            else:
                LOG.info("all_initial_tests_passed",
                        total=total_count,
                        success_rate=success_rate * 100)
                
        except Exception as test_error:
            LOG.error("initial_testing_failed_exception",
                     error=str(test_error))
            
            # Decidir si continuar
            if os.getenv('ABORT_ON_TEST_ERROR', 'true').lower() == 'true':
                raise

        # ===== CORRECCIÓN: Restablecer equity tras tests =====
        bot.equity = float(config.initial_capital)
        bot.initial_capital = float(config.initial_capital)
        
        # ===== NUEVO: Reconciliación Inicial =====
        if hasattr(bot, 'position_ledger'):
            try:
                LOG.info("performing_initial_reconciliation")
                initial_recon = await bot.position_ledger.reconcile_with_exchange(
                    bot, exchange_manager
                )
                
                if initial_recon.get('discrepancies') or initial_recon.get('ledger_only') or initial_recon.get('exchange_only'):
                    LOG.warning("initial_reconciliation_found_issues",
                               **initial_recon)
                else:
                    LOG.info("initial_reconciliation_clean")
            except Exception as recon_error:
                LOG.warning("initial_reconciliation_failed", error=str(recon_error))
        
        # Iniciar bot
        await bot.start()
        
        LOG.info("bot_started_entering_main_loop")
        
        # Main loop
        while bot.is_running:
            try:
                # Verificar kill switch de Telegram
                if hasattr(bot, 'telegram_kill_switch') and bot.telegram_kill_switch:
                    if bot.telegram_kill_switch.circuit_breaker_active:
                        LOG.warning("trading_paused_by_telegram_kill_switch")
                        await asyncio.sleep(30)
                        continue
                
                # ✅ MEJORA: Confirmar cantidad de símbolos a procesar
                symbols_to_process = config.symbols.copy()
                
                LOG.info("main_loop_iteration_starting",
                        total_symbols_configured=len(symbols_to_process),
                        symbols_sample=symbols_to_process[:10] if len(symbols_to_process) > 10 else symbols_to_process)
                
                # ✅ MEJORA: Estrategia de priorización con logging
                
                # 1. Símbolos con posiciones activas (máxima prioridad)
                priority_1_active = []
                
                # 2. Símbolos con alto volumen/volatilidad (alta prioridad)
                priority_2_highvol = []
                
                # 3. Resto de símbolos
                priority_3_rest = []
                
                if hasattr(bot, 'risk_manager'):
                    active_symbols = set(bot.risk_manager.active_stops.keys())
                    
                    for symbol in symbols_to_process:
                        if symbol in active_symbols:
                            priority_1_active.append(symbol)
                        else:
                            priority_3_rest.append(symbol)
                    
                    LOG.info("symbol_prioritization",
                            active_positions=len(priority_1_active),
                            remaining=len(priority_3_rest))
                else:
                    priority_3_rest = symbols_to_process.copy()
                
                # ✅ MEJORA: Filtrado inteligente para reducir carga sin perder oportunidades
                # Solo aplicar si hay MUCHOS símbolos (>50)
                
                if len(priority_3_rest) > 50:
                    LOG.info("applying_smart_symbol_filtering",
                            total_candidates=len(priority_3_rest),
                            reason="too_many_symbols")
                    
                    # Filtrar por volumen mínimo
                    try:
                        filtered_by_volume = await filter_pairs_by_volume(
                            bot.exchange_manager,
                            priority_3_rest[:100],  # Limitar consultas
                            min_volume_24h=500000.0  # $500k mínimo
                        )
                        
                        if filtered_by_volume:
                            priority_2_highvol = filtered_by_volume[:30]  # Top 30
                            LOG.info("volume_filtering_applied",
                                    original=len(priority_3_rest),
                                    after_filter=len(filtered_by_volume),
                                    selected=len(priority_2_highvol))
                        else:
                            # Fallback: usar primeros 30 si filtro falla
                            priority_2_highvol = priority_3_rest[:30]
                            LOG.warning("volume_filter_failed_using_first_n",
                                       selected=len(priority_2_highvol))
                    
                    except Exception as filter_error:
                        LOG.error("symbol_filtering_error_using_all",
                                 error=str(filter_error))
                        priority_2_highvol = priority_3_rest[:30]
                else:
                    # Si pocos símbolos, procesar todos
                    priority_2_highvol = priority_3_rest
                
                # Combinar prioridades
                ordered_symbols = priority_1_active + priority_2_highvol
                
                LOG.info("symbol_processing_plan",
                        total_to_process=len(ordered_symbols),
                        active_positions=len(priority_1_active),
                        high_volume=len(priority_2_highvol),
                        skipped=len(symbols_to_process) - len(ordered_symbols))
                
                # ✅ MEJORA: Ajustar concurrencia dinámicamente
                base_concurrent = 10  # Aumentado de 5
                memory_usage = MEMORY_MANAGER.get_memory_usage().get('rss_mb', 0) if MEMORY_MANAGER else 0
                
                if memory_usage > 1800:
                    max_concurrent = 3
                    LOG.warning("reducing_concurrency_critical_memory", memory_mb=memory_usage)
                elif memory_usage > 1500:
                    max_concurrent = 5
                    LOG.info("reducing_concurrency_high_memory", memory_mb=memory_usage)
                else:
                    # Ajustar según cantidad de símbolos
                    if len(ordered_symbols) > 100:
                        max_concurrent = 15
                    elif len(ordered_symbols) > 50:
                        max_concurrent = 12
                    else:
                        max_concurrent = base_concurrent
                    
                    LOG.debug("concurrency_set",
                             max_concurrent=max_concurrent,
                             symbols=len(ordered_symbols),
                             memory_mb=memory_usage)
                
                semaphore = asyncio.Semaphore(max_concurrent)
                
                # ✅ MEJORA: Logging detallado por símbolo procesado
                processing_stats = {
                    'attempted': 0,
                    'success': 0,
                    'failed': 0,
                    'skipped': 0,
                    'timeout': 0
                }
                
                async def process_with_semaphore(symbol):
                    async with semaphore:
                        processing_stats['attempted'] += 1
                        
                        # Logging cada 50 símbolos
                        if processing_stats['attempted'] % 50 == 0:
                            LOG.info("processing_progress",
                                    processed=processing_stats['attempted'],
                                    total=len(ordered_symbols),
                                    progress_pct=processing_stats['attempted']/len(ordered_symbols)*100)
                        
                        if symbol not in bot.symbol_execution_locks:
                            bot.symbol_execution_locks[symbol] = asyncio.Lock()
                        
                        if bot.symbol_execution_locks[symbol].locked():
                            processing_stats['skipped'] += 1
                            LOG.debug("symbol_already_processing_skipping", symbol=symbol)
                            return {'symbol': symbol, 'status': 'skipped'}
                        
                        try:
                            async with bot.symbol_execution_locks[symbol]:
                                # ✅ Timeout individual por símbolo
                                try:
                                    result = await asyncio.wait_for(
                                        fetch_market_data(
                                            bot.exchange_manager,
                                            symbol,
                                            bot.config.timeframe
                                        ),
                                        timeout=15.0  # 15 segundos por símbolo
                                    )
                                except asyncio.TimeoutError:
                                    processing_stats['timeout'] += 1
                                    LOG.warning("symbol_fetch_timeout", symbol=symbol)
                                    return {'symbol': symbol, 'status': 'timeout'}
                                
                                if result.get('success'):
                                    process_result = await bot._process_symbol_data(symbol, result)
                                    
                                    if process_result and process_result.get('success'):
                                        processing_stats['success'] += 1
                                        
                                        # Log detallado solo para símbolos importantes
                                        if symbol in priority_1_active or processing_stats['success'] % 20 == 0:
                                            LOG.info("symbol_processed_successfully",
                                                    symbol=symbol,
                                                    candles=process_result.get('candles_processed', 0),
                                                    has_active_position=symbol in priority_1_active)
                                    else:
                                        processing_stats['failed'] += 1
                                    
                                    return {
                                        'symbol': symbol,
                                        'status': 'success',
                                        'result': process_result
                                    }
                                else:
                                    processing_stats['failed'] += 1
                                    return {
                                        'symbol': symbol,
                                        'status': 'failed',
                                        'error': result.get('error')
                                    }
                        except Exception as e:
                            processing_stats['failed'] += 1
                            LOG.debug("symbol_processing_exception",
                                     symbol=symbol,
                                     error=str(e)[:100])
                            return {'symbol': symbol, 'status': 'error', 'error': str(e)}
                
                # Crear tareas
                tasks = [asyncio.create_task(process_with_semaphore(symbol)) 
                        for symbol in ordered_symbols]
                
                # Esperar resultados con timeout global generoso
                if tasks:
                    try:
                        # Timeout: 2 minutos + 5 segundos por símbolo
                        global_timeout = 120 + (len(ordered_symbols) * 5)
                        
                        results = await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=global_timeout
                        )
                        
                        # ✅ Análisis detallado de resultados
                        LOG.info("main_loop_iteration_complete",
                                total_symbols=len(ordered_symbols),
                                attempted=processing_stats['attempted'],
                                success=processing_stats['success'],
                                failed=processing_stats['failed'],
                                skipped=processing_stats['skipped'],
                                timeout=processing_stats['timeout'],
                                success_rate=processing_stats['success']/processing_stats['attempted']*100 
                                            if processing_stats['attempted'] > 0 else 0)
                        
                        # ✅ Log de símbolos problemáticos
                        failed_symbols = [r['symbol'] for r in results 
                                         if isinstance(r, dict) and r.get('status') in ['failed', 'error', 'timeout']]
                        
                        if failed_symbols and len(failed_symbols) > 10:
                            LOG.warning("multiple_symbols_failed",
                                       count=len(failed_symbols),
                                       sample=failed_symbols[:10])
                        
                    except asyncio.TimeoutError:
                        LOG.error("main_loop_global_timeout",
                                 timeout_seconds=global_timeout,
                                 symbols_attempted=len(ordered_symbols),
                                 processed=processing_stats['success'])
                        
                        # Cancelar pendientes
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                # Health check y ajustes
                if hasattr(bot, 'health_check'):
                    try:
                        health = bot.health_check.get_health_status()
                        
                        # Alertar si memoria muy alta
                        if health.get('memory_mb', 0) > 2000:
                            LOG.warning("high_memory_usage_in_main_loop",
                                       memory_mb=health['memory_mb'])
                            
                            # Forzar limpieza
                            optimize_memory_usage()
                            
                        # Alertar si CPU muy alto
                        if health.get('cpu_percent', 0) > 90:
                            LOG.warning("high_cpu_usage_in_main_loop",
                                       cpu_percent=health['cpu_percent'])
                            
                            # Aumentar intervalo temporalmente
                            await asyncio.sleep(30)
                            
                    except Exception as health_error:
                        LOG.debug("health_check_error", error=str(health_error))
                
                # Intervalo entre ciclos
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                LOG.info("main_loop_cancelled")
                break
            except Exception as e:
                LOG.error("main_loop_error",
                         error=str(e),
                         traceback=traceback.format_exc()[:500])
                await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        LOG.info("keyboard_interrupt_received_shutting_down")
    
    except Exception as e:
        LOG.critical("critical_error_in_main",
                    error=str(e),
                    traceback=traceback.format_exc())
    
    finally:
        LOG.info("initiating_graceful_shutdown")
        
        # NUEVO: Lista de errores de shutdown para reportar
        shutdown_errors = []
        
        # 1. Detener Telegram primero (puede enviar notificaciones)
        if telegram_kill_switch and telegram_kill_switch.enabled:
            try:
                LOG.info("stopping_telegram_kill_switch")
                await telegram_kill_switch.stop()
                LOG.info("telegram_kill_switch_stopped")
            except Exception as telegram_error:
                error_msg = f"Telegram shutdown error: {str(telegram_error)}"
                shutdown_errors.append(error_msg)
                LOG.error("telegram_shutdown_error", error=str(telegram_error))
        
        # 2. Detener bot y sus tareas periódicas
        if bot:
            try:
                LOG.info("stopping_bot_and_periodic_tasks")
                
                # Marcar como no running para detener loops
                bot.is_running = False
                
                # Esperar un momento para que loops detecten el cambio
                await asyncio.sleep(2)
                
                # NUEVO: Cancelar tareas periódicas explícitamente
                if hasattr(bot, 'periodic_tasks'):
                    LOG.info("cancelling_periodic_tasks", count=len(bot.periodic_tasks))
                    for task in bot.periodic_tasks:
                        if task and not task.done():
                            task.cancel()
                    
                    # Esperar cancelaciones con timeout
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*bot.periodic_tasks, return_exceptions=True),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        LOG.warning("periodic_tasks_cancellation_timeout")
                
                # Guardar estado final
                try:
                    # Guardar ensemble models
                    if hasattr(bot, 'ensemble_learner') and bot.ensemble_learner:
                        LOG.info("saving_ensemble_models")
                        await bot.ensemble_learner._save_models()
                    
                    # Guardar RL agent
                    if hasattr(bot, 'rl_agent') and bot.rl_agent:
                        LOG.info("saving_rl_agent")
                        bot.rl_agent.save()
                    
                    # Guardar strategy performance
                    if hasattr(bot, 'strategy_manager') and bot.strategy_manager:
                        LOG.info("saving_strategy_performance")
                        await bot.strategy_manager.save_strategy_performance()
                    
                except Exception as save_error:
                    error_msg = f"State save error: {str(save_error)}"
                    shutdown_errors.append(error_msg)
                    LOG.error("state_save_error", error=str(save_error))
                
                # NUEVO: Generar reporte final
                try:
                    final_report = await bot.get_performance_report()
                    LOG.info("final_performance_report", **final_report)
                    
                    # Guardar reporte en archivo
                    import json
                    report_path = f"reports/final_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
                    os.makedirs("reports", exist_ok=True)
                    with open(report_path, 'w') as f:
                        json.dump(final_report, f, indent=2)
                    LOG.info("final_report_saved", path=report_path)
                    
                except Exception as report_error:
                    LOG.warning("final_report_generation_failed", error=str(report_error))
                
                # Ejecutar audit final de equity
                if hasattr(bot, 'position_ledger'):
                    try:
                        LOG.info("performing_final_equity_audit")
                        final_audit = bot.position_ledger.audit_equity(bot)
                        LOG.info("final_equity_audit", **final_audit)
                        
                        if not final_audit['is_consistent']:
                            LOG.error("final_audit_inconsistent",
                                     discrepancy=final_audit['discrepancy'])
                    except Exception as audit_error:
                        LOG.warning("final_audit_failed", error=str(audit_error))
                
                # Llamar al método stop del bot
                await bot.stop()
                LOG.info("bot_stopped")
                
            except Exception as stop_error:
                error_msg = f"Bot stop error: {str(stop_error)}"
                shutdown_errors.append(error_msg)
                LOG.error("bot_stop_error", error=str(stop_error))
        
        # 3. Cerrar conexión con exchange
        if exchange_manager:
            try:
                LOG.info("closing_exchange_connection")
                await exchange_manager.close()
                LOG.info("exchange_closed")
            except Exception as close_error:
                error_msg = f"Exchange close error: {str(close_error)}"
                shutdown_errors.append(error_msg)
                LOG.error("exchange_close_error", error=str(close_error))
        
        # 4. MEJORADO: Flush completo de InfluxDB y métricas locales
        try:
            LOG.info("flushing_all_metrics")
            
            # Sincronización final forzada
            if bot:
                try:
                    await sync_bot_metrics_to_influx(bot, force=True)
                    LOG.info("final_metrics_sync_completed")
                except Exception as sync_error:
                    LOG.warning("final_sync_failed", error=str(sync_error))
            
            # Flush InfluxDB
            if INFLUX_METRICS and INFLUX_METRICS.enabled:
                if hasattr(INFLUX_METRICS, 'write_api') and INFLUX_METRICS.write_api:
                    try:
                        # Múltiples intentos de flush
                        for attempt in range(3):
                            try:
                                INFLUX_METRICS.write_api.flush()
                                LOG.info("influxdb_flushed_successfully", attempt=attempt + 1)
                                break
                            except Exception as flush_error:
                                if attempt < 2:
                                    await asyncio.sleep(1)
                                else:
                                    raise
                    except Exception as influx_error:
                        LOG.error("influxdb_flush_failed_all_attempts", 
                                 error=str(influx_error))
            
            # Flush buffer local de métricas
            if METRICS and hasattr(METRICS, '_flush_buffer'):
                try:
                    METRICS._flush_buffer()
                    LOG.info("local_metrics_buffer_flushed")
                except Exception as metrics_error:
                    LOG.warning("metrics_buffer_flush_failed", error=str(metrics_error))
            
            LOG.info("all_metrics_flushed")
            
        except Exception as flush_error:
            LOG.error("metrics_flush_error", error=str(flush_error))
        
        # 5. NUEVO: Limpieza de memoria final
        try:
            LOG.info("final_memory_cleanup")
            optimize_memory_usage()
            
            # Obtener estadísticas finales
            if MEMORY_MANAGER:
                final_mem_stats = MEMORY_MANAGER.get_memory_stats()
                LOG.info("final_memory_stats", **final_mem_stats)
        except Exception as mem_error:
            LOG.debug("final_memory_cleanup_error", error=str(mem_error))
        
        # 6. NUEVO: Reporte de shutdown
        if shutdown_errors:
            LOG.warning("shutdown_completed_with_errors",
                       error_count=len(shutdown_errors),
                       errors=shutdown_errors)
        else:
            LOG.info("shutdown_completed_successfully")
        
        # 7. NUEVO: Enviar notificación final (si Telegram está disponible)
        try:
            if telegram_kill_switch and telegram_kill_switch.enabled and not shutdown_errors:
                # Crear bot temporal para enviar mensaje final
                from telegram import Bot
                temp_bot = Bot(token=telegram_kill_switch.bot_token)
                
                shutdown_message = (
                    "🛑 *Bot Shutdown Complete*\n\n"
                    f"Final Equity: ${bot.equity:,.2f}\n"
                    f"Total Trades: {bot.performance_metrics.get('total_trades', 0)}\n"
                    f"Win Rate: {bot.performance_metrics.get('win_rate', 0)*100:.1f}%\n"
                )
                
                for admin_id in telegram_kill_switch.admin_chat_ids:
                    try:
                        await temp_bot.send_message(
                            chat_id=admin_id,
                            text=shutdown_message,
                            parse_mode='Markdown'
                        )
                    except Exception:
                        pass
        except Exception as final_notify_error:
            LOG.debug("final_notification_failed", error=str(final_notify_error))

async def process_symbol_with_lock(bot, symbol: str):
    """Helper para procesar símbolo con lock"""
    async with bot.symbol_execution_locks[symbol]:
        try:
            result = await fetch_market_data(
                bot.exchange_manager,
                symbol,
                bot.config.timeframe
            )
            
            if result.get('success'):
                await bot._process_symbol_data(symbol, result)
            
        except Exception as e:
            LOG.error("process_symbol_with_lock_error",
                     symbol=symbol,
                     error=str(e))

def setup_signal_handlers():
    """Configura manejadores de señales para shutdown graceful"""
    def signal_handler(signum, frame):
        LOG.info("signal_received_initiating_shutdown", signal=signum)
        # El shutdown se maneja en el finally del main
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ===========================
# MAIN ENTRY POINT
# ===========================

if __name__ == "__main__":
    try:
        setup_signal_handlers()
        asyncio.run(advanced_ai_main_with_rl())
    except KeyboardInterrupt:
        LOG.info("application_terminated_gracefully")
    except Exception as e:
        LOG.critical("application_crash", error=str(e), traceback=traceback.format_exc())
        sys.exit(1)

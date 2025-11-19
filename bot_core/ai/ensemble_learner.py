import os
import joblib
import json
import logging
import shutil
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams
from bot_core.ai.models import LSTMPredictor, AttentionNetwork
from bot_core.ai.feature_processor import FeatureProcessor

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, classification_report, log_loss
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    # Define dummy classes if ML libraries are not available
    class VotingClassifier: pass
    class RandomForestClassifier: pass
    class LogisticRegression: pass
    class XGBClassifier: pass
    class RandomizedSearchCV: pass
    class TimeSeriesSplit: pass
    def precision_score(*args, **kwargs): return 0.0
    def log_loss(*args, **kwargs): return 0.0
    def classification_report(*args, **kwargs): return ""
    def compute_class_weight(*args, **kwargs): return []
    def compute_sample_weight(*args, **kwargs): return []
    class nn:
        class Module: pass
    class optim: pass
    class TensorDataset: pass
    class DataLoader: pass

logger = get_logger(__name__)

# --- Standalone Helper Functions (Pickle-safe for Multiprocessing) ---

def _create_fresh_models(config: AIEnsembleStrategyParams, num_features: int, device) -> Dict[str, Any]:
    """Creates a fresh set of untrained model instances based on configuration and dynamic feature count."""
    hp = config.hyperparameters
    xgb_config = hp.xgboost
    rf_config = hp.random_forest
    lr_config = hp.logistic_regression
    lstm_config = hp.lstm
    attn_config = hp.attention

    # Determine class weight setting for Sklearn models
    cw_option = 'balanced' if config.training.use_class_weighting else None

    models = {
        'gb': XGBClassifier(
            n_estimators=xgb_config.n_estimators,
            max_depth=xgb_config.max_depth,
            learning_rate=xgb_config.learning_rate,
            subsample=xgb_config.subsample,
            colsample_bytree=xgb_config.colsample_bytree,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
            # XGBoost handles weights via fit(sample_weight=...), not init
        ),
        'technical': VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(
                n_estimators=rf_config.n_estimators,
                max_depth=rf_config.max_depth,
                min_samples_leaf=rf_config.min_samples_leaf,
                class_weight=cw_option,
                random_state=42
            )),
            ('lr', LogisticRegression(
                max_iter=lr_config.max_iter,
                C=lr_config.C,
                class_weight=cw_option,
                random_state=42
            ))
        ], voting='soft')
    }

    if ML_AVAILABLE and torch.cuda.is_available():
        # Only create PyTorch models if available
        models['lstm'] = LSTMPredictor(
            num_features, lstm_config.hidden_dim, lstm_config.num_layers, lstm_config.dropout
        ).to(device)
        models['attention'] = AttentionNetwork(
            num_features, attn_config.hidden_dim, attn_config.num_layers, attn_config.nhead, attn_config.dropout
        ).to(device)
    
    return models

def _optimize_hyperparameters(model, X, y, param_dist, n_iter, logger_instance, fit_params=None):
    """Runs RandomizedSearchCV to find better hyperparameters."""
    try:
        # Use TimeSeriesSplit to prevent data leakage (future peeking)
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='neg_log_loss',
            cv=tscv,
            n_jobs=1, # Already in a subprocess
            random_state=42,
            verbose=0
        )
        search.fit(X, y, **(fit_params or {}))
        logger_instance.info(f"Hyperparameter optimization complete. Best score: {search.best_score_:.4f}")
        return search.best_estimator_
    except Exception as e:
        logger_instance.warning(f"Hyperparameter optimization failed: {e}. Using default parameters.")
        model.fit(X, y, **(fit_params or {}))
        return model

def _optimize_ensemble_weights(predictions: Dict[str, np.ndarray], y_true: np.ndarray, iterations: int = 1000) -> Dict[str, float]:
    """
    Finds optimal weights for the ensemble using Random Search to minimize Log Loss.
    predictions: Dict of model_name -> prob_array (N, 3)
    y_true: array (N,)
    """
    model_names = list(predictions.keys())
    if not model_names:
        return {}
    
    best_score = float('inf')
    best_weights = {name: 1.0/len(model_names) for name in model_names}
    
    # Convert dict to list of arrays for fast vectorized math
    # shape: (num_models, num_samples, num_classes)
    pred_stack = np.array([predictions[name] for name in model_names])
    
    for _ in range(iterations):
        # Generate random weights summing to 1
        w = np.random.dirichlet(np.ones(len(model_names)))
        
        # Weighted average: sum(w[i] * pred_stack[i])
        # w shape: (num_models,)
        # pred_stack shape: (num_models, num_samples, num_classes)
        # Result: (num_samples, num_classes)
        ensemble_probs = np.tensordot(w, pred_stack, axes=([0],[0]))
        
        # Clip to avoid log(0)
        ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)
        
        # Calculate Log Loss
        score = log_loss(y_true, ensemble_probs)
        
        if score < best_score:
            best_score = score
            best_weights = {name: float(w[i]) for i, name in enumerate(model_names)}
            
    return best_weights

def train_ensemble_task(symbol: str, df: pd.DataFrame, config: AIEnsembleStrategyParams) -> bool:
    """
    Standalone function to train models in a separate process.
    Returns True if training and saving were successful.
    """
    # Re-initialize logger for the worker process
    worker_logger = get_logger(f"trainer_{symbol}")
    worker_logger.info(f"Starting training task for {symbol} with {len(df)} records.")

    if not ML_AVAILABLE:
        worker_logger.error("ML libraries not available. Skipping training.")
        return False

    try:
        # 1. Prepare Data
        # Process (Lag generation + Normalization)
        df_proc = FeatureProcessor.process_data(df, config)
        num_features = df_proc.shape[1]
        
        # Create Labels
        labels = FeatureProcessor.create_labels(df, config)
        
        # Align Data (drop NaNs from rolling windows and shifting)
        common_index = df_proc.index.intersection(labels.index)
        if len(common_index) < 100:
            worker_logger.warning("Insufficient data after alignment.")
            return False
            
        X = df_proc.loc[common_index].values
        y = labels.loc[common_index].values.astype(int)

        # Split Train/Val
        split_idx = int(len(X) * (1 - config.training.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # --- Class Weighting Calculation ---
        sample_weights = None
        torch_weights = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.training.use_class_weighting:
            # For XGBoost (Sample Weights)
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            
            # For PyTorch (Class Weights Tensor)
            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
            
            # Map to fixed size 3 array [Sell(0), Hold(1), Buy(2)] to handle missing classes in small datasets
            weight_map = {c: w for c, w in zip(unique_classes, class_weights)}
            final_weights = [weight_map.get(c, 1.0) for c in [0, 1, 2]]
            torch_weights = torch.FloatTensor(final_weights).to(device)
            
            worker_logger.info(f"Using Class Weights: {final_weights}")

        # Create Models with dynamic feature count
        models = _create_fresh_models(config, num_features, device)
        trained_models = {}
        metrics = {}
        feature_importance = {}

        # 2. Train Scikit-Learn/XGBoost Models
        for name, model in models.items():
            if name in ['lstm', 'attention']: continue
            
            worker_logger.info(f"Training {name}...")
            
            fit_params = {}
            if name == 'gb' and sample_weights is not None:
                fit_params['sample_weight'] = sample_weights

            if config.training.auto_tune_models and name == 'gb':
                # Example param grid for XGBoost
                param_dist = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
                model = _optimize_hyperparameters(model, X_train, y_train, param_dist, config.training.n_iter_search, worker_logger, fit_params)
            else:
                model.fit(X_train, y_train, **fit_params)
            
            trained_models[name] = model
            
            # Evaluate
            preds = model.predict(X_val)
            prec = precision_score(y_val, preds, average=None, zero_division=0)
            metrics[name] = {'precision': prec.tolist()}
            
            # Feature Importance (if available)
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = model.feature_importances_.tolist()

        # 3. Train PyTorch Models
        seq_len = config.features.sequence_length
        X_seq_train, y_seq_train = FeatureProcessor.create_sequences(X_train, y_train, seq_len)
        X_seq_val, y_seq_val = FeatureProcessor.create_sequences(X_val, y_val, seq_len)

        if len(X_seq_train) > 0:
            train_tensor = TensorDataset(torch.FloatTensor(X_seq_train).to(device), torch.LongTensor(y_seq_train).to(device))
            train_loader = DataLoader(train_tensor, batch_size=config.training.batch_size, shuffle=True)
            
            for name in ['lstm', 'attention']:
                if name not in models: continue
                
                model = models[name]
                optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
                
                # Apply weights to loss function if calculated
                if torch_weights is not None:
                    criterion = nn.CrossEntropyLoss(weight=torch_weights)
                else:
                    criterion = nn.CrossEntropyLoss()
                
                worker_logger.info(f"Training {name} (PyTorch)...")
                model.train()
                for epoch in range(config.training.epochs):
                    epoch_loss = 0.0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    
                    # Simple early stopping check could go here
                
                trained_models[name] = model
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    val_inputs = torch.FloatTensor(X_seq_val).to(device)
                    outputs = model(val_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    prec = precision_score(y_seq_val, predicted.cpu().numpy(), average=None, zero_division=0)
                    metrics[name] = {'precision': prec.tolist()}

        # 4. Dynamic Weight Optimization (Auto-Tune)
        optimized_weights = {}
        if config.ensemble_weights.auto_tune:
            worker_logger.info("Optimizing ensemble weights on validation set...")
            val_preds = {}
            
            # Collect predictions from flat models
            if 'gb' in trained_models:
                val_preds['gb'] = trained_models['gb'].predict_proba(X_val)
            if 'technical' in trained_models:
                val_preds['technical'] = trained_models['technical'].predict_proba(X_val)
            
            # Collect predictions from sequence models (aligned)
            if len(X_seq_val) > 0:
                # Sequence models lose (seq_len - 1) samples at the start
                valid_indices = slice(seq_len - 1, None)
                
                # Truncate flat model preds to match sequence length
                for k in list(val_preds.keys()):
                    val_preds[k] = val_preds[k][valid_indices]
                
                y_val_opt = y_val[valid_indices]
                
                for name in ['lstm', 'attention']:
                    if name in trained_models:
                        model = trained_models[name]
                        model.eval()
                        with torch.no_grad():
                            inputs = torch.FloatTensor(X_seq_val).to(device)
                            probs = model(inputs).cpu().numpy()
                            val_preds[name] = probs
            else:
                # If sequence data is too short, we can't use NNs in optimization properly
                # or we just use flat models. For now, assume we use what we have.
                y_val_opt = y_val
            
            # Run Optimization
            if val_preds:
                optimized_weights = _optimize_ensemble_weights(val_preds, y_val_opt)
                worker_logger.info(f"Optimized Weights: {optimized_weights}")

        # 5. Walk-Forward Validation (Profitability Check)
        worker_logger.info("Performing Walk-Forward Validation on Validation Set...")
        
        # Aggregate Probabilities using Optimized Weights if available, else Config Weights
        num_val = len(y_val)
        ensemble_probs = np.zeros((num_val, 3))
        
        # Helper to get weight
        def get_weight(name, default_weight):
            if optimized_weights:
                return optimized_weights.get(name, 0.0)
            return default_weight

        config_weights = config.ensemble_weights
        
        # Sklearn Models
        if 'gb' in trained_models:
            probs = trained_models['gb'].predict_proba(X_val)
            w = get_weight('gb', config_weights.xgboost)
            ensemble_probs += probs * w
            
        if 'technical' in trained_models:
            probs = trained_models['technical'].predict_proba(X_val)
            w = get_weight('technical', config_weights.technical_ensemble)
            ensemble_probs += probs * w
            
        # NN Models (Alignment required)
        valid_nn_indices = slice(seq_len - 1, None)
        
        if len(X_seq_val) > 0:
            def add_nn_probs(model_name, default_weight):
                if model_name in trained_models:
                    model = trained_models[model_name]
                    model.eval()
                    with torch.no_grad():
                        inputs = torch.FloatTensor(X_seq_val).to(device)
                        probs = model(inputs).cpu().numpy()
                        # Add to the slice of ensemble_probs
                        target_len = len(ensemble_probs[valid_nn_indices])
                        if len(probs) == target_len:
                            w = get_weight(model_name, default_weight)
                            ensemble_probs[valid_nn_indices] += probs * w
            
            add_nn_probs('lstm', config_weights.lstm)
            add_nn_probs('attention', config_weights.attention)

        # Determine Final Predictions on the subset where all models could predict
        eval_probs = ensemble_probs[valid_nn_indices]
        final_preds = np.argmax(eval_probs, axis=1) # 0, 1, 2
        
        # Map to Signal: 0->-1 (Short), 1->0 (Hold), 2->1 (Long)
        signals = np.zeros_like(final_preds)
        signals[final_preds == 0] = -1
        signals[final_preds == 2] = 1
        
        # Get Market Returns for the validation period
        val_indices = common_index[split_idx:]
        eval_indices = val_indices[valid_nn_indices]
        
        if len(eval_indices) != len(signals):
            worker_logger.warning("Shape mismatch in validation. Skipping profitability check.")
        else:
            # Calculate Next-Candle Returns
            full_returns = df['close'].pct_change().shift(-1)
            eval_returns = full_returns.loc[eval_indices].fillna(0.0).values
            
            # Strategy Returns
            strategy_returns = signals * eval_returns
            
            # Metrics
            total_return = np.sum(strategy_returns)
            winning_returns = strategy_returns[strategy_returns > 0]
            losing_returns = strategy_returns[strategy_returns < 0]
            
            gross_profit = np.sum(winning_returns)
            gross_loss = np.abs(np.sum(losing_returns))
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
            
            mean_ret = np.mean(strategy_returns)
            std_ret = np.std(strategy_returns)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
            
            metrics['ensemble'] = {
                'profit_factor': float(profit_factor),
                'sharpe': float(sharpe),
                'total_return': float(total_return),
                'win_rate': float(len(winning_returns) / len(strategy_returns)) if len(strategy_returns) > 0 else 0.0
            }
            
            worker_logger.info(f"Validation Results: PF={profit_factor:.2f}, Sharpe={sharpe:.4f}, Ret={total_return:.4f}")
            
            # Check Thresholds
            if profit_factor < config.training.min_profit_factor:
                worker_logger.warning(f"Model rejected: Profit Factor {profit_factor:.2f} < {config.training.min_profit_factor}")
                return False
                
            if sharpe < config.training.min_sharpe_ratio:
                worker_logger.warning(f"Model rejected: Sharpe {sharpe:.4f} < {config.training.min_sharpe_ratio}")
                return False

        # 6. Atomic Save Artifacts
        safe_symbol = symbol.replace('/', '_')
        final_dir = os.path.join(config.model_path, safe_symbol)
        temp_dir = os.path.join(config.model_path, f"{safe_symbol}_temp")
        
        # Clean up any existing temp dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save sklearn models
            for name, model in trained_models.items():
                if name in ['lstm', 'attention']:
                    torch.save(model.state_dict(), os.path.join(temp_dir, f"{name}.pth"))
                else:
                    joblib.dump(model, os.path.join(temp_dir, f"{name}.joblib"))
            
            # Save Metadata
            meta = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'feature_importance': feature_importance,
                'feature_columns': config.feature_columns,
                'num_features': num_features,
                'optimized_weights': optimized_weights # Save the dynamic weights
            }
            with open(os.path.join(temp_dir, "metadata.json"), 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Atomic Swap
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            
            # Rename temp to final
            os.rename(temp_dir, final_dir)
            
            worker_logger.info(f"Training complete for {symbol}. Artifacts saved atomically to {final_dir}")
            return True
            
        except Exception as e:
            worker_logger.error(f"Failed to save artifacts for {symbol}: {e}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False

    except Exception as e:
        worker_logger.error(f"Training failed for {symbol}: {str(e)}", exc_info=True)
        return False

class EnsembleLearner:
    """
    Manages the lifecycle of AI models: loading, inference, and state management.
    """
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.symbol_models: Dict[str, Dict[str, Any]] = {}
        self.device = torch.device("cuda" if ML_AVAILABLE and torch.cuda.is_available() else "cpu")
        logger.info(f"EnsembleLearner initialized. Device: {self.device}")

    @property
    def is_trained(self) -> bool:
        return bool(self.symbol_models)

    def has_valid_model(self, symbol: str) -> bool:
        """Checks if a valid model is currently loaded in memory for the symbol."""
        return symbol in self.symbol_models

    def get_last_training_time(self, symbol: str) -> Optional[datetime]:
        """Checks the metadata file to find the last training timestamp."""
        try:
            safe_symbol = symbol.replace('/', '_')
            meta_path = os.path.join(self.config.model_path, safe_symbol, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data['timestamp'])
        except Exception:
            pass
        return None

    async def reload_models(self, symbol: str):
        """Loads trained models from disk into memory, ensuring config consistency."""
        if not ML_AVAILABLE:
            return

        safe_symbol = symbol.replace('/', '_')
        load_dir = os.path.join(self.config.model_path, safe_symbol)
        
        if not os.path.exists(load_dir):
            logger.warning(f"No model directory found for {symbol}")
            return

        try:
            meta_path = os.path.join(load_dir, "metadata.json")
            if not os.path.exists(meta_path):
                logger.warning(f"Metadata missing for {symbol}, cannot load models.")
                return

            # Load Metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # --- CONFIG CONSISTENCY CHECK ---
            saved_features = meta.get('feature_columns', [])
            current_features = self.config.feature_columns
            
            if saved_features != current_features:
                logger.warning(
                    f"Feature mismatch for {symbol}. "
                    f"Saved: {len(saved_features)} features, Current: {len(current_features)} features. "
                    "Model will be discarded to trigger retrain."
                )
                if symbol in self.symbol_models:
                    del self.symbol_models[symbol]
                return
            
            expected_num_features = FeatureProcessor.get_expected_feature_count(self.config)
            saved_num_features = meta.get('num_features', len(saved_features))
            
            if expected_num_features != saved_num_features:
                 logger.warning(
                    f"Feature count mismatch for {symbol}. "
                    f"Saved: {saved_num_features}, Expected: {expected_num_features}. "
                    "Model will be discarded to trigger retrain."
                )
                 if symbol in self.symbol_models:
                    del self.symbol_models[symbol]
                 return
            # --------------------------------

            models = {}
            # Load Sklearn Models
            for name in ['gb', 'technical']:
                path = os.path.join(load_dir, f"{name}.joblib")
                if os.path.exists(path):
                    models[name] = joblib.load(path)

            # Load PyTorch Models
            if 'lstm' in self.config.ensemble_weights.dict():
                path = os.path.join(load_dir, "lstm.pth")
                if os.path.exists(path):
                    lstm = LSTMPredictor(expected_num_features, self.config.hyperparameters.lstm.hidden_dim, 
                                         self.config.hyperparameters.lstm.num_layers, 
                                         self.config.hyperparameters.lstm.dropout).to(self.device)
                    lstm.load_state_dict(torch.load(path, map_location=self.device))
                    lstm.eval()
                    models['lstm'] = lstm

            if 'attention' in self.config.ensemble_weights.dict():
                path = os.path.join(load_dir, "attention.pth")
                if os.path.exists(path):
                    attn = AttentionNetwork(expected_num_features, self.config.hyperparameters.attention.hidden_dim,
                                            self.config.hyperparameters.attention.num_layers,
                                            self.config.hyperparameters.attention.nhead,
                                            self.config.hyperparameters.attention.dropout).to(self.device)
                    attn.load_state_dict(torch.load(path, map_location=self.device))
                    attn.eval()
                    models['attention'] = attn

            self.symbol_models[symbol] = {
                'models': models,
                'meta': meta
            }
            logger.info(f"Models reloaded successfully for {symbol}")

        except Exception as e:
            logger.error(f"Failed to reload models for {symbol}: {e}")

    async def predict(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Runs inference using the ensemble of models."""
        if symbol not in self.symbol_models:
            await self.reload_models(symbol)
            if symbol not in self.symbol_models:
                return {}

        entry = self.symbol_models[symbol]
        models = entry['models']
        meta = entry['meta']
        
        # Prepare Data
        df_proc = FeatureProcessor.process_data(df, self.config)
        X = df_proc.iloc[-1:].values # Last row for prediction
        
        # Prepare Sequences for NN
        seq_len = self.config.features.sequence_length
        if len(df_proc) >= seq_len:
            X_seq = df_proc.iloc[-seq_len:].values.reshape(1, seq_len, -1)
        else:
            X_seq = None

        votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        config_weights = self.config.ensemble_weights
        
        # Determine Weights (Dynamic vs Static)
        optimized_weights = meta.get('optimized_weights')
        use_dynamic = config_weights.auto_tune and optimized_weights
        
        def get_weight(name, default_weight):
            if use_dynamic:
                return optimized_weights.get(name, 0.0)
            return default_weight

        # Store individual predictions for variance calculation
        model_predictions = []
        
        # Inference
        try:
            # XGBoost
            if 'gb' in models:
                probs = models['gb'].predict_proba(X)[0]
                w = get_weight('gb', config_weights.xgboost)
                votes['sell'] += probs[0] * w
                votes['hold'] += probs[1] * w
                votes['buy'] += probs[2] * w
                model_predictions.append((w, probs))

            # Technical Ensemble
            if 'technical' in models:
                probs = models['technical'].predict_proba(X)[0]
                w = get_weight('technical', config_weights.technical_ensemble)
                votes['sell'] += probs[0] * w
                votes['hold'] += probs[1] * w
                votes['buy'] += probs[2] * w
                model_predictions.append((w, probs))

            # PyTorch Models
            if X_seq is not None:
                with torch.no_grad():
                    tensor_in = torch.FloatTensor(X_seq).to(self.device)
                    
                    if 'lstm' in models:
                        probs = models['lstm'](tensor_in).cpu().numpy()[0]
                        w = get_weight('lstm', config_weights.lstm)
                        votes['sell'] += probs[0] * w
                        votes['hold'] += probs[1] * w
                        votes['buy'] += probs[2] * w
                        model_predictions.append((w, probs))
                        
                    if 'attention' in models:
                        probs = models['attention'](tensor_in).cpu().numpy()[0]
                        w = get_weight('attention', config_weights.attention)
                        votes['sell'] += probs[0] * w
                        votes['hold'] += probs[1] * w
                        votes['buy'] += probs[2] * w
                        model_predictions.append((w, probs))

        except Exception as e:
            logger.error(f"Inference error for {symbol}: {e}")
            return {}

        # Normalize votes
        total_weight = sum(votes.values())
        if total_weight > 0:
            for k in votes:
                votes[k] /= total_weight

        # Determine Action
        best_action = max(votes, key=votes.get)
        confidence = votes[best_action]

        # --- Disagreement Penalty ---
        if model_predictions and config_weights.disagreement_penalty > 0:
            action_map = {'sell': 0, 'hold': 1, 'buy': 2}
            best_action_idx = action_map[best_action]
            
            # Only consider models that actually contributed
            relevant_probs = [p[best_action_idx] for w, p in model_predictions if w > 0]
            
            if len(relevant_probs) > 1:
                std_dev = np.std(relevant_probs)
                penalty = std_dev * config_weights.disagreement_penalty
                original_conf = confidence
                confidence = max(0.0, confidence - penalty)
                if penalty > 0.05:
                    logger.debug(f"Confidence penalized for {symbol}", 
                                 original=original_conf, 
                                 penalty=penalty, 
                                 std_dev=std_dev, 
                                 final=confidence)

        # Extract top features from XGBoost if available
        top_features = {}
        if 'gb' in models and hasattr(models['gb'], 'feature_importances_'):
            imps = models['gb'].feature_importances_
            cols = FeatureProcessor.get_feature_names(self.config)
            indices = np.argsort(imps)[::-1][:5]
            for idx in indices:
                if idx < len(cols):
                    top_features[cols[idx]] = float(imps[idx])

        return {
            'action': best_action,
            'confidence': round(confidence, 4),
            'model_version': meta['timestamp'],
            'active_weights': votes,
            'top_features': top_features,
            'metrics': meta.get('metrics', {}),
            'optimized_weights': optimized_weights if use_dynamic else None
        }

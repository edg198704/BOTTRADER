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
        ensemble_probs = np.tensordot(w, pred_stack, axes=([0],[0]))
        
        # Clip to avoid log(0)
        ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)
        
        # Calculate Log Loss
        score = log_loss(y_true, ensemble_probs)
        
        if score < best_score:
            best_score = score
            best_weights = {name: float(w[i]) for i, name in enumerate(model_names)}
            
    return best_weights

def _load_saved_models(symbol: str, model_path: str, config: AIEnsembleStrategyParams, device) -> Optional[Dict[str, Any]]:
    """
    Loads trained models and metadata from disk. 
    Returns None if not found or if feature configuration mismatches.
    """
    safe_symbol = symbol.replace('/', '_')
    load_dir = os.path.join(model_path, safe_symbol)
    
    if not os.path.exists(load_dir):
        return None

    try:
        meta_path = os.path.join(load_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return None

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # --- CONFIG CONSISTENCY CHECK ---
        # We check if the model's required features are available in the current config.
        # The model might use a subset (due to feature selection), which is fine.
        saved_active_features = meta.get('active_feature_columns', meta.get('feature_columns', []))
        
        # Get all currently available features from config
        current_available_features = FeatureProcessor.get_feature_names(config)
        
        # Check if saved features are a subset of current available features
        if not set(saved_active_features).issubset(set(current_available_features)):
            return None
        
        # The number of features the model expects
        saved_num_features = meta.get('num_features', len(saved_active_features))
        # --------------------------------

        models = {}
        # Load Sklearn Models
        for name in ['gb', 'technical']:
            path = os.path.join(load_dir, f"{name}.joblib")
            if os.path.exists(path):
                models[name] = joblib.load(path)

        # Load PyTorch Models
        if 'lstm' in config.ensemble_weights.dict():
            path = os.path.join(load_dir, "lstm.pth")
            if os.path.exists(path):
                lstm = LSTMPredictor(saved_num_features, config.hyperparameters.lstm.hidden_dim, 
                                     config.hyperparameters.lstm.num_layers, 
                                     config.hyperparameters.lstm.dropout).to(device)
                lstm.load_state_dict(torch.load(path, map_location=device))
                lstm.eval()
                models['lstm'] = lstm

        if 'attention' in config.ensemble_weights.dict():
            path = os.path.join(load_dir, "attention.pth")
            if os.path.exists(path):
                attn = AttentionNetwork(saved_num_features, config.hyperparameters.attention.hidden_dim,
                                        config.hyperparameters.attention.num_layers,
                                        config.hyperparameters.attention.nhead,
                                        config.hyperparameters.attention.dropout).to(device)
                attn.load_state_dict(torch.load(path, map_location=device))
                attn.eval()
                models['attention'] = attn

        return {
            'models': models,
            'meta': meta
        }

    except Exception:
        return None

def _evaluate_ensemble(models: Dict[str, Any], 
                       X: np.ndarray, 
                       y: np.ndarray, 
                       X_seq: np.ndarray, 
                       y_seq: np.ndarray, 
                       config: AIEnsembleStrategyParams, 
                       device, 
                       market_returns: np.ndarray, 
                       weights_override: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Evaluates a set of models on a dataset and returns performance metrics.
    """
    seq_len = config.features.sequence_length
    num_samples = len(y)
    ensemble_probs = np.zeros((num_samples, 3))
    
    # Helper to get weight
    config_weights = config.ensemble_weights
    def get_weight(name, default_weight):
        if weights_override:
            return weights_override.get(name, 0.0)
        return default_weight

    # Sklearn Models
    if 'gb' in models:
        probs = models['gb'].predict_proba(X)
        w = get_weight('gb', config_weights.xgboost)
        ensemble_probs += probs * w
        
    if 'technical' in models:
        probs = models['technical'].predict_proba(X)
        w = get_weight('technical', config_weights.technical_ensemble)
        ensemble_probs += probs * w
        
    # NN Models (Alignment required)
    # Sequence models lose (seq_len - 1) samples at the start
    valid_nn_indices = slice(seq_len - 1, None)
    
    if len(X_seq) > 0:
        def add_nn_probs(model_name, default_weight):
            if model_name in models:
                model = models[model_name]
                model.eval()
                with torch.no_grad():
                    inputs = torch.FloatTensor(X_seq).to(device)
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
    
    # Align market returns
    # X was aligned with y. y was aligned with market_returns.
    # We need the subset corresponding to valid_nn_indices
    eval_returns = market_returns[valid_nn_indices]
    
    if len(eval_returns) != len(signals):
        return {}

    # Strategy Returns
    strategy_returns = signals * eval_returns
    
    winning_returns = strategy_returns[strategy_returns > 0]
    losing_returns = strategy_returns[strategy_returns < 0]
    
    gross_profit = np.sum(winning_returns)
    gross_loss = np.abs(np.sum(losing_returns))
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0
    
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
    
    return {
        'profit_factor': float(profit_factor),
        'sharpe': float(sharpe),
        'total_return': float(np.sum(strategy_returns)),
        'win_rate': float(len(winning_returns) / len(strategy_returns)) if len(strategy_returns) > 0 else 0.0
    }

def train_ensemble_task(symbol: str, df: pd.DataFrame, config: AIEnsembleStrategyParams) -> bool:
    """
    Standalone function to train models in a separate process.
    Implements Champion/Challenger logic to ensure model quality.
    Returns True if training and saving were successful.
    """
    worker_logger = get_logger(f"trainer_{symbol}")
    worker_logger.info(f"Starting training task for {symbol} with {len(df)} records.")

    if not ML_AVAILABLE:
        worker_logger.error("ML libraries not available. Skipping training.")
        return False

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Prepare Data
        df_proc = FeatureProcessor.process_data(df, config)
        all_feature_names = list(df_proc.columns)
        labels = FeatureProcessor.create_labels(df, config)
        
        # Align Data
        common_index = df_proc.index.intersection(labels.index)
        if len(common_index) < 100:
            worker_logger.warning("Insufficient data after alignment.")
            return False
            
        X_full = df_proc.loc[common_index].values
        y = labels.loc[common_index].values.astype(int)

        # --- FEATURE SELECTION ---
        selected_indices = list(range(X_full.shape[1])) # Default to all
        active_feature_names = all_feature_names

        if config.features.use_feature_selection:
            worker_logger.info("Running Feature Selection...")
            try:
                # Use a lightweight Random Forest for selection
                selector = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
                selector.fit(X_full, y)
                importances = selector.feature_importances_
                
                # Select top K features
                k = min(config.features.max_active_features, len(all_feature_names))
                top_k_indices = np.argsort(importances)[::-1][:k]
                
                # Sort indices to maintain relative order (important for time series structure if any)
                selected_indices = sorted(top_k_indices)
                active_feature_names = [all_feature_names[i] for i in selected_indices]
                
                worker_logger.info(f"Selected {len(active_feature_names)} features out of {len(all_feature_names)}.")
                worker_logger.debug(f"Active features: {active_feature_names}")
                
            except Exception as e:
                worker_logger.error(f"Feature selection failed: {e}. Using all features.")

        # Subset X to selected features
        X = X_full[:, selected_indices]
        num_features = X.shape[1]

        # Calculate Market Returns for Evaluation (Next candle return)
        full_returns = df['close'].pct_change().shift(-1)
        market_returns = full_returns.loc[common_index].fillna(0.0).values

        # Split Train/Val
        split_idx = int(len(X) * (1 - config.training.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        returns_val = market_returns[split_idx:]

        # Prepare Sequences for NN Evaluation
        seq_len = config.features.sequence_length
        X_seq_val, y_seq_val = FeatureProcessor.create_sequences(X_val, y_val, seq_len)

        # --- CHAMPION EVALUATION ---
        champion_metrics = None
        champion_data = _load_saved_models(symbol, config.model_path, config, device)
        
        if champion_data:
            worker_logger.info("Evaluating existing Champion model on new validation data...")
            champion_models = champion_data['models']
            champion_weights = champion_data['meta'].get('optimized_weights')
            
            # Note: Champion might use different features. 
            # _load_saved_models ensures we can generate them, but we need to extract them correctly here.
            # However, for simplicity in this architecture, we re-evaluate the champion using ITS OWN feature set logic
            # if we were doing strict versioning. 
            # Current simplification: We only compare if the champion is compatible with current config superset.
            # If feature selection changed drastically, the champion might be using features we just discarded.
            # To handle this robustly, we would need to reconstruct the champion's X from df_proc using its meta.
            
            champ_features = champion_data['meta'].get('active_feature_columns', champion_data['meta'].get('feature_columns'))
            
            # Reconstruct X for champion
            try:
                champ_indices = [all_feature_names.index(f) for f in champ_features if f in all_feature_names]
                if len(champ_indices) == len(champ_features):
                    X_champ_full = X_full[:, champ_indices]
                    X_champ_val = X_champ_full[split_idx:]
                    X_champ_seq_val, y_champ_seq_val = FeatureProcessor.create_sequences(X_champ_val, y_val, seq_len)
                    
                    champion_metrics = _evaluate_ensemble(
                        champion_models, X_champ_val, y_val, X_champ_seq_val, y_champ_seq_val, 
                        config, device, returns_val, weights_override=champion_weights
                    )
                    worker_logger.info(f"Champion Metrics: PF={champion_metrics.get('profit_factor', 0):.2f}, Sharpe={champion_metrics.get('sharpe', 0):.4f}")
            except Exception as e:
                worker_logger.warning(f"Could not evaluate champion due to feature mismatch: {e}")

        # --- CHALLENGER TRAINING ---
        # Class Weighting
        sample_weights = None
        torch_weights = None
        if config.training.use_class_weighting:
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
            weight_map = {c: w for c, w in zip(unique_classes, class_weights)}
            final_weights = [weight_map.get(c, 1.0) for c in [0, 1, 2]]
            torch_weights = torch.FloatTensor(final_weights).to(device)

        # Create & Train Models
        models = _create_fresh_models(config, num_features, device)
        trained_models = {}
        feature_importance = {}

        # Train Sklearn/XGBoost
        for name, model in models.items():
            if name in ['lstm', 'attention']: continue
            worker_logger.info(f"Training {name}...")
            fit_params = {}
            if name == 'gb' and sample_weights is not None:
                fit_params['sample_weight'] = sample_weights

            if config.training.auto_tune_models and name == 'gb':
                param_dist = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
                model = _optimize_hyperparameters(model, X_train, y_train, param_dist, config.training.n_iter_search, worker_logger, fit_params)
            else:
                model.fit(X_train, y_train, **fit_params)
            
            trained_models[name] = model
            if hasattr(model, 'feature_importances_'):
                # Map importances back to active feature names
                imps = model.feature_importances_.tolist()
                if len(imps) == len(active_feature_names):
                    feature_importance[name] = dict(zip(active_feature_names, imps))

        # Train PyTorch
        X_seq_train, y_seq_train = FeatureProcessor.create_sequences(X_train, y_train, seq_len)
        if len(X_seq_train) > 0:
            train_tensor = TensorDataset(torch.FloatTensor(X_seq_train).to(device), torch.LongTensor(y_seq_train).to(device))
            train_loader = DataLoader(train_tensor, batch_size=config.training.batch_size, shuffle=True)
            
            for name in ['lstm', 'attention']:
                if name not in models: continue
                model = models[name]
                optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
                criterion = nn.CrossEntropyLoss(weight=torch_weights) if torch_weights is not None else nn.CrossEntropyLoss()
                
                worker_logger.info(f"Training {name} (PyTorch)...")
                model.train()
                for epoch in range(config.training.epochs):
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                trained_models[name] = model

        # Optimize Weights
        optimized_weights = {}
        if config.ensemble_weights.auto_tune:
            worker_logger.info("Optimizing ensemble weights...")
            val_preds = {}
            if 'gb' in trained_models: val_preds['gb'] = trained_models['gb'].predict_proba(X_val)
            if 'technical' in trained_models: val_preds['technical'] = trained_models['technical'].predict_proba(X_val)
            
            if len(X_seq_val) > 0:
                valid_indices = slice(seq_len - 1, None)
                for k in list(val_preds.keys()): val_preds[k] = val_preds[k][valid_indices]
                y_val_opt = y_val[valid_indices]
                for name in ['lstm', 'attention']:
                    if name in trained_models:
                        model = trained_models[name]
                        model.eval()
                        with torch.no_grad():
                            val_preds[name] = model(torch.FloatTensor(X_seq_val).to(device)).cpu().numpy()
            else:
                y_val_opt = y_val
            
            if val_preds:
                optimized_weights = _optimize_ensemble_weights(val_preds, y_val_opt)

        # --- CHALLENGER EVALUATION ---
        challenger_metrics = _evaluate_ensemble(
            trained_models, X_val, y_val, X_seq_val, y_seq_val, 
            config, device, returns_val, weights_override=optimized_weights
        )
        
        pf = challenger_metrics.get('profit_factor', 0)
        sharpe = challenger_metrics.get('sharpe', 0)
        worker_logger.info(f"Challenger Metrics: PF={pf:.2f}, Sharpe={sharpe:.4f}")

        # --- GATING & COMPARISON ---
        # 1. Absolute Gates
        if pf < config.training.min_profit_factor:
            worker_logger.warning(f"Challenger rejected: PF {pf:.2f} < {config.training.min_profit_factor}")
            return False
        if sharpe < config.training.min_sharpe_ratio:
            worker_logger.warning(f"Challenger rejected: Sharpe {sharpe:.4f} < {config.training.min_sharpe_ratio}")
            return False

        # 2. Champion vs Challenger
        if champion_metrics:
            champ_sharpe = champion_metrics.get('sharpe', 0)
            improvement_needed = config.training.min_improvement_pct
            
            is_better = False
            if champ_sharpe <= 0:
                if sharpe > champ_sharpe + 0.05: 
                    is_better = True
            else:
                if sharpe > champ_sharpe * (1 + improvement_needed):
                    is_better = True
            
            if not is_better:
                worker_logger.warning(f"Challenger failed to beat Champion. Champ Sharpe: {champ_sharpe:.4f}, Challenger: {sharpe:.4f}")
                return False
            else:
                worker_logger.info(f"Challenger defeated Champion! ({sharpe:.4f} vs {champ_sharpe:.4f})")

        # --- SAVE ARTIFACTS ---
        safe_symbol = symbol.replace('/', '_')
        final_dir = os.path.join(config.model_path, safe_symbol)
        temp_dir = os.path.join(config.model_path, f"{safe_symbol}_temp")
        
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            for name, model in trained_models.items():
                if name in ['lstm', 'attention']:
                    torch.save(model.state_dict(), os.path.join(temp_dir, f"{name}.pth"))
                else:
                    joblib.dump(model, os.path.join(temp_dir, f"{name}.joblib"))
            
            meta = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {'ensemble': challenger_metrics},
                'feature_importance': feature_importance,
                'feature_columns': config.feature_columns, # Full list from config
                'active_feature_columns': active_feature_names, # Actual subset used
                'num_features': num_features,
                'optimized_weights': optimized_weights
            }
            with open(os.path.join(temp_dir, "metadata.json"), 'w') as f:
                json.dump(meta, f, indent=2)
            
            if os.path.exists(final_dir): shutil.rmtree(final_dir)
            os.rename(temp_dir, final_dir)
            
            worker_logger.info(f"Training complete. Artifacts saved to {final_dir}")
            return True
            
        except Exception as e:
            worker_logger.error(f"Failed to save artifacts: {e}")
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
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
        return symbol in self.symbol_models

    def get_last_training_time(self, symbol: str) -> Optional[datetime]:
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

        try:
            loaded_data = _load_saved_models(symbol, self.config.model_path, self.config, self.device)
            
            if loaded_data:
                self.symbol_models[symbol] = loaded_data
                logger.info(f"Models reloaded successfully for {symbol}")
            else:
                # If load failed (e.g. config mismatch), remove existing model to force retrain
                if symbol in self.symbol_models:
                    del self.symbol_models[symbol]
                logger.warning(f"Could not load valid models for {symbol} (possible config mismatch).")

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
        
        # --- FILTER FEATURES ---
        # The model might use a subset of features defined in metadata
        active_features = meta.get('active_feature_columns', meta.get('feature_columns'))
        
        # Ensure we have all required columns
        if not set(active_features).issubset(set(df_proc.columns)):
            logger.error(f"Inference failed: Missing features for {symbol}. Model expects {active_features}")
            return {}
            
        # Subset to active features in correct order
        df_proc = df_proc[active_features]
        # -----------------------

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
                confidence = max(0.0, confidence - penalty)

        # Extract top features from XGBoost if available
        top_features = {}
        if 'gb' in models and hasattr(models['gb'], 'feature_importances_'):
            # Need to map back to names since we subsetted
            imps = models['gb'].feature_importances_
            cols = active_features
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

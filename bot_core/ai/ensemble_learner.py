import os
import joblib
import json
import logging
import shutil
import numpy as np
import pandas as pd
import asyncio
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from bot_core.logger import get_logger
from bot_core.config import AIEnsembleStrategyParams
from bot_core.ai.models import LSTMPredictor, AttentionNetwork
from bot_core.ai.feature_processor import FeatureProcessor
from bot_core.ai.regime_detector import MarketRegimeDetector

# ML Imports with safe fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, classification_report, log_loss
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
    from sklearn.isotonic import IsotonicRegression
    from xgboost import XGBClassifier
    from scipy.optimize import minimize
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
    class IsotonicRegression: pass
    class IsolationForest: pass
    def precision_score(*args, **kwargs): return 0.0
    def log_loss(*args, **kwargs): return 0.0
    def classification_report(*args, **kwargs): return ""
    def compute_class_weight(*args, **kwargs): return []
    def compute_sample_weight(*args, **kwargs): return []
    def minimize(*args, **kwargs): return None
    class nn:
        class Module: pass
    class optim: pass
    class TensorDataset: pass
    class DataLoader: pass

logger = get_logger(__name__)

# --- Helper Classes ---

class MulticlassCalibrator:
    """Calibrates probabilities for multiclass classification using One-vs-Rest Isotonic Regression."""
    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrators = {} # class_idx -> regressor

    def fit(self, X_probs, y):
        # X_probs: (N, n_classes)
        # y: (N,)
        n_classes = X_probs.shape[1]
        for i in range(n_classes):
            # Binary target: 1 if sample is class i, else 0
            y_binary = (y == i).astype(int)
            # Input: probability of class i
            X_col = X_probs[:, i]
            
            reg = IsotonicRegression(out_of_bounds='clip')
            reg.fit(X_col, y_binary)
            self.calibrators[i] = reg
            
    def predict(self, X_probs):
        n_samples, n_classes = X_probs.shape
        calibrated = np.zeros_like(X_probs)
        
        for i in range(n_classes):
            if i in self.calibrators:
                calibrated[:, i] = self.calibrators[i].predict(X_probs[:, i])
            else:
                calibrated[:, i] = X_probs[:, i]
        
        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        return calibrated / row_sums

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

def _train_torch_model(model, X_train, y_train, X_val, y_val, config, device, class_weights=None):
    """
    Robust training loop for PyTorch models with Early Stopping and LR Scheduling.
    """
    if len(X_train) == 0 or len(X_val) == 0:
        return model

    # Hyperparameters
    epochs = config.training.epochs
    batch_size = config.training.batch_size
    lr = config.training.learning_rate
    patience = config.training.early_stopping_patience
    
    # Setup
    train_tensor = TensorDataset(torch.FloatTensor(X_train).to(device), torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    
    # Validation tensor (full batch for eval)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(1, patience // 2))

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            outputs = model(b_x)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
        
        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Early stopping triggered
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model

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

def _optimize_ensemble_weights(predictions: Dict[str, np.ndarray], y_true: np.ndarray, method: str = 'slsqp', sample_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Finds optimal weights for the ensemble using Scipy Optimization (SLSQP) or Random Search.
    Minimizes Log Loss, optionally weighted by sample_weights.
    predictions: Dict of model_name -> prob_array (N, 3)
    y_true: array (N,)
    sample_weights: array (N,) or None
    """
    model_names = list(predictions.keys())
    if not model_names:
        return {}
    
    n_models = len(model_names)
    # Convert dict to list of arrays for fast vectorized math
    # shape: (num_models, num_samples, num_classes)
    pred_stack = np.array([predictions[name] for name in model_names])
    
    # Default equal weights
    best_weights = {name: 1.0/n_models for name in model_names}

    def loss_func(weights):
        # Normalize weights to sum to 1 (just in case, though constraints handle this)
        w = np.array(weights)
        w = w / np.sum(w)
        
        # Weighted average: sum(w[i] * pred_stack[i])
        ensemble_probs = np.tensordot(w, pred_stack, axes=([0],[0]))
        
        # Clip to avoid log(0)
        ensemble_probs = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)
        return log_loss(y_true, ensemble_probs, sample_weight=sample_weights)

    if method == 'slsqp' and ML_AVAILABLE:
        try:
            # Constraints: sum(w) = 1
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            # Bounds: 0 <= w <= 1
            bounds = [(0, 1)] * n_models
            # Initial guess: equal weights
            x0 = np.ones(n_models) / n_models
            
            res = minimize(loss_func, x0, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-4)
            
            if res.success:
                optimized_w = res.x / np.sum(res.x) # Ensure strict sum to 1
                best_weights = {name: float(optimized_w[i]) for i, name in enumerate(model_names)}
                return best_weights
        except Exception as e:
            # Fallback to random search if optimization fails
            pass

    # Fallback: Random Search (Monte Carlo)
    best_score = float('inf')
    iterations = 1000
    
    for _ in range(iterations):
        w = np.random.dirichlet(np.ones(n_models))
        score = loss_func(w)
        
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
        saved_active_features = meta.get('active_feature_columns', meta.get('feature_columns', []))
        current_available_features = FeatureProcessor.get_feature_names(config)
        
        if not set(saved_active_features).issubset(set(current_available_features)):
            return None
        
        saved_num_features = meta.get('num_features', len(saved_active_features))
        # --------------------------------

        models = {}
        # Load Sklearn Models
        for name in ['gb', 'technical']:
            path = os.path.join(load_dir, f"{name}.joblib")
            if os.path.exists(path):
                models[name] = joblib.load(path)

        # Load Calibrator
        calibrator = None
        calib_path = os.path.join(load_dir, "calibrator.joblib")
        if os.path.exists(calib_path):
            calibrator = joblib.load(calib_path)

        # Load Drift Detector
        drift_detector = None
        drift_path = os.path.join(load_dir, "drift_detector.joblib")
        if os.path.exists(drift_path):
            drift_detector = joblib.load(drift_path)

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
            'calibrator': calibrator,
            'drift_detector': drift_detector,
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
                       weights_override: Optional[Dict[str, float]] = None,
                       calibrator: Optional[MulticlassCalibrator] = None) -> Dict[str, Any]:
    """
    Evaluates a set of models on a dataset and returns performance metrics.
    Applies calibration if provided.
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
    valid_nn_indices = slice(seq_len - 1, None)
    
    if len(X_seq) > 0:
        def add_nn_probs(model_name, default_weight):
            if model_name in models:
                model = models[model_name]
                model.eval()
                with torch.no_grad():
                    inputs = torch.FloatTensor(X_seq).to(device)
                    probs = model(inputs).cpu().numpy()
                    target_len = len(ensemble_probs[valid_nn_indices])
                    if len(probs) == target_len:
                        w = get_weight(model_name, default_weight)
                        ensemble_probs[valid_nn_indices] += probs * w
        
        add_nn_probs('lstm', config_weights.lstm)
        add_nn_probs('attention', config_weights.attention)

    # Determine Final Predictions
    eval_probs = ensemble_probs[valid_nn_indices]
    
    # Apply Calibration if available
    if calibrator:
        eval_probs = calibrator.predict(eval_probs)

    final_preds = np.argmax(eval_probs, axis=1) # 0, 1, 2
    
    # Map to Signal: 0->-1 (Short), 1->0 (Hold), 2->1 (Long)
    signals = np.zeros_like(final_preds)
    signals[final_preds == 0] = -1
    signals[final_preds == 2] = 1
    
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

def train_ensemble_task(symbol: str, df: pd.DataFrame, config: AIEnsembleStrategyParams, leader_df: Optional[pd.DataFrame] = None) -> bool:
    """
    Standalone function to train models in a separate process.
    Implements Walk-Forward Validation (Stacking) with Purging for robust weight optimization.
    """
    worker_logger = get_logger(f"trainer_{symbol}")
    worker_logger.info(f"Starting training task for {symbol} with {len(df)} records.")

    if not ML_AVAILABLE:
        worker_logger.error("ML libraries not available. Skipping training.")
        return False

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Prepare Data (Including Leader Features)
        df_proc = FeatureProcessor.process_data(df, config, leader_df=leader_df)
        all_feature_names = list(df_proc.columns)
        labels = FeatureProcessor.create_labels(df, config)
        
        common_index = df_proc.index.intersection(labels.index)
        if len(common_index) < 200:
            worker_logger.warning("Insufficient data after alignment.")
            return False
            
        X_full = df_proc.loc[common_index].values
        y = labels.loc[common_index].values.astype(int)

        # --- FEATURE SELECTION ---
        selected_indices = list(range(X_full.shape[1]))
        active_feature_names = all_feature_names

        if config.features.use_feature_selection:
            try:
                selector = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
                selector.fit(X_full, y)
                importances = selector.feature_importances_
                k = min(config.features.max_active_features, len(all_feature_names))
                top_k_indices = np.argsort(importances)[::-1][:k]
                selected_indices = sorted(top_k_indices)
                active_feature_names = [all_feature_names[i] for i in selected_indices]
            except Exception as e:
                worker_logger.error(f"Feature selection failed: {e}. Using all features.")

        X = X_full[:, selected_indices]
        num_features = X.shape[1]

        full_returns = df['close'].pct_change().shift(-1)
        market_returns = full_returns.loc[common_index].fillna(0.0).values

        # --- DATA SPLITTING (Dev / Test) ---
        # Dev set is used for CV (Stacking) and Final Training
        # Test set is strictly for Champion/Challenger evaluation
        test_split_idx = int(len(X) * (1 - config.training.validation_split))
        
        X_dev, X_test = X[:test_split_idx], X[test_split_idx:]
        y_dev, y_test = y[:test_split_idx], y[test_split_idx:]
        returns_test = market_returns[test_split_idx:]
        
        # Prepare Test Sequences
        seq_len = config.features.sequence_length
        X_seq_test, y_seq_test = FeatureProcessor.create_sequences(X_test, y_test, seq_len)

        # --- CHAMPION EVALUATION ---
        champion_metrics = None
        champion_data = _load_saved_models(symbol, config.model_path, config, device)
        
        if champion_data:
            champion_models = champion_data['models']
            champion_weights = champion_data['meta'].get('optimized_weights')
            champion_calibrator = champion_data.get('calibrator')
            champ_features = champion_data['meta'].get('active_feature_columns', champion_data['meta'].get('feature_columns'))
            
            try:
                champ_indices = [all_feature_names.index(f) for f in champ_features if f in all_feature_names]
                if len(champ_indices) == len(champ_features):
                    X_champ_full = X_full[:, champ_indices]
                    X_champ_test = X_champ_full[test_split_idx:]
                    X_champ_seq_test, y_champ_seq_test = FeatureProcessor.create_sequences(X_champ_test, y_test, seq_len)
                    
                    champion_metrics = _evaluate_ensemble(
                        champion_models, X_champ_test, y_test, X_champ_seq_test, y_champ_seq_test, 
                        config, device, returns_test, weights_override=champion_weights, calibrator=champion_calibrator
                    )
                    worker_logger.info(f"Champion Metrics: PF={champion_metrics.get('profit_factor', 0):.2f}, Sharpe={champion_metrics.get('sharpe', 0):.4f}")
            except Exception as e:
                worker_logger.warning(f"Could not evaluate champion: {e}")

        # --- WALK-FORWARD VALIDATION (STACKING) WITH PURGING ---
        # Generate Out-Of-Sample predictions for X_dev to optimize weights
        
        oos_predictions = {} # model_name -> list of arrays
        oos_indices = []
        
        # Only perform CV if we have enough data
        use_cv = len(X_dev) > 500 and config.training.cv_splits > 1
        
        if use_cv:
            worker_logger.info(f"Starting Walk-Forward Validation with {config.training.cv_splits} splits (Purged)...")
            tscv = TimeSeriesSplit(n_splits=config.training.cv_splits)
            horizon = config.features.labeling_horizon
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev)):
                # PURGING: Remove samples from end of train set that overlap with validation labels
                # The labels in train_idx[-horizon:] contain information about prices in val_idx
                if len(train_idx) > horizon:
                    train_idx = train_idx[:-horizon]
                
                X_fold_train, X_fold_val = X_dev[train_idx], X_dev[val_idx]
                y_fold_train, y_fold_val = y_dev[train_idx], y_dev[val_idx]
                
                # Create fresh models for this fold (Fast training, no hyperparam opt)
                fold_models = _create_fresh_models(config, num_features, device)
                
                # Train Fold Models
                # Sklearn
                for name in ['gb', 'technical']:
                    fold_models[name].fit(X_fold_train, y_fold_train)
                    preds = fold_models[name].predict_proba(X_fold_val)
                    if name not in oos_predictions: oos_predictions[name] = []
                    oos_predictions[name].append(preds)
                
                # PyTorch
                X_seq_tr, y_seq_tr = FeatureProcessor.create_sequences(X_fold_train, y_fold_train, seq_len)
                X_seq_val, y_seq_val = FeatureProcessor.create_sequences(X_fold_val, y_fold_val, seq_len)
                
                if len(X_seq_tr) > 0 and len(X_seq_val) > 0:
                    for name in ['lstm', 'attention']:
                        model = fold_models[name]
                        # Use robust training loop even for CV, but with fewer epochs if desired
                        # Here we use full config epochs but rely on early stopping to cut it short
                        _train_torch_model(model, X_seq_tr, y_seq_tr, X_seq_val, y_seq_val, config, device)
                        
                        model.eval()
                        with torch.no_grad():
                            preds = model(torch.FloatTensor(X_seq_val).to(device)).cpu().numpy()
                            if name not in oos_predictions: oos_predictions[name] = []
                            oos_predictions[name].append(preds)
                
                # Store indices for alignment (adjust for sequence length loss if needed)
                # Note: X_seq_val corresponds to X_fold_val[seq_len-1:]
                # We align everything to the sequence-valid portion
                valid_slice = slice(seq_len - 1, None)
                oos_indices.append(val_idx[valid_slice])

        # --- OPTIMIZE WEIGHTS & CALIBRATE ---
        optimized_weights = {}
        regime_weights = {}
        calibrator = None
        
        if use_cv and oos_indices:
            try:
                # Concatenate OOS predictions
                combined_indices = np.concatenate(oos_indices)
                y_oos = y_dev[combined_indices]
                
                # Calculate sample weights for OOS data if class weighting is enabled
                oos_sample_weights = None
                if config.training.use_class_weighting:
                    oos_sample_weights = compute_sample_weight(class_weight='balanced', y=y_oos)
                
                combined_preds = {}
                valid_models = []
                for name, pred_list in oos_predictions.items():
                    # Ensure we have predictions for all folds
                    if len(pred_list) == len(oos_indices):
                        # Slice each prediction array to match the sequence-valid portion
                        # Sklearn preds are full length of val_idx, NN preds are already sliced
                        # We need to slice Sklearn preds to match NN/indices
                        sliced_list = []
                        for i, p in enumerate(pred_list):
                            if name in ['gb', 'technical']:
                                sliced_list.append(p[seq_len-1:])
                            else:
                                sliced_list.append(p)
                        
                        combined_preds[name] = np.concatenate(sliced_list)
                        valid_models.append(name)
                
                if combined_preds:
                    # 1. Global Weight Optimization
                    if config.ensemble_weights.auto_tune:
                        optimized_weights = _optimize_ensemble_weights(
                            combined_preds, y_oos, 
                            method=config.ensemble_weights.optimization_method,
                            sample_weights=oos_sample_weights
                        )
                    else:
                        cw = config.ensemble_weights
                        optimized_weights = {
                            'gb': cw.xgboost, 'technical': cw.technical_ensemble,
                            'lstm': cw.lstm, 'attention': cw.attention
                        }
                    
                    # 2. Regime-Specific Weight Optimization
                    if config.ensemble_weights.use_regime_specific_weights:
                        worker_logger.info("Optimizing weights per market regime...")
                        # Instantiate detector to get regimes for the OOS data
                        detector = MarketRegimeDetector(config)
                        
                        dev_indices = common_index[:test_split_idx]
                        oos_timestamps = dev_indices[combined_indices]
                        
                        # Get regimes for the full DF, then slice
                        full_regimes = detector.get_regime_series(df)
                        oos_regimes = full_regimes.loc[oos_timestamps].values
                        
                        unique_regimes = np.unique(oos_regimes)
                        for regime in unique_regimes:
                            # Filter data for this regime
                            mask = (oos_regimes == regime)
                            if np.sum(mask) > 50: # Minimum samples to optimize
                                regime_y = y_oos[mask]
                                regime_preds = {k: v[mask] for k, v in combined_preds.items()}
                                
                                # Also filter sample weights if they exist
                                regime_sw = oos_sample_weights[mask] if oos_sample_weights is not None else None
                                
                                r_weights = _optimize_ensemble_weights(
                                    regime_preds, regime_y, 
                                    method=config.ensemble_weights.optimization_method,
                                    sample_weights=regime_sw
                                )
                                regime_weights[regime] = r_weights
                                worker_logger.info(f"Optimized weights for {regime}: {r_weights}")
                            else:
                                worker_logger.debug(f"Insufficient samples for {regime} weight optimization, using global.")

                    # Fit Calibrator on OOS data
                    if config.training.calibration_method != 'none':
                        worker_logger.info(f"Calibrating ensemble using {config.training.calibration_method}...")
                        pred_stack = np.array([combined_preds[name] for name in valid_models])
                        weights = np.array([optimized_weights.get(name, 0.0) for name in valid_models])
                        if weights.sum() > 0: weights /= weights.sum()
                        ensemble_probs_oos = np.tensordot(weights, pred_stack, axes=([0],[0]))
                        
                        calibrator = MulticlassCalibrator(method=config.training.calibration_method)
                        calibrator.fit(ensemble_probs_oos, y_oos)
            except Exception as e:
                worker_logger.error(f"CV Weight Optimization failed: {e}", exc_info=True)

        # Fallback if CV failed or not used
        if not optimized_weights:
             cw = config.ensemble_weights
             optimized_weights = {
                'gb': cw.xgboost, 'technical': cw.technical_ensemble,
                'lstm': cw.lstm, 'attention': cw.attention
            }

        # --- FINAL TRAINING (Full Dev Set) ---
        worker_logger.info("Training final models on full development set...")
        final_models = _create_fresh_models(config, num_features, device)
        trained_models = {}
        feature_importance = {}
        
        # Weights for final training
        sample_weights = None
        torch_weights = None
        if config.training.use_class_weighting:
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_dev)
            unique_classes = np.unique(y_dev)
            class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_dev)
            weight_map = {c: w for c, w in zip(unique_classes, class_weights)}
            final_weights = [weight_map.get(c, 1.0) for c in [0, 1, 2]]
            torch_weights = torch.FloatTensor(final_weights).to(device)

        # Train Sklearn/XGBoost
        for name, model in final_models.items():
            if name in ['lstm', 'attention']: continue
            fit_params = {}
            if name == 'gb' and sample_weights is not None:
                fit_params['sample_weight'] = sample_weights

            if config.training.auto_tune_models and name == 'gb':
                param_dist = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
                model = _optimize_hyperparameters(model, X_dev, y_dev, param_dist, config.training.n_iter_search, worker_logger, fit_params)
            else:
                model.fit(X_dev, y_dev, **fit_params)
            
            trained_models[name] = model
            if hasattr(model, 'feature_importances_'):
                imps = model.feature_importances_.tolist()
                if len(imps) == len(active_feature_names):
                    feature_importance[name] = dict(zip(active_feature_names, imps))

        # Train PyTorch
        # We need a validation set for early stopping. 
        # We can use the last chunk of X_dev as internal validation.
        val_size = int(len(X_dev) * 0.15) # 15% internal validation
        if val_size > seq_len:
            X_final_train = X_dev[:-val_size]
            y_final_train = y_dev[:-val_size]
            X_final_val = X_dev[-val_size:]
            y_final_val = y_dev[-val_size:]
            
            X_seq_tr, y_seq_tr = FeatureProcessor.create_sequences(X_final_train, y_final_train, seq_len)
            X_seq_val, y_seq_val = FeatureProcessor.create_sequences(X_final_val, y_final_val, seq_len)
            
            for name in ['lstm', 'attention']:
                if name not in final_models: continue
                model = final_models[name]
                _train_torch_model(model, X_seq_tr, y_seq_tr, X_seq_val, y_seq_val, config, device, torch_weights)
                trained_models[name] = model
        else:
            # Fallback if data too small: train without early stopping on full set
            X_seq_dev, y_seq_dev = FeatureProcessor.create_sequences(X_dev, y_dev, seq_len)
            if len(X_seq_dev) > 0:
                train_tensor = TensorDataset(torch.FloatTensor(X_seq_dev).to(device), torch.LongTensor(y_seq_dev).to(device))
                train_loader = DataLoader(train_tensor, batch_size=config.training.batch_size, shuffle=True)
                
                for name in ['lstm', 'attention']:
                    if name not in final_models: continue
                    model = final_models[name]
                    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
                    criterion = nn.CrossEntropyLoss(weight=torch_weights) if torch_weights is not None else nn.CrossEntropyLoss()
                    
                    model.train()
                    for epoch in range(config.training.epochs):
                        for batch_X, batch_y in train_loader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                    trained_models[name] = model

        # --- DRIFT DETECTION TRAINING ---
        drift_detector = None
        if config.drift.enabled:
            worker_logger.info("Training Drift Detector (Isolation Forest)...")
            try:
                drift_detector = IsolationForest(
                    contamination=config.drift.contamination, 
                    random_state=42, 
                    n_jobs=1
                )
                drift_detector.fit(X_dev)
            except Exception as e:
                worker_logger.error(f"Drift detector training failed: {e}")

        # --- CHALLENGER EVALUATION (Using Test Set) ---
        challenger_metrics = _evaluate_ensemble(
            trained_models, X_test, y_test, X_seq_test, y_seq_test, 
            config, device, returns_test, weights_override=optimized_weights, calibrator=calibrator
        )
        
        pf = challenger_metrics.get('profit_factor', 0)
        sharpe = challenger_metrics.get('sharpe', 0)
        worker_logger.info(f"Challenger Metrics: PF={pf:.2f}, Sharpe={sharpe:.4f}")

        # --- GATING ---
        if pf < config.training.min_profit_factor:
            worker_logger.warning(f"Challenger rejected: PF {pf:.2f} < {config.training.min_profit_factor}")
            return False
        if sharpe < config.training.min_sharpe_ratio:
            worker_logger.warning(f"Challenger rejected: Sharpe {sharpe:.4f} < {config.training.min_sharpe_ratio}")
            return False

        if champion_metrics:
            champ_sharpe = champion_metrics.get('sharpe', 0)
            improvement_needed = config.training.min_improvement_pct
            is_better = False
            if champ_sharpe <= 0:
                if sharpe > champ_sharpe + 0.05: is_better = True
            else:
                if sharpe > champ_sharpe * (1 + improvement_needed): is_better = True
            
            if not is_better:
                worker_logger.warning(f"Challenger failed to beat Champion. Champ Sharpe: {champ_sharpe:.4f}, Challenger: {sharpe:.4f}")
                return False

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
            
            if calibrator:
                joblib.dump(calibrator, os.path.join(temp_dir, "calibrator.joblib"))
            
            if drift_detector:
                joblib.dump(drift_detector, os.path.join(temp_dir, "drift_detector.joblib"))

            meta = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {'ensemble': challenger_metrics},
                'feature_importance': feature_importance,
                'feature_columns': config.feature_columns,
                'active_feature_columns': active_feature_names,
                'num_features': num_features,
                'optimized_weights': optimized_weights,
                'regime_weights': regime_weights
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
    Uses a ThreadPoolExecutor to prevent blocking the main asyncio loop during CPU-bound operations.
    """
    def __init__(self, config: AIEnsembleStrategyParams):
        self.config = config
        self.symbol_models: Dict[str, Dict[str, Any]] = {}
        self.device = torch.device("cuda" if ML_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info(f"EnsembleLearner initialized. Device: {self.device}")

    async def close(self):
        self.executor.shutdown(wait=True)
        logger.info("EnsembleLearner executor shut down.")

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
        if not ML_AVAILABLE:
            return

        try:
            loop = asyncio.get_running_loop()
            loaded_data = await loop.run_in_executor(
                self.executor, 
                _load_saved_models, 
                symbol, 
                self.config.model_path, 
                self.config, 
                self.device
            )
            
            if loaded_data:
                self.symbol_models[symbol] = loaded_data
                logger.info(f"Models reloaded successfully for {symbol}")
            else:
                if symbol in self.symbol_models:
                    del self.symbol_models[symbol]
                logger.warning(f"Could not load valid models for {symbol} (possible config mismatch).")

        except Exception as e:
            logger.error(f"Failed to reload models for {symbol}: {e}")

    def _predict_sync(self, df: pd.DataFrame, symbol: str, regime: Optional[str] = None, leader_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        entry = self.symbol_models.get(symbol)
        if not entry:
            return {}

        models = entry['models']
        meta = entry['meta']
        calibrator = entry.get('calibrator')
        drift_detector = entry.get('drift_detector')
        
        # Prepare Data (Including Leader Features)
        df_proc = FeatureProcessor.process_data(df, self.config, leader_df=leader_df)
        active_features = meta.get('active_feature_columns', meta.get('feature_columns'))
        
        if not set(active_features).issubset(set(df_proc.columns)):
            logger.error(f"Inference failed: Missing features for {symbol}. Model expects {active_features}")
            return {}
            
        df_proc = df_proc[active_features]
        X = df_proc.iloc[-1:].values
        
        # --- DRIFT DETECTION ---
        is_anomaly = False
        anomaly_score = 0.0
        if drift_detector and self.config.drift.enabled:
            try:
                # predict returns -1 for outlier, 1 for inlier
                pred = drift_detector.predict(X)[0]
                is_anomaly = (pred == -1)
                # decision_function returns negative for outliers
                anomaly_score = drift_detector.decision_function(X)[0]
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")

        seq_len = self.config.features.sequence_length
        if len(df_proc) >= seq_len:
            X_seq = df_proc.iloc[-seq_len:].values.reshape(1, seq_len, -1)
        else:
            X_seq = None

        votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        config_weights = self.config.ensemble_weights
        
        # --- Weight Selection Logic ---
        optimized_weights = meta.get('optimized_weights')
        regime_weights = meta.get('regime_weights', {})
        
        use_dynamic = config_weights.auto_tune and optimized_weights
        active_weight_map = optimized_weights if use_dynamic else {}
        
        # Override with regime specific weights if available
        if config_weights.use_regime_specific_weights and regime and regime in regime_weights:
            active_weight_map = regime_weights[regime]
        
        def get_weight(name, default_weight):
            if active_weight_map:
                return active_weight_map.get(name, 0.0)
            return default_weight

        model_predictions = []
        
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

        # --- APPLY CALIBRATION ---
        if calibrator:
            # Calibrator expects (N, 3) array
            raw_probs = np.array([[votes['sell'], votes['hold'], votes['buy']]])
            calibrated_probs = calibrator.predict(raw_probs)[0]
            votes['sell'] = calibrated_probs[0]
            votes['hold'] = calibrated_probs[1]
            votes['buy'] = calibrated_probs[2]

        # Determine Action
        best_action = max(votes, key=votes.get)
        confidence = votes[best_action]

        # --- Disagreement Penalty ---
        if model_predictions and config_weights.disagreement_penalty > 0:
            action_map = {'sell': 0, 'hold': 1, 'buy': 2}
            best_action_idx = action_map[best_action]
            relevant_probs = [p[best_action_idx] for w, p in model_predictions if w > 0]
            if len(relevant_probs) > 1:
                std_dev = np.std(relevant_probs)
                penalty = std_dev * config_weights.disagreement_penalty
                confidence = max(0.0, confidence - penalty)

        top_features = {}
        if 'gb' in models and hasattr(models['gb'], 'feature_importances_'):
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
            'optimized_weights': active_weight_map if active_weight_map else None,
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score)
        }

    async def predict(self, df: pd.DataFrame, symbol: str, regime: Optional[str] = None, leader_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if symbol not in self.symbol_models:
            await self.reload_models(symbol)
            if symbol not in self.symbol_models:
                return {}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._predict_sync, df, symbol, regime, leader_df)

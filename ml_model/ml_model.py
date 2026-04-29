# ml_model/ml_model.py
import logging
from collections import deque
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    xgb = None  # type: ignore


def ml_available() -> bool:
    return _HAS_XGB


class RollingML:
    """
    Rolling XGBoost overlay with walk-forward validation and periodic retraining.
    - 15-feature set (indicators + price momentum + EMA ratios)
    - 80/20 train/test split — logs out-of-sample accuracy on each fit
    - Retrain trigger: call fit() again after N new bars
    """

    FEATURE_COLS = [
        "rsi", "adx", "macd_hist", "macd_norm",
        "bb_pct", "atr_pct", "vol_osc",
        "stoch_k", "stoch_d",
        "ret_1", "ret_3", "ret_5",
        "ema21_vs_50", "ema50_vs_200",
        "close_vs_vwap",
    ]

    def __init__(self, max_train_rows: int = 5_000, target_horizon: int = 3, retrain_every: int = 100):
        self.max_train_rows = int(max_train_rows)
        self.target_horizon = int(target_horizon)
        self.retrain_every = int(retrain_every)
        self.model = None
        self.trained = False
        self._bars_since_train = 0
        self._last_train_len = 0
        self._close_window: deque = deque(maxlen=6)

        if not _HAS_XGB:
            logging.info("RollingML: xgboost not available; ML disabled.")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].replace(0, np.nan)
        ema200 = df.get("EMA_200", close)
        ema50 = df.get("EMA_50", close)
        ema21 = df.get("EMA_21", close)
        vwap = df.get("VWAP", close)

        X = pd.DataFrame({
            "rsi":           df.get("RSI", pd.Series(50, index=df.index)),
            "adx":           df.get("ADX", pd.Series(0, index=df.index)),
            "macd_hist":     df.get("MACD_HIST", pd.Series(0, index=df.index)),
            "macd_norm":     (df.get("MACD", pd.Series(0, index=df.index))
                              / close).fillna(0),
            "bb_pct":        df.get("BB_PCT", pd.Series(0.5, index=df.index)),
            "atr_pct":       df.get("ATR_pct", pd.Series(0, index=df.index)),
            "vol_osc":       df.get("Vol_Osc", pd.Series(0, index=df.index)),
            "stoch_k":       df.get("Stoch_RSI_K", pd.Series(50, index=df.index)),
            "stoch_d":       df.get("Stoch_RSI_D", pd.Series(50, index=df.index)),
            "ret_1":         close.pct_change(1),
            "ret_3":         close.pct_change(3),
            "ret_5":         close.pct_change(5),
            "ema21_vs_50":   (ema21 / ema50.replace(0, np.nan) - 1.0).fillna(0),
            "ema50_vs_200":  (ema50 / ema200.replace(0, np.nan) - 1.0).fillna(0),
            "close_vs_vwap": (close / vwap.replace(0, np.nan) - 1.0).fillna(0),
        }, index=df.index)

        return X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    def _features_and_target(self, df: pd.DataFrame):
        X = self._build_features(df)
        ret_fwd = df["Close"].shift(-self.target_horizon) / df["Close"] - 1.0
        y = (ret_fwd > 0).astype(int)

        # Drop last horizon rows (no label yet)
        X = X.iloc[:-self.target_horizon].tail(self.max_train_rows)
        y = y.iloc[:-self.target_horizon].tail(self.max_train_rows)
        return X, y

    def fit(self, df: pd.DataFrame):
        if not _HAS_XGB:
            self.trained = False
            return

        if df is None or len(df) < 200:
            self.trained = False
            return

        try:
            X, y = self._features_and_target(df)
            if X.empty or y.empty or y.nunique() < 2:
                self.trained = False
                return

            # Walk-forward split: train on first 80%, validate on last 20%
            split = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

            dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
            dval = xgb.DMatrix(X_val.values, label=y_val.values)

            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 4,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 3,
                "gamma": 0.1,
                "seed": 42,
                "verbosity": 0,
            }
            evals_result = {}
            self.model = xgb.train(
                params, dtrain,
                num_boost_round=400,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                evals_result=evals_result,
                verbose_eval=False,
            )
            self.trained = True
            self._last_train_len = len(df)
            self._bars_since_train = 0

            # Out-of-sample accuracy
            preds = (self.model.predict(dval) >= 0.5).astype(int)
            acc = (preds == y_val.values).mean()
            best_round = self.model.best_iteration
            logging.info("RollingML trained: rows=%d split=%d OOS_acc=%.3f best_round=%d",
                         len(X), split, acc, best_round)

        except Exception as e:
            self.model = None
            self.trained = False
            logging.warning("RollingML.fit error: %s", e)

    def should_retrain(self, current_df_len: int) -> bool:
        """Returns True if enough new bars have arrived since last train."""
        return (current_df_len - self._last_train_len) >= self.retrain_every

    def tick(self):
        """Call once per bar to track retrain cadence."""
        self._bars_since_train += 1

    def update_close(self, close: float) -> None:
        """Feed each bar's close so predict_signal can compute momentum features."""
        if close and close > 0:
            self._close_window.append(float(close))

    def predict_signal(self, row: pd.Series) -> int:
        """
        Returns +1 (long), -1 (short), or 0 (no signal).
        Uses tighter confidence bands (0.60 / 0.40) to reduce noise.
        """
        if not _HAS_XGB or not self.trained or self.model is None:
            return 0

        try:
            close = float(row.get("Close", 1) or 1)
            ema200 = row.get("EMA_200") or close
            ema50 = row.get("EMA_50") or close
            ema21 = row.get("EMA_21") or close
            vwap = row.get("VWAP") or close

            w = list(self._close_window)
            ret_1 = (w[-1] / w[-2] - 1.0) if len(w) >= 2 else 0.0
            ret_3 = (w[-1] / w[-4] - 1.0) if len(w) >= 4 else 0.0
            ret_5 = (w[-1] / w[-6] - 1.0) if len(w) >= 6 else 0.0

            X = np.array([[
                float(row.get("RSI", 50) or 50),
                float(row.get("ADX", 0) or 0),
                float(row.get("MACD_HIST", 0) or 0),
                float((row.get("MACD", 0) or 0) / (close or 1)),
                float(row.get("BB_PCT", 0.5) or 0.5),
                float(row.get("ATR_pct", 0) or 0),
                float(row.get("Vol_Osc", 0) or 0),
                float(row.get("Stoch_RSI_K", 50) or 50),
                float(row.get("Stoch_RSI_D", 50) or 50),
                ret_1,
                ret_3,
                ret_5,
                float((ema21 / (ema50 or 1)) - 1.0),
                float((ema50 / (ema200 or 1)) - 1.0),
                float((close / (vwap or 1)) - 1.0),
            ]], dtype=float)
            dtest = xgb.DMatrix(X)
            p = float(self.model.predict(dtest)[0])
            if p >= 0.60:
                return +1
            elif p <= 0.40:
                return -1
            return 0
        except Exception as e:
            logging.warning("RollingML.predict_signal error: %s", e)
            return 0
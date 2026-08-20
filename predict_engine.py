"""
predict_engine.py - Dual Model Stock Prediction Engine
Model v4.1 (37 features) for long predictions
Model v5 (41 features) for short predictions
"""

import os
import json
import time
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
from openpyxl import Workbook

logger = logging.getLogger(__name__)

# --- Directories ---
MODEL_DIR = Path(__file__).parent / 'model'
PREDICTION_HISTORY_DIR = Path('/app/data/predictions')
PREDICTION_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Lists ---
FEATURES_LONG = [
    'MA_spread', 'c_vs_MA5', 'c_vs_MA20', 'c_vs_MA60',
    'MA5_slope', 'MA20_slope', 'vol_ratio', 'vol_trend',
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
    'vol_10d', 'vol_20d', 'RSI', 'MACD_DIF', 'MACD_Signal', 'MACD_Hist',
    'dist_high20', 'BB_pos', 'K_raw', 'vol_conv',
    'foreign_3d', 'foreign_5d', 'trust_3d', 'trust_5d', 'inst_momentum',
    'foreign_consec_buy', 'trust_consec_buy',
    'taiex_above_MA20', 'taiex_trend',
    'margin_chg_5d', 'short_chg_5d', 'margin_usage', 'short_ratio',
    'industry_momentum',
]

FEATURES_SHORT = FEATURES_LONG + [
    'bb_width', 'max_gain_20d', 'max_loss_20d', 'vol_breakout_days',
]

FINMIND_API_URL = 'https://api.finmindtrade.com/api/v4/data'

# --- Module-level engine singleton ---
_engine_instance: Optional['PredictEngine'] = None


def init_engine(token: str) -> 'PredictEngine':
    """Initialize the global engine singleton."""
    global _engine_instance
    _engine_instance = PredictEngine(token)
    return _engine_instance


def get_engine() -> 'PredictEngine':
    """Return the global engine singleton."""
    if _engine_instance is None:
        raise RuntimeError("Engine not initialized. Call init_engine(token) first.")
    return _engine_instance


class PredictEngine:
    """Dual model stock prediction engine using FinMind API."""

    def __init__(self, token: str):
        self.token = token
        self.api_call_count = 0
        self.model_long = None
        self.model_short = None
        self.reg_long = None
        self.reg_short = None
        self.stock_info: pd.DataFrame = pd.DataFrame()

        self._load_model()
        self._load_stock_info()
        self.backfill()

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------
    def _load_model(self):
        """Load long (v4.1) and short (v5) classification + regression models."""
        long_path = MODEL_DIR / 'stock_model_v4.pkl'
        short_path = MODEL_DIR / 'stock_model_v5.pkl'

        if long_path.exists():
            with open(long_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.model_long = data.get('classifier') or data.get('model')
                self.reg_long = data.get('regressor')
            else:
                self.model_long = data
            logger.info("Loaded long model (v4.1) from %s", long_path)
        else:
            logger.warning("Long model not found: %s", long_path)

        if short_path.exists():
            with open(short_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.model_short = data.get('classifier') or data.get('model')
                self.reg_short = data.get('regressor')
            else:
                self.model_short = data
            logger.info("Loaded short model (v5) from %s", short_path)
        else:
            logger.warning("Short model not found: %s", short_path)

    # ------------------------------------------------------------------
    # Stock Info
    # ------------------------------------------------------------------
    def _load_stock_info(self):
        """Load Taiwan stock info from FinMind."""
        try:
            params = {
                'dataset': 'TaiwanStockInfo',
                'token': self.token,
            }
            resp = self._api_get(params)
            if resp is not None and len(resp) > 0:
                self.stock_info = resp[['stock_id', 'stock_name', 'industry_category']].copy()
                self.stock_info = self.stock_info.drop_duplicates(subset='stock_id')
                logger.info("Loaded %d stock info records", len(self.stock_info))
        except Exception as e:
            logger.error("Failed to load stock info: %s", e)

    # ------------------------------------------------------------------
    # API Helpers
    # ------------------------------------------------------------------
    def _api_get(self, params: dict) -> Optional[pd.DataFrame]:
        """Call FinMind API with rate limiting."""
        self.api_call_count += 1
        if self.api_call_count % 30 == 0:
            time.sleep(1.5)
        elif self.api_call_count % 5 == 0:
            time.sleep(0.3)

        try:
            resp = requests.get(FINMIND_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') != 200:
                logger.warning("FinMind API error: %s", data.get('msg', ''))
                return None
            df = pd.DataFrame(data.get('data', []))
            return df if len(df) > 0 else None
        except Exception as e:
            logger.error("API request failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Data Fetching
    # ------------------------------------------------------------------
    def _fetch_price(self, stock_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch daily price data for a stock."""
        params = {
            'dataset': 'TaiwanStockPrice',
            'data_id': stock_id,
            'start_date': start_date,
            'end_date': end_date,
            'token': self.token,
        }
        df = self._api_get(params)
        if df is not None and len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            for col in ['open', 'max', 'min', 'close', 'Trading_Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _fetch_institutional(self, stock_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch institutional investor buy/sell data."""
        params = {
            'dataset': 'TaiwanStockInstitutionalInvestorsBuySell',
            'data_id': stock_id,
            'start_date': start_date,
            'end_date': end_date,
            'token': self.token,
        }
        df = self._api_get(params)
        if df is not None and len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df['buy'] = pd.to_numeric(df.get('buy', 0), errors='coerce').fillna(0)
            df['sell'] = pd.to_numeric(df.get('sell', 0), errors='coerce').fillna(0)
            df['net'] = df['buy'] - df['sell']
        return df

    def _fetch_margin(self, stock_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch margin trading data."""
        params = {
            'dataset': 'TaiwanStockMarginPurchaseShortSale',
            'data_id': stock_id,
            'start_date': start_date,
            'end_date': end_date,
            'token': self.token,
        }
        df = self._api_get(params)
        if df is not None and len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            for col in ['MarginPurchaseBuy', 'MarginPurchaseSell', 'MarginPurchaseTodayBalance',
                        'MarginPurchaseLimit', 'ShortSaleBuy', 'ShortSaleSell',
                        'ShortSaleTodayBalance']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

    def _fetch_taiex(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch TAIEX index data."""
        params = {
            'dataset': 'TaiwanStockPrice',
            'data_id': 'TAIEX',
            'start_date': start_date,
            'end_date': end_date,
            'token': self.token,
        }
        df = self._api_get(params)
        if df is not None and len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Top 300 Stocks
    # ------------------------------------------------------------------
    def _get_top300(self, end_date: str) -> List[str]:
        """Get top 300 stocks by recent trading volume."""
        start = (pd.Timestamp(end_date) - timedelta(days=10)).strftime('%Y-%m-%d')
        params = {
            'dataset': 'TaiwanStockPrice',
            'start_date': start,
            'end_date': end_date,
            'token': self.token,
        }
        df = self._api_get(params)
        if df is None or len(df) == 0:
            logger.warning("Cannot get top 300 stocks")
            return []

        df['Trading_Volume'] = pd.to_numeric(df['Trading_Volume'], errors='coerce').fillna(0)
        vol_sum = df.groupby('stock_id')['Trading_Volume'].sum().reset_index()
        vol_sum = vol_sum.sort_values('Trading_Volume', ascending=False)

        # Filter to common stocks (4-digit numeric IDs)
        vol_sum = vol_sum[vol_sum['stock_id'].str.match(r'^\d{4}$')]
        top300 = vol_sum.head(300)['stock_id'].tolist()
        logger.info("Got top %d stocks by volume", len(top300))
        return top300

    # ------------------------------------------------------------------
    # Feature Engineering - Long (37 features)
    # ------------------------------------------------------------------
    def _compute_features_long(self, price_df: pd.DataFrame,
                               inst_df: Optional[pd.DataFrame],
                               margin_df: Optional[pd.DataFrame],
                               taiex_df: Optional[pd.DataFrame],
                               industry_ret: float = 0.0) -> Optional[Dict[str, float]]:
        """Compute 37 features for long model."""
        if price_df is None or len(price_df) < 60:
            return None

        close = price_df['close'].values
        volume = price_df['Trading_Volume'].values
        high = price_df['max'].values
        low = price_df['min'].values
        n = len(close)

        try:
            # Moving averages
            ma5 = pd.Series(close).rolling(5).mean().values
            ma20 = pd.Series(close).rolling(20).mean().values
            ma60 = pd.Series(close).rolling(60).mean().values

            c = close[-1]
            features = {}

            # MA features
            features['MA_spread'] = (ma5[-1] - ma20[-1]) / ma20[-1] if ma20[-1] != 0 else 0
            features['c_vs_MA5'] = (c - ma5[-1]) / ma5[-1] if ma5[-1] != 0 else 0
            features['c_vs_MA20'] = (c - ma20[-1]) / ma20[-1] if ma20[-1] != 0 else 0
            features['c_vs_MA60'] = (c - ma60[-1]) / ma60[-1] if ma60[-1] != 0 else 0
            features['MA5_slope'] = (ma5[-1] - ma5[-5]) / ma5[-5] if ma5[-5] != 0 else 0
            features['MA20_slope'] = (ma20[-1] - ma20[-5]) / ma20[-5] if ma20[-5] != 0 else 0

            # Volume features
            vol_ma20 = np.mean(volume[-20:]) if n >= 20 else np.mean(volume)
            features['vol_ratio'] = volume[-1] / vol_ma20 if vol_ma20 != 0 else 1
            vol_ma5 = np.mean(volume[-5:])
            vol_ma20_val = np.mean(volume[-20:])
            features['vol_trend'] = vol_ma5 / vol_ma20_val if vol_ma20_val != 0 else 1

            # Returns
            features['ret_1d'] = (close[-1] - close[-2]) / close[-2] if close[-2] != 0 else 0
            features['ret_3d'] = (close[-1] - close[-4]) / close[-4] if n >= 4 and close[-4] != 0 else 0
            features['ret_5d'] = (close[-1] - close[-6]) / close[-6] if n >= 6 and close[-6] != 0 else 0
            features['ret_10d'] = (close[-1] - close[-11]) / close[-11] if n >= 11 and close[-11] != 0 else 0
            features['ret_20d'] = (close[-1] - close[-21]) / close[-21] if n >= 21 and close[-21] != 0 else 0

            # Volatility
            rets = pd.Series(close).pct_change().dropna().values
            features['vol_10d'] = np.std(rets[-10:]) if len(rets) >= 10 else 0
            features['vol_20d'] = np.std(rets[-20:]) if len(rets) >= 20 else 0

            # RSI (14-day)
            deltas = np.diff(close[-15:])
            gains = np.maximum(deltas, 0)
            losses = np.abs(np.minimum(deltas, 0))
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            features['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = pd.Series(close).ewm(span=12).mean().values
            ema26 = pd.Series(close).ewm(span=26).mean().values
            dif = ema12 - ema26
            signal = pd.Series(dif).ewm(span=9).mean().values
            features['MACD_DIF'] = dif[-1] / c if c != 0 else 0
            features['MACD_Signal'] = signal[-1] / c if c != 0 else 0
            features['MACD_Hist'] = (dif[-1] - signal[-1]) / c if c != 0 else 0

            # Distance from 20-day high
            high20 = np.max(high[-20:])
            features['dist_high20'] = (c - high20) / high20 if high20 != 0 else 0

            # Bollinger Band position
            std20 = np.std(close[-20:])
            bb_upper = ma20[-1] + 2 * std20
            bb_lower = ma20[-1] - 2 * std20
            bb_range = bb_upper - bb_lower
            features['BB_pos'] = (c - bb_lower) / bb_range if bb_range != 0 else 0.5

            # Stochastic K
            low20 = np.min(low[-20:])
            high20_val = np.max(high[-20:])
            k_range = high20_val - low20
            features['K_raw'] = (c - low20) / k_range if k_range != 0 else 0.5

            # Volume convergence
            vol5 = np.mean(volume[-5:])
            vol20 = np.mean(volume[-20:])
            features['vol_conv'] = vol5 / vol20 if vol20 != 0 else 1

            # Institutional features
            if inst_df is not None and len(inst_df) > 0:
                foreign = inst_df[inst_df['name'].str.contains('外資', na=False)]
                trust = inst_df[inst_df['name'].str.contains('投信', na=False)]

                foreign_daily = foreign.groupby('date')['net'].sum().reset_index()
                trust_daily = trust.groupby('date')['net'].sum().reset_index()

                f_vals = foreign_daily['net'].values
                t_vals = trust_daily['net'].values

                features['foreign_3d'] = np.sum(f_vals[-3:]) / 1e6 if len(f_vals) >= 3 else 0
                features['foreign_5d'] = np.sum(f_vals[-5:]) / 1e6 if len(f_vals) >= 5 else 0
                features['trust_3d'] = np.sum(t_vals[-3:]) / 1e6 if len(t_vals) >= 3 else 0
                features['trust_5d'] = np.sum(t_vals[-5:]) / 1e6 if len(t_vals) >= 5 else 0
                features['inst_momentum'] = (features['foreign_5d'] + features['trust_5d'])

                # Consecutive buy days
                features['foreign_consec_buy'] = self._count_consec_positive(f_vals)
                features['trust_consec_buy'] = self._count_consec_positive(t_vals)
            else:
                features['foreign_3d'] = 0
                features['foreign_5d'] = 0
                features['trust_3d'] = 0
                features['trust_5d'] = 0
                features['inst_momentum'] = 0
                features['foreign_consec_buy'] = 0
                features['trust_consec_buy'] = 0

            # TAIEX features
            if taiex_df is not None and len(taiex_df) >= 20:
                taiex_close = taiex_df['close'].values
                taiex_ma20 = np.mean(taiex_close[-20:])
                features['taiex_above_MA20'] = 1 if taiex_close[-1] > taiex_ma20 else 0
                features['taiex_trend'] = (taiex_close[-1] - taiex_close[-20]) / taiex_close[-20] if taiex_close[-20] != 0 else 0
            else:
                features['taiex_above_MA20'] = 0
                features['taiex_trend'] = 0

            # Margin features
            if margin_df is not None and len(margin_df) >= 5:
                margin_bal = margin_df['MarginPurchaseTodayBalance'].values
                short_bal = margin_df['ShortSaleTodayBalance'].values
                margin_limit = margin_df['MarginPurchaseLimit'].values

                features['margin_chg_5d'] = (margin_bal[-1] - margin_bal[-5]) / (margin_bal[-5] + 1)
                features['short_chg_5d'] = (short_bal[-1] - short_bal[-5]) / (short_bal[-5] + 1)
                features['margin_usage'] = margin_bal[-1] / margin_limit[-1] if margin_limit[-1] != 0 else 0
                total_shares = volume[-1] if volume[-1] != 0 else 1
                features['short_ratio'] = short_bal[-1] / total_shares
            else:
                features['margin_chg_5d'] = 0
                features['short_chg_5d'] = 0
                features['margin_usage'] = 0
                features['short_ratio'] = 0

            # Industry momentum
            features['industry_momentum'] = industry_ret

            return features

        except Exception as e:
            logger.error("Feature computation error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Feature Engineering - Short (41 features)
    # ------------------------------------------------------------------
    def _compute_features_short(self, price_df: pd.DataFrame,
                                inst_df: Optional[pd.DataFrame],
                                margin_df: Optional[pd.DataFrame],
                                taiex_df: Optional[pd.DataFrame],
                                industry_ret: float = 0.0) -> Optional[Dict[str, float]]:
        """Compute 41 features for short model (37 base + 4 additional)."""
        features = self._compute_features_long(price_df, inst_df, margin_df, taiex_df, industry_ret)
        if features is None:
            return None

        close = price_df['close'].values
        volume = price_df['Trading_Volume'].values

        try:
            # bb_width = (20-day std * 2) / 20-day mean
            std20 = np.std(close[-20:])
            mean20 = np.mean(close[-20:])
            features['bb_width'] = (std20 * 2) / mean20 if mean20 != 0 else 0

            # max_gain_20d
            close_20ago = close[-21] if len(close) >= 21 else close[0]
            max_close_20 = np.max(close[-20:])
            features['max_gain_20d'] = (max_close_20 - close_20ago) / close_20ago if close_20ago != 0 else 0

            # max_loss_20d
            min_close_20 = np.min(close[-20:])
            features['max_loss_20d'] = (min_close_20 - close_20ago) / close_20ago if close_20ago != 0 else 0

            # vol_breakout_days: days in past 20 where volume > 1.5x 20-day avg
            vol_20 = volume[-20:]
            avg_vol_20 = np.mean(vol_20)
            features['vol_breakout_days'] = int(np.sum(vol_20 > 1.5 * avg_vol_20))

            return features

        except Exception as e:
            logger.error("Short feature computation error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _count_consec_positive(arr) -> int:
        """Count consecutive positive values from the end of array."""
        count = 0
        for v in reversed(arr):
            if v > 0:
                count += 1
            else:
                break
        return count

    def _get_stock_name(self, stock_id: str) -> str:
        """Get stock name from info."""
        if self.stock_info is not None and len(self.stock_info) > 0:
            match = self.stock_info[self.stock_info['stock_id'] == stock_id]
            if len(match) > 0:
                return match.iloc[0]['stock_name']
        return stock_id

    def _get_industry(self, stock_id: str) -> str:
        """Get industry category for a stock."""
        if self.stock_info is not None and len(self.stock_info) > 0:
            match = self.stock_info[self.stock_info['stock_id'] == stock_id]
            if len(match) > 0:
                return match.iloc[0].get('industry_category', 'Unknown')
        return 'Unknown'

    # ------------------------------------------------------------------
    # Trading Days
    # ------------------------------------------------------------------
    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """Get trading days using 0050 as reference."""
        df = self._fetch_price('0050', start_date, end_date)
        if df is not None and len(df) > 0:
            return df['date'].dt.strftime('%Y-%m-%d').tolist()
        return []

    # ------------------------------------------------------------------
    # Prediction History
    # ------------------------------------------------------------------
    def _get_history_path(self, date_str: str) -> Path:
        """Get prediction history file path for a date."""
        return PREDICTION_HISTORY_DIR / f"pred_{date_str}.json"

    def get_existing_predictions(self, date_str: str) -> Optional[Dict]:
        """Get existing predictions for a date."""
        path = self._get_history_path(date_str)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _save_predictions(self, date_str: str, data: Dict):
        """Save predictions to history."""
        path = self._get_history_path(date_str)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, target_date: Optional[str] = None) -> Optional[Path]:
        """
        Run dual model prediction.
        Returns path to Excel output file.
        """
        if self.model_long is None or self.model_short is None:
            logger.error("Models not loaded. Cannot predict.")
            return None

        today = target_date or datetime.now().strftime('%Y-%m-%d')
        logger.info("Running prediction for %s", today)

        # Check if already predicted
        existing = self.get_existing_predictions(today)
        if existing:
            logger.info("Predictions already exist for %s", today)

        # Date ranges
        end_date = today
        start_price = (pd.Timestamp(today) - timedelta(days=180)).strftime('%Y-%m-%d')  # 120 trading days ~180 calendar
        start_inst = (pd.Timestamp(today) - timedelta(days=35)).strftime('%Y-%m-%d')  # 20 trading days
        start_margin = start_inst

        # Get top 300 stocks
        top300 = self._get_top300(end_date)
        if not top300:
            logger.error("No stocks found")
            return None

        # Fetch TAIEX
        taiex_df = self._fetch_taiex(start_price, end_date)

        # Compute industry returns for momentum
        industry_returns = {}

        # Collect results
        long_results = []
        short_results = []

        for i, stock_id in enumerate(top300):
            if (i + 1) % 50 == 0:
                logger.info("Processing %d/%d stocks...", i + 1, len(top300))

            try:
                # Fetch data
                price_df = self._fetch_price(stock_id, start_price, end_date)
                if price_df is None or len(price_df) < 60:
                    continue

                inst_df = self._fetch_institutional(stock_id, start_inst, end_date)
                margin_df = self._fetch_margin(stock_id, start_margin, end_date)

                # Get industry momentum
                industry = self._get_industry(stock_id)
                industry_ret = industry_returns.get(industry, 0.0)

                # Compute features for both models
                feat_long = self._compute_features_long(price_df, inst_df, margin_df, taiex_df, industry_ret)
                feat_short = self._compute_features_short(price_df, inst_df, margin_df, taiex_df, industry_ret)

                if feat_long is None or feat_short is None:
                    continue

                last_close = price_df['close'].values[-1]
                stock_name = self._get_stock_name(stock_id)

                # Long model prediction
                X_long = pd.DataFrame([feat_long])[FEATURES_LONG]
                long_prob = self.model_long.predict_proba(X_long)[0][1]

                # Regression for max gain
                future_max_pct = 0.0
                if self.reg_long is not None:
                    future_max_pct = self.reg_long.predict(X_long)[0]

                long_results.append({
                    'stock_id': stock_id,
                    'stock_name': stock_name,
                    'close': last_close,
                    'prob': long_prob,
                    'max_gain': future_max_pct,
                    'target_price': last_close * (1 + future_max_pct),
                })

                # Short model prediction
                X_short = pd.DataFrame([feat_short])[FEATURES_SHORT]
                short_prob = self.model_short.predict_proba(X_short)[0][1]

                # Regression for max loss
                future_min_pct = 0.0
                if self.reg_short is not None:
                    future_min_pct = self.reg_short.predict(X_short)[0]

                short_results.append({
                    'stock_id': stock_id,
                    'stock_name': stock_name,
                    'close': last_close,
                    'prob': short_prob,
                    'max_loss': future_min_pct,
                    'target_price': last_close * (1 + future_min_pct),
                })

            except Exception as e:
                logger.error("Error processing %s: %s", stock_id, e)
                continue

        if not long_results and not short_results:
            logger.error("No valid predictions generated")
            return None

        # Sort and get top 10
        long_top10 = sorted(long_results, key=lambda x: x['prob'], reverse=True)[:10]
        short_top10 = sorted(short_results, key=lambda x: x['prob'], reverse=True)[:10]

        # Generate Excel
        output_path = PREDICTION_HISTORY_DIR / f"prediction_{today}.xlsx"
        self._write_excel(output_path, long_top10, short_top10)

        # Save history JSON
        history_data = {
            'date': today,
            'long_top10': long_top10,
            'short_top10': short_top10,
            'total_stocks': len(top300),
            'processed': len(long_results),
        }
        self._save_predictions(today, history_data)

        logger.info("Prediction complete. Output: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Excel Output
    # ------------------------------------------------------------------
    def _write_excel(self, path: Path, long_top10: List[Dict], short_top10: List[Dict]):
        """Write prediction results to Excel with two sheets."""
        wb = Workbook()

        # Sheet 1: 做多Top10
        ws_long = wb.active
        ws_long.title = '做多Top10'
        ws_long.append(['代號', '名稱', '收盤', '起漲機率', '預估最高漲幅', '預估目標價'])
        for item in long_top10:
            ws_long.append([
                item['stock_id'],
                item['stock_name'],
                round(item['close'], 2),
                f"{item['prob']:.1%}",
                f"{item['max_gain']:.1%}",
                round(item['target_price'], 2),
            ])

        # Sheet 2: 做空Top10
        ws_short = wb.create_sheet('做空Top10')
        ws_short.append(['代號', '名稱', '收盤', '做空機率', '預估最大跌幅', '預估目標價'])
        for item in short_top10:
            ws_short.append([
                item['stock_id'],
                item['stock_name'],
                round(item['close'], 2),
                f"{item['prob']:.1%}",
                f"{item['max_loss']:.1%}",
                round(item['target_price'], 2),
            ])

        wb.save(path)
        logger.info("Excel saved: %s", path)

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------
    def review(self, prediction_date: Optional[str] = None) -> Optional[Dict]:
        """
        Review predictions from ~20 trading days ago.
        Compare predicted vs actual performance.
        """
        today = datetime.now().strftime('%Y-%m-%d')

        if prediction_date is None:
            # Find prediction from ~20 trading days ago
            start_check = (datetime.now() - timedelta(days=40)).strftime('%Y-%m-%d')
            trading_days = self.get_trading_days(start_check, today)
            if len(trading_days) < 20:
                logger.warning("Not enough trading days for review")
                return None
            prediction_date = trading_days[-21] if len(trading_days) >= 21 else trading_days[0]

        # Load predictions
        predictions = self.get_existing_predictions(prediction_date)
        if predictions is None:
            logger.warning("No predictions found for %s", prediction_date)
            return None

        review_results = {
            'prediction_date': prediction_date,
            'review_date': today,
            'long_review': [],
            'short_review': [],
        }

        # Review long predictions
        for item in predictions.get('long_top10', []):
            stock_id = item['stock_id']
            pred_close = item['close']

            # Fetch actual price data after prediction
            price_df = self._fetch_price(stock_id, prediction_date, today)
            if price_df is None or len(price_df) < 2:
                continue

            actual_max = price_df['max'].max()
            actual_close = price_df['close'].values[-1]
            actual_gain = (actual_max - pred_close) / pred_close

            review_results['long_review'].append({
                'stock_id': stock_id,
                'stock_name': item['stock_name'],
                'pred_prob': item['prob'],
                'pred_gain': item['max_gain'],
                'actual_max_gain': actual_gain,
                'actual_return': (actual_close - pred_close) / pred_close,
                'hit': actual_gain >= 0.05,  # 5% gain threshold
            })

        # Review short predictions
        for item in predictions.get('short_top10', []):
            stock_id = item['stock_id']
            pred_close = item['close']

            price_df = self._fetch_price(stock_id, prediction_date, today)
            if price_df is None or len(price_df) < 2:
                continue

            actual_min = price_df['min'].min()
            actual_close = price_df['close'].values[-1]
            actual_loss = (actual_min - pred_close) / pred_close

            review_results['short_review'].append({
                'stock_id': stock_id,
                'stock_name': item['stock_name'],
                'pred_prob': item['prob'],
                'pred_loss': item['max_loss'],
                'actual_max_loss': actual_loss,
                'actual_return': (actual_close - pred_close) / pred_close,
                'hit': actual_loss <= -0.05,  # 5% drop threshold
            })

        # Compute hit rates
        long_hits = sum(1 for r in review_results['long_review'] if r['hit'])
        long_total = len(review_results['long_review'])
        short_hits = sum(1 for r in review_results['short_review'] if r['hit'])
        short_total = len(review_results['short_review'])

        review_results['long_hit_rate'] = long_hits / long_total if long_total > 0 else 0
        review_results['short_hit_rate'] = short_hits / short_total if short_total > 0 else 0

        # Save review
        review_path = PREDICTION_HISTORY_DIR / f"review_{prediction_date}.json"
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(review_results, f, ensure_ascii=False, indent=2)

        logger.info("Review complete for %s: long hit=%d/%d, short hit=%d/%d",
                    prediction_date, long_hits, long_total, short_hits, short_total)
        return review_results

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------
    def backfill(self):
        """
        Backfill missing trading days' predictions on init.
        Uses 0050 to determine trading days.
        """
        today = datetime.now().strftime('%Y-%m-%d')
        start_check = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        trading_days = self.get_trading_days(start_check, today)
        if not trading_days:
            logger.info("No trading days to backfill")
            return

        missing_days = []
        for day in trading_days:
            if not self._get_history_path(day).exists():
                missing_days.append(day)

        if not missing_days:
            logger.info("No missing predictions to backfill")
            return

        logger.info("Backfilling %d missing trading days: %s", len(missing_days), missing_days)
        for day in missing_days:
            try:
                self.predict(target_date=day)
            except Exception as e:
                logger.error("Backfill failed for %s: %s", day, e)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup_old_predictions(self):
        """Remove prediction records older than 30 trading days."""
        today = datetime.now().strftime('%Y-%m-%d')
        start_check = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        trading_days = self.get_trading_days(start_check, today)
        if len(trading_days) <= 30:
            return

        # Keep only last 30 trading days
        cutoff_date = trading_days[-30]

        removed = 0
        for f in PREDICTION_HISTORY_DIR.glob('pred_*.json'):
            date_str = f.stem.replace('pred_', '')
            if date_str < cutoff_date:
                f.unlink()
                removed += 1

        for f in PREDICTION_HISTORY_DIR.glob('review_*.json'):
            date_str = f.stem.replace('review_', '')
            if date_str < cutoff_date:
                f.unlink()
                removed += 1

        for f in PREDICTION_HISTORY_DIR.glob('prediction_*.xlsx'):
            date_str = f.stem.replace('prediction_', '')
            if date_str < cutoff_date:
                f.unlink()
                removed += 1

        if removed > 0:
            logger.info("Cleaned up %d old prediction files (before %s)", removed, cutoff_date)

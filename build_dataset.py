#!/usr/bin/env python3
"""
build_dataset.py — 建訓練用特徵檔（三重門檻標籤 + 多資料源）

===============================================================================
這一版修掉的快取 bug（很重要）
===============================================================================
原本快取檔名包含完整日期範圍，而日期是用「今天」算的：

    end = today ; start = end − 5年
    cache_key = f"{dataset}_{sid}_{start}_{end}"   ← 包含今天的日期

結果隔天重跑時 end 變了，**整個快取全部失效**，等於每天都從零開始抓。
實測：3327 次呼叫只命中 14 次快取，然後撞上每小時上限等 40 分鐘。

修法三處：
  1. 快取鍵只取到「年-月」→ 同月內重跑都會命中
  2. --as-of 可以固定結束日 → 跨月重跑也會命中，快取鍵完全穩定
  3. --max-per-hour 可調 → 依你的 FinMind 方案調高，大幅縮短時間

代價：同月內重跑會用到「稍微舊幾天」的資料。建 5 年訓練集完全不影響。

===============================================================================
API 用量
===============================================================================
每檔股票的呼叫數 = 4（價量/法人/信用/除權息）+ 選配資料源數。
1057 檔 × 7 = 約 7400 次。用 --no-holding --no-revenue --no-per 可降到 4200。
快取修好之後這個成本只付一次。

===============================================================================
三種進場方式
===============================================================================
    close      從 close[t] 起算。**不可執行**（收盤後才有訊號），僅作上界對照。
    next_open  從 open[t+1] 起算。真實但會追高。
    pullback   在 close[t] 掛限價單，有效 --fill-days 天，回檔到才成交。

實測（8%/4%、15日、top-5）：close 進場超額 +0.94%，next_open −0.08%。
差別在於贏的那段大多發生在收盤到隔日開盤之間，你參與不到。

===============================================================================
已處理的偏誤
===============================================================================
1. 觀察窗一律從進場日之後起算（pullback 連成交當天也不算，更保守）
2. 候選池沿歷史逐季探測取聯集 + 每日依「截至當日」的 20 日均額重排
   → 沒有存活者偏差
3. 排名用成交**金額**不是股數（股數會讓低價高換手的投機股排前面）
4. 還原股價（台股 7~8 月除息集中，未還原會製造假跌幅）
5. 產業動能跨股票橫斷面計算
6. 除權息與月營收逐檔抓（FinMind 不支援這兩個的全市場查詢）
7. 月營收延後 15 天才視為可用（實際公布是次月 10 日前，直接對齊 date
   會讓模型提前 10~40 天看到還沒公布的數字）

===============================================================================
用法
===============================================================================
    python build_dataset.py --token TOKEN --years 5 --cache-dir .cache \\
        --entry next_open --horizon 15 --up-pct 0.06 --dn-pct 0.12 \\
        --as-of 2026-08-24 --max-per-hour 5000 --out model/f_6_12.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

try:
    from features import (FEATURES_ALL, FEATURES_HOLDING, FEATURES_REVENUE,
                          FEATURES_VALUE, MIN_HISTORY, compute_feature_frame,
                          industry_momentum_panel)
except ImportError:
    sys.exit("找不到 features.py —— 請放在同一個資料夾")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("build_dataset")

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
HORIZON, UP_PCT, DN_PCT, FILL_DAYS = 15, 0.06, 0.12, 3
UNIVERSE_SIZE, VOL_LOOKBACK = 300, 20
MAX_PER_HOUR = 1800


class ApiError(RuntimeError):
    pass


class RateLimited(ApiError):
    pass


class Client:
    """時間窗限流。失敗會拋錯而不是回 None —— 回 None 會讓呼叫端用 0 填補，
    導致被限流之後所有籌碼特徵靜默歸零。"""

    def __init__(self, token: str, max_per_hour: int = MAX_PER_HOUR,
                 cache_dir: Optional[Path] = None):
        self.token = token
        self.max_per_hour = max_per_hour
        self._calls: deque = deque()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_calls = self.n_cached = 0

    def _throttle(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 3600:
            self._calls.popleft()
        if len(self._calls) >= self.max_per_hour:
            wait = 3600 - (now - self._calls[0]) + 1
            logger.warning("本地限流：等待 %.0f 秒（每小時上限 %d，"
                           "可用 --max-per-hour 調高）", wait, self.max_per_hour)
            time.sleep(max(wait, 0))
        self._calls.append(time.monotonic())
        time.sleep(0.05)

    def _cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{key.replace('/', '_').replace(':', '_')}.parquet"

    def get(self, dataset: str, start: str, end: str,
            data_id: Optional[str] = None, retries: int = 3):
        # 快取鍵只取到「年-月」。原本用完整日期，導致隔天重跑時 end 變了、
        # 整個快取全部失效（實測 3327 次呼叫只命中 14 次）。
        cp = self._cache_path(
            f"{dataset}_{data_id or 'ALL'}_{start[:7]}_{end[:7]}")
        if cp and cp.exists():
            self.n_cached += 1
            try:
                return pd.read_parquet(cp)
            except Exception:
                cp.unlink(missing_ok=True)

        params = {"dataset": dataset, "start_date": start,
                  "end_date": end, "token": self.token}
        if data_id:
            params["data_id"] = data_id
        last: Optional[Exception] = None
        for attempt in range(retries):
            self._throttle()
            self.n_calls += 1
            try:
                r = requests.get(FINMIND_API_URL, params=params, timeout=90)
                if r.status_code in (402, 429):
                    raise RateLimited(f"HTTP {r.status_code}：額度用盡")
                r.raise_for_status()
                j = r.json()
                if j.get("status") != 200:
                    msg = str(j.get("msg", ""))
                    if "limit" in msg.lower() or "上限" in msg:
                        raise RateLimited(msg)
                    raise ApiError(f"status={j.get('status')} msg={msg}")
                rows = j.get("data", [])
                if not rows:
                    return None
                df = pd.DataFrame(rows)
                if cp is not None:
                    try:
                        df.to_parquet(cp, index=False)
                    except Exception as ex:
                        logger.debug("快取寫入失敗：%s", ex)
                return df
            except RateLimited:
                raise
            except Exception as ex:
                last = ex
                if attempt < retries - 1:
                    time.sleep(2.0 * (attempt + 1))
        raise ApiError(f"{dataset}/{data_id} 重試 {retries} 次後失敗：{last}")

    def get_soft(self, dataset: str, start: str, end: str,
                 data_id: Optional[str] = None):
        """選配資料源用：失敗只回 None 不中斷（額度用盡仍然往上拋）。"""
        try:
            return self.get(dataset, start, end, data_id)
        except RateLimited:
            raise
        except ApiError as ex:
            logger.debug("%s/%s 取用失敗：%s", dataset, data_id, ex)
            return None


# =============================================================================
# 抓取
# =============================================================================

def fetch_stock_info(cli: Client) -> pd.DataFrame:
    df = cli.get("TaiwanStockInfo", "2000-01-01", "2100-01-01")
    if df is None:
        raise ApiError("取不到股票清單")
    keep = [c for c in ("stock_id", "stock_name", "industry_category")
            if c in df.columns]
    return df[keep].drop_duplicates(subset="stock_id").reset_index(drop=True)


def fetch_price(cli: Client, sid: str, start: str, end: str,
                adjusted: bool = True) -> Optional[pd.DataFrame]:
    ds = "TaiwanStockPriceAdj" if adjusted else "TaiwanStockPrice"
    try:
        df = cli.get(ds, start, end, sid)
    except RateLimited:
        raise
    except ApiError:
        if not adjusted:
            raise
        logger.debug("%s 還原股價失敗，退回未還原", sid)
        df = cli.get("TaiwanStockPrice", start, end, sid)
    if df is None or len(df) == 0:
        return None
    need = ("date", "close", "max", "min", "Trading_Volume")
    if any(c not in df.columns for c in need):
        logger.warning("%s 價量資料缺欄位，跳過", sid)
        return None

    cols = list(need) + (["open"] if "open" in df.columns else [])
    out = df[cols].copy()
    out["date"] = pd.to_datetime(out["date"])
    for c in ("close", "max", "min", "Trading_Volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["open"] = (pd.to_numeric(df["open"], errors="coerce")
                   if "open" in df.columns else out["close"])
    # 選股池排名要用成交金額，不能用股數
    out["Trading_money"] = (pd.to_numeric(df["Trading_money"], errors="coerce")
                            if "Trading_money" in df.columns
                            else out["close"] * out["Trading_Volume"])
    # 重複日期在這裡就去掉，讓特徵與標籤看到同一個序列
    return (out.dropna(subset=["date", "close", "max", "min", "Trading_Volume"])
            .sort_values("date").drop_duplicates("date", keep="last")
            .reset_index(drop=True))


def build_universe(panel: pd.DataFrame, size: int,
                   lookback: int = VOL_LOOKBACK) -> pd.DataFrame:
    """每日動態選股池，依 lookback 日的平均成交**金額**排名（trailing，無前視）。"""
    col = "Trading_money" if "Trading_money" in panel.columns else "Trading_Volume"
    if col == "Trading_Volume":
        logger.warning("沒有 Trading_money，退回股數排名 —— 會偏向低價投機股")
    wide = panel.pivot_table(index="date", columns="stock_id",
                             values=col, aggfunc="last").sort_index()
    avg = wide.rolling(lookback, min_periods=lookback).mean()
    mask = avg.rank(axis=1, ascending=False, method="first") <= size
    sel = mask.stack().rename("u").reset_index()
    sel = sel[sel["u"]].drop(columns="u")
    logger.info("選股池（依%s）：%d 個 (日期,股票) 組合，%d 個交易日",
                "成交金額" if col == "Trading_money" else "成交股數",
                len(sel), sel["date"].nunique())
    return sel


def extract_ex_dates(df) -> List[pd.Timestamp]:
    """除權息交易日。FinMind 的 TaiwanStockDividend 不支援全市場查詢
    （不給 data_id 只會回零星幾筆），必須逐檔抓。"""
    if df is None or len(df) == 0:
        return []
    cols = [c for c in df.columns
            if "ExDividendTradingDate" in c or "ExRightTradingDate" in c]
    vals: List[pd.Timestamp] = []
    for c in cols:
        d = pd.to_datetime(df[c], errors="coerce").dropna()
        vals.extend([v for v in d if v > pd.Timestamp("1990-01-01")])
    return sorted(set(vals))


# =============================================================================
# 標籤
# =============================================================================

def _barrier(fh, fl, close_end, entry_px, up_pct, dn_pct, horizon):
    """給定進場後的觀察窗，判定先碰哪一邊。
    同日兩邊都觸及時保守判失敗（日內無法分辨先後，寧可低估）。"""
    up_hit = fh >= (entry_px * (1 + up_pct))[:, None]
    dn_hit = fl <= (entry_px * (1 - dn_pct))[:, None]
    BIG = horizon + 10
    uf = np.where(up_hit.any(1), up_hit.argmax(1), BIG)
    df_ = np.where(dn_hit.any(1), dn_hit.argmax(1), BIG)
    win, lose = uf < df_, df_ < uf
    tie = (uf == df_) & (uf < BIG)
    timeout = (uf >= BIG) & (df_ >= BIG)

    m = fh.shape[0]
    label = np.zeros(m)
    label[win] = 1.0
    days = np.minimum(uf, df_) + 1.0
    days[timeout] = float(horizon)
    ret = np.full(m, np.nan)
    ret[win] = up_pct
    ret[lose | tie] = -dn_pct
    ret[timeout] = close_end[timeout] / entry_px[timeout] - 1.0
    oc = np.array(["timeout"] * m, dtype=object)
    oc[win] = "up"
    oc[lose | tie] = "down"
    return label, days, ret, oc


def triple_barrier(price_df, horizon, up_pct, dn_pct,
                   entry="next_open", fill_days=FILL_DAYS) -> pd.DataFrame:
    """三重門檻標籤。觀察窗一律從進場日之後起算 —— 訊號當天的高低點在收盤時
    已知，用它就是洩漏。pullback 模式連成交當天也不算（無法確定日內先後）。"""
    d = price_df.set_index("date").sort_index()
    close = d["close"].to_numpy(float)
    high, low = d["max"].to_numpy(float), d["min"].to_numpy(float)
    open_ = d["open"].to_numpy(float) if "open" in d.columns else close.copy()
    n, T = len(close), horizon
    cols = ["tb_label", "tb_days", "tb_ret", "tb_entry", "tb_gap", "tb_fill_days"]
    out = pd.DataFrame(index=d.index, dtype=float)

    if entry == "pullback":
        K = max(int(fill_days), 1)
        W = K + T
        if n <= W:
            for c in cols:
                out[c] = np.nan
            out["tb_outcome"] = None
            return out
        m = n - W
        fh_all = np.lib.stride_tricks.sliding_window_view(high[1:], W)[:m]
        fl_all = np.lib.stride_tricks.sliding_window_view(low[1:], W)[:m]
        limit = close[:m]
        fill_mask = fl_all[:, :K] <= limit[:, None]
        filled = fill_mask.any(1)
        fo = np.where(filled, fill_mask.argmax(1), 0)
        idx = fo[:, None] + 1 + np.arange(T)[None, :]
        fh = np.take_along_axis(fh_all, idx, axis=1)
        fl = np.take_along_axis(fl_all, idx, axis=1)
        end_pos = np.clip(np.arange(m) + 1 + fo + T, 0, n - 1)
        label, days, ret, oc = _barrier(fh, fl, close[end_pos], limit,
                                        up_pct, dn_pct, T)
        label = np.where(filled, label, np.nan)
        days = np.where(filled, days, np.nan)
        ret = np.where(filled, ret, np.nan)
        entry_px = np.where(filled, limit, np.nan)
        fill_d = np.where(filled, fo + 1.0, np.nan)
        oc = np.where(filled, oc, None)
        gap = open_[1:m + 1] / close[:m] - 1.0
    else:
        if n <= T:
            for c in cols:
                out[c] = np.nan
            out["tb_outcome"] = None
            return out
        m = n - T
        fh = np.lib.stride_tricks.sliding_window_view(high[1:], T)[:m]
        fl = np.lib.stride_tricks.sliding_window_view(low[1:], T)[:m]
        prev_close, nxt_open = close[:m], open_[1:m + 1]
        gap = nxt_open / prev_close - 1.0
        entry_px = nxt_open if entry == "next_open" else prev_close
        bad = ~np.isfinite(entry_px) | (entry_px <= 0)
        if bad.any():
            entry_px = np.where(bad, prev_close, entry_px)
        label, days, ret, oc = _barrier(fh, fl, close[T:T + m], entry_px,
                                        up_pct, dn_pct, T)
        fill_d = np.zeros(m)

    pad = np.full(n - m, np.nan)
    out["tb_label"] = np.concatenate([label, pad])
    out["tb_days"] = np.concatenate([days, pad])
    out["tb_ret"] = np.concatenate([ret, pad])
    out["tb_entry"] = np.concatenate([entry_px, pad])
    out["tb_gap"] = np.concatenate([gap, pad])
    out["tb_fill_days"] = np.concatenate([fill_d, pad])
    out["tb_outcome"] = list(oc) + [None] * (n - m)
    return out


def forward_targets(price_df, horizon) -> pd.DataFrame:
    """舊式前瞻標籤（對照用）。一律從 t+1 起算。"""
    d = price_df.set_index("date").sort_index()
    close, high, low = d["close"], d["max"], d["min"]
    fh = high.shift(-1).rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
    fl = low.shift(-1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
    out = pd.DataFrame(index=d.index)
    out["future_max"] = fh / close - 1.0
    out["future_min"] = fl / close - 1.0
    out[f"future_ret_{horizon}d"] = close.shift(-horizon) / close - 1.0
    return out


def short_feasibility(index, margin_df, ex_dates: Iterable, horizon) -> pd.DataFrame:
    """做空可行性（現階段只做多，欄位留給之後）。
    未含「平盤下不得放空」—— FinMind 無逐日個股清單，這是樂觀上界。"""
    out = pd.DataFrame(index=index)
    if (margin_df is not None and len(margin_df)
            and "ShortSaleTodayBalance" in margin_df.columns):
        m = margin_df.copy()
        m["date"] = pd.to_datetime(m["date"])
        sb = (pd.to_numeric(m.groupby("date")["ShortSaleTodayBalance"].last(),
                            errors="coerce").reindex(index).ffill().fillna(0.0))
        out["short_has_supply"] = sb.rolling(20, min_periods=1).max() > 0
    else:
        out["short_has_supply"] = False
    ex = pd.DatetimeIndex(sorted(set(pd.to_datetime(list(ex_dates)))))
    if len(ex) == 0:
        out["short_no_recall"] = True
    else:
        n = len(index)
        blocked = np.zeros(n, dtype=bool)
        ex_in = ex[(ex >= index[0]) & (ex <= index[-1])]
        if len(ex_in):
            ep = np.asarray(index.searchsorted(ex_in, side="left"))
            for i in range(n):
                lo, hi = i + 1, min(i + horizon, n - 1)
                if lo <= hi:
                    blocked[i] = bool(((ep >= lo) & (ep <= hi)).any())
        out["short_no_recall"] = ~blocked
    out["short_feasible"] = out["short_has_supply"] & out["short_no_recall"]
    return out


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="建訓練用特徵檔")
    ap.add_argument("--token", required=True)
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--as-of", default=None,
                    help="固定資料結束日（YYYY-MM-DD）。給了之後快取鍵完全穩定，"
                         "跨月重跑也會命中；不給則用今天")
    ap.add_argument("--max-per-hour", type=int, default=MAX_PER_HOUR,
                    help="每小時 API 上限。依你的 FinMind 方案調高可大幅縮短時間")
    ap.add_argument("--universe", type=int, default=UNIVERSE_SIZE)
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="候選聯集上限（0=不限）。超過時依進榜季數裁減")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--up-pct", type=float, default=UP_PCT)
    ap.add_argument("--dn-pct", type=float, default=DN_PCT)
    ap.add_argument("--entry", choices=["close", "next_open", "pullback"],
                    default="next_open")
    ap.add_argument("--fill-days", type=int, default=FILL_DAYS)
    ap.add_argument("--no-holding", action="store_true",
                    help="不抓股東持股分級（每檔省 1 次 API，但少 4 個特徵）")
    ap.add_argument("--no-revenue", action="store_true", help="不抓月營收")
    ap.add_argument("--no-per", action="store_true", help="不抓本益比")
    ap.add_argument("--out", default="model/f_6_12.pkl")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--no-adjust", action="store_true")
    ap.add_argument("--market-id", default="TAIEX")
    args = ap.parse_args()

    end = (pd.Timestamp(args.as_of).normalize() if args.as_of
           else pd.Timestamp.today().normalize())
    start = end - pd.Timedelta(days=int(args.years * 365 + MIN_HISTORY * 1.6 + 60))
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    entry_desc = {"close": "當日收盤（不可執行）", "next_open": "隔日開盤",
                  "pullback": f"回檔限價（{args.fill_days}日有效）"}[args.entry]
    logger.info("區間 %s ~ %s（含 %d 日暖機）%s", s, e, MIN_HISTORY,
                "" if args.as_of else "　※ 未指定 --as-of，跨月重跑快取會失效")
    logger.info("起漲：%d 日內先漲 %.0f%% 成功、先跌 %.0f%% 失敗；進場 = %s",
                args.horizon, args.up_pct * 100, args.dn_pct * 100, entry_desc)
    extra = [n for n, off in (("持股分級", args.no_holding),
                              ("月營收", args.no_revenue),
                              ("本益比", args.no_per)) if not off]
    logger.info("額外資料源：%s　每小時上限 %d",
                "、".join(extra) if extra else "無", args.max_per_hour)

    cli = Client(args.token, max_per_hour=args.max_per_hour,
                 cache_dir=args.cache_dir)

    # ---- 1. 股票清單 ----
    info = fetch_stock_info(cli)
    if "industry_category" not in info.columns:
        sys.exit("TaiwanStockInfo 沒有 industry_category 欄位")
    # 一律 astype(str)：型別不同會讓每一檔都靜默變成 Unknown
    _sids = info["stock_id"].astype(str)
    industry_map = {sid: (ind if isinstance(ind, str) and ind else "Unknown")
                    for sid, ind in zip(_sids, info["industry_category"])}
    name_map = (dict(zip(_sids, info["stock_name"]))
                if "stock_name" in info.columns else {})
    known = set(_sids)
    logger.info("股票清單 %d 檔，%d 個產業", len(info), len(set(industry_map.values())))

    # ---- 2. 候選：沿歷史逐季探測取聯集（避免候選池的存活者偏差）----
    n_cand = int(args.universe * 1.6)
    quarters = pd.date_range(start, end, freq="QS")
    if len(quarters) == 0:
        quarters = pd.DatetimeIndex([start])
    cand_set: set = set()
    cand_freq: Dict[str, int] = {}
    for q in quarters:
        q_end = min(q + pd.Timedelta(days=30), end)
        try:
            probe = cli.get("TaiwanStockPrice", q.strftime("%Y-%m-%d"),
                            q_end.strftime("%Y-%m-%d"))
        except RateLimited as ex:
            logger.error("額度用盡，中止：%s", ex)
            sys.exit(1)
        except ApiError as ex:
            logger.warning("%s 探測失敗：%s", q.date(), ex)
            continue
        if probe is None or len(probe) == 0 or "stock_id" not in probe.columns:
            continue
        probe = probe[probe["stock_id"].astype(str).str.match(r"^\d{4}$")].copy()
        if len(probe) == 0:
            continue
        if "Trading_money" in probe.columns:
            probe["_t"] = pd.to_numeric(probe["Trading_money"], errors="coerce").fillna(0)
        elif all(c in probe.columns for c in ("close", "Trading_Volume")):
            probe["_t"] = (pd.to_numeric(probe["close"], errors="coerce").fillna(0)
                           * pd.to_numeric(probe["Trading_Volume"], errors="coerce").fillna(0))
        else:
            continue
        top = (probe.groupby("stock_id")["_t"].sum()
               .nlargest(n_cand).index.astype(str).tolist())
        before = len(cand_set)
        cand_set.update(top)
        for sid in top:
            cand_freq[sid] = cand_freq.get(sid, 0) + 1
        logger.info("  %s 探測：新增 %d（累計 %d）",
                    q.date(), len(cand_set) - before, len(cand_set))
    if not cand_set:
        sys.exit("所有季度探測都失敗")
    if args.max_candidates and len(cand_set) > args.max_candidates:
        keep = pd.Series(cand_freq).nlargest(args.max_candidates).index.tolist()
        logger.warning("候選 %d 超過上限，依進榜季數裁到 %d", len(cand_set), len(keep))
        cand_set = set(keep)
    cands = sorted(cand_set)
    per_stock = 4 + len(extra)
    est = 2 + len(quarters) + len(cands) * per_stock
    logger.info("候選 %d 檔；預估 API 約 %d 次（每檔 %d 次）；"
                "以上限 %d/小時估算約需 %.1f 小時（快取命中會大幅縮短）",
                len(cands), est, per_stock, args.max_per_hour,
                est / max(args.max_per_hour, 1))
    miss = [c for c in cands if c not in known]
    if miss:
        logger.warning("%d 檔候選不在今日清單中（可能已下市）", len(miss))

    # ---- 3. 大盤 ----
    market = None
    for mid in (args.market_id, "0050"):
        try:
            market = cli.get("TaiwanStockPrice", s, e, mid)
        except RateLimited as ex:
            # RateLimited 是 ApiError 子類，不先攔會被當一般失敗吞掉 →
            # 訓練集 taiex_* 全 0 而線上有真值 → skew
            logger.error("額度用盡，中止：%s", ex)
            sys.exit(1)
        except ApiError as ex:
            logger.warning("大盤 %s 失敗：%s", mid, ex)
            market = None
        if market is not None and len(market):
            logger.info("大盤使用 %s（%d 筆）", mid, len(market))
            break
    if market is None or len(market) == 0:
        logger.error("大盤缺失 → taiex_* 恆為 0 而線上有真值。這是 skew。")
        market = None
    elif any(c not in market.columns for c in ("date", "close")):
        sys.exit(f"大盤缺 date/close：{list(market.columns)}")
    else:
        market = market[["date", "close"]].copy()

    # ---- 4. 價量 ----
    prices: Dict[str, pd.DataFrame] = {}
    for i, sid in enumerate(cands, 1):
        if i % 50 == 0:
            logger.info("價量 %d/%d（API %d，快取 %d）",
                        i, len(cands), cli.n_calls, cli.n_cached)
        try:
            df = fetch_price(cli, sid, s, e, adjusted=not args.no_adjust)
        except RateLimited as ex:
            logger.error("額度用盡，中止：%s", ex)
            sys.exit(1)
        except ApiError as ex:
            logger.warning("%s 跳過：%s", sid, ex)
            continue
        if df is not None and len(df) >= MIN_HISTORY + args.horizon + args.fill_days:
            prices[sid] = df
    logger.info("取得 %d 檔可用價量", len(prices))
    if not prices:
        sys.exit("沒有可用的價量資料")

    # ---- 5. 選股池 ----
    panel = pd.concat([d.assign(stock_id=sid) for sid, d in prices.items()],
                      ignore_index=True)
    universe = build_universe(panel, args.universe)
    active = sorted(universe["stock_id"].unique())
    logger.info("曾進入選股池 %d 檔", len(active))

    # ---- 6. 產業動能 ----
    ind_panel = industry_momentum_panel(panel, industry_map, lookback=5)
    logger.info("產業動能：%d 個產業 × %d 日", ind_panel.shape[1], ind_panel.shape[0])

    # ---- 7. 逐檔：籌碼 + 選配資料源 + 特徵 + 標籤 ----
    cover = {"div": 0, "holding": 0, "revenue": 0, "per": 0}
    rows: List[pd.DataFrame] = []
    for i, sid in enumerate(active, 1):
        if i % 50 == 0:
            logger.info("特徵 %d/%d（API %d，快取 %d）",
                        i, len(active), cli.n_calls, cli.n_cached)
        px = prices.get(sid)
        if px is None:
            continue
        try:
            inst = cli.get("TaiwanStockInstitutionalInvestorsBuySell", s, e, sid)
            marg = cli.get("TaiwanStockMarginPurchaseShortSale", s, e, sid)
            divx = cli.get("TaiwanStockDividend", s, e, sid)
            # 選配：失敗只是少幾個特徵，不該讓整檔股票被丟掉
            hold = None if args.no_holding else cli.get_soft(
                "TaiwanStockHoldingSharesPer", s, e, sid)
            rev = None if args.no_revenue else cli.get_soft(
                "TaiwanStockMonthRevenue", s, e, sid)
            per = None if args.no_per else cli.get_soft("TaiwanStockPER", s, e, sid)
        except RateLimited as ex:
            logger.error("額度用盡，中止：%s", ex)
            sys.exit(1)
        except ApiError as ex:
            # 核心籌碼失敗就跳過這檔：用 0 填補會讓模型學到線上不存在的分佈
            logger.warning("%s 籌碼失敗，跳過：%s", sid, ex)
            continue

        ex_list = extract_ex_dates(divx)
        cover["div"] += bool(ex_list)
        cover["holding"] += bool(hold is not None and len(hold))
        cover["revenue"] += bool(rev is not None and len(rev))
        cover["per"] += bool(per is not None and len(per))

        ind = industry_map.get(sid, "Unknown") or "Unknown"
        im = ind_panel[ind] if ind in ind_panel.columns else None
        feats = compute_feature_frame(px, inst_df=inst, margin_df=marg,
                                      market_df=market, industry_momentum=im,
                                      holding_df=hold, revenue_df=rev, per_df=per)
        if len(feats) == 0:
            continue

        tb = triple_barrier(px, args.horizon, args.up_pct, args.dn_pct,
                            entry=args.entry, fill_days=args.fill_days)
        row = (feats.join(tb, how="left")
               .join(forward_targets(px, args.horizon), how="left")
               .join(short_feasibility(feats.index, marg, ex_list, args.horizon),
                     how="left"))
        row["stock_id"] = sid
        row["industry"] = ind
        row["stock_name"] = name_map.get(sid, sid)
        row["close"] = px.set_index("date")["close"].reindex(row.index)
        row = row.reset_index()
        if "date" not in row.columns:
            row = row.rename(columns={row.columns[0]: "date"})
        # merge 而非逐列 tuple 比對：不依賴迭代 dtype，也快得多
        row = row.merge(universe, on=["stock_id", "date"], how="inner")
        if len(row):
            rows.append(row)

    if not rows:
        sys.exit("沒有產生任何特徵列")
    df = pd.concat(rows, ignore_index=True)

    # ---- 8. 收尾 ----
    n0 = len(df)
    df = df.dropna(subset=FEATURES_ALL, how="any")
    n_sig = len(df)
    df = df.dropna(subset=["tb_label"])     # pullback 未成交在這裡被剔除
    fill_rate = len(df) / max(n_sig, 1)
    df = df.sort_values(["date", "stock_id"]).reset_index(drop=True)

    print(f"\n{'='*74}\n資料集摘要\n{'='*74}")
    print(f"  列數/股票/交易日 : {len(df):,} / {df['stock_id'].nunique()} / "
          f"{df['date'].nunique()}")
    print(f"  日期範圍         : {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  每日平均檔數     : {len(df)/max(df['date'].nunique(),1):.1f}")
    print(f"  特徵數           : {len(FEATURES_ALL)}")
    print(f"  進場方式         : {entry_desc}")
    print(f"  API 呼叫         : {cli.n_calls}（快取命中 {cli.n_cached}）")
    if cli.n_calls and cli.n_cached / max(cli.n_calls + cli.n_cached, 1) < 0.3:
        print(f"    註：快取命中率低。加 --as-of {e} 可讓之後重跑完全命中。")

    n_act = max(len(active), 1)
    print(f"\n  資料源涵蓋率（佔 {n_act} 檔）:")
    for key, label, off in (("div", "除權息", False),
                            ("holding", "股東持股分級", args.no_holding),
                            ("revenue", "月營收", args.no_revenue),
                            ("per", "本益比", args.no_per)):
        if off:
            print(f"    {label:<14} 未啟用")
            continue
        c = cover[key] / n_act
        flag = "" if c >= 0.8 else ("  ⚠ 涵蓋率過低，相關特徵不可信"
                                    if c < 0.5 else "  △ 涵蓋率偏低")
        print(f"    {label:<14}{c:>6.0%}{flag}")

    const = [c for c in FEATURES_ALL if df[c].nunique(dropna=False) <= 1]
    zero = [c for c in FEATURES_ALL if (df[c] == 0).mean() > 0.95]
    print(f"\n  常數特徵 : {const or '無'}")
    print(f"  幾乎全零 : {[c for c in zero if c not in const] or '無'}")
    if const or zero:
        print("    ⚠ taiex_* 在單日內本來就是常數（正常）。若 strength/holding/")
        print("      revenue/value 那幾組出現在上面，代表沒抓到。")

    if args.entry == "pullback":
        print(f"\n  限價單成交率 {fill_rate:.1%}"
              f"，平均等待 {df['tb_fill_days'].mean():.1f} 日")

    oc = df["tb_outcome"].value_counts(normalize=True)
    print(f"\n  【起漲標籤】{args.horizon} 日內 漲 {args.up_pct:.0%} vs 跌 {args.dn_pct:.0%}")
    print(f"    成功 {oc.get('up',0):.1%}（← 基準）／失敗 {oc.get('down',0):.1%}"
          f"／到期 {oc.get('timeout',0):.1%}")
    print(f"    平均持有 {df['tb_days'].mean():.1f} 日"
          f"，平均毛報酬 {df['tb_ret'].mean():+.2%}（隨機選股期望值）")
    if oc.get("up", 0) > 0.5:
        print("    註：基準 > 50%，門檻偏鬆，建議調高 --up-pct")
    elif oc.get("up", 0) < 0.08:
        print("    註：基準 < 8%，正樣本太少，建議調低 --up-pct")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(df, f)
    print(f"\n已存檔 → {out}（{out.stat().st_size/1e6:.1f} MB）")

    meta = {
        "built_at": pd.Timestamp.now().isoformat(),
        "as_of": e,
        "rows": len(df), "stocks": int(df["stock_id"].nunique()),
        "dates": int(df["date"].nunique()),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "horizon": args.horizon, "up_pct": args.up_pct, "dn_pct": args.dn_pct,
        "entry": args.entry, "fill_days": args.fill_days,
        "fill_rate": round(float(fill_rate), 4),
        "base_success_rate": float(oc.get("up", 0)),
        "universe_size": args.universe, "universe_ranked_by": "Trading_money",
        "adjusted_price": not args.no_adjust,
        "coverage": {k: round(v / n_act, 3) for k, v in cover.items()},
        "features": FEATURES_ALL, "constant_features": const,
        "new_feature_groups": {"holding": FEATURES_HOLDING,
                               "revenue": FEATURES_REVENUE,
                               "value": FEATURES_VALUE},
        "api_calls": cli.n_calls, "cache_hits": cli.n_cached,
        "known_limitation": "門檻用盤中高低判定，隱含能在門檻價成交；"
                            "停損遇跳空實際成交更差。月營收已延後 15 天避免前視。",
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"中介資料 → {out.with_suffix('.meta.json')}")
    print(f"\n下一步：\n  python train_score.py --data {out} --seeds 42,7,2024")


if __name__ == "__main__":
    main()

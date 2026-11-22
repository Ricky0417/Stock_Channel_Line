"""
features.py — 特徵計算的唯一實作（訓練與推論共用）

整輪除錯裡最貴的 bug —— 8 個籌碼特徵在推論時恆為 0，佔模型 13~18% 的分裂
決策 —— 根因是「同一件事被寫了兩遍」：訓練一份、推論一份，各自漂移。
所以這裡只有一個公開函式，訓練取整張表、推論取 .iloc[-1]，同一段程式碼。

===============================================================================
特徵分組（共 69 個）
===============================================================================
  價量技術 (41)  原本就有的均線/報酬/波動/量/技術指標/法人/信用/大盤
  籌碼正規化 (9)  法人與信用除以 20 日均量 → 跨股票可比
  強勢路徑 (7)   ★新增：收盤在日內的位置、漲停次數、連續上漲、跳空
  籌碼集中 (4)   ★新增：大戶/散戶持股比例與變化（TaiwanStockHoldingSharesPer）
  基本面 (4)     ★新增：月營收年增/月增/連續成長（TaiwanStockMonthRevenue）
  估值 (4)       ★新增：PER/PBR/殖利率/本益比一年分位（TaiwanStockPER）

===============================================================================
為什麼加這些
===============================================================================
IC 分析顯示現有 50 個特徵裡只有 6 個 |IC| > 0.05，37 個是雜訊。RSI 的 IC 是
−0.00005，字面上的零。原因是它們全都是價量的重新組合 —— 全台灣每個看盤軟體
都有的東西，任何邊際優勢早就被套利掉了。

新增的四組刻意選了**不同維度的資訊**：
  * 強勢路徑：門檻式/非線性/路徑式，而非連續型指標。漲停是台股特有的狀態
    （買盤強到掛不進去），收盤在日內高點也代表買盤強度 —— 我們有 high/low/
    close 卻從來沒用過收盤在區間裡的位置，這是個疏漏。
  * 籌碼集中：大戶持股比例是「誰在持有」，不是「價格怎麼動」。
  * 基本面與估值：真正的新資訊，跟價量的相關性低。

===============================================================================
前視偏誤的處理
===============================================================================
  * 月營收：FinMind 的 date 是營收所屬月份的月底，但公布是次月 10 日前。
    直接對齊 date 會讓模型提前看到還沒公布的數字。這裡一律延後到
    **次月 15 日**才可用（保守）。
  * 股東持股分級：每週公布，reindex 到交易日後 ffill，只用過去的值。
  * 所有 rolling 都是 trailing。

===============================================================================
刻意保留的原始定義怪癖（改了會讓既有模型的門檻失效）
===============================================================================
  MA5_slope/MA20_slope 是 4 期斜率、RSI 用簡單平均非 Wilder、vol_conv 與
  vol_trend 完全相同、short_ratio 分母是當日量、波動率用母體標準差(ddof=0)、
  bb_width 是 2σ/mean 而 BB_pos 分母是 4σ。MACD 用 adjust=False 讓數值與
  歷史長度無關（adjust=True 在截斷序列上會算出不同的值）。
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# 特徵清單
# =============================================================================

FEATURES_37 = [
    "MA_spread", "c_vs_MA5", "c_vs_MA20", "c_vs_MA60",
    "MA5_slope", "MA20_slope", "vol_ratio", "vol_trend",
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "vol_10d", "vol_20d", "RSI", "MACD_DIF", "MACD_Signal", "MACD_Hist",
    "dist_high20", "BB_pos", "K_raw", "vol_conv",
    "foreign_3d", "foreign_5d", "trust_3d", "trust_5d", "inst_momentum",
    "foreign_consec_buy", "trust_consec_buy",
    "taiex_above_MA20", "taiex_trend",
    "margin_chg_5d", "short_chg_5d", "margin_usage", "short_ratio",
    "industry_momentum",
]
EXTRA_4 = ["bb_width", "max_gain_20d", "max_loss_20d", "vol_breakout_days"]

FEATURES_NORM = [
    "foreign_3d_norm", "foreign_5d_norm", "foreign_20d_norm",
    "trust_3d_norm", "trust_5d_norm", "trust_20d_norm",
    "inst_momentum_norm", "short_ratio_norm", "margin_bal_norm",
]

# ★ 強勢路徑：門檻式/路徑式，捕捉「真的有東西在推」的狀態
FEATURES_STRENGTH = [
    "close_pos_day",      # 收盤在當日高低區間的位置（1=收最高）
    "close_pos_5d",       # 近5日平均
    "limit_up_5d",        # 近5日漲停次數
    "limit_up_20d",       # 近20日漲停次數
    "consec_up",          # 連續上漲天數
    "up_days_10d",        # 近10日上漲天數比例
    "gap_up_5d",          # 近5日向上跳空次數
]

# ★ 籌碼集中度（TaiwanStockHoldingSharesPer，每週）
FEATURES_HOLDING = [
    "big_holder_pct",     # 400張以上持股比例
    "big_holder_chg_4w",  # 4週變化（正 = 籌碼集中中）
    "small_holder_pct",   # 未滿10張（散戶）持股比例
    "small_holder_chg_4w",
]

# ★ 基本面（TaiwanStockMonthRevenue，每月，已處理公布時點）
FEATURES_REVENUE = [
    "rev_yoy",            # 最新可用月營收的年增率
    "rev_mom",            # 月增率
    "rev_yoy_3m",         # 近3月營收合計的年增率
    "rev_growth_streak",  # 連續年增為正的月數
]

# ★ 估值（TaiwanStockPER，每日）
FEATURES_VALUE = [
    "per_val", "pbr_val", "div_yield", "per_pctile_1y",
]

FEATURES_41 = FEATURES_37 + EXTRA_4          # 舊契約（v4/v5）
FEATURES_ALL = (FEATURES_41 + FEATURES_NORM + FEATURES_STRENGTH
                + FEATURES_HOLDING + FEATURES_REVENUE + FEATURES_VALUE)

MIN_HISTORY = 120   # 暖機。推論端至少要 MIN_HISTORY + 1 根 K 棒

FOREIGN_TOKENS = ("外資", "Foreign_Investor", "Foreign_Dealer_Self")
TRUST_TOKENS = ("投信", "Investment_Trust")
_MARGIN_COLS = ("MarginPurchaseTodayBalance", "MarginPurchaseLimit",
                "ShortSaleTodayBalance")

LIMIT_UP_THRESHOLD = 0.095   # 台股漲跌幅上限 10%，用 9.5% 判定漲停
BIG_HOLDER_MIN_SHARES = 400_001   # 400 張以上 = 大戶
SMALL_HOLDER_MAX_SHARES = 10_000  # 未滿 10 張 = 散戶
REVENUE_PUB_LAG_DAYS = 15    # 月營收：次月 10 日前公布，保守用 15 天


# =============================================================================
# 小工具
# =============================================================================

def _safe_div(a, b, fill: float = 0.0):
    """b 為 0/NaN 時回傳 fill。兩邊必須是索引相同的 Series ——
    若一邊是 ndarray、另一邊是 DatetimeIndex 的 Series，pandas 會以聯集對齊，
    產出全 NaN 再被 fillna 變成一整欄「看起來很合理的常數」，完全靜默。"""
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        raise TypeError("_safe_div 只接受 Series，避免索引對齊造成靜默錯誤")
    if not a.index.equals(b.index):
        raise ValueError("_safe_div 的兩個 Series 索引不一致，會靜默錯位")
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def _consec_positive(s: pd.Series) -> pd.Series:
    """從當日往前數，連續為正的天數。以「非正值的累積次數」分組歸零計數器。"""
    pos = s > 0
    return pos.groupby((~pos).cumsum()).cumsum().astype(float)


def _breakout_days(vol: pd.Series, window: int = 20,
                   mult: float = 1.5) -> pd.Series:
    """過去 window 天裡，有幾天的量 > mult × 該窗口的平均量。
    注意是「窗口內每天 vs 該窗口自己的平均」，不是「每天 vs 自己的移動平均」。"""
    def _count(a: np.ndarray) -> float:
        m = a.mean()
        return float((a > mult * m).sum()) if m > 0 else 0.0
    return vol.rolling(window).apply(_count, raw=True)


def _reindex_daily(df: Optional[pd.DataFrame], index: pd.DatetimeIndex,
                   value_cols: Iterable[str], how: str) -> pd.DataFrame:
    """對齊到交易日索引。how 必填 —— 把存量欄位誤走 sum 會把餘額加總，
    數字看起來合理但全錯。
        'zero'  流量（買賣超）→ 同日相加，缺漏補 0
        'ffill' 存量（餘額）→ 同日取最後，缺漏沿用前一日
    """
    cols = list(value_cols)
    if how not in ("zero", "ffill"):
        raise ValueError(f"how 必須是 'zero' 或 'ffill'，收到 {how!r}")
    if df is None or len(df) == 0:
        return pd.DataFrame(0.0, index=index, columns=cols)

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    have = [c for c in cols if c in d.columns]
    lost = [c for c in cols if c not in have]
    if lost:
        logger.warning("資料缺少欄位 %s，以 0 計", lost)
    for c in have:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    if not have:
        return pd.DataFrame(0.0, index=index, columns=cols)

    g = (d.groupby("date")[have].sum() if how == "zero"
         else d.sort_values("date").groupby("date")[have].last())
    g = g.reindex(index)
    g = g.fillna(0.0) if how == "zero" else g.ffill().fillna(0.0)
    for c in lost:
        g[c] = 0.0
    return g[cols]


# =============================================================================
# 分區塊
# =============================================================================

def _inst_net_series(inst_df, index: pd.DatetimeIndex):
    """外資/投信逐日淨股數。舊版用 str.contains('外資')，但 FinMind v4 回傳的是
    Foreign_Investor / Investment_Trust，篩選為空 → 7 個特徵永遠是 0。"""
    zero = pd.Series(0.0, index=index)
    if inst_df is None or len(inst_df) == 0 or "name" not in inst_df.columns:
        return zero, zero, False

    d = inst_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    if "net" not in d.columns:
        def _col(name: str) -> pd.Series:
            # 不能用 d.get(name, 0)：欄位不存在時回傳純量，沒有 .fillna
            if name in d.columns:
                return pd.to_numeric(d[name], errors="coerce").fillna(0)
            logger.warning("法人資料缺少 %s 欄位，以 0 計", name)
            return pd.Series(0.0, index=d.index)
        d["net"] = _col("buy") - _col("sell")
    d["net"] = pd.to_numeric(d["net"], errors="coerce").fillna(0)

    names = d["name"].astype(str)
    fm = names.str.contains("|".join(FOREIGN_TOKENS), case=False, na=False, regex=True)
    tm = names.str.contains("|".join(TRUST_TOKENS), case=False, na=False, regex=True)
    if not fm.any() and not tm.any():
        logger.warning("法人名稱比對不到任何列。實際值=%s", sorted(names.unique())[:8])
        return zero, zero, False

    f = d[fm].groupby("date")["net"].sum().reindex(index).fillna(0.0)
    t = d[tm].groupby("date")["net"].sum().reindex(index).fillna(0.0)
    return f, t, True


def _institutional_block(f_net, t_net, vol_ma20, ok: bool) -> pd.DataFrame:
    idx = f_net.index
    out = pd.DataFrame(index=idx)
    old = ["foreign_3d", "foreign_5d", "trust_3d", "trust_5d", "inst_momentum",
           "foreign_consec_buy", "trust_consec_buy"]
    new = ["foreign_3d_norm", "foreign_5d_norm", "foreign_20d_norm",
           "trust_3d_norm", "trust_5d_norm", "trust_20d_norm", "inst_momentum_norm"]
    if not ok:
        for c in old + new:
            out[c] = 0.0
        return out

    # 舊定義：絕對股數/1e6。除數必須與訓練一致（LightGBM 記絕對門檻）
    f6, t6 = f_net / 1e6, t_net / 1e6
    out["foreign_3d"] = f6.rolling(3).sum()
    out["foreign_5d"] = f6.rolling(5).sum()
    out["trust_3d"] = t6.rolling(3).sum()
    out["trust_5d"] = t6.rolling(5).sum()
    out["inst_momentum"] = out["foreign_5d"] + out["trust_5d"]
    out["foreign_consec_buy"] = _consec_positive(f_net)
    out["trust_consec_buy"] = _consec_positive(t_net)
    # 新定義：除以 20 日均量 →「買了幾天的量」，跨股票可比
    for k in (3, 5, 20):
        out[f"foreign_{k}d_norm"] = _safe_div(f_net.rolling(k).sum(), vol_ma20)
        out[f"trust_{k}d_norm"] = _safe_div(t_net.rolling(k).sum(), vol_ma20)
    out["inst_momentum_norm"] = out["foreign_5d_norm"] + out["trust_5d_norm"]
    return out


def _margin_block(margin_df, index, volume, vol_ma20) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    m = _reindex_daily(margin_df, index, _MARGIN_COLS, how="ffill")
    mb, sb, ml = (m["MarginPurchaseTodayBalance"], m["ShortSaleTodayBalance"],
                  m["MarginPurchaseLimit"])
    # 原版：(bal[-1]-bal[-5])/(bal[-5]+1)，跨 4 期。用 _safe_div 而非裸除：
    # 分母雖有 +1，餘額為負（資料異常）時仍可能為 0，fillna 清不掉 inf
    out["margin_chg_5d"] = _safe_div(mb - mb.shift(4), mb.shift(4) + 1)
    out["short_chg_5d"] = _safe_div(sb - sb.shift(4), sb.shift(4) + 1)
    out["margin_usage"] = _safe_div(mb, ml)
    out["short_ratio"] = _safe_div(sb, volume)          # 原版：當日量
    out["short_ratio_norm"] = _safe_div(sb, vol_ma20)   # 新：20日均量
    out["margin_bal_norm"] = _safe_div(mb, vol_ma20)
    return out


def _market_block(market_df, index) -> pd.DataFrame:
    """大盤特徵。同一天對所有股票相同（正常，不是 bug）。"""
    out = pd.DataFrame(index=index)
    if market_df is None or len(market_df) == 0:
        logger.warning("大盤資料缺失 → taiex_* 全為 0")
        out["taiex_above_MA20"] = 0.0
        out["taiex_trend"] = 0.0
        return out
    d = market_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    # 先 groupby 去重再 reindex：重複日期會讓 reindex 直接拋
    close = (pd.to_numeric(d.groupby("date")["close"].last(), errors="coerce")
             .sort_index().reindex(index).ffill())
    out["taiex_above_MA20"] = (close > close.rolling(20).mean()).astype(float)
    out["taiex_trend"] = ((close - close.shift(19)) / close.shift(19)).fillna(0.0)
    return out


def _strength_block(close, high, low, open_, index) -> pd.DataFrame:
    """★ 強勢路徑特徵。

    這些是門檻式/路徑式的，跟現有的連續型指標互補：
      * 收盤在日內區間的位置 —— 我們有 high/low/close 卻從沒用過，是個疏漏。
        收在最高附近 = 買盤把價格頂到收盤，強度訊號。
      * 漲停 —— 台股 10% 漲跌幅限制讓漲停成為一個離散狀態（買盤強到掛不進去），
        連續價格特徵抓不到這個非線性。
    """
    out = pd.DataFrame(index=index)
    rng = high - low
    out["close_pos_day"] = _safe_div(close - low, rng, fill=0.5)
    out["close_pos_5d"] = out["close_pos_day"].rolling(5).mean()

    ret1 = close.pct_change(fill_method=None)
    is_lu = (ret1 >= LIMIT_UP_THRESHOLD).astype(float)
    out["limit_up_5d"] = is_lu.rolling(5).sum()
    out["limit_up_20d"] = is_lu.rolling(20).sum()

    out["consec_up"] = _consec_positive(ret1.fillna(0.0))
    out["up_days_10d"] = (ret1 > 0).astype(float).rolling(10).mean()

    # 向上跳空：開盤 > 前一日收盤 × 1.02
    gap = _safe_div(open_ - close.shift(1), close.shift(1))
    out["gap_up_5d"] = (gap > 0.02).astype(float).rolling(5).sum()
    return out


def _holding_block(holding_df, index) -> pd.DataFrame:
    """★ 籌碼集中度（TaiwanStockHoldingSharesPer，每週）。

    HoldingSharesLevel 是字串區間（"400,001-600,000"、"1-999" 等）。
    取開頭的數字判斷級距：>= 400,001 股（400張）算大戶，< 10,001 算散戶。
    每週資料 reindex 到交易日後 ffill —— 只用已公布的值，沒有前視。
    """
    out = pd.DataFrame(index=index)
    cols = FEATURES_HOLDING
    if holding_df is None or len(holding_df) == 0:
        for c in cols:
            out[c] = 0.0
        return out

    d = holding_df.copy()
    need = {"date", "HoldingSharesLevel", "percent"}
    if not need.issubset(d.columns):
        logger.warning("持股分級資料缺欄位（有 %s），以 0 計", list(d.columns)[:8])
        for c in cols:
            out[c] = 0.0
        return out

    d["date"] = pd.to_datetime(d["date"])
    d["percent"] = pd.to_numeric(d["percent"], errors="coerce")
    lvl = d["HoldingSharesLevel"].astype(str)
    # 跳過合計列
    d = d[~lvl.str.contains("total|合計|總計", case=False, na=False)].copy()
    lvl = d["HoldingSharesLevel"].astype(str)

    def _low(s: str) -> float:
        m = re.search(r"[\d,]+", s)
        return float(m.group().replace(",", "")) if m else np.nan

    d["_low"] = lvl.map(_low)
    d = d.dropna(subset=["_low", "percent"])

    big = (d[d["_low"] >= BIG_HOLDER_MIN_SHARES]
           .groupby("date")["percent"].sum())
    small = (d[d["_low"] < SMALL_HOLDER_MAX_SHARES]
             .groupby("date")["percent"].sum())

    b = big.sort_index().reindex(index).ffill()
    s = small.sort_index().reindex(index).ffill()
    out["big_holder_pct"] = b.fillna(0.0)
    out["small_holder_pct"] = s.fillna(0.0)
    # 4 週 ≈ 20 個交易日
    out["big_holder_chg_4w"] = (b - b.shift(20)).fillna(0.0)
    out["small_holder_chg_4w"] = (s - s.shift(20)).fillna(0.0)
    return out


def _revenue_block(rev_df, index) -> pd.DataFrame:
    """★ 月營收（TaiwanStockMonthRevenue）。

    【前視偏誤處理】FinMind 的 date 是營收所屬月份的月底，但實際公布是
    次月 10 日前。直接對齊 date 會讓模型提前 10~40 天看到還沒公布的數字。
    這裡一律延後 REVENUE_PUB_LAG_DAYS(15) 天才視為可用。
    """
    out = pd.DataFrame(index=index)
    cols = FEATURES_REVENUE
    if rev_df is None or len(rev_df) == 0 or "revenue" not in rev_df.columns:
        for c in cols:
            out[c] = 0.0
        return out

    d = rev_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["revenue"] = pd.to_numeric(d["revenue"], errors="coerce")
    d = d.dropna(subset=["revenue"]).sort_values("date")
    if len(d) < 13:
        for c in cols:
            out[c] = 0.0
        return out

    d = d.drop_duplicates("date", keep="last").set_index("date")
    rev = d["revenue"]
    yoy = rev / rev.shift(12) - 1.0
    mom = rev / rev.shift(1) - 1.0
    yoy3 = rev.rolling(3).sum() / rev.rolling(3).sum().shift(12) - 1.0
    streak = _consec_positive(yoy.fillna(-1.0))

    m = pd.DataFrame({"rev_yoy": yoy, "rev_mom": mom,
                      "rev_yoy_3m": yoy3, "rev_growth_streak": streak})
    m = m.replace([np.inf, -np.inf], np.nan)
    # ★ 公布時點：營收月月底 + 15 天之後才可用
    m.index = m.index + pd.Timedelta(days=REVENUE_PUB_LAG_DAYS)
    m = m[~m.index.duplicated(keep="last")].sort_index()

    for c in cols:
        out[c] = m[c].reindex(index.union(m.index)).ffill().reindex(index).fillna(0.0)
    return out


def _value_block(per_df, index) -> pd.DataFrame:
    """★ 估值（TaiwanStockPER：PER / PBR / 殖利率）。

    除了原始值，還算「本益比在自己過去一年的分位」—— 相對自己的歷史便宜或貴，
    比絕對水準更可比（不同產業的合理 PER 差很多）。
    """
    out = pd.DataFrame(index=index)
    if per_df is None or len(per_df) == 0:
        for c in FEATURES_VALUE:
            out[c] = 0.0
        return out

    d = per_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date", keep="last").set_index("date")

    def _col(name: str) -> pd.Series:
        if name not in d.columns:
            return pd.Series(np.nan, index=index)
        return (pd.to_numeric(d[name], errors="coerce")
                .reindex(index.union(d.index)).ffill().reindex(index))

    per = _col("PER").replace([np.inf, -np.inf], np.nan)
    pbr = _col("PBR").replace([np.inf, -np.inf], np.nan)
    dy = _col("dividend_yield").replace([np.inf, -np.inf], np.nan)

    out["per_val"] = per.clip(-100, 300).fillna(0.0)
    out["pbr_val"] = pbr.clip(0, 50).fillna(0.0)
    out["div_yield"] = dy.clip(0, 30).fillna(0.0)
    # 一年（約 250 交易日）內的分位，trailing
    out["per_pctile_1y"] = (per.rolling(250, min_periods=60)
                            .rank(pct=True).fillna(0.5))
    return out


# =============================================================================
# 主函式
# =============================================================================

def compute_feature_frame(
    price_df: pd.DataFrame,
    inst_df: Optional[pd.DataFrame] = None,
    margin_df: Optional[pd.DataFrame] = None,
    market_df: Optional[pd.DataFrame] = None,
    industry_momentum: Optional[pd.Series] = None,
    holding_df: Optional[pd.DataFrame] = None,
    revenue_df: Optional[pd.DataFrame] = None,
    per_df: Optional[pd.DataFrame] = None,
    min_history: int = MIN_HISTORY,
) -> pd.DataFrame:
    """單一檔股票每個交易日的全部特徵（FEATURES_ALL，69 個）。

    price_df 需含 date/close/max/min/Trading_Volume（有 open 更好，
    強勢路徑特徵的跳空判定會用到）。建議用還原股價。
    industry_momentum 由呼叫端跨股票橫斷面算好傳入（單檔算不出產業動能）。

    推論 = .iloc[-1].to_dict()；訓練 = 整張表。同一段程式碼，這是重點。
    """
    # 用 <= 而不是 <：最後會做 f.iloc[min_history:]，剛好等於會切出空表
    if price_df is None or len(price_df) <= min_history:
        return pd.DataFrame(columns=FEATURES_ALL)
    need = ("date", "close", "max", "min", "Trading_Volume")
    missing = [c for c in need if c not in price_df.columns]
    if missing:
        raise KeyError(f"price_df 缺少欄位：{missing}")

    p = price_df.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").drop_duplicates("date", keep="last")
    for c in ("close", "max", "min", "Trading_Volume"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    if "open" in p.columns:
        p["open"] = pd.to_numeric(p["open"], errors="coerce")
    p = p.dropna(subset=["close", "max", "min", "Trading_Volume"])
    if len(p) <= min_history:
        return pd.DataFrame(columns=FEATURES_ALL)

    p = p.set_index("date")
    idx = p.index
    close, high, low, vol = p["close"], p["max"], p["min"], p["Trading_Volume"]
    open_ = p["open"] if "open" in p.columns else close.copy()
    f = pd.DataFrame(index=idx)

    # ---- 均線 ----
    ma5, ma20, ma60 = (close.rolling(5).mean(), close.rolling(20).mean(),
                       close.rolling(60).mean())
    f["MA_spread"] = _safe_div(ma5 - ma20, ma20)
    f["c_vs_MA5"] = _safe_div(close - ma5, ma5)
    f["c_vs_MA20"] = _safe_div(close - ma20, ma20)
    f["c_vs_MA60"] = _safe_div(close - ma60, ma60)
    f["MA5_slope"] = _safe_div(ma5 - ma5.shift(4), ma5.shift(4))    # 原版是 4 期
    f["MA20_slope"] = _safe_div(ma20 - ma20.shift(4), ma20.shift(4))

    # ---- 量 ----
    vol_ma5, vol_ma20 = vol.rolling(5).mean(), vol.rolling(20).mean()
    f["vol_ratio"] = _safe_div(vol, vol_ma20, fill=1.0)
    f["vol_trend"] = _safe_div(vol_ma5, vol_ma20, fill=1.0)
    f["vol_conv"] = f["vol_trend"]      # 與 vol_trend 相同，訓練時 drop 一個

    # ---- 報酬 ----
    for name, k in (("ret_1d", 1), ("ret_3d", 3), ("ret_5d", 5),
                    ("ret_10d", 10), ("ret_20d", 20)):
        f[name] = _safe_div(close - close.shift(k), close.shift(k))

    # ---- 波動（母體標準差 ddof=0；fill_method=None 不用前值填停牌）----
    rets = close.pct_change(fill_method=None)
    f["vol_10d"] = rets.rolling(10).std(ddof=0).fillna(0.0)
    f["vol_20d"] = rets.rolling(20).std(ddof=0).fillna(0.0)

    # ---- RSI（14 期簡單平均，非 Wilder）----
    delta = close.diff()
    ag = delta.clip(lower=0).rolling(14).mean()
    al = (-delta).clip(lower=0).rolling(14).mean()
    f["RSI"] = (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(100 - 100 / 101)

    # ---- MACD（adjust=False 讓數值與歷史長度無關）----
    dif = (close.ewm(span=12, adjust=False).mean()
           - close.ewm(span=26, adjust=False).mean())
    sig = dif.ewm(span=9, adjust=False).mean()
    f["MACD_DIF"] = _safe_div(dif, close)
    f["MACD_Signal"] = _safe_div(sig, close)
    f["MACD_Hist"] = _safe_div(dif - sig, close)

    # ---- 價格位置 ----
    high20, low20 = high.rolling(20).max(), low.rolling(20).min()
    f["dist_high20"] = _safe_div(close - high20, high20)
    std20 = close.rolling(20).std(ddof=0)
    f["BB_pos"] = _safe_div(close - (ma20 - 2 * std20), 4 * std20, fill=0.5)
    f["K_raw"] = _safe_div(close - low20, high20 - low20, fill=0.5)

    # ---- 籌碼 / 大盤 ----
    f_net, t_net, inst_ok = _inst_net_series(inst_df, idx)
    f = f.join(_institutional_block(f_net, t_net, vol_ma20, inst_ok))
    f = f.join(_market_block(market_df, idx))
    f = f.join(_margin_block(margin_df, idx, vol, vol_ma20))

    # ---- 產業動能（舊版恆為 0，因為 industry_returns 是永遠空的 dict）----
    if industry_momentum is not None and len(industry_momentum):
        im = pd.Series(industry_momentum)
        im.index = pd.to_datetime(im.index)
        f["industry_momentum"] = im.reindex(idx).fillna(0.0).astype(float)
    else:
        f["industry_momentum"] = 0.0

    # ---- 空方額外 4 個 ----
    # bb_width 是 2σ/mean（半寬），BB_pos 分母是 4σ（全寬）—— 不一致但照原版
    f["bb_width"] = _safe_div(2 * std20, close.rolling(20).mean())
    c20 = close.shift(20)
    f["max_gain_20d"] = _safe_div(close.rolling(20).max() - c20, c20)
    f["max_loss_20d"] = _safe_div(close.rolling(20).min() - c20, c20)
    f["vol_breakout_days"] = _breakout_days(vol).fillna(0.0)

    # ---- ★ 新增四組 ----
    f = f.join(_strength_block(close, high, low, open_, idx))
    f = f.join(_holding_block(holding_df, idx))
    f = f.join(_revenue_block(revenue_df, idx))
    f = f.join(_value_block(per_df, idx))

    f = f[FEATURES_ALL]
    f = f.iloc[min_history:]
    return f.replace([np.inf, -np.inf], np.nan)


def compute_latest(price_df: pd.DataFrame, **kwargs) -> Optional[Dict[str, float]]:
    """推論用：只取最後一天。刻意做成薄包裝而非另一份實作 ——
    兩份實作就是 skew 的定義。"""
    frame = compute_feature_frame(price_df, **kwargs)
    if len(frame) == 0:
        return None
    row = frame.iloc[-1]
    arr = row.to_numpy(dtype="float64", na_value=np.nan)
    if not np.all(np.isfinite(arr)):
        bad = [c for c in frame.columns if not np.isfinite(row[c])]
        logger.warning("最新一列特徵含 NaN/Inf：%s", bad)
        return None
    return {c: float(row[c]) for c in frame.columns}


def industry_momentum_panel(price_panel: pd.DataFrame,
                            industry_map: Dict[str, str],
                            lookback: int = 5) -> pd.DataFrame:
    """每個「日期 × 產業」的動能。只用截至當日的資料。"""
    p = price_panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["close"] = pd.to_numeric(p["close"], errors="coerce")
    wide = p.pivot_table(index="date", columns="stock_id",
                         values="close", aggfunc="last").sort_index()
    # fill_method=None：預設 'pad' 會用前值填停牌，讓報酬看起來是 0 而非缺值
    ret = wide.pct_change(lookback, fill_method=None)
    industries = pd.Series({c: (industry_map.get(c) or "Unknown")
                            for c in wide.columns})
    return pd.DataFrame({ind: ret[list(cols.index)].mean(axis=1)
                         for ind, cols in industries.groupby(industries)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    print(f"特徵總數：{len(FEATURES_ALL)}")
    for name, group in (("價量技術", FEATURES_41), ("籌碼正規化", FEATURES_NORM),
                        ("★強勢路徑", FEATURES_STRENGTH),
                        ("★籌碼集中", FEATURES_HOLDING),
                        ("★基本面", FEATURES_REVENUE), ("★估值", FEATURES_VALUE)):
        print(f"  {name:<12}{len(group):>3} 個")
    rng = np.random.default_rng(0)
    n = 400
    px = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n))),
    })
    px["open"] = px["close"] * (1 + rng.normal(0, 0.004, n))
    px["max"] = np.maximum(px["close"], px["open"]) * (1 + abs(rng.normal(0, 0.008, n)))
    px["min"] = np.minimum(px["close"], px["open"]) * (1 - abs(rng.normal(0, 0.008, n)))
    px["Trading_Volume"] = rng.integers(1e6, 5e7, n).astype(float)
    fr = compute_feature_frame(px)
    print(f"\n合成測試：輸出 {fr.shape[0]} 列 × {fr.shape[1]} 欄")
    print(f"含 NaN：{[c for c in fr.columns if fr[c].isna().any()] or '無'}")
    print(f"常數欄：{[c for c in fr.columns if fr[c].nunique(dropna=False) <= 1]}")
    print("（籌碼/大盤/產業/持股/營收/估值 無輸入資料，恆為 0 是預期的）")

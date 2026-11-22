#!/usr/bin/env python3
"""
predict_v2.py v5 — 用 stock_model_v12 產生每日名單（做多），並推送到 Telegram

===============================================================================
v5 新增：--telegram 推送
===============================================================================
跑完把名單推到 TG，跟舊的 /predict 一樣方便。

**不需要動 bot.py。** 這裡是直接呼叫 Telegram 的 HTTP API
（api.telegram.org/bot<TOKEN>/sendDocument），跟 bot 那支程式完全獨立 ——
只是共用同一個 bot token。

chat_id 的解析順序：
    1. --chat-id（逗號分隔可多個）
    2. 環境變數 TG_CHAT_ID / TELEGRAM_CHAT_ID
    3. data/watchlist.json 的 key（bot.py 就是用 chat_id 當 key）

訊息用純文字不用 Markdown —— 股票名稱可能含 `*`（全額交割股標記），
Markdown 會把它當粗體語法，輕則排版亂掉重則 400 錯誤。

推送失敗不會讓當天的名單白算：檔案先寫完才送，送失敗只記 log。

===============================================================================
v4：--daemon 常駐模式（給 Docker 用）
===============================================================================
    python predict_v2.py --daemon --at 18:30 --telegram

每個工作日 18:30 自己跑，log 印到 stdout 讓 docker logs 收。

排程時間用 Asia/Taipei 算，不依賴容器的 TZ —— python:3.11-slim 沒裝 tzdata
時 TZ 環境變數是沒作用的，會靜默用 UTC，18:30 變成台灣清晨 2:30 而且不報錯。

常駐時自動開 --strict-today：今天的收盤資料還沒更新就當失敗、重試，
而不是靜默輸出昨天的名單當成今天的。連假期間會看到 4 次重試後放棄，正常。

失敗不會殺掉容器：所有錯誤路徑都拋 RunError，daemon 接住、記 log、
等 --retry-wait-min 再試，最多 --retries 次。

===============================================================================
v3：三個修正
===============================================================================
1. **只抓目標日期在池子裡的股票**（每日成本從 2830 降到約 1800）
2. **名稱註記** —— 帶 `*` 多為全額交割股。**不過濾、不影響排序**，只標示。
   實測 2026-09-24 第一名就是「5314 世紀*」。這類股票買進要先繳足現金、
   流動性差，回測不知道這件事，實際滑價比回測數字大。
3. **已存在的檔案會檢查評分池大小** —— 踩過一次：--universe 30 試跑留下
   47 檔池子的名單，之後正式跑被靜默跳過，完全沒有異狀可察覺。

===============================================================================
跟舊 predict_engine.py 的四個關鍵差異（每一個都是踩過的坑）
===============================================================================
1. **特徵由 features.py 計算** —— 跟訓練同一份實作。舊版自己抄一份，7 個法人
   特徵因為只比對中文名稱而恆為 0，加上 industry_momentum 讀永遠空的 dict，
   共 8 個死特徵，佔模型 13~18% 的分裂決策。
2. **還原股價（TaiwanStockPriceAdj）** —— 跟訓練一致。舊版用未還原，除權息
   當天出現假跌幅，7~9 月旺季全市場受影響。
3. **選股池用成交金額排名** —— 直接呼叫訓練用的 build_universe。
4. **用 Booster.predict() 取機率** —— LGBMClassifier.predict() 回的是 0/1
   標籤，排序會變成「從一堆 1 裡面隨便挑」，而且完全不會報錯。

===============================================================================
FinMind 全市場查詢的限制
===============================================================================
`TaiwanStockPrice` 不給 data_id 時，**只有 start_date == end_date 才會回傳
全市場資料**。給日期區間會回空集合，status 還是 200 —— 靜默失敗。
作法：先試小區間，回空就退回逐日查詢。

===============================================================================
快取
===============================================================================
build_dataset 的快取鍵含「年-月」，同月重跑會回該月第一次跑的舊資料 ——
對建訓練集是對的，對每日預測是錯的。所以這支自己管增量快取。

===============================================================================
用法
===============================================================================
    python predict_v2.py --model model/stock_model_v12.pkl --max-per-hour 4800
    python predict_v2.py --daemon --at 18:30 --telegram --max-per-hour 4800
    python predict_v2.py --backfill-from 2026-07-25 --backfill-to 2026-08-21

查 FinMind 每小時上限：
    python -c "import os,requests;r=requests.get('https://api.web.finmindtrade.com/v2/user_info',headers={'Authorization':'Bearer '+os.environ['FINMIND_TOKEN']}).json();print(r.get('api_request_limit'),r.get('user_count'))"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

try:
    from features import (FEATURES_ALL, MIN_HISTORY, compute_feature_frame,
                          industry_momentum_panel)
    from build_dataset import (ApiError, Client, RateLimited, build_universe,
                               fetch_stock_info)
except ImportError as ex:
    sys.exit(f"需要 features.py 與 build_dataset.py 在同一資料夾：{ex}")

# force=True：build_dataset 在 module 層級也呼叫了 basicConfig，不加的話
# 這裡的設定會被忽略，log 只剩 HH:MM:SS 沒有日期 —— daemon 跑幾個月之後
# 翻 log 會分不出是哪一天。
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", force=True)
logger = logging.getLogger("predict_v2")

FAST_SETS = ("TaiwanStockInstitutionalInvestorsBuySell",
             "TaiwanStockMarginPurchaseShortSale")
SLOW_SETS = ("TaiwanStockHoldingSharesPer", "TaiwanStockMonthRevenue",
             "TaiwanStockPER")
HISTORY_DAYS = 760      # 餵飽 per_pctile_1y 的 rolling(250) 與月營收 shift(12)
POOL_SHRINK_TOL = 0.8   # 已存在的評分池小於預期的這個比例就重算
TG_API = "https://api.telegram.org"
SEP = "=" * 78

# 排程時區。python:3.11-slim 沒裝 tzdata 時 ZoneInfo 會拋錯，
# 這時退回系統時間並警告 —— 不能靜默用 UTC。
try:
    from zoneinfo import ZoneInfo
    TZ: Optional["ZoneInfo"] = ZoneInfo("Asia/Taipei")
except Exception:                                    # noqa: BLE001
    TZ = None


class RunError(RuntimeError):
    """單次執行失敗。daemon 接住繼續，單次模式印出來後結束。"""


def now_tpe() -> datetime:
    return datetime.now(TZ) if TZ is not None else datetime.now()


def name_flag(name: str) -> str:
    """名稱上的交易註記。**只標示，不過濾。**

    台股慣例：名稱帶 `*` 多半是全額交割股（買進要先繳足現金、流動性差、
    買賣價差大）。回測只看到有價格有成交金額，當成一般股票計算，
    所以實際滑價會比回測數字大。
    """
    return "⚠全額交割/注意股（名稱帶*）" if "*" in str(name) else ""


# =============================================================================
# Telegram（直接打 HTTP API，不經過 bot.py）
# =============================================================================

def resolve_chat_ids(args) -> List[str]:
    raw = (args.chat_id or os.environ.get("TG_CHAT_ID")
           or os.environ.get("TELEGRAM_CHAT_ID"))
    if raw:
        return [x.strip() for x in str(raw).split(",") if x.strip()]
    # 退路：bot.py 的 watchlist.json 就是用 chat_id 當 key
    wl = Path(args.out_dir).parent / "watchlist.json"
    if wl.exists():
        try:
            d = json.loads(wl.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                ids = [str(k) for k in d if str(k).lstrip("-").isdigit()]
                if ids:
                    logger.info("chat_id 取自 %s：%s", wl.name, ids)
                    return ids
        except Exception as ex:                       # noqa: BLE001
            logger.debug("讀不到 watchlist.json：%s", ex)
    return []


def tg_send(token: str, chat_id: str, text: str,
            file_path: Optional[Path] = None) -> None:
    """純文字，不用 parse_mode —— 股名可能含 `*`，Markdown 會當成語法。"""
    if file_path and Path(file_path).exists():
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{TG_API}/bot{token}/sendDocument",
                data={"chat_id": chat_id, "caption": text[:1024]},
                files={"document": (Path(file_path).name, f)}, timeout=60)
    else:
        r = requests.post(f"{TG_API}/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": text[:4096]},
                          timeout=30)
    if r.status_code != 200:
        # 不要把 URL 印出來 —— 裡面含 token
        raise RuntimeError(f"HTTP {r.status_code}：{r.text[:200]}")
    if not r.json().get("ok"):
        raise RuntimeError(str(r.json())[:200])


def build_message(ds: str, picks: List[Dict], cfg: Dict, n_pool: int) -> str:
    up, dn = float(cfg.get("up", 0.12)), float(cfg.get("dn", 0.08))
    lines = [f"📈 {ds} 做多名單（評分池 {n_pool} 檔）", ""]
    for p in picks:
        c = p["close_adj"]
        lines.append(f"{p['rank']}. {p['stock_id']} {p['stock_name']}　"
                     f"{c:.2f} → 目標 {c*(1+up):.2f} ／ 停損 {c*(1-dn):.2f}")
    flagged = [p for p in picks if p.get("flag")]
    if flagged:
        lines.append("")
        lines.append("⚠ " + "、".join(p["stock_id"] for p in flagged)
                     + " 名稱帶 *（多為全額交割股）：需先繳足現金、"
                       "流動性差、滑價比回測大")
    lines += ["",
              f"規則：隔日開盤買進 → +{up:.0%} 出場 → -{dn:.0%} 停損 "
              f"→ 最多 {cfg.get('horizon')} 個交易日",
              "目標/停損價以今日收盤估算，實際請用明日開盤價重算",
              "分數僅供同日排序，絕對值無意義。目前為紙上交易階段。"]
    return "\n".join(lines)


# =============================================================================
# 全市場行情（決定候選池）
# =============================================================================

def fetch_market_snapshot(cli: Client, end_ts: pd.Timestamp,
                          want_days: int = 6,
                          max_lookback: int = 25) -> Optional[pd.DataFrame]:
    """FinMind 的 TaiwanStockPrice 不給 data_id 時，只有單一日期查詢才會回傳
    全市場資料；給區間會回空集合而且 status 仍是 200（靜默失敗）。"""
    lo = (end_ts - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    hi = end_ts.strftime("%Y-%m-%d")
    try:
        df = cli.get("TaiwanStockPrice", lo, hi)
    except RateLimited:
        raise
    except ApiError as ex:
        logger.debug("區間查詢失敗：%s", ex)
        df = None
    if df is not None and len(df) > 200 and "stock_id" in df.columns:
        logger.info("全市場行情：區間查詢成功（%d 筆）", len(df))
        return df

    logger.info("區間查詢回空 → 改用逐日查詢（FinMind 全市場只吃單一日期）")
    frames: List[pd.DataFrame] = []
    d = end_ts
    for _ in range(max_lookback):
        if len(frames) >= want_days:
            break
        ds = d.strftime("%Y-%m-%d")
        d -= pd.Timedelta(days=1)
        if pd.Timestamp(ds).weekday() >= 5:
            continue
        try:
            day = cli.get("TaiwanStockPrice", ds, ds)
        except RateLimited:
            raise
        except ApiError as ex:
            logger.debug("%s 查詢失敗：%s", ds, ex)
            continue
        if day is not None and len(day) > 200 and "stock_id" in day.columns:
            frames.append(day)
            logger.info("  %s：%d 筆（含權證，之後只留 4 碼普通股）", ds, len(day))
    return pd.concat(frames, ignore_index=True) if frames else None


# =============================================================================
# 增量快取
# =============================================================================

class Store:
    """每個 (資料集, 股票) 一個 parquet，存完整歷史，每次只補抓新的那幾天。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.n_full = self.n_incr = self.n_skip = 0

    def _path(self, dataset: str, sid: str) -> Path:
        d = self.root / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{sid}.parquet"

    @staticmethod
    def _norm(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or len(df) == 0 or "date" not in df.columns:
            return None
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        return d if len(d) else None

    def fetch(self, cli: Client, dataset: str, sid: str,
              start: str, end: str, soft: bool = False,
              refresh_days: int = 0) -> Optional[pd.DataFrame]:
        p = self._path(dataset, sid)
        cached = None
        if p.exists():
            try:
                cached = self._norm(pd.read_parquet(p))
            except Exception:
                p.unlink(missing_ok=True)

        end_ts = pd.Timestamp(end)
        getter = cli.get_soft if soft else cli.get

        if cached is not None:
            last = cached["date"].max()
            if refresh_days > 0 and (end_ts - last).days <= refresh_days:
                self.n_skip += 1
                return cached
            if last >= end_ts:
                self.n_skip += 1
                return cached
            nxt = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            fresh = self._norm(getter(dataset, nxt, end, sid))
            self.n_incr += 1
            if fresh is None:
                return cached
            # 用「整列」去重而不是只看 date —— 法人與持股分級同一天有多列
            merged = (pd.concat([cached, fresh], ignore_index=True)
                      .drop_duplicates().sort_values("date")
                      .reset_index(drop=True))
            try:
                merged.to_parquet(p, index=False)
            except Exception as ex:
                logger.debug("快取寫入失敗 %s/%s：%s", dataset, sid, ex)
            return merged

        full = self._norm(getter(dataset, start, end, sid))
        self.n_full += 1
        if full is None:
            return None
        try:
            full.sort_values("date").reset_index(drop=True).to_parquet(p, index=False)
        except Exception as ex:
            logger.debug("快取寫入失敗 %s/%s：%s", dataset, sid, ex)
        return full


# =============================================================================
# 模型
# =============================================================================

def load_model(path: Path) -> Dict:
    if not path.exists():
        raise RunError(f"找不到模型檔 {path}")
    with open(path, "rb") as f:
        m = pickle.load(f)
    if not isinstance(m, dict) or "boosters" not in m or "features" not in m:
        raise RunError(f"{path} 不是 train_final.py 產生的模型檔")
    cfg = m.get("config", {})
    logger.info("模型 %s　訓練至 %s　%d 個特徵　%d 個種子",
                m.get("schema", "?"), m.get("train_end", "?"),
                len(m["features"]), len(m["boosters"]))
    logger.info("策略：目標 +%.0f%%　停損 -%.0f%%　最多抱 %s 天　每天前 %s 名　%s 進場",
                cfg.get("up", 0) * 100, cfg.get("dn", 0) * 100,
                cfg.get("horizon"), cfg.get("top_n"), cfg.get("entry"))
    oos = m.get("oos") or {}
    if oos:
        logger.info("走查成績：命中 %.1f%%（隨機 %.1f%%，倍數 %.2f）",
                    oos.get("hit_rate", 0) * 100,
                    (oos.get("rand_hit_rate") or 0) * 100,
                    oos.get("multiple", float("nan")))
    return m


def score(model: Dict, X: pd.DataFrame) -> np.ndarray:
    """多個種子的機率取平均。用 Booster.predict()，它一定回傳機率。"""
    feats = model["features"]
    miss = [c for c in feats if c not in X.columns]
    if miss:
        raise RunError(f"特徵表缺少模型需要的欄位：{miss}")
    return np.mean([b.predict(X[feats]) for b in model["boosters"]], axis=0)


# =============================================================================
# 單次執行
# =============================================================================

def run_once(args) -> List[str]:
    """跑一次，回傳寫出的日期。失敗一律拋 RunError，讓 daemon 能接住。"""
    token = args.token or os.environ.get("FINMIND_TOKEN")
    if not token:
        raise RunError("缺少 token：請設定環境變數 FINMIND_TOKEN")

    model = load_model(Path(args.model))
    cfg = model.get("config", {})
    TOP_N = int(cfg.get("top_n", 5))
    UP, DN = float(cfg.get("up", 0.12)), float(cfg.get("dn", 0.08))
    train_end = pd.Timestamp(model["train_end"]) if model.get("train_end") else None

    explicit = bool(args.date or args.backfill_from)
    if args.backfill_from:
        d0 = pd.Timestamp(args.backfill_from)
        d1 = (pd.Timestamp(args.backfill_to) if args.backfill_to
              else pd.Timestamp(now_tpe().date()))
        want_range, end = (d0, d1), d1
    elif args.date:
        end = pd.Timestamp(args.date)
        want_range = (end, end)
    else:
        end = pd.Timestamp(now_tpe().date())
        want_range = (end, end)

    if train_end is not None and want_range[0] <= train_end:
        logger.warning("⚠ 目標日期 %s 早於或等於模型訓練結束日 %s —— "
                       "那段對模型是訓練期內，不能當樣本外證據",
                       want_range[0].date(), train_end.date())

    start = end - pd.Timedelta(days=int(args.history_days))
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    logger.info("資料區間 %s ~ %s；目標日期 %s ~ %s",
                s, e, want_range[0].date(), want_range[1].date())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Client 的快取關掉 —— 它的鍵含年-月，對每日預測會回舊資料。用 Store 自己管。
    cli = Client(token, max_per_hour=args.max_per_hour, cache_dir=None)
    store = Store(Path(args.cache_dir))

    # ---- 1. 股票清單 ----
    try:
        info = fetch_stock_info(cli)
    except (ApiError, RateLimited) as ex:
        raise RunError(f"取不到股票清單：{ex}") from ex
    sids = info["stock_id"].astype(str)
    ind_col = (info["industry_category"] if "industry_category" in info.columns
               else pd.Series(["Unknown"] * len(info)))
    industry_map = {sid: (ind if isinstance(ind, str) and ind else "Unknown")
                    for sid, ind in zip(sids, ind_col)}
    name_map = (dict(zip(sids, info["stock_name"]))
                if "stock_name" in info.columns else {})

    # ---- 2. 候選 ----
    try:
        probe = fetch_market_snapshot(cli, end)
    except RateLimited as ex:
        raise RunError(f"額度用盡：{ex}") from ex
    if probe is None:
        raise RunError("取不到全市場行情（非交易日或收盤資料未更新？）")
    probe = probe[probe["stock_id"].astype(str).str.match(r"^\d{4}$")].copy()
    if "Trading_money" in probe.columns:
        probe["_t"] = pd.to_numeric(probe["Trading_money"], errors="coerce").fillna(0)
    else:
        probe["_t"] = (pd.to_numeric(probe.get("close"), errors="coerce").fillna(0)
                       * pd.to_numeric(probe.get("Trading_Volume"),
                                       errors="coerce").fillna(0))
    n_cand = int(args.universe * 1.6)
    cands = (probe.groupby("stock_id")["_t"].sum()
             .nlargest(n_cand).index.astype(str).tolist())
    logger.info("候選 %d 檔（近期成交金額前 %d）", len(cands), n_cand)

    # ---- 3. 大盤 ----
    market = None
    for mid in (args.market_id, "0050"):
        try:
            market = cli.get("TaiwanStockPrice", s, e, mid)
        except RateLimited as ex:
            raise RunError(f"額度用盡：{ex}") from ex
        except ApiError as ex:
            logger.warning("大盤 %s 失敗：%s", mid, ex)
            market = None
        if market is not None and len(market):
            logger.info("大盤使用 %s（%d 筆）", mid, len(market))
            break
    if market is None or len(market) == 0:
        # 訓練時大盤有真值，線上補 0 就是 train-serve skew，寧可中止
        raise RunError("取不到大盤資料，補 0 會造成 skew")
    market = market[["date", "close"]].copy()

    # ---- 4. 逐檔抓價量 ----
    prices: Dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for i, sid in enumerate(cands, 1):
        if i % 50 == 0:
            logger.info("價量 %d/%d（API %d，%.0f 分）",
                        i, len(cands), cli.n_calls, (time.time() - t0) / 60)
        try:
            raw = store.fetch(cli, "TaiwanStockPriceAdj", sid, s, e)
        except RateLimited as ex:
            raise RunError(f"額度用盡，已完成 {i-1} 檔：{ex}") from ex
        except ApiError as ex:
            logger.warning("%s 價量失敗：%s", sid, ex)
            continue
        if raw is None or len(raw) < MIN_HISTORY + 2:
            continue
        need = ("date", "close", "max", "min", "Trading_Volume")
        if any(c not in raw.columns for c in need):
            continue
        d = raw[[c for c in list(need) + ["open", "Trading_money"]
                 if c in raw.columns]].copy()
        for c in ("close", "max", "min", "Trading_Volume"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["open"] = (pd.to_numeric(d["open"], errors="coerce")
                     if "open" in d.columns else d["close"])
        d["Trading_money"] = (pd.to_numeric(d["Trading_money"], errors="coerce")
                              if "Trading_money" in d.columns
                              else d["close"] * d["Trading_Volume"])
        d = (d.dropna(subset=["date", "close", "max", "min", "Trading_Volume"])
             .sort_values("date").drop_duplicates("date", keep="last")
             .reset_index(drop=True))
        if len(d) >= MIN_HISTORY + 2:
            prices[sid] = d
    logger.info("可用價量 %d 檔", len(prices))
    if not prices:
        raise RunError("沒有可用的價量資料")

    # ---- 5. 選股池 ----
    panel = pd.concat([d.assign(stock_id=sid) for sid, d in prices.items()],
                      ignore_index=True)
    universe = build_universe(panel, args.universe)
    universe["date"] = pd.to_datetime(universe["date"])
    uni_dates = sorted(universe["date"].unique())
    if not uni_dates:
        raise RunError("選股池是空的（資料量不足 20 個交易日？）")
    latest = pd.Timestamp(uni_dates[-1])

    targets = [pd.Timestamp(d) for d in uni_dates
               if want_range[0] <= d <= want_range[1]]
    if not targets and not explicit:
        if args.strict_today:
            # daemon 用這個：今天的資料還沒到就拋錯讓外面重試，
            # 而不是靜默產生「昨天的名單」當成今天的
            raise RunError(f"{want_range[0].date()} 還沒有行情"
                           f"（資料最後一天是 {latest.date()}）")
        logger.info("%s 沒有行情（非交易日或收盤未更新），改用最近交易日 %s",
                    want_range[0].date(), latest.date())
        targets = [latest]
    if not targets:
        raise RunError(f"目標區間內沒有交易日（資料最後一天是 {latest.date()}）")

    # 已存在的檔案：除了「在不在」，還要看當初的評分池夠不夠大。
    # 踩過一次：--universe 30 試跑留下 47 檔池子的名單，正式跑時被靜默跳過。
    uni_counts = universe.groupby("date")["stock_id"].size()
    if not args.overwrite:
        todo, n_skip = [], 0
        for d in targets:
            p = out_dir / f"pick_{d.strftime('%Y-%m-%d')}.json"
            if not p.exists():
                todo.append(d)
                continue
            try:
                prev_n = int(json.loads(p.read_text(encoding="utf-8")).get("n_pool", 0))
            except Exception:
                todo.append(d)
                continue
            exp_n = int(uni_counts.get(d, 0))
            if exp_n and prev_n < exp_n * POOL_SHRINK_TOL:
                logger.warning("%s 已存在，但當初評分池只有 %d 檔（這次 %d 檔）→ 重算",
                               d.date(), prev_n, exp_n)
                todo.append(d)
            else:
                n_skip += 1
        if n_skip:
            logger.info("跳過已存在且評分池足夠的 %d 天（--overwrite 可強制重算）",
                        n_skip)
        targets = todo
    if not targets:
        logger.info("沒有需要計算的日期。")
        return []
    logger.info("要產生 %d 天：%s ~ %s", len(targets),
                targets[0].date(), targets[-1].date())

    # ★ 只取「目標日期當天在池子裡」的股票。之前抓的是全期曾進池的 455 檔，
    #   但後面 merge 只留當天的 ~300 檔，白抓 155 檔 × 5 個資料集。
    tset = set(targets)
    active = sorted(universe.loc[universe["date"].isin(tset), "stock_id"].unique())
    logger.info("目標日期的選股池共 %d 檔（全期曾進池 %d 檔）",
                len(active), universe["stock_id"].nunique())

    # ---- 6. 產業動能 ----
    ind_panel = industry_momentum_panel(panel, industry_map, lookback=5)

    # ---- 7. 逐檔特徵 ----
    feats_needed = model["features"]
    rows: List[pd.DataFrame] = []
    for i, sid in enumerate(active, 1):
        if i % 50 == 0:
            logger.info("特徵 %d/%d（API %d，%.0f 分）",
                        i, len(active), cli.n_calls, (time.time() - t0) / 60)
        px = prices.get(sid)
        if px is None:
            continue
        try:
            inst = store.fetch(cli, FAST_SETS[0], sid, s, e)
            marg = store.fetch(cli, FAST_SETS[1], sid, s, e)
            hold = store.fetch(cli, SLOW_SETS[0], sid, s, e, soft=True,
                               refresh_days=args.slow_refresh_days)
            rev = store.fetch(cli, SLOW_SETS[1], sid, s, e, soft=True,
                              refresh_days=args.slow_refresh_days)
            per = store.fetch(cli, SLOW_SETS[2], sid, s, e, soft=True,
                              refresh_days=args.slow_refresh_days)
        except RateLimited as ex:
            raise RunError(f"額度用盡，已完成 {i-1} 檔：{ex}") from ex
        except ApiError as ex:
            # 補 0 會讓這檔的特徵分佈跟訓練時不一樣，寧可整檔跳過
            logger.warning("%s 籌碼失敗，跳過：%s", sid, ex)
            continue

        ind = industry_map.get(sid, "Unknown") or "Unknown"
        im = ind_panel[ind] if ind in ind_panel.columns else None
        f = compute_feature_frame(px, inst_df=inst, margin_df=marg,
                                  market_df=market, industry_momentum=im,
                                  holding_df=hold, revenue_df=rev, per_df=per)
        if len(f) == 0:
            continue
        f = f.loc[f.index.isin(tset)]
        if len(f) == 0:
            continue
        f = f.copy()
        f["stock_id"] = sid
        f["close"] = px.set_index("date")["close"].reindex(f.index)
        f.index.name = "date"
        rows.append(f.reset_index())

    if not rows:
        raise RunError("沒有算出任何特徵（目標日期太接近資料起點？暖機需 120 天）")
    allf = pd.concat(rows, ignore_index=True)
    allf["date"] = pd.to_datetime(allf["date"])
    allf = allf.merge(universe, on=["stock_id", "date"], how="inner")

    n_before = len(allf)
    allf = allf.dropna(subset=feats_needed, how="any")
    if n_before - len(allf):
        logger.info("因特徵含 NaN 剔除 %d 列（訓練時也是這樣處理）",
                    n_before - len(allf))
    if len(allf) == 0:
        raise RunError("所有列的特徵都含 NaN，無法評分")

    # ---- 8. 評分 + 輸出 ----
    allf["score"] = score(model, allf)
    written: List[str] = []
    last_payload: Optional[Dict] = None
    flagged_total = 0
    for d in targets:
        g = allf[allf["date"] == d]
        if len(g) < TOP_N:
            logger.warning("%s 只有 %d 檔可評分，跳過", d.date(), len(g))
            continue
        g = g.sort_values("score", ascending=False)
        ds = d.strftime("%Y-%m-%d")

        picks = []
        for r, (_, row) in enumerate(g.head(TOP_N).iterrows()):
            sid = str(row["stock_id"])
            nm = str(name_map.get(sid, sid))
            fl = name_flag(nm)
            if fl:
                flagged_total += 1
            picks.append({"rank": r + 1, "stock_id": sid, "stock_name": nm,
                          "flag": fl, "score": float(row["score"]),
                          "close_adj": float(row["close"]),
                          "vol_20d": float(row["vol_20d"])})

        # 整個評分池都存下來 —— compare_review.py 要靠它建同波動對照組，
        # 也讓之後想比 top-10 時可以直接重算，不用重跑
        pool = [{"stock_id": str(a), "score": float(b), "vol_20d": float(c),
                 "close_adj": float(x)}
                for a, b, c, x in zip(g["stock_id"], g["score"],
                                      g["vol_20d"], g["close"])]

        (out_dir / f"pick_{ds}.json").write_text(json.dumps({
            "date": ds,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "predict_v2",
            "model": {"schema": model.get("schema"),
                      "train_end": model.get("train_end"),
                      "config": cfg, "features": len(feats_needed)},
            "in_training_period": bool(train_end is not None and d <= train_end),
            "n_pool": len(g), "picks": picks, "pool": pool,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        _write_excel(out_dir / f"pick_{ds}.xlsx", ds, picks, cfg, model, len(g))
        written.append(ds)
        last_payload = {"ds": ds, "picks": picks, "n_pool": len(g)}
        logger.info("%s：%s", ds, "　".join(
            f"{p['stock_id']} {p['stock_name']}({p['score']:.3f})" for p in picks))

    logger.info("完成 %d 天　API %d 次（完整 %d／增量 %d／快取 %d）",
                len(written), cli.n_calls, store.n_full, store.n_incr, store.n_skip)
    if flagged_total:
        logger.warning("名單中有 %d 檔名稱帶 `*`（多為全額交割股）："
                       "買進要先繳足現金、流動性差，實際滑價比回測大。"
                       "已照常列入排名。", flagged_total)

    # ---- 9. 推 Telegram（檔案已經寫完，送失敗不影響名單）----
    if args.telegram and last_payload:
        _notify(args, cfg, last_payload, out_dir)
    return written


def _notify(args, cfg: Dict, payload: Dict, out_dir: Path) -> None:
    bot_token = (args.bot_token or os.environ.get("BOT_TOKEN")
                 or os.environ.get("TELEGRAM_BOT_TOKEN"))
    if not bot_token:
        logger.warning("要推 Telegram 但找不到 BOT_TOKEN，跳過推送（名單已存檔）")
        return
    chat_ids = resolve_chat_ids(args)
    if not chat_ids:
        logger.warning("要推 Telegram 但找不到 chat_id，跳過推送（名單已存檔）。"
                       "請設定 TG_CHAT_ID 或用 --chat-id")
        return
    ds = payload["ds"]
    text = build_message(ds, payload["picks"], cfg, payload["n_pool"])
    xlsx = out_dir / f"pick_{ds}.xlsx"
    for cid in chat_ids:
        try:
            tg_send(bot_token, cid, text, xlsx if xlsx.exists() else None)
            logger.info("已推送到 chat %s", cid)
        except Exception as ex:                       # noqa: BLE001
            # 推送失敗不該讓當天的名單白算，也不該殺掉 daemon
            logger.warning("推送到 chat %s 失敗：%s", cid, ex)


def _write_excel(path: Path, ds: str, picks: List[Dict], cfg: Dict,
                 model: Dict, n_pool: int) -> None:
    try:
        from openpyxl import Workbook
    except ImportError:
        logger.warning("沒有 openpyxl，跳過 Excel（JSON 還是有產生）")
        return
    up, dn = float(cfg.get("up", 0.12)), float(cfg.get("dn", 0.08))
    wb = Workbook()
    ws = wb.active
    ws.title = "做多名單"
    ws.append(["排名", "代號", "名稱", "收盤(還原)", "模型分數",
               "目標價(+%.0f%%)" % (up * 100), "停損價(-%.0f%%)" % (dn * 100),
               "20日波動", "註記"])
    for p in picks:
        c = p["close_adj"]
        # 寫數字不寫字串 —— 字串在 Excel 裡不能排序也不能算
        ws.append([p["rank"], p["stock_id"], p["stock_name"],
                   round(c, 2), round(p["score"], 4),
                   round(c * (1 + up), 2), round(c * (1 - dn), 2),
                   round(p["vol_20d"], 4), p.get("flag", "")])
    for col, w in zip("ABCDEFGHI", (6, 10, 14, 12, 10, 14, 14, 10, 28)):
        ws.column_dimensions[col].width = w

    ws2 = wb.create_sheet("說明")
    for line in [
        ["日期", ds], ["模型", str(model.get("schema"))],
        ["訓練資料截止", str(model.get("train_end"))],
        ["評分池檔數", n_pool], [],
        ["操作規則"],
        ["進場", "隔日開盤買進（不是今天收盤）"],
        ["停利", f"漲到 +{up:.0%} 出場"],
        ["停損", f"跌到 -{dn:.0%} 出場"],
        ["最長持有", f"{cfg.get('horizon')} 個交易日，到期平倉"], [],
        ["注意"],
        ["分數只能用來比較同一天的股票排名，絕對值沒有意義"],
        ["目標價/停損價是用今日收盤估的，實際請以明日開盤成交價重算"],
        ["名稱帶 * 多為全額交割股：要先繳足現金、流動性差、滑價大"],
        ["先紙上交易，累積幾週後用 compare_review.py 檢視成效"],
    ]:
        ws2.append(line)
    ws2.column_dimensions["A"].width = 18
    ws2.column_dimensions["B"].width = 50
    wb.save(path)


# =============================================================================
# main / daemon
# =============================================================================

def _next_run(at_hh: int, at_mm: int) -> datetime:
    """下一個工作日的 at 時刻。"""
    n = now_tpe()
    t = n.replace(hour=at_hh, minute=at_mm, second=0, microsecond=0)
    if t <= n:
        t += timedelta(days=1)
    while t.weekday() >= 5:            # 週六日往後推
        t += timedelta(days=1)
    return t


def main():
    ap = argparse.ArgumentParser(description="用 stock_model_v12 產生每日名單")
    ap.add_argument("--token", default=None, help="FinMind，預設讀 FINMIND_TOKEN")
    ap.add_argument("--model", default="model/stock_model_v12.pkl")
    ap.add_argument("--out-dir", default="data/predictions")
    ap.add_argument("--cache-dir", default=".cache_live")
    ap.add_argument("--universe", type=int, default=300)
    ap.add_argument("--max-per-hour", type=int, default=1800)
    ap.add_argument("--slow-refresh-days", type=int, default=7)
    ap.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    ap.add_argument("--date", default=None)
    ap.add_argument("--backfill-from", default=None)
    ap.add_argument("--backfill-to", default=None)
    ap.add_argument("--market-id", default="TAIEX")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--strict-today", action="store_true",
                    help="今天的行情還沒更新就報錯，不要退回用昨天的")
    # telegram
    ap.add_argument("--telegram", action="store_true", help="跑完推送到 TG")
    ap.add_argument("--bot-token", default=None, help="預設讀 BOT_TOKEN")
    ap.add_argument("--chat-id", default=None,
                    help="逗號分隔。沒給就讀 TG_CHAT_ID，再沒有就從 "
                         "data/watchlist.json 的 key 取")
    # daemon
    ap.add_argument("--daemon", action="store_true",
                    help="常駐，每個工作日固定時間跑一次（給 Docker 用）")
    ap.add_argument("--at", default="18:30", help="daemon 的執行時刻 HH:MM（台北）")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--retry-wait-min", type=int, default=30)
    ap.add_argument("--run-now", action="store_true",
                    help="daemon 啟動時先跑一次，不等到 --at")
    args = ap.parse_args()

    if TZ is None:
        logger.warning("找不到時區資料（容器缺 tzdata），排程改用系統時間。"
                       "建議 Dockerfile 加 `apt-get install -y tzdata`，"
                       "否則 --at 會依 UTC 計算。")

    if not args.daemon:
        try:
            run_once(args)
        except RunError as ex:
            logger.error("%s", ex)
            sys.exit(1)
        return

    try:
        hh, mm = (int(x) for x in args.at.split(":"))
    except Exception:
        sys.exit(f"--at 格式應為 HH:MM，收到 {args.at!r}")

    logger.info("常駐模式啟動：每週一~五 %02d:%02d（台北）執行", hh, mm)
    if args.telegram:
        ids = resolve_chat_ids(args)
        logger.info("推送目標 chat_id：%s", ids or "（找不到，啟動後會警告）")
    # 常駐時要求「今天的」資料，避免收盤未更新就靜默輸出昨天的名單
    args.strict_today = True

    if args.run_now:
        _attempt(args)

    while True:
        nxt = _next_run(hh, mm)
        wait = (nxt - now_tpe()).total_seconds()
        logger.info("下次執行 %s（%.1f 小時後）",
                    nxt.strftime("%Y-%m-%d %H:%M"), wait / 3600)
        time.sleep(max(wait, 1))
        _attempt(args)


def _attempt(args) -> None:
    """跑一次，失敗就等一下重試。不讓例外殺掉容器。"""
    for k in range(1, args.retries + 1):
        try:
            written = run_once(args)
            if written:
                logger.info("✓ 產生 %s", "、".join(written))
            else:
                logger.info("沒有新的日期需要產生（可能已經跑過了）")
            return
        except RunError as ex:
            logger.warning("第 %d/%d 次失敗：%s", k, args.retries, ex)
        except Exception as ex:                       # noqa: BLE001
            logger.exception("第 %d/%d 次發生未預期錯誤：%s", k, args.retries, ex)
        if k < args.retries:
            logger.info("%d 分鐘後重試", args.retry_wait_min)
            time.sleep(args.retry_wait_min * 60)
    logger.error("重試 %d 次都失敗，今天放棄（可能是休市日，明天會再試）",
                 args.retries)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
compare_review.py v2 — 舊 /predict 名單 vs 新 /pick 名單，用同一把尺比較

===============================================================================
v2 修掉的不公平比較：--top-n
===============================================================================
v1 直接拿兩邊「原本的名單」比：

    舊名單 177 筆 / 36 天  →  每天 10 檔
    新名單  99 筆 / 21 天  →  每天  5 檔

**舊的是 top-10，新的是 top-5。** 只要模型排序有任何能力，前 5 名本來就會
比前 10 名好 —— 新模型量到的一部分優勢可能只是「挑得少」，不是「排得準」。

v2 讓兩邊用**相同檔數**：
  * 新名單：predict_v2 的 JSON 存了完整評分池，直接依分數重算前 N 名
  * 舊名單：本來就依機率排序，取前 N 筆

預設 N 取模型設定的 top_n（5）。想看 10 檔就 --top-n 10。

===============================================================================
為什麼不能直接比兩邊原本的「命中率」
===============================================================================
舊 predict_engine.review() 的命中定義是：

    hit_target = max(預測漲幅 × 0.70, 0.03)

**門檻取決於模型自己的預測值。** 預測 +3% 的股票門檻是 3%，預測 +30% 的是 21%。
模型只要把預測值壓低，命中率就會變高。而新模型根本不預測漲幅。

實測這個差別有多大：同一批舊名單、同一段期間 ——
    用舊規則量：命中 57.6%，看起來很棒
    用同一把尺量：倍數 0.96，輸給隨機

所以這支把兩份名單都用同一套交易規則跑一次（規則從模型 pkl 讀，不寫死）：

    隔日開盤買進 → 漲到 +12% 出場 → 跌到 -8% 停損 → 最多抱 20 個交易日

===============================================================================
第三欄才是真正的裁判
===============================================================================
報表有三欄：舊名單 / 新名單 / **同波動隨機**。

實測 2026-07-27~08-21 那 20 天，隨機組的命中率是 49.5%、年化 +82% ——
那段行情好到隨便買都會摸到 +12%。只看絕對命中率會嚴重誤判。

**能跨期間轉移的是倍數，不是水準**：回測期模型 44.0% / 隨機 37.8% = 1.16，
樣本外 57.6% / 49.5% = 1.16。命中率差了 13 個百分點，倍數一模一樣。

===============================================================================
資料從哪來
===============================================================================
  舊名單   data/predictions/pred_YYYY-MM-DD.json 的 long_top10
           （也吃舊格式 predict_YYYYMMDD.json）
  新名單   data/predictions/pick_YYYY-MM-DD.json 的 pool（依分數重排）
  還原股價 .cache_live/TaiwanStockPriceAdj/*.parquet（predict_v2 建立的）
           快取缺的股票才打 API

**兩邊一律用還原股價評分。** 舊名單當初記錄的 close 是未還原的，7~8 月除權息
旺季會製造假跌幅；用還原股價重算對兩邊都公平。

===============================================================================
用法
===============================================================================
    python compare_review.py --from 2026-07-25 --to 2026-08-21
    python compare_review.py --from 2026-07-25 --to 2026-08-21 --top-n 10

輸出：console 表格 + data/predictions/compare_起_迄.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from build_dataset import ApiError, Client, RateLimited, triple_barrier
except ImportError as ex:
    sys.exit(f"需要 build_dataset.py 在同一資料夾：{ex}")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("compare_review")

VOL_BUCKETS = 5
RAND_REPEAT = 20
TRADING_DAYS_PER_YEAR = 250.0
DEFAULT_COST = 0.0046          # 手續費兩邊打折約 0.16% + 證交稅 0.3%
SEP = "=" * 78


# =============================================================================
# 讀名單
# =============================================================================

def load_new(out_dir: Path) -> Dict[pd.Timestamp, Dict]:
    out: Dict[pd.Timestamp, Dict] = {}
    for p in sorted(out_dir.glob("pick_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out[pd.Timestamp(d["date"])] = d
        except Exception as ex:
            logger.warning("%s 讀取失敗：%s", p.name, ex)
    return out


def load_old(out_dir: Path) -> Dict[pd.Timestamp, List[Dict]]:
    """舊 predict_engine 的 long_top10（已依機率排序）。兩種檔名格式都吃。"""
    out: Dict[pd.Timestamp, List[Dict]] = {}

    def _take(data: Dict, ds: str) -> None:
        items = data.get("long_top10")
        if items is None:
            items = [{"stock_id": i.get("stock_id"),
                      "stock_name": i.get("stock_name", ""),
                      "prob": i.get("prob_long", i.get("prob")),
                      "max_gain": i.get("pred_max", i.get("max_gain")),
                      "close": i.get("close")}
                     for i in data.get("long", [])]
        rows = [{"stock_id": str(i.get("stock_id", "")).strip(),
                 "stock_name": i.get("stock_name", ""),
                 "prob": i.get("prob"), "max_gain": i.get("max_gain"),
                 "close": i.get("close")}
                for i in (items or []) if i.get("stock_id")]
        if rows:
            out[pd.Timestamp(ds)] = rows

    for p in sorted(out_dir.glob("pred_*.json")):
        m = re.match(r"pred_(\d{4}-\d{2}-\d{2})\.json$", p.name)
        if m:
            try:
                _take(json.loads(p.read_text(encoding="utf-8")), m.group(1))
            except Exception as ex:
                logger.warning("%s 讀取失敗：%s", p.name, ex)
    for p in sorted(out_dir.glob("predict_*.json")):
        m = re.match(r"predict_(\d{8})\.json$", p.name)
        if not m:
            continue
        raw = m.group(1)
        ds = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        if pd.Timestamp(ds) in out:
            continue                      # 新格式優先
        try:
            _take(json.loads(p.read_text(encoding="utf-8")), ds)
        except Exception as ex:
            logger.warning("%s 讀取失敗：%s", p.name, ex)
    return out


# =============================================================================
# 價格
# =============================================================================

class Prices:
    """先讀 predict_v2 的還原股價快取，缺的才打 API。"""

    def __init__(self, cache_dir: Path, cli: Optional[Client]):
        self.dir = Path(cache_dir) / "TaiwanStockPriceAdj"
        self.cli = cli
        self._mem: Dict[str, Optional[pd.DataFrame]] = {}
        self.n_cache = self.n_api = self.n_fail = 0

    def get(self, sid: str, start: str, end: str) -> Optional[pd.DataFrame]:
        if sid in self._mem:
            return self._mem[sid]
        df = None
        p = self.dir / f"{sid}.parquet"
        if p.exists():
            try:
                df = pd.read_parquet(p)
                self.n_cache += 1
            except Exception:
                df = None
        if df is None and self.cli is not None:
            try:
                df = self.cli.get("TaiwanStockPriceAdj", start, end, sid)
                self.n_api += 1
            except RateLimited:
                raise
            except ApiError as ex:
                logger.debug("%s 抓取失敗：%s", sid, ex)
                df = None
        if df is None or len(df) == 0:
            self.n_fail += 1
            self._mem[sid] = None
            return None
        need = ("date", "close", "max", "min")
        if any(c not in df.columns for c in need):
            self._mem[sid] = None
            return None
        d = df[[c for c in list(need) + ["open"] if c in df.columns]].copy()
        d["date"] = pd.to_datetime(d["date"])
        for c in ("close", "max", "min"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["open"] = (pd.to_numeric(d["open"], errors="coerce")
                     if "open" in d.columns else d["close"])
        d = (d.dropna(subset=["date", "close", "max", "min"])
             .sort_values("date").drop_duplicates("date", keep="last")
             .reset_index(drop=True))
        self._mem[sid] = d if len(d) else None
        return self._mem[sid]


def vol_20d(px: pd.DataFrame) -> pd.Series:
    """跟 features.py 一致：母體標準差(ddof=0)，停牌不用前值填。"""
    r = px.set_index("date")["close"].pct_change(fill_method=None)
    return r.rolling(20).std(ddof=0)


# =============================================================================
# 評估
# =============================================================================

def outcome_at(px: Optional[pd.DataFrame], date: pd.Timestamp, H: int,
               up: float, dn: float) -> Optional[Dict]:
    """某檔股票在某一天進場，照規則跑完之後的結果。

    直接用 build_dataset.triple_barrier —— 跟訓練同一份實作。
    """
    if px is None or len(px) < H + 2:
        return None
    tb = triple_barrier(px, H, up, dn, entry="next_open")
    if date not in tb.index:
        return None
    row = tb.loc[date]
    if not np.isfinite(row.get("tb_ret", np.nan)):
        return None       # 觀察窗還沒跑完
    return {"ret": float(row["tb_ret"]), "days": float(row["tb_days"]),
            "outcome": row["tb_outcome"], "entry": float(row["tb_entry"])}


def agg(rows: List[Dict], cost: float) -> Dict[str, float]:
    if not rows:
        return {}
    ret = np.array([r["ret"] for r in rows], float)
    days = np.array([r["days"] for r in rows], float)
    oc = np.array([r["outcome"] for r in rows], dtype=object)
    mu = float(np.nanmean(ret))
    ad = float(np.nanmean(days))
    turns = TRADING_DAYS_PER_YEAR / max(ad, 1.0)
    return {"n": len(rows),
            "hit": float((oc == "up").mean()),
            "stop": float((oc == "down").mean()),
            "tmo": float((oc == "timeout").mean()),
            "gross": mu, "net": mu - cost, "days": ad, "turns": turns,
            "annual": (mu - cost) * turns,
            "win": float((ret > 0).mean())}


def main():
    ap = argparse.ArgumentParser(description="兩份名單用同一把尺比較")
    ap.add_argument("--out-dir", default="data/predictions")
    ap.add_argument("--cache-dir", default=".cache_live")
    ap.add_argument("--model", default="model/stock_model_v12.pkl",
                    help="從模型讀交易規則（目標/停損/天數/檔數）")
    ap.add_argument("--from", dest="d_from", default=None)
    ap.add_argument("--to", dest="d_to", default=None)
    ap.add_argument("--top-n", type=int, default=None,
                    help="兩邊各取前 N 檔比較。預設用模型設定的檔數。"
                         "舊名單是 top-10、新名單是 top-5，不對齊就不公平")
    ap.add_argument("--cost", type=float, default=DEFAULT_COST)
    ap.add_argument("--vol-buckets", type=int, default=VOL_BUCKETS)
    ap.add_argument("--token", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    # ---- 交易規則從模型讀，不寫死 ----
    H, UP, DN, MODEL_TOPN = 20, 0.12, 0.08, 5
    mp = Path(args.model)
    if mp.exists():
        try:
            with open(mp, "rb") as f:
                cfg = pickle.load(f).get("config", {})
            H = int(cfg.get("horizon", H))
            UP, DN = float(cfg.get("up", UP)), float(cfg.get("dn", DN))
            MODEL_TOPN = int(cfg.get("top_n", MODEL_TOPN))
        except Exception as ex:
            logger.warning("讀不到模型設定，用預設值：%s", ex)
    TOPN = int(args.top_n) if args.top_n else MODEL_TOPN

    logger.info("評分規則：隔日開盤買進 → +%.0f%% 出場 → -%.0f%% 停損 → 最多 %d 天",
                UP * 100, DN * 100, H)
    logger.info("兩邊都取前 %d 檔比較（成本 %.2f%%／來回，不含滑價）",
                TOPN, args.cost * 100)

    out_dir = Path(args.out_dir)
    new_all, old_all = load_new(out_dir), load_old(out_dir)
    logger.info("找到 新名單 %d 天、舊名單 %d 天", len(new_all), len(old_all))
    if not new_all and not old_all:
        sys.exit(f"{out_dir} 裡沒有任何名單")

    # 原始名單長度（提醒使用者兩邊本來就不一樣長）
    raw_new = [len(v.get("picks", [])) for v in new_all.values()]
    raw_old = [len(v) for v in old_all.values()]
    if raw_new and raw_old:
        logger.info("原始名單長度：舊 %d 檔／天，新 %d 檔／天 → 統一取前 %d 檔",
                    int(np.median(raw_old)), int(np.median(raw_new)), TOPN)

    dates = sorted(set(new_all) | set(old_all))
    if args.d_from:
        dates = [d for d in dates if d >= pd.Timestamp(args.d_from)]
    if args.d_to:
        dates = [d for d in dates if d <= pd.Timestamp(args.d_to)]
    if not dates:
        sys.exit("指定區間內沒有名單")

    token = args.token or os.environ.get("FINMIND_TOKEN")
    cli = Client(token, max_per_hour=1800, cache_dir=None) if token else None
    if cli is None:
        logger.info("沒有 token —— 只用快取裡的股票，缺的會被跳過")
    prices = Prices(Path(args.cache_dir), cli)

    s = (min(dates) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    e = (max(dates) + pd.Timedelta(days=120)).strftime("%Y-%m-%d")

    rng = np.random.default_rng(0)
    res: Dict[str, List[Dict]] = {"old": [], "new": [], "rand": []}
    per_date: List[Dict] = []
    old_detail: List[Dict] = []
    pending = 0
    short_new = short_old = 0

    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        day_rows: Dict[str, List[Dict]] = {"old": [], "new": [], "rand": []}
        pool_vol: Dict[str, float] = {}

        # ---- 新名單：從完整評分池依分數重排，取前 TOPN ----
        newd = new_all.get(d)
        new_ids: List[str] = []
        if newd:
            pool = newd.get("pool") or []
            for it in pool:
                pool_vol[str(it["stock_id"])] = float(it["vol_20d"])
            if pool:
                ranked = sorted(pool, key=lambda x: -float(x["score"]))
                new_ids = [str(i["stock_id"]) for i in ranked[:TOPN]]
            else:
                # 舊版 JSON 沒存 pool，只能用當初的 picks
                new_ids = [str(i["stock_id"])
                           for i in newd.get("picks", [])][:TOPN]
            if len(new_ids) < TOPN:
                short_new += 1
            for sid in new_ids:
                o = outcome_at(prices.get(sid, s, e), d, H, UP, DN)
                if o:
                    day_rows["new"].append(o)

        # ---- 舊名單：本來就依機率排序，取前 TOPN ----
        oldd = old_all.get(d)
        old_ids: List[str] = []
        if oldd:
            sel = oldd[:TOPN]
            if len(sel) < TOPN:
                short_old += 1
            for it in sel:
                sid = it["stock_id"]
                old_ids.append(sid)
                px = prices.get(sid, s, e)
                o = outcome_at(px, d, H, UP, DN)
                if o:
                    day_rows["old"].append(o)
                    mg = it.get("max_gain")
                    if px is not None and mg is not None:
                        w = px[px["date"] > d].head(H)
                        if len(w):
                            actual = float(w["max"].max()) / o["entry"] - 1.0
                            thr = max(float(mg) * 0.70, 0.03)
                            old_detail.append({
                                "date": ds, "stock_id": sid,
                                "pred_gain": float(mg), "old_threshold": thr,
                                "actual_max_gain": actual,
                                "old_rule_hit": bool(actual >= thr),
                                "same_rule_outcome": o["outcome"],
                                "same_rule_ret": o["ret"]})
                if sid not in pool_vol and px is not None:
                    v = vol_20d(px)
                    if d in v.index and np.isfinite(v.loc[d]):
                        pool_vol[sid] = float(v.loc[d])

        # ---- 同波動隨機對照 ----
        ref = new_ids or old_ids
        if pool_vol and ref:
            pv = pd.Series(pool_vol).dropna()
            if len(pv) >= args.vol_buckets * 2:
                try:
                    vb = pd.qcut(pv, args.vol_buckets, labels=False,
                                 duplicates="drop")
                except Exception:
                    vb = pd.Series(0, index=pv.index)
            else:
                vb = pd.Series(0, index=pv.index)
            for x in ref:
                if x not in vb.index:
                    continue
                b = vb.loc[x]
                cand = vb.index[vb.to_numpy() == b].tolist()
                if not cand:
                    continue
                for _ in range(RAND_REPEAT):
                    o = outcome_at(prices.get(str(rng.choice(cand)), s, e),
                                   d, H, UP, DN)
                    if o:
                        day_rows["rand"].append(o)

        if not day_rows["new"] and not day_rows["old"]:
            pending += 1
            continue
        for k in res:
            res[k].extend(day_rows[k])
        per_date.append({
            "date": ds, "old_n": len(day_rows["old"]), "new_n": len(day_rows["new"]),
            "old_ret": float(np.mean([r["ret"] for r in day_rows["old"]]))
            if day_rows["old"] else np.nan,
            "new_ret": float(np.mean([r["ret"] for r in day_rows["new"]]))
            if day_rows["new"] else np.nan,
            "rand_ret": float(np.mean([r["ret"] for r in day_rows["rand"]]))
            if day_rows["rand"] else np.nan})

    if pending:
        logger.info("%d 天的觀察窗還沒跑完（不足 %d 個交易日），已排除", pending, H)
    if short_new or short_old:
        logger.info("名單不足 %d 檔的天數：新 %d 天、舊 %d 天",
                    TOPN, short_new, short_old)
    logger.info("價格來源：快取 %d 檔、API %d 檔、取不到 %d 檔",
                prices.n_cache, prices.n_api, prices.n_fail)

    A = agg(res["old"], args.cost)
    B = agg(res["new"], args.cost)
    R = agg(res["rand"], args.cost)
    if not A and not B:
        sys.exit("沒有任何已到期的交易可以評估")

    print(f"\n{SEP}\n同一把尺：隔日開盤買進 → +{UP:.0%} 出場 → "
          f"-{DN:.0%} 停損 → 最多 {H} 天　｜　兩邊各取前 {TOPN} 檔\n{SEP}")
    print(f"  期間 {per_date[0]['date']} ~ {per_date[-1]['date']}"
          f"（{len(per_date)} 個交易日）\n")

    def col(x: Dict, k: str, pct=True, sign=False) -> str:
        if not x or k not in x or not np.isfinite(x.get(k, np.nan)):
            return f"{'—':>12}"
        v = x[k]
        if not pct:
            return f"{v:>12.1f}"
        return f"{v:>+12.2%}" if sign else f"{v:>12.1%}"

    print(f"  {'':<22}{'舊 /predict':>12}{'新 /pick':>12}{'同波動隨機':>12}")
    print(f"  {'-'*58}")
    print(f"  {'交易筆數':<22}{A.get('n',0):>12}{B.get('n',0):>12}{R.get('n',0):>12}")
    print(f"  {'摸到 +%.0f%% 出場' % (UP*100):<20}"
          f"{col(A,'hit')}{col(B,'hit')}{col(R,'hit')}")
    print(f"  {'摸到 -%.0f%% 停損' % (DN*100):<20}"
          f"{col(A,'stop')}{col(B,'stop')}{col(R,'stop')}")
    print(f"  {'到期平倉':<22}{col(A,'tmo')}{col(B,'tmo')}{col(R,'tmo')}")
    print(f"  {'-'*58}")
    print(f"  {'每筆毛':<22}{col(A,'gross',sign=True)}{col(B,'gross',sign=True)}"
          f"{col(R,'gross',sign=True)}")
    print(f"  {'每筆淨（扣 %.2f%%）' % (args.cost*100):<19}"
          f"{col(A,'net',sign=True)}{col(B,'net',sign=True)}{col(R,'net',sign=True)}")
    print(f"  {'平均持有（天）':<20}{col(A,'days',pct=False)}"
          f"{col(B,'days',pct=False)}{col(R,'days',pct=False)}")

    rh = R.get("hit")
    print(f"\n  {'倍數（命中 ÷ 隨機）':<20}", end="")
    for x in (A, B, R):
        m = (x.get("hit") / rh) if (rh and x.get("hit") is not None) else None
        print(f"{m:>12.2f}" if m else f"{'—':>12}", end="")
    print()

    # 有效樣本：同一天的 N 檔會一起漲跌，獨立觀測接近「天數」而不是「筆數」
    nd = len(per_date)
    if rh and B:
        se = float(np.sqrt(max(rh * (1 - rh), 1e-9) / max(nd, 1)))
        gap = B["hit"] - rh
        print(f"\n  新名單贏隨機 {gap*100:+.1f} 個百分點；"
              f"以 {nd} 天為獨立觀測估，標準誤約 {se*100:.1f} 個百分點"
              f" → 約 {abs(gap)/se if se else 0:.1f} 個標準誤")
        if abs(gap) < 2 * se:
            print(f"  **統計上還不算數。** 同一天的 {TOPN} 檔會一起漲跌，"
                  f"獨立觀測接近天數而非筆數。")

    print(f"\n{SEP}\n判讀\n{SEP}")
    if rh:
        for nm, x in (("舊 /predict", A), ("新 /pick", B)):
            if not x:
                print(f"  {nm}：沒有資料")
                continue
            m = x["hit"] / rh
            v = ("贏過同波動隨機" if m >= 1.05 else
                 "跟隨機差不多" if m >= 0.98 else "輸給隨機")
            print(f"  {nm}：倍數 {m:.2f} → {v}")
    print(f"\n  ・**看倍數，不要看年化。** 這段期間隨機組年化就 "
          f"{R.get('annual', 0):+.0%}，那是行情不是能力，週期一換就消失。")
    print(f"  ・能跨期間轉移的是倍數：回測期 44.0%/37.8% = 1.16，")
    print(f"    樣本外 57.6%/49.5% = 1.16。命中率差 13 個百分點，倍數一樣。")
    print(f"  ・要穩定分辨 1.16 和 1.00，大約需要累積半年以上。")

    if old_detail:
        od = pd.DataFrame(old_detail)
        print(f"\n{SEP}\n附錄：舊 /review 規則下的舊名單命中率\n{SEP}")
        print(f"  命中 {od['old_rule_hit'].mean():.1%}"
              f"（{int(od['old_rule_hit'].sum())}/{len(od)}）"
              f"　平均門檻 {od['old_threshold'].mean():.1%}")
        print(f"  註：舊規則門檻 =「預測漲幅 × 70%」，由模型自己的預測值決定，")
        print(f"      壓低預測就能拉高命中率。這個數字不能跟上面的表互比 ——")
        print(f"      只是讓你對得上以前看慣的那個數字。")

    rep = Path(args.report) if args.report else (
        out_dir / f"compare_{per_date[0]['date']}_{per_date[-1]['date']}_top{TOPN}.xlsx")
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "總比較"
        ws.append(["項目", "舊 /predict", "新 /pick", "同波動隨機"])
        for label, key, kind in [("交易筆數", "n", 0), ("摸到目標出場", "hit", 1),
                                 ("停損出場", "stop", 1), ("到期平倉", "tmo", 1),
                                 ("每筆毛", "gross", 1), ("每筆淨", "net", 1),
                                 ("平均持有天數", "days", 2), ("年周轉", "turns", 2),
                                 ("年化淨", "annual", 1)]:
            vals = []
            for x in (A, B, R):
                v = x.get(key)
                vals.append(None if v is None or not np.isfinite(v)
                            else (round(v, 4) if kind == 1 else
                                  round(v, 2) if kind == 2 else int(v)))
            ws.append([label] + vals)
            if kind == 1:
                for c in "BCD":
                    ws[f"{c}{ws.max_row}"].number_format = "0.0%"
        if rh:
            ws.append(["倍數（命中÷隨機）"] +
                      [round(x["hit"] / rh, 2) if x.get("hit") else None
                       for x in (A, B, R)])
        ws.append([])
        ws.append(["規則", f"隔日開盤買 / +{UP:.0%} 停利 / -{DN:.0%} 停損 / 最多 {H} 天"])
        ws.append(["兩邊取前 N 檔", TOPN])
        ws.append(["成本", args.cost])
        ws[f"B{ws.max_row}"].number_format = "0.00%"
        ws.append(["提醒", "看倍數不要看年化；年化受這段行情影響極大"])
        for col_, w in zip("ABCD", (20, 14, 14, 14)):
            ws.column_dimensions[col_].width = w

        ws2 = wb.create_sheet("逐日")
        ws2.append(["日期", "舊檔數", "新檔數", "舊平均報酬", "新平均報酬",
                    "隨機平均報酬"])
        for r in per_date:
            ws2.append([r["date"], r["old_n"], r["new_n"]] +
                       [None if not np.isfinite(r[k]) else round(r[k], 4)
                        for k in ("old_ret", "new_ret", "rand_ret")])
        for c in "DEF":
            for i in range(2, ws2.max_row + 1):
                ws2[f"{c}{i}"].number_format = "0.0%"
        for col_, w in zip("ABCDEF", (12, 9, 9, 13, 13, 14)):
            ws2.column_dimensions[col_].width = w

        if old_detail:
            ws3 = wb.create_sheet("舊規則對照")
            od = pd.DataFrame(old_detail)
            ws3.append(list(od.columns))
            for _, r in od.iterrows():
                ws3.append([bool(r[c]) if isinstance(r[c], (bool, np.bool_))
                            else r[c] for c in od.columns])
            for col_, w in zip("ABCDEFG", (12, 10, 12, 14, 16, 16, 14)):
                ws3.column_dimensions[col_].width = w

        wb.save(rep)
        print(f"\n報表 → {rep}")
    except ImportError:
        logger.warning("沒有 openpyxl，跳過 Excel")


if __name__ == "__main__":
    main()

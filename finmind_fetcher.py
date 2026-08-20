#!/usr/bin/env python3
"""
FinMind 股價資料抓取模組
取代原本從證交所 API 抓資料的部分，回傳格式與 channel_engine.py / chart_drawer.py 相容
（Date 為 index，欄位為 Open/High/Low/Close/Volume）

使用方式：
    export FINMIND_TOKEN="你的token"   # 建議用環境變數，不要寫死在程式碼裡
    python3 -c "from finmind_fetcher import fetch_stock_ohlc; print(fetch_stock_ohlc('2408'))"
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_SNAPSHOT_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_tick_snapshot"

# 期貨關鍵字對照表：使用者輸入 → FinMind data_id + 中文名稱
FUTURES_ALIASES = {
    '台指': ('TX', '台指期'),
    '台指期': ('TX', '台指期'),
    'tx': ('TX', '台指期'),
    'TX': ('TX', '台指期'),
    '小台': ('MTX', '小型台指期'),
    '小台指': ('MTX', '小型台指期'),
    'mtx': ('MTX', '小型台指期'),
    'MTX': ('MTX', '小型台指期'),
    '電子期': ('TE', '電子期貨'),
    'te': ('TE', '電子期貨'),
    'TE': ('TE', '電子期貨'),
    '金融期': ('TF', '金融期貨'),
    'tf': ('TF', '金融期貨'),
    'TF': ('TF', '金融期貨'),
}

def is_futures_keyword(text: str) -> bool:
    """判斷輸入是否為期貨關鍵字"""
    return text.strip() in FUTURES_ALIASES

def get_futures_info(text: str) -> tuple:
    """回傳 (data_id, display_name)，找不到回傳 (None, None)"""
    return FUTURES_ALIASES.get(text.strip(), (None, None))

# 股票代號 -> 中文名稱 的記憶體快取。基本資料（股名）幾乎不會變動，
# 用簡單的 dict 快取就好，避免每次分析同一檔股票都重打一次 API，
# 也不需要處理過期問題（重啟 bot 就會自然清空重抓）。
_stock_name_cache: dict = {}

# 股名 -> 股號 的反查表（啟動時載入一次）
_stock_name_to_id: dict = {}
_stock_info_loaded: bool = False


def _load_stock_info(token: Optional[str] = None):
    """載入全部台股清單到記憶體，建立股名→股號對照表"""
    global _stock_name_to_id, _stock_info_loaded, _stock_name_cache
    if _stock_info_loaded:
        return
    
    token = token or os.environ.get("FINMIND_TOKEN")
    if not token:
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    params = {"dataset": "TaiwanStockInfo"}
    
    try:
        resp = requests.get(FINMIND_API_URL, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            return
        data = resp.json().get("data", [])
        for row in data:
            sid = row.get('stock_id', '')
            sname = row.get('stock_name', '')
            if sid and sname:
                _stock_name_to_id[sname] = sid
                _stock_name_cache[sid] = sname
        _stock_info_loaded = True
    except Exception:
        pass


def search_stock_by_name(query: str, token: Optional[str] = None) -> Optional[tuple]:
    """
    用股名搜尋股號。支援完全比對和部分比對。

    Args:
        query: 使用者輸入的股名，例如「南亞科」「台積電」

    Returns:
        (stock_id, stock_name) 或 None
    """
    _load_stock_info(token)
    
    if not _stock_name_to_id:
        return None
    
    # 完全比對
    if query in _stock_name_to_id:
        return (_stock_name_to_id[query], query)
    
    # 部分比對（股名包含輸入的文字）
    matches = [(sid, name) for name, sid in _stock_name_to_id.items() if query in name]
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # 多個匹配，優先選完全匹配長度最短的（最精確）
        matches.sort(key=lambda x: len(x[1]))
        return matches[0]
    
    return None


def fetch_stock_name(stock_id: str, token: Optional[str] = None) -> Optional[str]:
    """
    取得股票中文名稱（來自 FinMind TaiwanStockInfo 資料集）。

    Args:
        stock_id: 股票代號，例如 '2408'
        token: FinMind API token，預設從環境變數 FINMIND_TOKEN 讀取

    Returns:
        中文股名字串，例如 '南亞科'；查不到、或發生任何錯誤時回傳 None
        （設計成不拋例外，因為股名只是顯示用的附加資訊，抓不到也不該讓
        整個分析流程掛掉，呼叫端沒拿到名稱時 fallback 顯示代號就好）
    """
    if stock_id in _stock_name_cache:
        return _stock_name_cache[stock_id]

    token = token or os.environ.get("FINMIND_TOKEN")
    if not token:
        return None

    headers = {"Authorization": f"Bearer {token}"}
    params = {"dataset": "TaiwanStockInfo", "data_id": stock_id}

    try:
        resp = requests.get(FINMIND_API_URL, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json().get("data", [])
        if not data:
            return None
        # 同一股票代碼可能因為產業分類調整等原因有多筆歷史紀錄，取最新的一筆
        name = data[-1].get("stock_name")
        if name:
            _stock_name_cache[stock_id] = name
        return name
    except Exception:
        # 股名抓取失敗（連線問題、額度用完等）不影響主流程，回傳 None 讓
        # 呼叫端 fallback 用代號顯示即可
        return None


def fetch_stock_ohlc(
    stock_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    token: Optional[str] = None,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """
    從 FinMind 抓取台股日線 OHLCV 資料

    Args:
        stock_id: 股票代號，例如 '2408'
        start_date: 起始日期 'YYYY-MM-DD'，預設抓 lookback_days 天前
        end_date: 結束日期 'YYYY-MM-DD'，預設今天
        token: FinMind API token，預設從環境變數 FINMIND_TOKEN 讀取
        lookback_days: 沒指定 start_date 時，往回抓幾天資料（預設180天，約半年交易日）

    Returns:
        DataFrame，index 為 Date(datetime)，欄位為 Open/High/Low/Close/Volume
        （跟原本證交所版本輸出格式一致，可以直接餵給 calculate_channel / draw_channel_chart）

    Raises:
        ValueError: token 未提供、或 API 回傳錯誤
        requests.RequestException: 網路連線問題
    """
    token = token or os.environ.get("FINMIND_TOKEN")
    if not token:
        raise ValueError(
            "缺少 FinMind API token。請設定環境變數 FINMIND_TOKEN，"
            "或呼叫時傳入 token='...' 參數。"
        )

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
    }

    resp = requests.get(FINMIND_API_URL, headers=headers, params=params, timeout=10)

    if resp.status_code != 200:
        raise ValueError(f"FinMind API 回應錯誤 (HTTP {resp.status_code}): {resp.text}")

    payload = resp.json()

    if payload.get("status") != 200 and payload.get("msg") not in (None, "success"):
        # FinMind 有時會在 200 status code 底下，data 卻是空的（例如額度用完、股票代號錯誤）
        raise ValueError(f"FinMind API 回傳異常: {payload.get('msg', payload)}")

    data = payload.get("data", [])
    if not data:
        raise ValueError(
            f"FinMind 沒有回傳 {stock_id} 在 {start_date}~{end_date} 的資料，"
            f"請確認股票代號是否正確，或該區間是否有交易日。"
        )

    df = pd.DataFrame(data)

    # FinMind 欄位對照到原本證交所版本使用的欄位名稱
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume",
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    # FinMind 對興櫃/新掛牌首日等特殊情況，open 可能是 0，避免通道計算出錯先過濾掉
    df = df[df["Open"] > 0]

    return df


def fetch_realtime_snapshot(stock_id: str, token: Optional[str] = None) -> Optional[dict]:
    """
    抓取即時報價快照（約每10秒更新一次），使用 FinMind 的
    taiwan_stock_tick_snapshot 資料集。

    ⚠️ 這個資料集需要 Sponsor（付費）方案的 token 才能使用，
    Free / Backer 帳號呼叫會失敗（FinMind 會回傳錯誤訊息）。

    Args:
        stock_id: 股票代號，例如 '2408'
        token: FinMind API token，預設從環境變數 FINMIND_TOKEN 讀取

    Returns:
        dict，包含 close(最新成交價)、change_price(漲跌)、change_rate(漲跌%)、
        date(成交時間) 等欄位；非交易時段或查無資料時回傳 None

    Raises:
        ValueError: token 未提供、或 API 回傳錯誤（例如權限不足、非 Sponsor 方案）
        requests.RequestException: 網路連線問題
    """
    token = token or os.environ.get("FINMIND_TOKEN")
    if not token:
        raise ValueError(
            "缺少 FinMind API token。請設定環境變數 FINMIND_TOKEN，"
            "或呼叫時傳入 token='...' 參數。"
        )

    headers = {"Authorization": f"Bearer {token}"}
    params = {"data_id": stock_id}

    resp = requests.get(FINMIND_SNAPSHOT_URL, headers=headers, params=params, timeout=10)

    if resp.status_code != 200:
        raise ValueError(
            f"FinMind 即時報價 API 回應錯誤 (HTTP {resp.status_code}): {resp.text}\n"
            f"（提示：taiwan_stock_tick_snapshot 需要 Sponsor 付費方案，"
            f"請確認你的帳號等級是否有開通這個資料集）"
        )

    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        return None

    return data[0]  # 指定單一 data_id 時，回傳的 data 只會有一筆


def fetch_stock_ohlc_with_realtime(
    stock_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    token: Optional[str] = None,
    lookback_days: int = 180,
):
    """
    抓歷史日K，並嘗試把「今天盤中即時報價」併入成最後一根K棒，
    這樣通道計算（支撐/壓力線、力竭判斷）跟畫圖，都會把今天的走勢算進去，
    而不是只看到昨天收盤為止。

    運作邏輯：
    1. 先抓日K資料（跟 fetch_stock_ohlc 一樣）
    2. 如果日K資料「已經」包含今天（例如收盤後17:30已經更新），
       代表今天已經是正式收盤資料了，不需要、也不應該用即時資料去覆蓋
    3. 否則呼叫即時報價，如果确定是「今天」的快照、且已經有實際成交
       （open/high/low/close 都 > 0），就把它當成一根「還在形成中」的K棒
       接在日K資料最後面

    ⚠️ 注意：這根「即時K棒」的 High/Low 是「目前為止」的當日最高/最低，
    收盤前還會持續變動，所以用它算出來的通道位置、訊號，會隨盤中價格浮動，
    收盤後才會是當天的最終定案版本。

    Returns:
        (df, is_realtime, snapshot)
        - df: 跟 fetch_stock_ohlc 格式一樣的 DataFrame
        - is_realtime: True 代表最後一根K棒是即時、尚未收盤定案的資料；
          False 代表全部都是已收盤的日K（可能是盤後查詢、或即時資料不可用）
        - snapshot: 原始即時報價 dict（沒有即時資料時為 None），
          方便呼叫端直接拿去顯示漲跌幅等資訊，不用重複呼叫 API
    """
    df = fetch_stock_ohlc(stock_id, start_date=start_date, end_date=end_date,
                           token=token, lookback_days=lookback_days)

    today = pd.Timestamp.now().normalize()

    # 日K資料已經有今天了（通常是收盤後17:30已更新），不需要疊加即時資料
    if len(df) > 0 and df.index[-1] == today:
        return df, False, None

    try:
        snapshot = fetch_realtime_snapshot(stock_id, token=token)
    except Exception:
        # 即時資料抓不到（權限不足、連線問題等），退回純日K，不影響主流程
        return df, False, None

    if not snapshot:
        return df, False, None

    # 確認快照真的是「今天」的資料，避免週末/非交易日撈到舊快照當成今天處理
    snap_date_str = str(snapshot.get('date', ''))[:10]
    try:
        snap_date = pd.Timestamp(snap_date_str).normalize()
    except (ValueError, TypeError):
        return df, False, snapshot

    if snap_date != today:
        return df, False, snapshot

    o, h, l, c = snapshot.get('open'), snapshot.get('high'), snapshot.get('low'), snapshot.get('close')
    if not all(isinstance(v, (int, float)) and v > 0 for v in (o, h, l, c)):
        # 開盤前、或還沒有實際成交，不足以形成一根有意義的K棒
        return df, False, snapshot

    volume = snapshot.get('total_volume') or snapshot.get('volume') or 0

    today_row = pd.DataFrame(
        {'Open': [float(o)], 'High': [float(h)], 'Low': [float(l)],
         'Close': [float(c)], 'Volume': [float(volume)]},
        index=[today],
    )
    df = pd.concat([df, today_row])
    df.index.name = 'Date'

    return df, True, snapshot


def fetch_futures_ohlc(
    futures_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    token: Optional[str] = None,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """
    從 FinMind 抓取台灣期貨日線 OHLCV 資料（近月合約，一般交易時段）

    Args:
        futures_id: 期貨代碼，例如 'TX'（台指期）、'MTX'（小台）
        start_date: 起始日期 'YYYY-MM-DD'
        end_date: 結束日期 'YYYY-MM-DD'
        token: FinMind API token
        lookback_days: 往回抓幾天

    Returns:
        DataFrame，格式同 fetch_stock_ohlc（Date index + Open/High/Low/Close/Volume）
    """
    token = token or os.environ.get("FINMIND_TOKEN")
    if not token:
        raise ValueError("缺少 FinMind API token。")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "dataset": "TaiwanFuturesDaily",
        "data_id": futures_id,
        "start_date": start_date,
        "end_date": end_date,
    }

    resp = requests.get(FINMIND_API_URL, headers=headers, params=params, timeout=10)

    if resp.status_code != 200:
        raise ValueError(f"FinMind API 回應錯誤 (HTTP {resp.status_code}): {resp.text}")

    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        raise ValueError(f"FinMind 沒有回傳 {futures_id} 的期貨資料。")

    df = pd.DataFrame(data)

    # 只取一般交易時段（排除 after_market 盤後）
    if 'trading_session' in df.columns:
        df = df[df['trading_session'] == 'position']

    # 只取近月合約（contract_date 長度為6的純月份，排除價差合約如 202607/202608）
    if 'contract_date' in df.columns:
        df = df[df['contract_date'].str.len() == 6]

    if df.empty:
        raise ValueError(f"過濾後 {futures_id} 無有效資料（可能是非交易時段或資料格式異常）。")

    # 每天可能有多個月份合約，取成交量最大的那個（=近月主力合約）
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    idx_max_vol = df.groupby('date')['volume'].idxmax()
    df = df.loc[idx_max_vol]

    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df[df["Open"] > 0]

    return df


if __name__ == "__main__":
    import sys
    stock_id = sys.argv[1] if len(sys.argv) > 1 else "2408"

    df = fetch_stock_ohlc(stock_id)
    print(df.tail())
    print(f"\n共 {len(df)} 筆日K資料，區間 {df.index[0].date()} ~ {df.index[-1].date()}")

    print("\n--- 即時報價 (需 Sponsor 方案) ---")
    try:
        snapshot = fetch_realtime_snapshot(stock_id)
        if snapshot:
            print(snapshot)
        else:
            print("目前查無即時報價（可能是非交易時段）")
    except ValueError as e:
        print(f"即時報價抓取失敗: {e}")


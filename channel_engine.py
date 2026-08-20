#!/usr/bin/env python3
"""
台股趨勢通道計算引擎
邏輯：
- 多頭：找突破壓力的關鍵K → Low為支撐起點 → 連回測低點 → 等距平行往上
- 空頭：找跌破支撐的關鍵K → High為壓力起點 → 連反彈高點 → 等距平行往下
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class ChannelResult:
    """通道計算結果"""
    trend: str  # 'bullish' or 'bearish'
    slope: float
    intercept: float
    channel_width: float
    breakout_idx: int
    breakout_date: str
    breakout_price: float
    anchor_points: List[Tuple[int, float]]  # (idx, price)
    
    # Current levels
    current_line1: float  # Support (bull) / Resistance (bear)
    current_line2: float  # R1 (bull) / S1 (bear)
    current_line3: float  # R2 (bull) / S2 (bear)
    
    # Signal
    signal: str  # 'buy', 'sell', 'short', 'cover', 'neutral', 'exhaustion', ...
    signal_text: str
    position_pct: float  # 0-100, position within channel

    # 力竭（動能衰竭）判斷
    is_exhausted: bool = False          # 是否判定為力竭警訊
    exhaustion_peak_idx: Optional[int] = None   # 造成力竭判斷的前波高/低點 index
    exhaustion_peak_pct: Optional[float] = None  # 該轉折點當時在通道中的位置(%)

    # 是否為「已超出合理外推範圍」的通道（見 extend_channel_to_now 說明）：
    # True 代表 current_line1/2/3 是用『封頂後』的位置算出來的，不是把陡峭的
    # 斜率無限外推到今天算出來的數字，避免像是短短20天噴出段的斜率，
    # 外推30天後支撐線衝到比歷史最高價還高的離譜情況
    is_stale_extrapolation: bool = False


def find_breakout_k(df: pd.DataFrame, order: int = 5) -> Tuple[Optional[int], str]:
    """
    找到關鍵K（突破/跌破的那根K）
    Returns: (breakout_index, trend_direction)
    """
    low_idx = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
    high_idx = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
    
    if len(low_idx) < 3 or len(high_idx) < 3:
        return None, 'unknown'
    
    # Check recent trend: are the last 3-4 lows ascending or descending?
    recent_lows = low_idx[-4:] if len(low_idx) >= 4 else low_idx[-3:]
    recent_low_vals = [df['Low'].iloc[i] for i in recent_lows]
    
    recent_highs = high_idx[-4:] if len(high_idx) >= 4 else high_idx[-3:]
    recent_high_vals = [df['High'].iloc[i] for i in recent_highs]
    
    # Determine trend by checking if recent price is above or below the midpoint
    # and whether lows/highs are trending
    mid_price = (df['High'].max() + df['Low'].min()) / 2
    current_price = df['Close'].iloc[-1]
    
    # Check if highs are descending (bearish)
    highs_descending = all(recent_high_vals[i] >= recent_high_vals[i+1] 
                          for i in range(len(recent_high_vals)-1))
    
    # Check if lows are ascending (bullish)  
    lows_ascending = all(recent_low_vals[i] <= recent_low_vals[i+1] 
                        for i in range(len(recent_low_vals)-1))
    
    # More robust: check slope of recent highs and lows
    if len(recent_highs) >= 2:
        high_slope = np.polyfit(recent_highs.astype(float), recent_high_vals, 1)[0]
    else:
        high_slope = 0
        
    if len(recent_lows) >= 2:
        low_slope = np.polyfit(recent_lows.astype(float), recent_low_vals, 1)[0]
    else:
        low_slope = 0
    
    # Determine trend
    if low_slope > 0.3 and high_slope > 0:
        trend = 'bullish'
    elif high_slope < -0.3 and low_slope < 0:
        trend = 'bearish'
    elif low_slope > 0:
        trend = 'bullish'
    elif high_slope < 0:
        trend = 'bearish'
    else:
        # Fallback: use price position relative to recent range
        recent_range = df.iloc[-30:]
        if current_price > recent_range['Close'].mean():
            trend = 'bullish'
        else:
            trend = 'bearish'
    
    # Find breakout K
    if trend == 'bullish':
        breakout_idx = find_bullish_breakout(df, low_idx, high_idx, order)
    else:
        breakout_idx = find_bearish_breakout(df, low_idx, high_idx, order)
    
    return breakout_idx, trend


def find_bullish_breakout(df: pd.DataFrame, low_idx, high_idx, order: int) -> Optional[int]:
    """
    找多頭突破的關鍵K：
    - 前面有一段壓力區（前高）
    - 某根K突破該壓力，收盤站上
    - 成交量放大
    """
    # Look for the candle that broke above a previous resistance level
    # Strategy: find where price broke above a previous significant high
    
    # Get resistance levels (previous highs before the current uptrend)
    # Work backwards from recent lows to find where trend started
    
    if len(low_idx) < 2:
        return low_idx[-1] if len(low_idx) > 0 else 0
    
    # Find the lowest recent low (potential trend start area)
    # Then look for the breakout candle after it
    recent_low_indices = low_idx[low_idx > len(df) // 4]  # Skip very early data
    
    if len(recent_low_indices) < 2:
        return low_idx[-1]
    
    # Find the "base" low before uptrend acceleration
    # Look for where consecutive lows start ascending significantly
    for i in range(len(recent_low_indices) - 1):
        curr_low = df['Low'].iloc[recent_low_indices[i]]
        next_low = df['Low'].iloc[recent_low_indices[i+1]]
        
        # If next low is significantly higher, the breakout likely happened between them
        if next_low > curr_low * 1.1:  # 10% higher
            # The breakout K is between these two lows
            # Find the candle with biggest up move and volume in this range
            start = recent_low_indices[i]
            end = recent_low_indices[i+1]
            
            # Find resistance level (highest high before the start)
            pre_high = df['High'].iloc[max(0, start-20):start].max()
            
            # Find the first candle that closed above that resistance
            for j in range(start, end):
                if df['Close'].iloc[j] > pre_high:
                    return j
            
            # Fallback: use the low point itself
            return recent_low_indices[i]
    
    # Fallback: use the earliest ascending low
    return recent_low_indices[0]


def find_bearish_breakout(df: pd.DataFrame, low_idx, high_idx, order: int) -> Optional[int]:
    """
    找空頭跌破的關鍵K：
    - 前面有一段支撐區（前低）
    - 某根K跌破該支撐
    """
    if len(high_idx) < 2:
        return high_idx[-1] if len(high_idx) > 0 else len(df) - 1
    
    # Find the peak (highest high) then look for trend reversal
    peak_idx = high_idx[np.argmax([df['High'].iloc[i] for i in high_idx])]
    
    # After the peak, look for descending highs
    post_peak_highs = high_idx[high_idx > peak_idx]
    
    if len(post_peak_highs) >= 1:
        return peak_idx  # The peak itself is the start of downtrend
    
    return high_idx[-1]


def calculate_channel(df: pd.DataFrame, order: int = 5) -> Optional[ChannelResult]:
    """
    計算趨勢通道
    """
    breakout_idx, trend = find_breakout_k(df, order)
    
    if breakout_idx is None:
        return None
    
    low_idx = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
    high_idx = argrelextrema(df['High'].values, np.greater_equal, order=order)[0]
    
    last_idx = len(df) - 1
    
    if trend == 'bullish':
        return calculate_bullish_channel(df, breakout_idx, low_idx, high_idx, last_idx)
    else:
        return calculate_bearish_channel(df, breakout_idx, low_idx, high_idx, last_idx)


def find_last_swing_before_now(df, slope, intercept, last_idx, breakout_idx, lookback=20, mode='high'):
    """
    找出「目前這波下跌/反彈之前」最近一個轉折高點(或低點)的位置。

    不使用 argrelextrema，因為 argrelextrema 需要左右各 order 根K線才能
    確認轉折，最近幾天的轉折點往往還沒被確認，導致力竭偵測永遠慢半拍。
    這裡改用簡單、即時的做法：往回找 lookback 根K內，「相對通道線」最高(或最低)
    的位置 —— 注意一定要用相對通道線的距離來比較，不能直接比較原始價格，
    因為通道本身有斜率，原始價格會隨時間自然墊高/墊低，直接比較會抓錯轉折點。
    """
    start = max(breakout_idx, last_idx - lookback + 1)
    if start >= last_idx:
        return None

    window = df.iloc[start:last_idx + 1]
    if len(window) < 3:
        return None

    positions = np.arange(start, last_idx + 1, dtype=float)
    trend_at_pos = slope * positions + intercept

    if mode == 'high':
        rel_dist = window['High'].values - trend_at_pos
        rel_pos = int(np.argmax(rel_dist))
    else:
        rel_dist = window['Low'].values - trend_at_pos
        rel_pos = int(np.argmin(rel_dist))

    swing_iloc = start + rel_pos

    # 轉折點如果就是今天，代表現在還在噴出/趕底，還不能判斷力竭
    if swing_iloc >= last_idx:
        return None

    return swing_iloc


def check_bullish_exhaustion(df, slope, intercept, channel_width, breakout_idx, last_idx,
                              lookback=20, mid_threshold=50, near_threshold=60):
    """
    判斷多頭是否『力竭』：

    1. 目前價格已經跌破支撐下軌（而不只是接近）
    2. 這波下跌之前，最後一個反彈高點，並沒有摸到通道中軌(50%)，
       代表反彈力道明顯轉弱、量能不足以再挑戰上緣

    兩個條件都成立 → 明確力竭警訊
    只成立一部分（價格還沒破、但前波反彈很弱）→ 提醒轉弱，避免誤判為買點

    Returns: (is_exhausted, is_weak_warning, peak_iloc, peak_position_pct)
    """
    if channel_width <= 0:
        return False, False, None, None

    current_support = slope * last_idx + intercept
    current_price = df['Close'].iloc[-1]
    current_position_pct = (current_price - current_support) / channel_width * 100

    broke_support = current_price < current_support

    peak_iloc = find_last_swing_before_now(df, slope, intercept, last_idx, breakout_idx,
                                            lookback=lookback, mode='high')
    if peak_iloc is None:
        return False, False, None, None

    support_at_peak = slope * peak_iloc + intercept
    peak_position_pct = (df['High'].iloc[peak_iloc] - support_at_peak) / channel_width * 100

    # 確認高點之後確實一路走弱到現在（用「相對通道位置」比較，不能比較原始價格，
    # 因為通道有斜率，原始價格會隨時間自然墊高，比較原始價格會誤判）
    if not (current_position_pct < peak_position_pct):
        return False, False, None, None

    is_exhausted = broke_support and peak_position_pct < mid_threshold
    is_weak_warning = (not is_exhausted) and (current_position_pct <= 15) and (peak_position_pct < near_threshold)

    return is_exhausted, is_weak_warning, peak_iloc, peak_position_pct


def check_bearish_exhaustion(df, slope, intercept, channel_width, breakout_idx, last_idx,
                              lookback=20, mid_threshold=50, near_threshold=60):
    """
    判斷空頭是否『力竭』（跟多頭力竭對稱）：

    1. 目前價格已經站回壓力上軌之上
    2. 這波反彈之前，最後一個回測低點，並沒有摸到通道中軌(50%)，
       代表空方力道轉弱、賣壓不足以再破前低

    Returns: (is_exhausted, is_weak_warning, trough_iloc, trough_position_pct)
    """
    if channel_width <= 0:
        return False, False, None, None

    current_resist = slope * last_idx + intercept
    current_price = df['Close'].iloc[-1]
    # 0% = 在壓力上軌, 100% = 在 S1
    current_position_pct = (current_resist - current_price) / channel_width * 100

    broke_resist = current_price > current_resist

    trough_iloc = find_last_swing_before_now(df, slope, intercept, last_idx, breakout_idx,
                                              lookback=lookback, mode='low')
    if trough_iloc is None:
        return False, False, None, None

    resist_at_trough = slope * trough_iloc + intercept
    trough_position_pct = (resist_at_trough - df['Low'].iloc[trough_iloc]) / channel_width * 100

    # 確認低點之後確實一路反彈到現在（用「相對通道位置」比較，不能比較原始價格）
    if not (current_position_pct < trough_position_pct):
        return False, False, None, None

    is_exhausted = broke_resist and trough_position_pct < mid_threshold
    is_weak_warning = (not is_exhausted) and (current_position_pct <= 15) and (trough_position_pct < near_threshold)

    return is_exhausted, is_weak_warning, trough_iloc, trough_position_pct


def get_recent_direction(df, slope, intercept, channel_width, last_idx,
                          lookback=5, threshold_pct=5.0) -> str:
    """
    判斷價格「相對通道線」最近是往上偏移還是往下偏移。

    不是看原始價格的漲跌（因為通道本身有斜率，原始價格本來就會隨時間墊高/墊低），
    而是看「價格離通道基準線的距離」在過去 lookback 天內是擴大還是縮小：
    - 距離擴大（正值變大）→ 'rising'：價格正在往通道上緣的方向移動
    - 距離縮小（往負的方向變化）→ 'falling'：價格正在往通道下緣的方向移動
    - 變化不明顯 → 'flat'

    這個方向判斷對多頭、空頭通道都通用（因為兩者的基準線分別是支撐線／壓力線，
    邏輯完全一樣，差別只在呼叫端怎麼解讀 rising/falling 對應「順勢」還是「逆勢」）。
    """
    if channel_width <= 0:
        return 'flat'

    idx_past = max(0, last_idx - lookback)
    if idx_past == last_idx:
        return 'flat'

    offset_now = df['Close'].iloc[last_idx] - (slope * last_idx + intercept)
    offset_past = df['Close'].iloc[idx_past] - (slope * idx_past + intercept)

    diff_pct = (offset_now - offset_past) / channel_width * 100

    if diff_pct > threshold_pct:
        return 'rising'
    elif diff_pct < -threshold_pct:
        return 'falling'
    else:
        return 'flat'


def touched_level_today(df, level_value, last_idx) -> bool:
    """判斷「今天」這根K的高低範圍，是否有涵蓋到某個價位（例如中軌價）"""
    low = df['Low'].iloc[last_idx]
    high = df['High'].iloc[last_idx]
    return low <= level_value <= high

def calculate_bullish_channel(df, breakout_idx, low_idx, high_idx, last_idx) -> Optional[ChannelResult]:
    """多頭通道：低點連低點為基準線，等距往上"""
    
    # Support anchors: breakout K's low + subsequent pullback lows
    support_anchors = [breakout_idx]
    
    # Add pullback lows after breakout
    post_breakout_lows = low_idx[low_idx > breakout_idx]
    for li in post_breakout_lows:
        # Only add if it's higher than breakout (ascending lows)
        if df['Low'].iloc[li] >= df['Low'].iloc[breakout_idx] * 0.95:
            support_anchors.append(li)
    
    if len(support_anchors) < 2:
        # If only one anchor, try to use breakout + next significant low
        if len(post_breakout_lows) > 0:
            support_anchors.append(post_breakout_lows[0])
        else:
            return None
    
    # Fit support line
    x_pts = np.array(support_anchors[:4], dtype=float)  # Max 4 points
    y_pts = np.array([df['Low'].iloc[i] for i in support_anchors[:4]])
    
    if len(x_pts) >= 2:
        slope, intercept = np.polyfit(x_pts, y_pts, 1)
    else:
        return None
    
    # Channel width: first significant high after breakout minus support at that point
    post_breakout_highs = high_idx[high_idx > breakout_idx]
    
    if len(post_breakout_highs) == 0:
        return None
    
    # Use first major high
    first_high_idx = post_breakout_highs[0]
    support_at_high = slope * first_high_idx + intercept
    channel_width = df['High'].iloc[first_high_idx] - support_at_high
    
    if channel_width <= 0:
        # Try next high
        if len(post_breakout_highs) > 1:
            first_high_idx = post_breakout_highs[1]
            support_at_high = slope * first_high_idx + intercept
            channel_width = df['High'].iloc[first_high_idx] - support_at_high
        if channel_width <= 0:
            return None
    
    # Current levels
    current_support = slope * last_idx + intercept
    current_r1 = current_support + channel_width
    current_r2 = current_support + 2 * channel_width
    
    # Current price position
    current_price = df['Close'].iloc[-1]
    position_pct = (current_price - current_support) / channel_width * 100
    
    # 力竭檢查：跌破支撐 + 前波反彈沒摸到中軌 → 用來加強跌破時的訊號判斷，
    # 也用來決定圖上要不要標示力竭轉折點（X 標記）
    is_exhausted, is_weak_warning, peak_iloc, peak_pct = check_bullish_exhaustion(
        df, slope, intercept, channel_width, breakout_idx, last_idx)

    # Signal（第二版：通道寬度相對距離±near_frac + 方向判斷）
    signal, signal_text = get_bullish_signal_v2(
        df, slope, intercept, channel_width, last_idx, is_exhausted, breakout_idx)

    anchor_points = [(i, df['Low'].iloc[i]) for i in support_anchors[:4]]
    
    return ChannelResult(
        trend='bullish',
        slope=slope,
        intercept=intercept,
        channel_width=channel_width,
        breakout_idx=breakout_idx,
        breakout_date=df.index[breakout_idx].strftime('%Y-%m-%d'),
        breakout_price=df['Low'].iloc[breakout_idx],
        anchor_points=anchor_points,
        current_line1=current_support,
        current_line2=current_r1,
        current_line3=current_r2,
        signal=signal,
        signal_text=signal_text,
        position_pct=position_pct,
        is_exhausted=is_exhausted or is_weak_warning,
        exhaustion_peak_idx=peak_iloc if (is_exhausted or is_weak_warning) else None,
        exhaustion_peak_pct=peak_pct if (is_exhausted or is_weak_warning) else None,
    )


def calculate_bearish_channel(df, breakout_idx, low_idx, high_idx, last_idx) -> Optional[ChannelResult]:
    """空頭通道：高點連高點為基準線，等距往下"""
    
    # Resistance anchors: peak + subsequent lower highs
    resist_anchors = [breakout_idx]
    
    post_peak_highs = high_idx[high_idx > breakout_idx]
    for hi in post_peak_highs:
        # Only add if it's lower than peak (descending highs)
        if df['High'].iloc[hi] <= df['High'].iloc[breakout_idx]:
            resist_anchors.append(hi)
    
    if len(resist_anchors) < 2:
        if len(post_peak_highs) > 0:
            resist_anchors.append(post_peak_highs[0])
        else:
            # Fallback: 急殺無反彈（argrelextrema 找不到 lower high），
            # 改用 breakout 後前幾根K棒的最高 High 當第二錨點
            search_end = min(last_idx + 1, breakout_idx + 6)
            if search_end > breakout_idx + 1:
                post_highs = df['High'].iloc[breakout_idx+1:search_end]
                if len(post_highs) > 0:
                    fallback_pos = int(post_highs.values.argmax())
                    fallback_iloc = breakout_idx + 1 + fallback_pos
                    if fallback_iloc != breakout_idx:
                        resist_anchors.append(fallback_iloc)
            
            # 如果 fallback 也找不到，用回歸法直接擬合壓力線
            if len(resist_anchors) < 2:
                if last_idx - breakout_idx >= 3:
                    post_range = list(range(breakout_idx, last_idx + 1))
                    x_all = np.array(post_range, dtype=float)
                    y_all = np.array([df['High'].iloc[i] for i in post_range])
                    slope_fb, intercept_fb = np.polyfit(x_all, y_all, 1)
                    
                    if slope_fb < 0:  # 確認是下降趨勢
                        resist_line = slope_fb * x_all + intercept_fb
                        low_vals = np.array([df['Low'].iloc[i] for i in post_range])
                        distances = resist_line - low_vals
                        pos_distances = distances[distances > 0]
                        channel_width = float(np.percentile(pos_distances, 75)) if len(pos_distances) > 0 else float(np.max(distances))
                        
                        if channel_width > 0:
                            current_resist = slope_fb * last_idx + intercept_fb
                            current_s1 = current_resist - channel_width
                            current_s2 = current_resist - 2 * channel_width
                            current_price = df['Close'].iloc[-1]
                            position_pct = (current_resist - current_price) / channel_width * 100
                            
                            is_exhausted, is_weak_warning, trough_iloc, trough_pct = check_bearish_exhaustion(
                                df, slope_fb, intercept_fb, channel_width, breakout_idx, last_idx)
                            signal, signal_text = get_bearish_signal_v2(
                                df, slope_fb, intercept_fb, channel_width, last_idx, is_exhausted, breakout_idx)
                            
                            return ChannelResult(
                                trend='bearish',
                                slope=slope_fb,
                                intercept=intercept_fb,
                                channel_width=channel_width,
                                breakout_idx=breakout_idx,
                                breakout_date=df.index[breakout_idx].strftime('%Y-%m-%d'),
                                breakout_price=df['High'].iloc[breakout_idx],
                                anchor_points=[(breakout_idx, df['High'].iloc[breakout_idx])],
                                current_line1=current_resist,
                                current_line2=current_s1,
                                current_line3=current_s2,
                                signal=signal,
                                signal_text=signal_text,
                                position_pct=position_pct,
                                is_exhausted=is_exhausted or is_weak_warning,
                                exhaustion_peak_idx=trough_iloc if (is_exhausted or is_weak_warning) else None,
                                exhaustion_peak_pct=trough_pct if (is_exhausted or is_weak_warning) else None,
                            )
                
                return None
    
    # Fit resistance line (high-to-high)
    x_pts = np.array(resist_anchors[:4], dtype=float)
    y_pts = np.array([df['High'].iloc[i] for i in resist_anchors[:4]])
    
    if len(x_pts) >= 2:
        slope, intercept = np.polyfit(x_pts, y_pts, 1)
    else:
        return None
    
    # Channel width: resistance to first significant low after peak
    post_peak_lows = low_idx[low_idx > breakout_idx]
    
    if len(post_peak_lows) == 0:
        return None
    
    # Use first major low after peak
    first_low_idx = post_peak_lows[0]
    resist_at_low = slope * first_low_idx + intercept
    channel_width = resist_at_low - df['Low'].iloc[first_low_idx]
    
    if channel_width <= 0:
        if len(post_peak_lows) > 1:
            first_low_idx = post_peak_lows[1]
            resist_at_low = slope * first_low_idx + intercept
            channel_width = resist_at_low - df['Low'].iloc[first_low_idx]
        if channel_width <= 0:
            return None
    
    # Current levels
    current_resist = slope * last_idx + intercept
    current_s1 = current_resist - channel_width
    current_s2 = current_resist - 2 * channel_width
    
    # Current price position (0% = at resistance/top, 100% = at S1/bottom)
    current_price = df['Close'].iloc[-1]
    position_pct = (current_resist - current_price) / channel_width * 100
    
    # 力竭檢查：站回壓力上軌 + 前波回測沒摸到中軌 → 用來加強突破時的訊號判斷，
    # 也用來決定圖上要不要標示力竭轉折點（X 標記）
    is_exhausted, is_weak_warning, trough_iloc, trough_pct = check_bearish_exhaustion(
        df, slope, intercept, channel_width, breakout_idx, last_idx)

    # Signal（第二版：通道寬度相對距離±near_frac + 方向判斷）
    signal, signal_text = get_bearish_signal_v2(
        df, slope, intercept, channel_width, last_idx, is_exhausted, breakout_idx)

    anchor_points = [(i, df['High'].iloc[i]) for i in resist_anchors[:4]]
    
    return ChannelResult(
        trend='bearish',
        slope=slope,
        intercept=intercept,
        channel_width=channel_width,
        breakout_idx=breakout_idx,
        breakout_date=df.index[breakout_idx].strftime('%Y-%m-%d'),
        breakout_price=df['High'].iloc[breakout_idx],
        anchor_points=anchor_points,
        current_line1=current_resist,
        current_line2=current_s1,
        current_line3=current_s2,
        signal=signal,
        signal_text=signal_text,
        position_pct=position_pct,
        is_exhausted=is_exhausted or is_weak_warning,
        exhaustion_peak_idx=trough_iloc if (is_exhausted or is_weak_warning) else None,
        exhaustion_peak_pct=trough_pct if (is_exhausted or is_weak_warning) else None,
    )


def try_secondary_channel(df: pd.DataFrame, primary_result: ChannelResult,
                           order: int = 3, recent_window: int = 60):
    """
    當主要通道被跌破/突破時（例如多頭已跌破支撐超過3%），嘗試只用
    「最近一段資料」去找出是否已經開始形成一個新的反向通道
    （多頭跌破 → 找空頭通道；空頭突破 → 找多頭通道）。

    目的：跌破當下，原本的上升通道線還是有參考價值（可以看出跌了多深），
    但如果已經開始走出新的下降結構，把兩條通道疊在一起看，更容易判斷
    「這只是回檔」還是「真的已經轉空」。

    做法分三層（跟 try_realign_channel 一致）：
    1. 優先用「轉折點連轉折點」方式擬合，找出反向轉折發生在哪裡
    2. 轉折點不足時（反轉剛發生沒多久，資料本來就少），fallback 改用
       _fit_regression_channel 做線性回歸擬合，確保只要有 >=3 根K棒，
       就能先給出一條初步的反向通道，不會直接放棄
    3. 最後統一用 _refit_with_open_anchors 把基準線改成用開盤價重新擬合
       （反轉當下常常是跳空跌破/突破，影線容易失真，開盤價更能反映
       市場真正認同的新支撐/壓力位置）

    Args:
        df: 完整的日K資料（含即時K棒也沒關係，因為只是取最後 recent_window 根）
        primary_result: 主要通道的計算結果，用來判斷要找哪個方向的次要通道
        order: 找轉折點的靈敏度，預設比主通道更靈敏（3 vs 主通道的 5），
               因為次要通道的資料量本來就少，需要抓得更細才找得到轉折點
        recent_window: 只用最近幾根K線去找新通道，避免被更早、已經作廢的
               走勢干擾（新趨勢通常只需要近期資料就足夠判斷）

    Returns:
        ChannelResult | None
        資料真的太少（少於3根K）才會回傳 None，代表新趨勢還沒走出足夠的
        資料可以判斷，不影響原本圖表照常顯示。回傳的 ChannelResult 內部的
        index 已經自動轉換成跟原本 df 一致的座標，呼叫端可以直接跟
        primary_result 用同一個 df 一起畫圖。
    """
    if len(df) < 8:
        return None

    window = min(recent_window, len(df))
    sub_df = df.iloc[-window:]
    offset = len(df) - window
    last_idx = len(sub_df) - 1
    new_trend = 'bearish' if primary_result.trend == 'bullish' else 'bullish'

    secondary = None
    used_pivot_method = False
    try:
        low_idx = argrelextrema(sub_df['Low'].values, np.less_equal, order=order)[0]
        high_idx = argrelextrema(sub_df['High'].values, np.greater_equal, order=order)[0]

        if len(low_idx) >= 2 and len(high_idx) >= 2:
            if new_trend == 'bearish':
                breakout_idx = find_bearish_breakout(sub_df, low_idx, high_idx, order)
                if breakout_idx is not None:
                    secondary = calculate_bearish_channel(sub_df, breakout_idx, low_idx, high_idx, last_idx)
            else:
                breakout_idx = find_bullish_breakout(sub_df, low_idx, high_idx, order)
                if breakout_idx is not None:
                    secondary = calculate_bullish_channel(sub_df, breakout_idx, low_idx, high_idx, last_idx)
            used_pivot_method = secondary is not None
    except Exception:
        secondary = None
        used_pivot_method = False

    if secondary is None:
        # 轉折點不足（反轉剛發生沒多久很常見），改用線性回歸 fallback，
        # 從主通道判定的跌破/突破那根K開始算，確保新通道只用「反轉之後」的資料
        try:
            # 從「反轉之後」的資料重新擬合，避免摻入太多反轉前、已經不適用的資料
            fallback_start = max(0, last_idx - min(last_idx, recent_window // 2))
            secondary = _fit_regression_channel(sub_df, new_trend, fallback_start, last_idx)
        except Exception:
            secondary = None
    elif used_pivot_method:
        # 點對點方式找到的轉折K棒，基準線改用開盤價重新擬合
        refitted = _refit_with_open_anchors(sub_df, secondary)
        if refitted is not None:
            secondary = refitted

    if secondary is None:
        return None

    try:
        # sub_df 是 df 的最後 window 根，secondary 內部算出來的 index 是
        # 相對 sub_df（從0開始）的位置，這裡轉換成跟完整 df 一致的絕對位置，
        # 讓呼叫端可以直接疊加在同一張圖上，不用另外處理座標換算
        secondary.breakout_idx += offset
        secondary.intercept = secondary.intercept - secondary.slope * offset
        secondary.anchor_points = [(i + offset, p) for i, p in secondary.anchor_points]
        if secondary.exhaustion_peak_idx is not None:
            secondary.exhaustion_peak_idx += offset

        return secondary
    except Exception:
        # 次要通道找不到是正常情況（資料不足），不應該讓主流程掛掉
        return None

def channel_outside_ratio(df: pd.DataFrame, result: ChannelResult, lookback: int = 10) -> float:
    """
    計算「最近 lookback 根K線」中，收盤價已經跑出通道『順勢方向外側』的比例。

    背景：像 2327 國巨、2634 漢翔、2303 聯電、6182 合晶 這類噴出股，常常在
    上漲過程中斜率突然轉陡，導致原本用較平緩斜率畫出的通道，完全跟不上
    後面的走勢——K棒不再於通道內來回擺盪，而是持續貼著或跑到通道上緣（R2）
    之外。這個函式就是用來偵測「這種狀況是否已經發生」。

    只看順勢方向的外側（多頭只看是否超過 R2，空頭只看是否跌破 S2），
    刻意不看逆勢那一側，因為逆勢跌破/突破已經由 breakdown/breakout 的
    力竭邏輯處理，這裡不重複判斷，避免兩套邏輯互相干擾。

    Returns:
        0.0 ~ 1.0，代表比例。資料不足（例如通道才剛形成沒幾根K）時回傳 0.0，
        視為通道仍然有效，不觸發重新校正。
    """
    last_idx = len(df) - 1
    start = max(result.breakout_idx, last_idx - lookback + 1)
    if start > last_idx:
        return 0.0

    outside_count = 0
    total = 0
    for i in range(start, last_idx + 1):
        total += 1
        line_at_i = result.slope * i + result.intercept
        close = df['Close'].iloc[i]
        if result.trend == 'bullish':
            outer_edge = line_at_i + 2 * result.channel_width  # R2
            if close > outer_edge:
                outside_count += 1
        else:
            outer_edge = line_at_i - 2 * result.channel_width  # S2
            if close < outer_edge:
                outside_count += 1

    if total == 0:
        return 0.0
    return outside_count / total


def _refit_with_open_anchors(df: pd.DataFrame, result: ChannelResult) -> Optional[ChannelResult]:
    """
    把已經算好的通道（不管是點對點擬合還是回歸擬合出來的），改用『開盤價』
    重新擬合支撐/壓力基準線，取代原本用影線最低/最高點當基準的作法。

    背景：飆股常常先跳空、再急拉／急殺（例如 2327 國巨、6182 合晶這類案例），
    當天的影線（High/Low）可能只是短暫洗盤或衝高，不足以代表真正守住/攻破的
    價位；反而是開盤價（尤其跳空缺口那天的開盤）更能反映市場真正認同的
    支撐/壓力位置。這裡沿用已經找到的轉折K棒（result.anchor_points 的位置），
    改用這些K棒的開盤價重新擬合一條線；channel_width 仍然用影線對這條新線的
    偏離幅度決定，確保通道還是包得住整段K線走勢。

    Args:
        df: 跟 result 內部 index 同一套座標系統的 DataFrame（呼叫端自己負責，
            通常是還沒做 offset 轉換的 sub_df）
        result: calculate_bullish_channel / calculate_bearish_channel 或
            _fit_regression_channel 算出來的初版通道結果

    Returns:
        ChannelResult | None（改良失敗時回傳 None，呼叫端應該保留原本的 result
        繼續使用，不影響主流程）
    """
    last_idx = len(df) - 1
    idxs = sorted(set(int(i) for i, _ in result.anchor_points) | {int(result.breakout_idx), last_idx})
    idxs = [i for i in idxs if 0 <= i <= last_idx]
    if len(idxs) < 2:
        return None

    start_idx = idxs[0]
    x_pts = np.array(idxs, dtype=float)
    y_pts = df['Open'].iloc[idxs].values

    try:
        slope, intercept = np.polyfit(x_pts, y_pts, 1)
    except Exception:
        return None

    x_range = np.arange(start_idx, last_idx + 1, dtype=float)
    line_vals = slope * x_range + intercept

    try:
        if result.trend == 'bullish':
            highs = df['High'].iloc[start_idx:last_idx + 1].values
            residuals = highs - line_vals
            residuals = residuals[residuals > 0]
            if len(residuals) == 0:
                return None
            channel_width = float(np.percentile(residuals, 75))
            if channel_width <= 0:
                return None

            current_support = slope * last_idx + intercept
            current_r1 = current_support + channel_width
            current_r2 = current_support + 2 * channel_width
            current_price = df['Close'].iloc[-1]
            position_pct = (current_price - current_support) / channel_width * 100

            is_exhausted, is_weak_warning, peak_iloc, peak_pct = check_bullish_exhaustion(
                df, slope, intercept, channel_width, start_idx, last_idx)
            signal, signal_text = get_bullish_signal_v2(
                df, slope, intercept, channel_width, last_idx, is_exhausted, start_idx)

            anchor_points = [(i, df['Open'].iloc[i]) for i in idxs]

            return ChannelResult(
                trend='bullish', slope=slope, intercept=intercept, channel_width=channel_width,
                breakout_idx=start_idx, breakout_date=df.index[start_idx].strftime('%Y-%m-%d'),
                breakout_price=df['Open'].iloc[start_idx], anchor_points=anchor_points,
                current_line1=current_support, current_line2=current_r1, current_line3=current_r2,
                signal=signal, signal_text=signal_text, position_pct=position_pct,
                is_exhausted=is_exhausted or is_weak_warning,
                exhaustion_peak_idx=peak_iloc if (is_exhausted or is_weak_warning) else None,
                exhaustion_peak_pct=peak_pct if (is_exhausted or is_weak_warning) else None,
            )
        else:
            lows = df['Low'].iloc[start_idx:last_idx + 1].values
            residuals = line_vals - lows
            residuals = residuals[residuals > 0]
            if len(residuals) == 0:
                return None
            channel_width = float(np.percentile(residuals, 75))
            if channel_width <= 0:
                return None

            current_resist = slope * last_idx + intercept
            current_s1 = current_resist - channel_width
            current_s2 = current_resist - 2 * channel_width
            current_price = df['Close'].iloc[-1]
            position_pct = (current_resist - current_price) / channel_width * 100

            is_exhausted, is_weak_warning, trough_iloc, trough_pct = check_bearish_exhaustion(
                df, slope, intercept, channel_width, start_idx, last_idx)
            signal, signal_text = get_bearish_signal_v2(
                df, slope, intercept, channel_width, last_idx, is_exhausted, start_idx)

            anchor_points = [(i, df['Open'].iloc[i]) for i in idxs]

            return ChannelResult(
                trend='bearish', slope=slope, intercept=intercept, channel_width=channel_width,
                breakout_idx=start_idx, breakout_date=df.index[start_idx].strftime('%Y-%m-%d'),
                breakout_price=df['Open'].iloc[start_idx], anchor_points=anchor_points,
                current_line1=current_resist, current_line2=current_s1, current_line3=current_s2,
                signal=signal, signal_text=signal_text, position_pct=position_pct,
                is_exhausted=is_exhausted or is_weak_warning,
                exhaustion_peak_idx=trough_iloc if (is_exhausted or is_weak_warning) else None,
                exhaustion_peak_pct=trough_pct if (is_exhausted or is_weak_warning) else None,
            )
    except Exception:
        return None


def _fit_regression_channel(df: pd.DataFrame, trend: str, start_idx: int, last_idx: int) -> Optional[ChannelResult]:
    """
    當轉折點（argrelextrema）數量不足以用「低點連低點／高點連高點」的方式
    擬合通道時（常見於噴出段：漲勢又急又直，回檔次數本來就很少，找不到
    兩個以上的轉折點），改用最簡單、也最通用的線性回歸方式重新畫一條通道：

    - 多頭：對這段區間的『開盤價』做線性回歸當基準線（支撐），channel_width
      用同一段 High 偏離基準線的幅度（取75百分位，避免單一異常K棒把
      通道撐得過寬）決定
    - 空頭：對稱處理（開盤價做回歸當壓力線，Low 決定 channel_width）

    這保證只要區間內有 >= 3 根K棒，就一定能算出一條通道，不會因為
    轉折點不足而放棄（跟 calculate_bullish_channel/calculate_bearish_channel
    不同，那兩個函式依賴 argrelextrema 找到的轉折點，噴出段常常不夠用）。

    用開盤價而不是影線最低/最高點當基準線：飆股常常先跳空、再急拉/急殺，
    當天的影線可能只是短暫洗盤/衝高，不足以代表真正守住/攻破的價位，
    開盤價（尤其跳空缺口那天）更能反映市場真正認同的支撐/壓力位置。
    """
    x_pts = np.arange(start_idx, last_idx + 1, dtype=float)
    if len(x_pts) < 3:
        return None

    if trend == 'bullish':
        y_pts = df['Open'].iloc[start_idx:last_idx + 1].values
        slope, intercept = np.polyfit(x_pts, y_pts, 1)
        line_vals = slope * x_pts + intercept
        highs = df['High'].iloc[start_idx:last_idx + 1].values
        residuals = highs - line_vals
        residuals = residuals[residuals > 0]
        if len(residuals) == 0:
            return None
        channel_width = float(np.percentile(residuals, 75))
        if channel_width <= 0:
            return None

        current_support = slope * last_idx + intercept
        current_r1 = current_support + channel_width
        current_r2 = current_support + 2 * channel_width
        current_price = df['Close'].iloc[-1]
        position_pct = (current_price - current_support) / channel_width * 100

        is_exhausted, is_weak_warning, peak_iloc, peak_pct = check_bullish_exhaustion(
            df, slope, intercept, channel_width, start_idx, last_idx)
        signal, signal_text = get_bullish_signal_v2(
            df, slope, intercept, channel_width, last_idx, is_exhausted, start_idx)

        anchor_points = [(start_idx, df['Open'].iloc[start_idx]), (last_idx, df['Open'].iloc[last_idx])]

        return ChannelResult(
            trend='bullish', slope=slope, intercept=intercept, channel_width=channel_width,
            breakout_idx=start_idx, breakout_date=df.index[start_idx].strftime('%Y-%m-%d'),
            breakout_price=df['Open'].iloc[start_idx], anchor_points=anchor_points,
            current_line1=current_support, current_line2=current_r1, current_line3=current_r2,
            signal=signal, signal_text=signal_text, position_pct=position_pct,
            is_exhausted=is_exhausted or is_weak_warning,
            exhaustion_peak_idx=peak_iloc if (is_exhausted or is_weak_warning) else None,
            exhaustion_peak_pct=peak_pct if (is_exhausted or is_weak_warning) else None,
        )
    else:
        y_pts = df['Open'].iloc[start_idx:last_idx + 1].values
        slope, intercept = np.polyfit(x_pts, y_pts, 1)
        line_vals = slope * x_pts + intercept
        lows = df['Low'].iloc[start_idx:last_idx + 1].values
        residuals = line_vals - lows
        residuals = residuals[residuals > 0]
        if len(residuals) == 0:
            return None
        channel_width = float(np.percentile(residuals, 75))
        if channel_width <= 0:
            return None

        current_resist = slope * last_idx + intercept
        current_s1 = current_resist - channel_width
        current_s2 = current_resist - 2 * channel_width
        current_price = df['Close'].iloc[-1]
        position_pct = (current_resist - current_price) / channel_width * 100

        is_exhausted, is_weak_warning, trough_iloc, trough_pct = check_bearish_exhaustion(
            df, slope, intercept, channel_width, start_idx, last_idx)
        signal, signal_text = get_bearish_signal_v2(
            df, slope, intercept, channel_width, last_idx, is_exhausted, start_idx)

        anchor_points = [(start_idx, df['Open'].iloc[start_idx]), (last_idx, df['Open'].iloc[last_idx])]

        return ChannelResult(
            trend='bearish', slope=slope, intercept=intercept, channel_width=channel_width,
            breakout_idx=start_idx, breakout_date=df.index[start_idx].strftime('%Y-%m-%d'),
            breakout_price=df['Open'].iloc[start_idx], anchor_points=anchor_points,
            current_line1=current_resist, current_line2=current_s1, current_line3=current_s2,
            signal=signal, signal_text=signal_text, position_pct=position_pct,
            is_exhausted=is_exhausted or is_weak_warning,
            exhaustion_peak_idx=trough_iloc if (is_exhausted or is_weak_warning) else None,
            exhaustion_peak_pct=trough_pct if (is_exhausted or is_weak_warning) else None,
        )


def try_realign_channel(df: pd.DataFrame, primary_result: ChannelResult,
                         order: int = 3, recent_window: int = 40):
    """
    當 channel_outside_ratio 判斷原通道已經跟不上走勢時，只用「最近一段資料」
    重新找一次『同方向』（多頭→多頭／空頭→空頭）的通道，取代掉太平緩、已經
    不合身的舊通道。

    跟 try_secondary_channel 的差別：
    - try_secondary_channel：在「反轉」時去找『反方向』的新通道（例如多頭
      跌破後，看是否已經走出新的空頭結構）
    - try_realign_channel：在「同方向但斜率明顯不夠陡」時，重新對『同方向』
      做一次擬合，抓出更貼近最近噴出走勢的新支撐/壓力線

    做法分三層：
    1. 優先用跟主通道一樣的「轉折點連轉折點」方式擬合，找出『轉折發生在哪裡』
       （找得到就用，這一步只負責定位轉折K棒，不代表基準線的價位就是最終答案）
    2. 找不到足夠轉折點時（噴出段常見：漲勢又急又直，回檔次數少），
       fallback 改用 _fit_regression_channel 做線性回歸擬合，確保只要資料
       量夠，一定能給出一條可用的新通道，不會直接放棄
    3. 不管是哪一種方式找到轉折K棒，最後都用 _refit_with_open_anchors 把
       基準線改成用這些轉折K棒的『開盤價』重新擬合，取代影線最低/最高點
       ——飆股常跳空，影線容易是短暫洗盤/衝高，開盤價才是市場真正守住的
       關鍵價位（見 2327 國巨、6182 合晶這類案例）

    Args:
        df: 完整日K資料（含即時K棒也沒關係，只會取最後 recent_window 根）
        primary_result: 目前（已經跟不上走勢）的通道結果，用來決定同方向
        order: 找轉折點的靈敏度，比主通道更靈敏（同 try_secondary_channel）
        recent_window: 只用最近幾根K線重新擬合，避免被更早、已經作廢的
               平緩走勢干擾

    Returns:
        ChannelResult | None
        資料不足以重新形成通道時回傳 None，呼叫端應該退回使用原本的通道，
        不影響主流程。回傳的 ChannelResult 內部的 index 已經自動轉換成跟
        原本 df 一致的座標，可以直接跟原通道一起畫在同一張圖上。
    """
    if len(df) < 15:
        return None

    window = min(recent_window, len(df))
    sub_df = df.iloc[-window:]
    offset = len(df) - window
    last_idx = len(sub_df) - 1

    realigned = None
    used_pivot_method = False
    try:
        low_idx = argrelextrema(sub_df['Low'].values, np.less_equal, order=order)[0]
        high_idx = argrelextrema(sub_df['High'].values, np.greater_equal, order=order)[0]

        if len(low_idx) >= 2 and len(high_idx) >= 2:
            if primary_result.trend == 'bullish':
                breakout_idx = find_bullish_breakout(sub_df, low_idx, high_idx, order)
                if breakout_idx is not None:
                    realigned = calculate_bullish_channel(sub_df, breakout_idx, low_idx, high_idx, last_idx)
            else:
                breakout_idx = find_bearish_breakout(sub_df, low_idx, high_idx, order)
                if breakout_idx is not None:
                    realigned = calculate_bearish_channel(sub_df, breakout_idx, low_idx, high_idx, last_idx)
            used_pivot_method = realigned is not None
    except Exception:
        realigned = None
        used_pivot_method = False

    if realigned is None:
        # 轉折點不足，改用線性回歸 fallback（見 _fit_regression_channel 說明，
        # 這個 fallback 本身就已經是用開盤價當基準線，不需要再重新錨定）
        try:
            realigned = _fit_regression_channel(sub_df, primary_result.trend, 0, last_idx)
        except Exception:
            realigned = None
    elif used_pivot_method:
        # 點對點方式找到的轉折K棒，基準線改用開盤價重新擬合
        # （找不到就退回原本影線版本的 realigned，不影響主流程）
        refitted = _refit_with_open_anchors(sub_df, realigned)
        if refitted is not None:
            realigned = refitted

    if realigned is None:
        return None

    try:
        # sub_df 的 index 轉換成跟完整 df 一致的絕對位置（同 try_secondary_channel）
        realigned.breakout_idx += offset
        realigned.intercept = realigned.intercept - realigned.slope * offset
        realigned.anchor_points = [(i + offset, p) for i, p in realigned.anchor_points]
        if realigned.exhaustion_peak_idx is not None:
            realigned.exhaustion_peak_idx += offset
        return realigned
    except Exception:
        # 重新校正失敗是正常情況，不應該讓主流程掛掉，
        # 呼叫端應該保留原本的通道繼續使用
        return None


def is_long_term_choppy(df: pd.DataFrame, lookback: int = 120, efficiency_threshold: float = 0.25) -> bool:
    """
    判斷「長期」（預設最近120根K，約半年）走勢是不是比較像區間盤整，
    而不是走出一段有效的趨勢。

    背景：像 2449 京元電子這種股票，短期（例如最近兩三週）雖然走出一段清楚的
    下降趨勢，但拉長時間看，其實一直在一個區間裡上上下下、還沒真正脫離過去
    幾個月的整理格局。如果只看「近期方向」判斷多空，容易在區間股上誤判成
    「趨勢會繼續延續」，這個函式用一個粗略但常見的量化指標，讓呼叫端可以在
    通道判斷出來的訊號後面，多加一句「長期較偏區間整理」的提醒。

    做法：計算「效率係數」(efficiency ratio，概念上跟 Kaufman's Efficiency
    Ratio／KAMA 用的指標一樣) = 淨變動 / 總波動路徑
    - 淨變動：lookback 天內收盤價的淨變化幅度（頭尾價差的絕對值）
    - 總波動路徑：lookback 天內每天收盤價變化的絕對值加總（走了多少「路程」）
    - 比值越接近 1，代表走勢越像一條乾淨的直線趨勢；越接近 0，代表走了很多
      「來回路」但淨變化很小，比較像區間整理

    Args:
        df: 完整日K資料
        lookback: 要往回看多少根K棒（預設120根，約半年交易日）
        efficiency_threshold: 效率係數低於這個門檻，視為長期偏區間整理
            （0.25 是技術分析上常見、判斷「弱趨勢/盤整」的經驗值）

    Returns:
        True 代表長期比較像區間整理，呼叫端可以加註提醒；
        資料不足時保守回傳 False（不加註，避免誤報）
    """
    window = min(lookback, len(df))
    if window < 40:
        return False

    closes = df['Close'].iloc[-window:].values
    net_change = abs(closes[-1] - closes[0])
    total_path = np.sum(np.abs(np.diff(closes)))

    if total_path <= 0:
        return False

    efficiency = net_change / total_path
    return efficiency < efficiency_threshold


def extend_channel_to_now(df: pd.DataFrame, template: ChannelResult, last_idx: int,
                           max_width_multiple: float = 3.0) -> Optional[ChannelResult]:
    """
    拿一個已經算好的通道（slope/intercept/channel_width 不變），重新評估到
    『今天』（df 的 last_idx）的訊號、位置、力竭狀態。

    用途：像 rebase_broken_channel 這種情況，通道是用「崩跌前」的資料擬合的，
    但我們想知道「用這條線一路延伸到今天，現在的訊號/位置是什麼」，
    這個函式就是負責把 current_line1/2/3、signal、position_pct 等重新算一次，
    不動 slope/intercept/anchor_points（那些代表通道本身怎麼來的，不該變）。

    外推範圍上限（重要）：
    像噴出段這種短時間內急拉的線，斜率通常很陡，如果不管過了多久都一路照著
    同一個斜率往外推算到今天，時間一拉長，推算出來的支撐/壓力價位會遠遠
    超過股票歷史上出現過的最高價，變成沒有意義的數字（例如 4931 新盛力這
    個案例：噴出段只有短短十幾天，斜率極陡，外推超過一個月後，算出來的
    Support 甚至比歷史最高價還高）。

    這裡刻意不是用「外推了幾天」當上限（斜率越陡，同樣天數推算出的價位
    漲幅越誇張，天數上限完全防不住），而是直接限制「外推的價位漲幅本身
    不能超過通道寬度的固定倍數」（預設3倍）——不管斜率多陡，只要漲幅超過
    這個範圍，就不再繼續外推，數字會固定在上限位置。同時搭配一個寬鬆的
    天數上限（避免斜率極平緩、價位漲幅怎麼推都推不到上限的線，被永遠當成
    「仍然有效」而外推到很久以後）。超過上限後，current_line1/2/3 會固定在
    「上限位置」不再繼續外推，並標記 is_stale_extrapolation=True，讓呼叫端
    可以額外提醒使用者這條線已經過舊、數字僅供參考，應該優先參考反向的
    新通道（try_secondary_channel 算出來的）。

    Returns:
        ChannelResult | None（沒辦法評估時回傳 None，例如 channel_width<=0）
    """
    slope, intercept, width = template.slope, template.intercept, template.channel_width
    if width <= 0:
        return None

    anchor_idxs = [i for i, _ in template.anchor_points] if template.anchor_points else []
    last_anchor_idx = max(anchor_idxs) if anchor_idxs else template.breakout_idx
    fitted_span = max(last_anchor_idx - template.breakout_idx, 1)

    # 價位漲幅上限：外推的支撐/壓力最多只能比「擬合窗口結束時」的位置再高
    # （或低）MAX_EXTRAPOLATE_WIDTH_MULTIPLE 倍的通道寬度，不管斜率多陡
    max_price_move = max_width_multiple * width
    max_bars_by_value = (max_price_move / abs(slope)) if slope != 0 else float('inf')
    # 天數上限：給斜率很平緩、價位漲幅怎麼推都推不到上限的線一個保底範圍，
    # 避免這種線被永遠當成「仍在合理外推範圍內」而外推到很久以後
    max_bars_by_time = max(fitted_span * 2, 30)
    max_extrapolate = max(min(max_bars_by_value, max_bars_by_time), 5)

    eval_idx = min(last_idx, last_anchor_idx + int(max_extrapolate))
    is_stale = eval_idx < last_idx

    try:
        if template.trend == 'bullish':
            is_exhausted, is_weak, peak_iloc, peak_pct = check_bullish_exhaustion(
                df, slope, intercept, width, template.breakout_idx, eval_idx)
            signal, signal_text = get_bullish_signal_v2(df, slope, intercept, width, eval_idx, is_exhausted, template.breakout_idx)

            current_support = slope * eval_idx + intercept
            current_r1 = current_support + width
            current_r2 = current_support + 2 * width
            current_price = df['Close'].iloc[last_idx]
            position_pct = (current_price - current_support) / width * 100

            return ChannelResult(
                trend='bullish', slope=slope, intercept=intercept, channel_width=width,
                breakout_idx=template.breakout_idx, breakout_date=template.breakout_date,
                breakout_price=template.breakout_price, anchor_points=template.anchor_points,
                current_line1=current_support, current_line2=current_r1, current_line3=current_r2,
                signal=signal, signal_text=signal_text, position_pct=position_pct,
                is_exhausted=is_exhausted or is_weak,
                exhaustion_peak_idx=peak_iloc if (is_exhausted or is_weak) else None,
                exhaustion_peak_pct=peak_pct if (is_exhausted or is_weak) else None,
                is_stale_extrapolation=is_stale,
            )
        else:
            is_exhausted, is_weak, trough_iloc, trough_pct = check_bearish_exhaustion(
                df, slope, intercept, width, template.breakout_idx, eval_idx)
            signal, signal_text = get_bearish_signal_v2(df, slope, intercept, width, eval_idx, is_exhausted, template.breakout_idx)

            current_resist = slope * eval_idx + intercept
            current_s1 = current_resist - width
            current_s2 = current_resist - 2 * width
            current_price = df['Close'].iloc[last_idx]
            position_pct = (current_resist - current_price) / width * 100

            return ChannelResult(
                trend='bearish', slope=slope, intercept=intercept, channel_width=width,
                breakout_idx=template.breakout_idx, breakout_date=template.breakout_date,
                breakout_price=template.breakout_price, anchor_points=template.anchor_points,
                current_line1=current_resist, current_line2=current_s1, current_line3=current_s2,
                signal=signal, signal_text=signal_text, position_pct=position_pct,
                is_exhausted=is_exhausted or is_weak,
                exhaustion_peak_idx=trough_iloc if (is_exhausted or is_weak) else None,
                exhaustion_peak_pct=trough_pct if (is_exhausted or is_weak) else None,
                is_stale_extrapolation=is_stale,
            )
    except Exception:
        return None


def rebase_broken_channel(df: pd.DataFrame, result: ChannelResult,
                           stale_lookback: int = 10, stale_threshold: float = 0.6) -> Optional[ChannelResult]:
    """
    當通道已經跌破/突破時，回頭檢查『崩跌發生之前』原通道是不是其實早就
    已經跟不上走勢——也就是說，飆股常常是「緩漲 → 加速噴出 → 急殺」三段式
    走勢（例如 2327 國巨、6182 合晶），如果只看最原始、從緩漲階段就開始算的
    那條平緩通道，拿來當作「被跌破的通道」去找反向新通道，會失真：真正
    "剛被跌破的那條線"，其實是噴出段那條更陡的通道，不是緩漲階段的通道。

    做法：從主通道的起點往後找「崩跌前的最高點/最低點」（也就是噴出段的
    高峰），只用『高峰之前』的資料重新檢查是否需要 realign（同
    try_realign_channel 的邏輯），如果需要，就把重新校正過的陡峭通道
    延伸評估到今天（extend_channel_to_now），取代掉原本太平緩的通道。

    Args:
        df: 完整日K資料
        result: calculate_channel 算出來、且 signal 已經是 'breakdown' 或
            'breakout_up' 的原始通道

    Returns:
        ChannelResult | None
        None 代表「崩跌前沒有偵測到明顯加速」（例如本來就是走一條乾淨的
        通道，回檔後才反轉），呼叫端應該保留原本的 result 繼續使用，
        這是正常情況，不影響主流程。
    """
    last_idx = len(df) - 1
    peak_search_start = result.breakout_idx
    if peak_search_start >= last_idx - 5:
        return None

    try:
        if result.trend == 'bullish':
            segment = df['High'].iloc[peak_search_start:last_idx + 1].values
            peak_idx = int(np.argmax(segment)) + peak_search_start
        else:
            segment = df['Low'].iloc[peak_search_start:last_idx + 1].values
            peak_idx = int(np.argmin(segment)) + peak_search_start

        # 高/低點太靠近通道起點，代表根本沒有走出「加速噴出」的空間可以判斷
        if peak_idx <= peak_search_start + 5:
            return None

        df_pre_peak = df.iloc[:peak_idx + 1]
        ratio = channel_outside_ratio(df_pre_peak, result, lookback=stale_lookback)
        if ratio < stale_threshold:
            return None

        realigned = try_realign_channel(df_pre_peak, result)
        if realigned is None:
            return None

        # realigned 是用「崩跌前」的資料擬合的，重新評估到「今天」，
        # 才能正確反映出目前已經跌破的狀態（signal 會變成 breakdown/breakout_up）
        return extend_channel_to_now(df, realigned, last_idx)
    except Exception:
        return None


def get_bullish_peak_quality(df, slope, intercept, channel_width, breakout_idx, last_idx,
                              lookback=20, strong_threshold=85, mid_threshold=50) -> Optional[str]:
    """
    判斷「這次拉回之前，上一個反彈高點」摸到通道的哪個位置，用來決定中軌回測
    夠不夠格被當成買點（呼應軌道策略：每一段上漲理論上都該走到它該到的軌道，
    沒到就是動能不足的警訊）：

    - 'strong'：有摸到上軌附近（>=strong_threshold），維持原本邏輯，下軌/中軌
      都可以是買點
    - 'weak'：只摸到中軌~上軌之間，還沒到力竭的程度，但這次中軌回測要看有沒有
      站穩（收盤沒跌破中軌）才算數，站不穩就要等拉回下軌再買
    - 'exhausted'：連中軌都沒摸到，比 check_bullish_exhaustion「要等跌破支撐
      才確認力竭」的門檻更早示警，下軌也不建議買進
    - None：目前還沒有明確的前波高點可以比較（例如剛突破沒多久）
    """
    if channel_width <= 0:
        return None
    peak_iloc = find_last_swing_before_now(df, slope, intercept, last_idx, breakout_idx,
                                            lookback=lookback, mode='high')
    if peak_iloc is None:
        return None
    support_at_peak = slope * peak_iloc + intercept
    peak_position_pct = (df['High'].iloc[peak_iloc] - support_at_peak) / channel_width * 100
    if peak_position_pct >= strong_threshold:
        return 'strong'
    elif peak_position_pct >= mid_threshold:
        return 'weak'
    else:
        return 'exhausted'


def get_bullish_pullback_strength(df, slope, intercept, channel_width, breakout_idx, last_idx,
                                   lookback=20, shallow_threshold=50) -> bool:
    """
    判斷「這次反彈之前，上一個拉回低點」有沒有跌破中軌就轉而向上——如果連
    該回測到的中軌都沒跌破，代表買盤力道強、拉回不夠深，是多頭偏強勢的訊號
    （跟力竭剛好相反：力竭是「該到的高點沒到」，強勢是「該到的低點沒到」）
    """
    if channel_width <= 0:
        return False
    trough_iloc = find_last_swing_before_now(df, slope, intercept, last_idx, breakout_idx,
                                              lookback=lookback, mode='low')
    if trough_iloc is None:
        return False
    support_at_trough = slope * trough_iloc + intercept
    trough_position_pct = (df['Low'].iloc[trough_iloc] - support_at_trough) / channel_width * 100
    return trough_position_pct >= shallow_threshold


def get_bearish_trough_quality(df, slope, intercept, channel_width, breakout_idx, last_idx,
                                lookback=20, strong_threshold=85, mid_threshold=50) -> Optional[str]:
    """跟 get_bullish_peak_quality 對稱：判斷前一個回測低點摸到通道的哪個位置"""
    if channel_width <= 0:
        return None
    trough_iloc = find_last_swing_before_now(df, slope, intercept, last_idx, breakout_idx,
                                              lookback=lookback, mode='low')
    if trough_iloc is None:
        return None
    resist_at_trough = slope * trough_iloc + intercept
    trough_position_pct = (resist_at_trough - df['Low'].iloc[trough_iloc]) / channel_width * 100
    if trough_position_pct >= strong_threshold:
        return 'strong'
    elif trough_position_pct >= mid_threshold:
        return 'weak'
    else:
        return 'exhausted'


def get_bearish_rebound_strength(df, slope, intercept, channel_width, breakout_idx, last_idx,
                                  lookback=20, shallow_threshold=50) -> bool:
    """跟 get_bullish_pullback_strength 對稱：判斷前一波反彈高點有沒有站上中軌就轉而向下"""
    if channel_width <= 0:
        return False
    peak_iloc = find_last_swing_before_now(df, slope, intercept, last_idx, breakout_idx,
                                            lookback=lookback, mode='high')
    if peak_iloc is None:
        return False
    resist_at_peak = slope * peak_iloc + intercept
    peak_position_pct = (resist_at_peak - df['High'].iloc[peak_iloc]) / channel_width * 100
    return peak_position_pct >= shallow_threshold


def get_bullish_signal_v2(df, slope, intercept, channel_width, last_idx,
                           is_exhausted: bool, breakout_idx: int, near_frac: float = 0.15) -> Tuple[str, str]:
    """
    多頭通道訊號判斷（第二版）

    跟原本只看 position_pct（通道內百分比位置）不同，這版改用：
    1. 「價格距離軌道的相對位置，以通道寬度為單位」(±15%的通道寬度) 判斷是否
       觸及上/下軌，而不是用「股價的絕對百分比」——這點很關鍵：股價的絕對
       百分比不會隨通道寬窄縮放，遇到通道被重新校正成很陡（channel_width
       因此變窄）、但股價基期本身較高的股票（例如 4931 新盛力），用股價的
       3% 當門檻，會遠大於通道本身的寬度，導致明明已經明顯跌破支撐（例如
       位置在通道下方 -25%），卻因為換算成股價百分比還不到 3%，被誤判成
       「接近下軌、可以買進」，而不是「已經跌破」。改用通道寬度的相對比例
       之後，「近」「破」的判斷標準會自動隨通道本身的尺度縮放，不會再失真。
    2. 「最近價格走勢方向」（get_recent_direction）—— 同樣在中軌位置，
       往上觸及、還是往下觸及，代表的意義完全相反，訊號文字要分開
    3. 跌破支撐超過 near_frac 時，優先用力竭偵測（check_bullish_exhaustion，由
       呼叫端算好傳進來）給更精確的警訊文字，沒有力竭條件才用一般的跌破警訊
    4. 「前一段上漲有沒有摸到它該到的軌道」決定下軌/中軌算不算數的買點——
       前波沒摸到上軌，中軌買點要看這次回測有沒有站穩（收盤不破）才算數；
       前波連中軌都沒摸到，下軌也不建議買進，改為觀望；
       反過來，若上一次拉回沒跌破中軌就轉而向上，代表買盤偏強，上軌停利的
       語氣也會放緩
    """
    current_support = slope * last_idx + intercept
    current_r1 = current_support + channel_width
    current_mid = (current_support + current_r1) / 2
    current_price = df['Close'].iloc[-1]
    current_close = df['Close'].iloc[last_idx]

    if channel_width <= 0:
        pos = 0.0
    else:
        pos = (current_price - current_support) / channel_width  # 0=支撐, 1=R1

    near_support = pos >= -near_frac and pos <= near_frac
    near_r1 = abs(pos - 1.0) <= near_frac
    hard_break = pos < -near_frac

    direction = get_recent_direction(df, slope, intercept, channel_width, last_idx)
    touched_mid = touched_level_today(df, current_mid, last_idx)

    # 1) 明確跌破支撐超過 near_frac（通道寬度的比例）：最高優先權
    if hard_break:
        if is_exhausted:
            return 'exhaustion', '⚠️ 短線力竭，小心多轉空'
        return 'breakdown', '🚨 已跌破上升趨勢，留意多單停損！'

    # 2) 接近上軌（±near_frac，以通道寬度為單位）：先看前波拉回夠不夠深，
    #    拉回沒跌破中軌就轉強，代表買盤仍強，停利語氣放緩
    if near_r1:
        if get_bullish_pullback_strength(df, slope, intercept, channel_width, breakout_idx, last_idx):
            return 'sell_strong', '⚠️ 已接近多頭上軌，但近期拉回未跌破中軌顯示偏強，可留意突破續抱、不急停利'
        return 'sell', '⚠️ 已接近多頭上軌，停利出場或放空'

    # 3) 接近下軌（±near_frac，以通道寬度為單位）：前波反彈連中軌都沒摸到
    #    （力竭）的話，下軌不再是買點，改為觀望
    if near_support:
        peak_quality = get_bullish_peak_quality(df, slope, intercept, channel_width, breakout_idx, last_idx)
        if is_exhausted or peak_quality == 'exhausted':
            return 'watch_exhaustion', '⚠️ 前波反彈未過中軌，力竭警訊，下軌買盤薄弱，建議觀望'
        return 'buy', '✅ 已接近多頭下軌，可考慮建多單/空單回補'

    # 4) 今天觸及中軌，依前波高點強弱、方向分開處理
    if touched_mid:
        today_green = df['Close'].iloc[last_idx] > df['Open'].iloc[last_idx]
        if direction == 'rising':
            return 'watch_sell', '👀 已向上觸及上升中軌，可(部分)停利多單，若突破可續抱多單'
        elif direction == 'falling':
            peak_quality = get_bullish_peak_quality(df, slope, intercept, channel_width, breakout_idx, last_idx)
            if peak_quality == 'weak':
                # 前波有摸到中軌以上，但沒摸到上軌，屬於「轉弱待確認」：
                # 這次中軌回測有沒有站穩（收盤沒跌破中軌）才決定算不算買點
                if current_close >= current_mid:
                    return 'buy_mid', '✅ 前波未過上軌但中軌站穩收盤，可考慮輕倉試多，跌破中軌則出場'
                return 'watch_breakdown', '⚠️ 前波未過上軌，且中軌未能站穩，中軌不建議買進，請等拉回下軌'
            if peak_quality == 'exhausted':
                return 'watch_exhaustion', '⚠️ 前波反彈力道明顯不足（未過中軌），力竭疑慮，中軌不是買點，請保守觀望'
            if today_green:
                # 多方走勢雖轉弱，但今天並未真的跌破中軌，甚至收紅反彈，
                # 語氣不宜過度警示，先如實描述現況、持續觀察即可
                return 'watch_breakdown', '⚠️ 多方走勢轉弱，但今日觸及上升中軌未跌破且收紅反彈，若持續走弱才需留意多方力竭'
            return 'watch_breakdown', '⚠️ 已向下觸及上升中軌，若跌破請留意多方力竭'

    # 5) 其他情況：通道中間地帶，依方向給細緻提示
    # 這裡的 rising/falling 只是「最近5天相對通道基準線的短期漂移方向」，
    # 不代表真的跌破/站穩，那些情況已經由上面 near_support/hard_break 處理。
    # 順著原本多頭趨勢的漂移（rising）維持原本判斷即可；但逆勢漂移（falling，
    # 價格往支撐方向靠但還沒跌破、也還沒觸及支撐/中軌）只是短期現象，用
    # 「轉弱」這種確定性字眼會誤導——除非真的跌破支撐、或至少收黑加上真的
    # 跌破，才算數，這裡只給觀察用的提示，不下結論
    if direction == 'rising':
        return 'neutral_up', '📈 通道內偏強，續抱多單'
    elif direction == 'falling':
        return 'watch_reversal_down', '👀 近期價格略靠向支撐方向，但尚未跌破支撐，多頭格局未變，先觀察是否真的跌破再考慮動作'
    else:
        return 'neutral', '📊 通道中間，持股續抱或觀望'


def get_bearish_signal_v2(df, slope, intercept, channel_width, last_idx,
                           is_exhausted: bool, breakout_idx: int, near_frac: float = 0.15) -> Tuple[str, str]:
    """
    空頭通道訊號判斷（第二版），邏輯跟多頭對稱：
    - 壓力上軌：進場放空 / 慎防軋空的位置
    - 支撐下軌(S1)：回補出場的位置
    - 下降中軌：往下觸及＝順著空頭趨勢延續（可部分回補），
                往上觸及＝逆勢反彈（留意軋空/力竭）

    「近」「破」的判斷改用通道寬度的相對比例（±near_frac），理由同
    get_bullish_signal_v2：股價的絕對百分比不會隨通道寬窄縮放，改用
    通道寬度當基準，判斷標準才會正確隨通道本身的尺度縮放。

    同樣沿用「前一段走勢有沒有摸到它該到的軌道」判斷下一段回補/放空點站不
    站得住腳：前波回測沒摸到下軌，中軌回補點要看這次反彈有沒有守穩（收盤
    不站上）才算數；前波連中軌都沒摸到，上軌也不建議回補，改為觀望；反過來
    若上一次反彈沒站上中軌就轉而向下，代表空方偏強，上軌放空的語氣會加強。
    """
    current_resist = slope * last_idx + intercept
    current_s1 = current_resist - channel_width
    current_mid = (current_resist + current_s1) / 2
    current_price = df['Close'].iloc[-1]
    current_close = df['Close'].iloc[last_idx]

    if channel_width <= 0:
        pos = 0.0
    else:
        pos = (current_resist - current_price) / channel_width  # 0=壓力, 1=S1

    near_resist = pos >= -near_frac and pos <= near_frac
    near_s1 = abs(pos - 1.0) <= near_frac
    hard_break = pos < -near_frac

    direction = get_recent_direction(df, slope, intercept, channel_width, last_idx)
    touched_mid = touched_level_today(df, current_mid, last_idx)

    # 1) 明確突破壓力超過 near_frac（通道寬度的比例）：最高優先權（軋空警訊）
    if hard_break:
        if is_exhausted:
            return 'reverse_exhaustion', '⚠️ 空頭力竭，小心空轉多'
        return 'breakout_up', '🚨 已突破下降趨勢，留意空單停損！'

    # 2) 接近壓力上軌（±near_frac，以通道寬度為單位）：先看前波反彈夠不夠強，
    #    反彈沒站上中軌就轉弱，代表空方仍強，放空語氣加強
    if near_resist:
        if get_bearish_rebound_strength(df, slope, intercept, channel_width, breakout_idx, last_idx):
            return 'short_strong', '🔻 已接近空頭上軌，但近期反彈未站上中軌顯示偏弱，可留意跌破續抱、不急回補'
        return 'short', '🔻 已接近空頭上軌，可以放空或空單續抱'

    # 3) 接近支撐下軌 S1（±near_frac，以通道寬度為單位）：前波回測連中軌都
    #    沒摸到（力竭）的話，下軌不再是回補點，改為觀望
    if near_s1:
        trough_quality = get_bearish_trough_quality(df, slope, intercept, channel_width, breakout_idx, last_idx)
        if is_exhausted or trough_quality == 'exhausted':
            return 'watch_reverse_exhaustion', '⚠️ 前波回測未過中軌，力竭警訊，下軌賣壓薄弱，建議觀望'
        return 'cover', '💰 已接近空頭下軌，可考慮回補空單/找買點'

    # 4) 今天觸及中軌，依前波低點強弱、方向分開處理
    if touched_mid:
        today_green = df['Close'].iloc[last_idx] > df['Open'].iloc[last_idx]
        if direction == 'falling':
            return 'watch_cover', '👀 已向下觸及下降中軌，可(部分)回補空單，若跌破可續抱空單'
        elif direction == 'rising':
            trough_quality = get_bearish_trough_quality(df, slope, intercept, channel_width, breakout_idx, last_idx)
            if trough_quality == 'weak':
                # 前波有摸到中軌以下，但沒摸到下軌，屬於「轉弱待確認」：
                # 這次中軌反彈有沒有守穩（收盤沒站上中軌）才決定算不算回補/放空點
                if current_close <= current_mid:
                    return 'short_mid', '🔻 前波未過下軌但中軌守穩收盤，可考慮輕倉試空，站上中軌則出場'
                return 'watch_breakout', '⚠️ 前波未過下軌，且中軌未能守穩，中軌不建議放空，請等反彈上軌'
            if trough_quality == 'exhausted':
                return 'watch_reverse_exhaustion', '⚠️ 前波回測力道明顯不足（未過中軌），力竭疑慮，中軌不是空點，請保守觀望'
            if today_green:
                # 空頭格局沒有改變，但今天並未真的突破中軌，甚至收紅反彈，
                # 語氣不宜過度警示，先如實描述現況、持續觀察即可
                return 'watch_breakout', '⚠️ 空頭格局未變，但今日觸及下降中軌未跌破且收紅反彈，若持續走高才需留意空方力竭'
            return 'watch_breakout', '⚠️ 已向上觸及下降中軌，若突破請留意空方力竭'

    # 5) 其他情況
    # 同樣道理：這裡的 rising/falling 只是最近5天相對通道基準線的短期漂移，
    # 不代表真的突破/站穩。順著原本空頭趨勢的漂移（falling）維持原本判斷；
    # 逆勢漂移（rising，價格往壓力方向靠但還沒站上、也還沒觸及壓力/中軌）
    # 只是短期現象，用「轉強」這種確定性字眼會誤導——除非真的收紅且站上
    # 壓力（那屬於上面 hard_break/near_resist 的範圍），這裡只給觀察提示
    if direction == 'falling':
        return 'neutral_down', '📉 通道內偏弱，續抱空單'
    elif direction == 'rising':
        return 'watch_reversal_up', '👀 近期價格略靠向壓力方向，但尚未站上壓力，空頭格局未變，先觀察是否真的站穩再考慮動作'
    else:
        return 'neutral', '📊 通道中間，持空續抱或觀望'


def get_bullish_signal(position_pct, price, support, r1) -> Tuple[str, str]:
    """多頭訊號判斷（舊版，保留給需要單純用通道%位置判斷的地方相容使用）"""
    if position_pct <= 10:
        return 'buy', '✅ 接近支撐下軌，可以買進'
    elif position_pct >= 90:
        return 'sell', '⚠️ 接近壓力上軌，停利出場'
    elif position_pct <= 25:
        return 'watch_buy', '👀 靠近支撐，觀察買進機會'
    elif position_pct >= 75:
        return 'watch_sell', '👀 靠近壓力，注意停利'
    else:
        return 'neutral', '📊 通道中間，持股續抱或觀望'


def get_bearish_signal(position_pct, price, resist, s1) -> Tuple[str, str]:
    """空頭訊號判斷（舊版，保留給需要單純用通道%位置判斷的地方相容使用）"""
    if position_pct <= 10:
        return 'short', '🔻 接近壓力上軌，可以放空'
    elif position_pct >= 90:
        return 'cover', '💰 接近支撐下軌，回補停利'
    elif position_pct <= 25:
        return 'watch_short', '👀 靠近壓力，觀察放空機會'
    elif position_pct >= 75:
        return 'watch_cover', '👀 靠近支撐，注意回補'
    else:
        return 'neutral', '📊 通道中間，持空續抱或觀望'

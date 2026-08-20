#!/usr/bin/env python3
"""
圖表繪製模組
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 使用支援中文的字型，避免圖上中文字顯示成缺字方塊。
# 注意：Noto Sans CJK 的 .ttc 字型檔在 matplotlib 底下實際上只會被登記為
# 「Noto Sans CJK JP」這個名稱（即使裝的是完整的 CJK 字型包，繁體中文字
# 也涵蓋在同一個檔案裡，不影響顯示），所以把它也放進候選清單，
# 不能只依賴 TC/SC 這兩個名稱。
# 需要 Dockerfile 有安裝 fonts-noto-cjk 套件才會生效，本機沒裝的話
# matplotlib 會自動 fallback 回預設字型（中文一樣會是方塊，但不影響其他功能）。
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from io import BytesIO
from typing import Optional
from channel_engine import ChannelResult


def draw_channel_chart(df: pd.DataFrame, result: ChannelResult, ticker_name: str,
                        is_realtime_last: bool = False,
                        secondary_result: Optional[ChannelResult] = None,
                        secondary_label: str = '新') -> BytesIO:
    """
    繪製通道圖，回傳圖片 BytesIO

    Args:
        is_realtime_last: True 代表 df 最後一根K棒是「今天盤中即時、尚未收盤」的資料，
            會在圖上特別標示出來，跟已經收盤定案的日K做區隔
        secondary_result: 疊加顯示用的第二條通道，有兩種來源：
            1. 主通道被跌破/突破後找到的新反向通道
               （來自 channel_engine.try_secondary_channel）
            2. 走勢加速、原通道已跟不上時，被取代掉的「原本的通道」
               （來自 channel_engine.try_realign_channel，此時 result 是重新
               校正過的新通道，secondary_result 反而是舊的那條，用來對照）
            兩種情況都會用橘色虛線疊加畫在同一張圖上。
        secondary_label: 疊加通道的標籤前綴，預設「新」（用於情況1：新反向通道）；
            情況2（原通道被取代）應傳入「原」，圖上會顯示「原Support」等字樣，
            避免把「已經被取代的舊通道」誤標成「新通道」。
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2 = axes

    ax1.set_facecolor('#1a1a2e')
    ax2.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f1a')

    # Candlesticks (Taiwan: red=up, green=down)
    last_i = len(df) - 1
    for i in range(len(df)):
        color = '#ff4444' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#44cc44'
        is_forming = is_realtime_last and (i == last_i)
        line_color = '#ffdd00' if is_forming else color
        ax1.plot([i, i], [df['Low'].iloc[i], df['High'].iloc[i]], color=line_color,
                 linewidth=1.4 if is_forming else 0.6)
        body_bottom = min(df['Open'].iloc[i], df['Close'].iloc[i])
        body_top = max(df['Open'].iloc[i], df['Close'].iloc[i])
        if is_forming:
            # 即時K棒：用虛線黃框標示「還在形成中」，跟已收盤的K棒做區隔
            ax1.bar(i, body_top - body_bottom, bottom=body_bottom, width=0.6,
                    color=color, edgecolor='#ffdd00', linewidth=1.8, linestyle='--', alpha=0.85, hatch='//')
        else:
            ax1.bar(i, body_top - body_bottom, bottom=body_bottom, width=0.6, color=color, edgecolor=color)

    if is_realtime_last:
        ax1.annotate('⚡ 即時\n(尚未收盤)', xy=(last_i, df['High'].iloc[last_i]),
                     xytext=(last_i, df['High'].iloc[last_i] + (df['High'].max() - df['Low'].min()) * 0.04),
                     color='#ffdd00', fontsize=9, fontweight='bold', ha='center',
                     arrowprops=dict(arrowstyle='->', color='#ffdd00', lw=1))

    # Generate channel lines
    x_range = np.arange(len(df) + 15, dtype=float)
    
    if result.trend == 'bullish':
        line1 = result.slope * x_range + result.intercept  # Support
        line2 = line1 + result.channel_width  # R1
        line3 = line1 + 2 * result.channel_width  # R2
        line_color = '#00ffff'
        labels = ['Support', 'R1', 'R2']
    else:
        line1 = result.slope * x_range + result.intercept  # Resistance
        line2 = line1 - result.channel_width  # S1
        line3 = line1 - 2 * result.channel_width  # S2
        line_color = '#4488ff'
        labels = ['Resistance', 'S1', 'S2']

    # Draw from breakout K
    draw_start = max(0, result.breakout_idx - 3)
    x_draw = x_range[draw_start:]

    ax1.plot(x_draw, line1[draw_start:], color=line_color, linewidth=2.2,
             label=f'{labels[0]}: {result.current_line1:.0f}')
    ax1.plot(x_draw, line2[draw_start:], color=line_color, linewidth=2.2,
             label=f'{labels[1]}: {result.current_line2:.0f}')
    ax1.plot(x_draw, line3[draw_start:], color=line_color, linewidth=2.2,
             label=f'{labels[2]}: {result.current_line3:.0f}')

    # Fill zones
    ax1.fill_between(x_draw, line1[draw_start:], line2[draw_start:], alpha=0.05, color=line_color)
    ax1.fill_between(x_draw, line2[draw_start:], line3[draw_start:], alpha=0.05, color=line_color)

    # 中軌虛線（Support/Resistance 到 R1/S1 的中間），輔助判斷反彈/回測力道是否足夠
    mid_line = (line1 + line2) / 2
    ax1.plot(x_draw, mid_line[draw_start:], color=line_color, linewidth=1, linestyle=':', alpha=0.5)

    # Mark breakout K
    ax1.axvline(x=result.breakout_idx, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
    breakout_y = result.breakout_price
    ax1.scatter([result.breakout_idx], [breakout_y], color='yellow', marker='*', s=300, zorder=7,
               label=f'Breakout K ({result.breakout_date})')

    # Mark anchor points
    for idx, price in result.anchor_points[1:]:  # Skip breakout (already marked)
        ax1.scatter([idx], [price], color='white', marker='^' if result.trend == 'bullish' else 'v',
                   s=120, zorder=6, edgecolors='yellow', linewidths=1.5)

    # Secondary channel（跌破/突破後，嘗試找到的新反向通道，橘色虛線疊加顯示）
    if secondary_result is not None:
        sec_line1 = secondary_result.slope * x_range + secondary_result.intercept
        if secondary_result.trend == 'bullish':
            sec_line2 = sec_line1 + secondary_result.channel_width
            sec_line3 = sec_line1 + 2 * secondary_result.channel_width
            sec_labels = [f'{secondary_label}Support', f'{secondary_label}R1', f'{secondary_label}R2']
        else:
            sec_line2 = sec_line1 - secondary_result.channel_width
            sec_line3 = sec_line1 - 2 * secondary_result.channel_width
            sec_labels = [f'{secondary_label}Resistance', f'{secondary_label}S1', f'{secondary_label}S2']

        sec_color = '#ff8800'
        sec_draw_start = max(0, secondary_result.breakout_idx - 2)
        sec_x = x_range[sec_draw_start:]

        ax1.plot(sec_x, sec_line1[sec_draw_start:], color=sec_color, linewidth=1.8, linestyle='--',
                 label=f'{sec_labels[0]}: {secondary_result.current_line1:.0f}')
        ax1.plot(sec_x, sec_line2[sec_draw_start:], color=sec_color, linewidth=1.8, linestyle='--',
                 label=f'{sec_labels[1]}: {secondary_result.current_line2:.0f}')
        ax1.plot(sec_x, sec_line3[sec_draw_start:], color=sec_color, linewidth=1.8, linestyle='--',
                 label=f'{sec_labels[2]}: {secondary_result.current_line3:.0f}')
        ax1.scatter([secondary_result.breakout_idx], [secondary_result.breakout_price],
                   color=sec_color, marker='*', s=180, zorder=7,
                   label=f'{secondary_label}趨勢起點 ({secondary_result.breakout_date})')

    # Mark exhaustion swing point (未觸及中軌的轉折點，力竭訊號的依據)
    if result.is_exhausted and result.exhaustion_peak_idx is not None:
        ep_idx = result.exhaustion_peak_idx
        ep_price = df['High'].iloc[ep_idx] if result.trend == 'bullish' else df['Low'].iloc[ep_idx]
        ax1.scatter([ep_idx], [ep_price], color='#ff00ff', marker='x', s=220, zorder=8, linewidths=3)
        ax1.annotate(f'未達中軌 ({result.exhaustion_peak_pct:.0f}%)',
                     xy=(ep_idx, ep_price),
                     xytext=(ep_idx, ep_price + (result.channel_width * (0.08 if result.trend == 'bullish' else -0.08))),
                     color='#ff00ff', fontsize=9, fontweight='bold', ha='center',
                     arrowprops=dict(arrowstyle='->', color='#ff00ff', lw=1))

    # Current price line
    current_price = df['Close'].iloc[-1]
    ax1.axhline(y=current_price, color='orange', linewidth=0.8, linestyle=':', alpha=0.5)

    # Volume
    colors = ['#ff4444' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#44cc44' 
              for i in range(len(df))]
    ax2.bar(range(len(df)), df['Volume'], color=colors, alpha=0.7)
    
    # Highlight breakout volume
    ax2.bar(result.breakout_idx, df['Volume'].iloc[result.breakout_idx], color='yellow', alpha=0.9)

    # Right side labels
    last_x = len(df) + 10
    ax1.text(last_x, result.current_line1, f'{labels[0]}: {result.current_line1:.0f}',
             color=line_color, fontsize=10, va='center', fontweight='bold')
    ax1.text(last_x, result.current_line2, f'{labels[1]}: {result.current_line2:.0f}',
             color=line_color, fontsize=10, va='center', fontweight='bold')
    ax1.text(last_x, result.current_line3, f'{labels[2]}: {result.current_line3:.0f}',
             color=line_color, fontsize=10, va='center', fontweight='bold')
    ax1.text(len(df) + 1, current_price + (result.channel_width * 0.03),
             f'{current_price:.1f}', color='orange', fontsize=9)

    # Signal annotation
    signal_color = {
        'buy': '#00ff00', 'sell': '#ff6600', 'short': '#ff4444', 'cover': '#00ff00',
        'buy_mid': '#00ff00', 'short_mid': '#ff4444',
        'sell_strong': '#ff6600', 'short_strong': '#ff4444',
        'watch_buy': '#88ff88', 'watch_sell': '#ffaa44',
        'watch_short': '#ff8888', 'watch_cover': '#88ff88',
        'watch_reversal_up': '#88ff88', 'watch_reversal_down': '#ff8888',
        'neutral': '#aaaaaa',
        'exhaustion': '#ff00ff', 'watch_exhaustion': '#ff88ff',
        'reverse_exhaustion': '#ff00ff', 'watch_reverse_exhaustion': '#ff88ff',
    }
    sig_color = signal_color.get(result.signal, '#aaaaaa')
    # \ufe0f（emoji 顏色選擇符）是隱形字元，但 Noto Sans CJK 字型沒有這個
    # glyph，畫在圖上會多一個缺字方塊，這裡先過濾掉，不影響原本傳回給
    # Telegram 訊息用的 result.signal_text（那邊是原生 emoji 顯示不受影響）
    chart_signal_text = result.signal_text.replace('\ufe0f', '')
    ax1.text(len(df) // 2, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else df['High'].max() * 1.02,
             chart_signal_text, color=sig_color, fontsize=12, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor=sig_color, alpha=0.9))

    # Title
    trend_text = '多頭' if result.trend == 'bullish' else '空頭'
    realtime_suffix = '（含即時，尚未收盤）' if is_realtime_last else ''
    ax1.set_title(f'{ticker_name} - {trend_text} 等距平行通道{realtime_suffix}',
                  fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel('Price (TWD)', color='white')
    ax1.legend(loc='upper left' if result.trend == 'bullish' else 'upper right',
               fontsize=9, facecolor='#2a2a3e', labelcolor='white', framealpha=0.9)
    ax1.grid(True, alpha=0.15, color='gray')
    ax1.tick_params(colors='white')
    ax1.set_xlim(-2, len(df) + 15)

    ax2.set_ylabel('Volume', color='white')
    ax2.grid(True, alpha=0.15, color='gray')
    ax2.tick_params(colors='white')
    ax2.set_xlim(-2, len(df) + 15)

    # X-axis dates
    tick_positions = np.linspace(0, len(df) - 1, 10, dtype=int)
    date_labels = [df.index[i].strftime('%m/%d') for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(date_labels, color='white')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(date_labels, color='white')

    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

#!/usr/bin/env python3
"""
台股趨勢通道 Telegram Bot
功能：
1. 輸入股號 → 自動畫通道圖 + 判斷訊號
2. 管理觀察清單
3. 每日自動掃描 → 推送訊號
"""

import os
import logging
import json
import html
from datetime import datetime
from pathlib import Path

import pandas as pd
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from channel_engine import (
    calculate_channel, try_secondary_channel,
    channel_outside_ratio, try_realign_channel,
    is_long_term_choppy, rebase_broken_channel,
)
from chart_drawer import draw_channel_chart
from finmind_fetcher import (
    fetch_stock_ohlc, fetch_stock_ohlc_with_realtime, fetch_stock_name,
    fetch_futures_ohlc, is_futures_keyword, get_futures_info,
    search_stock_by_name,
)
from predict_engine import init_engine, get_engine

from dotenv import load_dotenv
load_dotenv()

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Config
BOT_TOKEN = os.getenv('BOT_TOKEN')
FINMIND_TOKEN = os.getenv('FINMIND_TOKEN')
WATCHLIST_FILE = '/app/data/watchlist.json'
SCAN_HOUR = int(os.getenv('SCAN_HOUR', '14'))  # 每日掃描時間（台灣時間14點=收盤後）
SCAN_MINUTE = int(os.getenv('SCAN_MINUTE', '30'))


def load_watchlist() -> dict:
    """載入觀察清單"""
    Path('/app/data').mkdir(parents=True, exist_ok=True)
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_watchlist(data: dict):
    """儲存觀察清單"""
    Path('/app/data').mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_stock_data(ticker_code: str) -> tuple:
    """
    取得股票資料 - 改用 FinMind API

    原本是自己爬 TWSE/TPEX 網頁的 JSON API，
    現在改成呼叫 finmind_fetcher.fetch_stock_ohlc()。
    回傳格式維持跟原本一樣：(df, name, code)，
    失敗時回傳 (None, None, code)，所以下面呼叫這個函式的地方完全不用改。

    name 現在會盡量帶入中文股名（例如「南亞科」），抓不到名稱時
    fallback 回傳「代號.TW」，不會讓整個查詢失敗。
    """
    code = ticker_code.replace('.TW', '').replace('.TWO', '')

    try:
        df = fetch_stock_ohlc(code, token=FINMIND_TOKEN)
    except Exception as e:
        logger.error(f"FinMind fetch error for {code}: {e}")
        return None, None, code

    if df is None or df.empty:
        return None, None, code

    stock_name = fetch_stock_name(code, token=FINMIND_TOKEN)
    name = stock_name if stock_name else f"{code}.TW"
    return df, name, code


def get_stock_data_realtime(ticker_code: str) -> tuple:
    """
    取得股票資料，並嘗試把「今天盤中即時報價」併入成通道計算/畫圖用的
    最後一根K棒（詳見 finmind_fetcher.fetch_stock_ohlc_with_realtime）。

    專門給 analyze_stock（單次查詢+畫圖）使用；觀察清單掃描
    （scan_watchlist / auto_scan）仍使用純日K版本的 get_stock_data，
    避免大量股票逐一打即時 API 太耗用量。

    Returns: (df, name, code, is_realtime, snapshot)
    """
    code = ticker_code.replace('.TW', '').replace('.TWO', '')

    try:
        df, is_realtime, snapshot = fetch_stock_ohlc_with_realtime(code, token=FINMIND_TOKEN)
    except Exception as e:
        logger.error(f"FinMind fetch error for {code}: {e}")
        return None, None, code, False, None

    if df is None or df.empty:
        return None, None, code, False, None

    stock_name = fetch_stock_name(code, token=FINMIND_TOKEN)
    name = stock_name if stock_name else f"{code}.TW"
    return df, name, code, is_realtime, snapshot


# 判斷「原通道是否已經跟不上走勢」的參數：
# 最近 CHANNEL_STALE_LOOKBACK 根K棒中，收盤價落在通道外側的比例
# 達到 CHANNEL_STALE_THRESHOLD 以上，就視為原通道太平緩、需要重新校正
CHANNEL_STALE_LOOKBACK = 10
CHANNEL_STALE_THRESHOLD = 0.6


def get_effective_channel(df: pd.DataFrame):
    """
    計算通道，並自動偵測「原通道是否已經跟不上走勢」，分兩種情況處理：

    情況1（還在原方向內，走勢正在加速）：
    像 2327 國巨、2634 漢翔、2303 聯電、6182 合晶 這類噴出股，常常在上漲
    過程中斜率突然加速，導致用早期較平緩K棒畫出的通道，完全跟不上後面
    噴出的走勢。用 channel_outside_ratio 偵測到後，改用 try_realign_channel
    只拿最近一段資料重新擬合，取代掉太平緩、已經不合身的舊通道。

    情況2（已經跌破/突破，急殺/急拉階段）：
    同樣是噴出股常見的模式——緩漲 → 加速噴出 → 急殺，三段式走勢。如果
    「原通道」在崩跌前根本沒抓到加速噴出那一段（例如主通道演算法挑到的
    起漲點太早、太平緩），拿去判斷跌破、找反向新通道時會失真：真正
    「剛被跌破的那條線」其實是噴出段那條更陡的通道，不是緩漲階段的通道。
    這裡用 rebase_broken_channel 回頭檢查崩跌前是否曾經加速過，如果有，
    改用「校正後、延伸到今天」的陡峭通道取代原本太平緩的通道，這樣後續
    （在 bot.py 呼叫 try_secondary_channel 時）找到的反向新通道，才會是
    真正緊貼著噴出段畫出來的，而不是從很久以前的緩漲起點延伸過來的。

    Returns:
        (result, old_result)
        - result: 實際要使用的通道結果
        - old_result: 情況1重新校正時，這是被取代掉的舊通道（用來疊圖對照，
          呼叫端可以用橘色虛線畫出來）；情況2或沒有校正時是 None
          （情況2直接把 result 換掉，不走疊圖對照這條路，因為情況2的
          呼叫端 bot.py 接下來還會用 result.signal 去找 try_secondary_channel，
          兩條線疊圖的效果是一樣的，不需要再多一個 old_result）
    """
    result = calculate_channel(df)
    if result is None:
        return None, None

    # 情況1：還在原方向內，檢查是否加速需要重新校正
    outside_ratio = channel_outside_ratio(df, result, lookback=CHANNEL_STALE_LOOKBACK)
    if outside_ratio >= CHANNEL_STALE_THRESHOLD:
        realigned = try_realign_channel(df, result)
        if realigned is not None:
            return realigned, result

    # 情況2：已經跌破/突破（含力竭變體），回頭檢查崩跌前是否曾經加速過
    if result.signal in ('breakdown', 'breakout_up', 'exhaustion', 'reverse_exhaustion'):
        rebased = rebase_broken_channel(df, result,
                                         stale_lookback=CHANNEL_STALE_LOOKBACK,
                                         stale_threshold=CHANNEL_STALE_THRESHOLD)
        if rebased is not None:
            result = rebased

    return result, None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    await update.message.reply_text(
        "🐟 台股趨勢通道 Bot\n\n"
        "📋 使用方式：\n"
        "• 直接輸入股號（例如 <code>2408</code>）→ 畫通道圖 + 訊號\n"
        "• <code>/watch 2408</code> → 加入觀察清單\n"
        "• <code>/unwatch 2408</code> → 移出觀察清單\n"
        "• <code>/list</code> → 查看觀察清單\n"
        "• <code>/scan</code> → 立即掃描觀察清單\n"
        "• <code>/help</code> → 使用說明\n\n"
        f"⏰ 每日自動掃描：台灣時間 {SCAN_HOUR}:{SCAN_MINUTE:02d}",
        parse_mode='HTML'
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    await update.message.reply_text(
        "📐 <b>通道邏輯：</b>\n"
        "• 多頭：從突破壓力的關鍵K起算，低點連低點為支撐，等距平行往上\n"
        "• 空頭：從跌破支撐的關鍵K起算，高點連高點為壓力，等距平行往下\n\n"
        "📊 <b>訊號規則：</b>\n"
        "• 多頭觸下軌 → ✅ 可以買進\n"
        "• 多頭觸上軌 → ⚠️ 停利出場\n"
        "• 空頭觸上軌 → 🔻 可以放空\n"
        "• 空頭觸下軌 → 💰 回補停利\n\n"
        "📋 <b>指令：</b>\n"
        "• 輸入股號 → 畫圖\n"
        "• <code>/watch 股號</code> → 加入觀察\n"
        "• <code>/unwatch 股號</code> → 移除觀察\n"
        "• <code>/list</code> → 觀察清單\n"
        "• <code>/scan</code> → 立即掃描",
        parse_mode='HTML'
    )


def get_next_day_levels(result):
    """
    計算「隔一個交易日」的通道線位置（Support/R1/R2 或 Resistance/S1/S2），
    而不是「今天」的位置。

    背景：通道線的 current_line1/2/3 是用「今天」這個時間點算出來的，但使用者
    看到訊息時「今天」已經收盤了，這幾個數字對於「接下來該怎麼操作」參考價值
    不大；改成算「隔一天」的位置，才能在還沒開盤前，先知道明天的支撐/壓力/
    R1/R2 大概在哪裡。

    數學上，通道是等距平行線，隔一天只是沿著斜率再往前推一根K棒的距離，
    所以只要把 slope 加一次到每條線上即可，不需要重新計算整條線。

    唯一的例外：如果這個通道是 is_stale_extrapolation（已經外推超出合理範圍，
    見 channel_engine.extend_channel_to_now 說明），代表「今天」本身這幾個數字
    就已經是被鎖住、不再繼續外推的結果，這種情況下「隔一天」也不應該再往外
    多推一格（會讓已經失真的數字更加失真），所以直接沿用跟今天一樣的凍結值。

    Returns:
        (line1_tomorrow, line2_tomorrow, line3_tomorrow)
    """
    delta = 0.0 if result.is_stale_extrapolation else result.slope
    return (
        result.current_line1 + delta,
        result.current_line2 + delta,
        result.current_line3 + delta,
    )


async def analyze_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """分析股票（直接輸入股號、期貨關鍵字、或股名）"""
    text = update.message.text.strip()
    
    # Check if it's a futures keyword (台指、TX、小台 etc.)
    is_futures = is_futures_keyword(text)
    
    if not is_futures:
        # Check if it's a stock code (numbers, possibly with .TW/.TWO)
        code = text.replace('.TW', '').replace('.TWO', '')
        if not code.isdigit() or len(code) < 4 or len(code) > 6:
            # Not a pure number — try stock name search
            search_result = search_stock_by_name(text, token=FINMIND_TOKEN)
            if search_result:
                text = search_result[0]  # Use the stock_id
            else:
                return  # Not a stock code or name, ignore
    
    await update.message.reply_text(f"⏳ 分析 {text} 中...")
    
    try:
        if is_futures:
            futures_id, futures_name = get_futures_info(text)
            try:
                df = fetch_futures_ohlc(futures_id, token=FINMIND_TOKEN)
            except Exception as e:
                logger.error(f"Futures fetch error for {futures_id}: {e}")
                await update.message.reply_text(f"❌ 找不到 {text} 的資料：{str(e)}")
                return
            name = futures_name
            symbol = futures_id
            is_realtime = False
            snapshot = None
        else:
            df, name, symbol, is_realtime, snapshot = get_stock_data_realtime(text)
        
        if df is None or len(df) < 30:
            await update.message.reply_text(f"❌ 找不到 {text} 的資料，或資料不足")
            return
        
        result, old_result = get_effective_channel(df)
        
        if result is None:
            await update.message.reply_text(f"❌ 無法計算 {text} 的通道（資料不足或趨勢不明確）")
            return
        
        # 兩種情況會疊加一條橘色虛線的第二通道：
        # 1. 走勢加速，原通道已跟不上，old_result 是被取代的舊通道（供對照）
        # 2. 主通道被跌破/突破/力竭時，嘗試找出是否已形成新的反向通道
        secondary_result = None
        secondary_label = '新'
        if old_result is not None:
            secondary_result = old_result
            secondary_label = '原'
        elif result.signal in ('breakdown', 'exhaustion', 'breakout_up', 'reverse_exhaustion'):
            secondary_result = try_secondary_channel(df, result)

        # Draw chart（如果今天有即時K棒，圖上最後一根會用不同標示畫出來；
        # 如果有找到反向通道或原通道對照，會用橘色虛線疊加畫上去）
        chart_buf = draw_channel_chart(df, result, f"{text} {name}",
                                        is_realtime_last=is_realtime,
                                        secondary_result=secondary_result,
                                        secondary_label=secondary_label)

        # 組即時報價那一行（直接沿用剛剛併資料時抓到的 snapshot，不用再打一次API）
        realtime_line = ""
        if snapshot and snapshot.get('close'):
            change_price = snapshot.get('change_price', 0) or 0
            change_rate = snapshot.get('change_rate', 0) or 0
            arrow = '🔺' if change_price > 0 else ('🔻' if change_price < 0 else '▪️')
            snap_time = snapshot.get('date', '')
            realtime_line = (
                f"\n⚡ 即時：{snapshot['close']:.2f} "
                f"{arrow}{change_price:+.2f} ({change_rate:+.2f}%)"
                f"{f' [{snap_time}]' if snap_time else ''}\n"
            )

        # Build caption
        # 改用 HTML 格式（parse_mode='HTML'）而不是舊版 Markdown。
        # 舊版 Markdown 的 *、_ 這些符號一旦沒有完美成對就會讓整則訊息送不出去，
        # 隨著這裡動態組合的內容越來越多（即時報價、通道校正說明等），
        # 很容易某個地方沒對齊就整包炸掉（Can't parse entities 錯誤）。
        # HTML 只有 &、<、> 需要跳脫，用 html.escape() 處理動態內容就能徹底避開這類問題。
        safe_text = html.escape(text)
        safe_name = html.escape(name)

        # 通道線顏色，跟 chart_drawer.py 畫圖時用的顏色對應，方便訊息文字跟圖上
        # 的線互相對照（多頭主通道＝青色、空頭主通道＝藍色、次要/對照通道一律橘色虛線）
        MAIN_COLOR_EMOJI = {'bullish': '🩵', 'bearish': '🔵'}
        SECONDARY_COLOR_EMOJI = '🟠'
        main_color = MAIN_COLOR_EMOJI[result.trend]

        # 🎯 那一行是「目前該怎麼操作」的策略建議，必須用目前實際在走的通道
        # 去算，而不是舊的、已經跌破/突破的主通道——主通道被跌破後，它自己的
        # signal_text（例如「已跌破上升趨勢，留意多單停損」）只是說明「曾經
        # 發生過跌破」這件事，跌破當下就該看了，跌破之後每天都還顯示同一句話
        # 就變成嚴重落後的資訊。真正該遵循的操作建議，是新通道（secondary_result，
        # secondary_label=='新' 時）自己算出來的 signal_text（例如「已接近空頭
        # 上軌，可以放空」），這才反映了現在的位置。
        # 情況1（secondary_label=='原'，走勢加速重新校正）不受影響，因為 result
        # 本身就已經是最新、還在同方向延續的通道。
        if secondary_result is not None and secondary_label == '新':
            primary_signal_text = secondary_result.signal_text
            strategy_note = f"（依{SECONDARY_COLOR_EMOJI}新通道判斷，原通道已跌破/突破，僅供對照）"
        else:
            primary_signal_text = result.signal_text
            strategy_note = ""
        safe_signal_text = html.escape(primary_signal_text)

        # 趨勢判斷：以「目前實際在走的通道」為準，而不是只看最早判斷出來的主通道方向。
        # 主通道被跌破/突破、且已經找到反向新通道時（secondary_label == '新'，
        # 也就是 try_secondary_channel 找到的真正反向通道），代表盤勢事實上已經
        # 轉向，此時「趨勢」欄位要跟著顯示新通道的方向，避免使用者看到「多頭」
        # 卻配一條向下噴出的通道，反而誤判。
        # 情況1（secondary_label == '原'，走勢加速、原通道被重新校正取代）方向
        # 沒有改變，只是斜率變陡，所以不受影響，繼續顯示 result.trend。
        effective_trend = result.trend
        if secondary_result is not None and secondary_label == '新':
            effective_trend = secondary_result.trend

        trend_text = '📈 多頭' if effective_trend == 'bullish' else '📉 空頭'
        realtime_note = "\n⚡ <b>圖表已包含今日盤中走勢（尚未收盤，數字仍會變動）</b>" if is_realtime else ""

        channel_note = ""
        if old_result is not None:
            channel_note = (
                "\n\n📐 偵測到走勢明顯加速，近期K棒多數已跑出原通道，"
                f"已切換為最新通道（{SECONDARY_COLOR_EMOJI} 橘色虛線為原通道，僅供對照）"
            )
        elif secondary_result is not None:
            if secondary_result.trend == 'bearish':
                sec_trend_label, l1, l2, l3 = '📉 新的下降通道', 'Resistance', 'S1', 'S2'
            else:
                sec_trend_label, l1, l2, l3 = '📈 新的上升通道', 'Support', 'R1', 'R2'
            sec_t1, sec_t2, sec_t3 = get_next_day_levels(secondary_result)
            channel_note = (
                f"\n\n{sec_trend_label}（{SECONDARY_COLOR_EMOJI} 橘色虛線，僅供參考，下列為隔一交易日位置）：\n"
                f"• {l1}：{sec_t1:.1f}\n"
                f"• {l2}：{sec_t2:.1f}\n"
                f"• {l3}：{sec_t3:.1f}"
            )
        elif result.signal in ('breakdown', 'exhaustion', 'breakout_up', 'reverse_exhaustion'):
            channel_note = "\n\n📐 新趨勢資料還不足以形成通道，先持續觀察"

        # 不管上面判斷出哪種訊號，只要長期（近半年）走勢比較像區間整理，
        # 就額外加註提醒——避免使用者只看到「近期方向」就誤判成趨勢會延續
        if is_long_term_choppy(df):
            channel_note += (
                "\n\n📦 長期（近半年）走勢較偏區間整理，近期方向僅供參考，"
                "慎防假突破/假跌破"
            )

        # 原通道是用「短時間內急拉/急殺」的窗口擬合出來的陡峭斜率，如果外推
        # 到今天已經超過合理範圍（見 channel_engine.extend_channel_to_now 說明），
        # Support/R1/R2 這幾個數字會失真（甚至可能超過歷史最高/最低價），
        # 提醒使用者這幾個數字僅供參考，不要直接拿來當進出場依據
        if result.is_stale_extrapolation:
            channel_note += (
                "\n\n⏳ 原通道形成時間較短，目前已外推超出合理範圍，"
                "Support/R1/R2 數字僅供參考，請以近期實際走勢及上方新通道（若有）為準"
            )
        
        t_line1, t_line2, t_line3 = get_next_day_levels(result)

        if result.trend == 'bullish':
            caption = (
                f"<b>{safe_text} {safe_name}</b>\n"
                f"趨勢：{trend_text}\n"
                f"關鍵K：{result.breakout_date}\n\n"
                f"📐 隔一交易日通道位置（{main_color} 主通道）：\n"
                f"• Support：{t_line1:.1f}\n"
                f"• R1：{t_line2:.1f}\n"
                f"• R2：{t_line3:.1f}\n\n"
                f"💰 收盤：{df['Close'].iloc[-1]:.1f}"
                f"{realtime_line}\n"
                f"📍 今日收盤位置：{result.position_pct:.0f}%\n\n"
                f"🎯 {safe_signal_text}{strategy_note}"
                f"{realtime_note}"
                f"{channel_note}"
            )
        else:
            caption = (
                f"<b>{safe_text} {safe_name}</b>\n"
                f"趨勢：{trend_text}\n"
                f"關鍵K：{result.breakout_date}\n\n"
                f"📐 隔一交易日通道位置（{main_color} 主通道）：\n"
                f"• Resistance：{t_line1:.1f}\n"
                f"• S1：{t_line2:.1f}\n"
                f"• S2：{t_line3:.1f}\n\n"
                f"💰 收盤：{df['Close'].iloc[-1]:.1f}"
                f"{realtime_line}\n"
                f"📍 今日收盤位置：{result.position_pct:.0f}%\n\n"
                f"🎯 {safe_signal_text}{strategy_note}"
                f"{realtime_note}"
                f"{channel_note}"
            )
        
        await update.message.reply_photo(
            photo=chart_buf,
            caption=caption,
            parse_mode='HTML'
        )
        
    except Exception as e:
        logger.error(f"Error analyzing {text}: {e}")
        await update.message.reply_text(f"❌ 分析 {text} 時發生錯誤：{str(e)}")


async def watch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """加入觀察清單"""
    if not context.args:
        await update.message.reply_text("用法：<code>/watch 2408</code>", parse_mode='HTML')
        return
    
    code = context.args[0].strip()
    chat_id = str(update.effective_chat.id)
    
    watchlist = load_watchlist()
    if chat_id not in watchlist:
        watchlist[chat_id] = []
    
    if code not in watchlist[chat_id]:
        watchlist[chat_id].append(code)
        save_watchlist(watchlist)
        await update.message.reply_text(f"✅ 已加入觀察清單：{code}")
    else:
        await update.message.reply_text(f"ℹ️ {code} 已在觀察清單中")


async def unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """移出觀察清單"""
    if not context.args:
        await update.message.reply_text("用法：<code>/unwatch 2408</code>", parse_mode='HTML')
        return
    
    code = context.args[0].strip()
    chat_id = str(update.effective_chat.id)
    
    watchlist = load_watchlist()
    if chat_id in watchlist and code in watchlist[chat_id]:
        watchlist[chat_id].remove(code)
        save_watchlist(watchlist)
        await update.message.reply_text(f"✅ 已移出觀察清單：{code}")
    else:
        await update.message.reply_text(f"ℹ️ {code} 不在觀察清單中")


async def list_watch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """查看觀察清單"""
    chat_id = str(update.effective_chat.id)
    watchlist = load_watchlist()
    
    stocks = watchlist.get(chat_id, [])
    if not stocks:
        await update.message.reply_text("📋 觀察清單是空的\n用 <code>/watch 股號</code> 加入", parse_mode='HTML')
    else:
        text = "📋 <b>觀察清單：</b>\n" + "\n".join(f"• {html.escape(s)}" for s in stocks)
        await update.message.reply_text(text, parse_mode='HTML')


async def scan_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """手動掃描觀察清單"""
    chat_id = str(update.effective_chat.id)
    watchlist = load_watchlist()
    
    stocks = watchlist.get(chat_id, [])
    if not stocks:
        await update.message.reply_text("📋 觀察清單是空的")
        return
    
    await update.message.reply_text(f"⏳ 掃描 {len(stocks)} 檔股票中...")
    
    signals = []
    for code in stocks:
        try:
            df, name, symbol = get_stock_data(code)
            if df is None or len(df) < 30:
                continue
            result, _ = get_effective_channel(df)
            if result is None:
                continue
            
            if result.signal in ('buy', 'sell', 'short', 'cover',
                                  'buy_mid', 'short_mid', 'sell_strong', 'short_strong',
                                  'breakdown', 'exhaustion', 'breakout_up', 'reverse_exhaustion',
                                  'watch_exhaustion', 'watch_reverse_exhaustion'):
                signals.append((code, name, result, df['Close'].iloc[-1], is_long_term_choppy(df)))
        except Exception as e:
            logger.error(f"Scan error for {code}: {e}")
    
    if signals:
        text = "🚨 <b>訊號提醒：</b>\n\n"
        for code, name, result, price, choppy in signals:
            choppy_note = "\n📦 長期較偏區間整理，僅供參考" if choppy else ""
            text += (f"<b>{html.escape(code)}</b> {html.escape(name)}\n"
                     f"{html.escape(result.signal_text)}\n"
                     f"收盤：{price:.1f}{choppy_note}\n\n")
        await update.message.reply_text(text, parse_mode='HTML')
    else:
        await update.message.reply_text("✅ 掃描完成，目前觀察清單中沒有觸發訊號的股票")


async def auto_scan(context: ContextTypes.DEFAULT_TYPE):
    """定時自動掃描所有用戶的觀察清單"""
    logger.info("Running auto scan...")
    watchlist = load_watchlist()
    
    for chat_id, stocks in watchlist.items():
        if not stocks:
            continue
        
        signals = []
        for code in stocks:
            try:
                df, name, symbol = get_stock_data(code)
                if df is None or len(df) < 30:
                    continue
                result, _ = get_effective_channel(df)
                if result is None:
                    continue
                
                if result.signal in ('buy', 'sell', 'short', 'cover',
                                      'buy_mid', 'short_mid', 'sell_strong', 'short_strong',
                                      'breakdown', 'exhaustion', 'breakout_up', 'reverse_exhaustion',
                                      'watch_exhaustion', 'watch_reverse_exhaustion',
                                      'watch_sell', 'watch_breakdown', 'watch_cover', 'watch_breakout',
                                      'watch_reversal_up', 'watch_reversal_down'):
                    signals.append((code, name, result, df['Close'].iloc[-1], is_long_term_choppy(df)))
            except Exception as e:
                logger.error(f"Auto scan error for {code}: {e}")
        
        if signals:
            text = f"📊 <b>每日掃描報告</b> ({datetime.now().strftime('%Y-%m-%d')})\n\n"
            for code, name, result, price, choppy in signals:
                choppy_note = "\n📦 長期較偏區間整理，僅供參考" if choppy else ""
                text += (f"<b>{html.escape(code)}</b> {html.escape(name)}\n"
                         f"{html.escape(result.signal_text)}\n"
                         f"收盤：{price:.1f}{choppy_note}\n\n")
            
            try:
                await context.bot.send_message(chat_id=int(chat_id), text=text, parse_mode='HTML')
            except Exception as e:
                logger.error(f"Failed to send to {chat_id}: {e}")
    
    logger.info("Auto scan complete")




async def predict_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """產出今日預測名單"""
    await update.message.reply_text("🔄 正在計算預測名單，請稍候（約 3~5 分鐘）...")
    engine = get_engine()
    if engine is None:
        await update.message.reply_text("❌ 預測引擎未初始化")
        return
    try:
        result_path = engine.predict()
        if result_path:
            await update.message.reply_document(
                document=open(result_path, 'rb'),
                caption="📊 今日選股預測名單"
            )
        else:
            await update.message.reply_text("❌ 預測失敗，請查看 log")
    except Exception as e:
        logger.error(f"predict error: {e}")
        await update.message.reply_text(f"❌ 預測錯誤: {e}")


async def review_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """檢核 20 天前的預測"""
    await update.message.reply_text("🔄 正在檢核歷史預測，請稍候...")
    engine = get_engine()
    if engine is None:
        await update.message.reply_text("❌ 預測引擎未初始化")
        return
    try:
        result_path = engine.review()
        if result_path:
            await update.message.reply_document(
                document=open(result_path, 'rb'),
                caption="📋 歷史預測檢核報告"
            )
        else:
            await update.message.reply_text("❌ 沒有可檢核的歷史記錄（需先執行 /predict 累積記錄）")
    except Exception as e:
        logger.error(f"review error: {e}")
        await update.message.reply_text(f"❌ 檢核錯誤: {e}")

def main():
    """Bot 主程式"""
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not set!")
        return

    if not FINMIND_TOKEN:
        logger.warning("FINMIND_TOKEN not set! FinMind 免費額度很低，建議設定 token 以取得較高的請求上限。")
    
    # Build application
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("watch", watch))
    app.add_handler(CommandHandler("unwatch", unwatch))
    app.add_handler(CommandHandler("list", list_watch))
    app.add_handler(CommandHandler("scan", scan_watchlist))
    
    app.add_handler(CommandHandler("predict", predict_handler))
    app.add_handler(CommandHandler("review", review_handler))

    # 初始化預測引擎
    init_engine(FINMIND_TOKEN)
    # Stock code handler (numbers only, 4-6 digits)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_stock))
    
    # Schedule daily scan
    scheduler = AsyncIOScheduler(timezone='Asia/Taipei')
    scheduler.add_job(
        auto_scan,
        'cron',
        hour=SCAN_HOUR,
        minute=SCAN_MINUTE,
        args=[app],
        id='daily_scan'
    )
    scheduler.start()
    logger.info(f"Daily scan scheduled at {SCAN_HOUR}:{SCAN_MINUTE:02d} Asia/Taipei")
    
    # Run bot
    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()

# 台股趨勢通道 Telegram Bot

自動分析台股趨勢通道，判斷進出場訊號。

## 功能

- 輸入股號 → 自動畫出等距平行通道 + 訊號判斷
- 觀察清單管理
- 每日自動掃描，主動推送訊號
- **選股預測**：AI 模型自動篩選做多/做空標的

## 使用方式

在 Telegram 對你的 Bot 說：

| 指令 | 功能 |
|------|------|
| `2408` | 直接輸入股號 |
| `南亞科` | 直接輸入股名（自動查找對應股號） |
| `台積電` | 支援模糊比對（輸入「台積」也能找到） |
| `台指` / `TX` | 台指期貨 |
| `小台` / `MTX` | 小型台指期 |
| `電子期` / `TE` | 電子期貨 |
| `金融期` / `TF` | 金融期貨 |
| `/watch 2408` | 加入觀察清單 |
| `/unwatch 2408` | 移出觀察清單 |
| `/list` | 查看觀察清單 |
| `/scan` | 立即掃描觀察清單 |
| `/predict` | 產出今日做多/做空 Top 10 預測名單（Excel） |
| `/review` | 檢核 20 天前的預測 vs 實際表現（Excel） |
| `/help` | 使用說明 |

## 通道邏輯

- **多頭**：找到突破壓力的關鍵K → 低點連低點為支撐起點 → 等距平行往上
- **空頭**：找到跌破支撐的關鍵K（峰值） → 高點連高點為壓力起點 → 等距平行往下

### 通道自我校正

- 走勢加速中：偵測到K棒跑出通道外側，切換為更陡的新通道
- 已跌破/突破：回頭檢查是否曾加速噴出，用校正後通道判斷
- 急殺無反彈：用 breakout 後的 High 做線性回歸擬合壓力線

### 訊號規則

| 趨勢 | 位置 | 訊號 |
|------|------|------|
| 多頭 | 觸下軌 | ✅ 可以買進 |
| 多頭 | 觸上軌 | ⚠️ 停利出場 |
| 空頭 | 觸上軌 | 🔻 可以放空 |
| 空頭 | 觸下軌 | 💰 回補停利 |

---

## 選股預測功能（v5 Dual Model）

基於 LightGBM 機器學習模型，自動篩選做多/做空標的。採用雙模型架構，各取所長。

### 模型架構

| 方向 | 模型檔 | 特徵數 | AUC | 說明 |
|------|--------|-------|-----|------|
| 做多 | `stock_model_v4.pkl` | 37 | 0.853 | 多頭期命中率高（回測 7~9/10）|
| 做空 | `stock_model_v5.pkl` | 41 | 0.901 | 精準率 60~65%，盤整/空頭期可靠 |

### 訓練資料

- **歷史範圍**：4 年（2022/8~2026/8）
- **股票池**：台股成交量 Top 500
- **標籤定義**：
  - 做多：前 10 天盤整 < 8% → 後 20 天最高漲幅 > 20%
  - 做空：前 10 天盤整 < 8% → 後 20 天最大跌幅 > 15%

### 特徵清單

**共用 37 個（做多模型）：**
- 技術面：MA_spread, c_vs_MA5/MA20/MA60, MA5/MA20_slope, vol_ratio, vol_trend, ret_1d/3d/5d/10d/20d, vol_10d/20d, RSI, MACD_DIF/Signal/Hist, dist_high20, BB_pos, K_raw, vol_conv
- 籌碼面：foreign_3d/5d, trust_3d/5d, inst_momentum, foreign/trust_consec_buy
- 大盤：taiex_above_MA20, taiex_trend
- 融資融券：margin_chg_5d, short_chg_5d, margin_usage, short_ratio
- 產業：industry_momentum

**做空模型額外 4 個（共 41 個）：**
- bb_width：布林帶寬度比
- max_gain_20d：過去 20 天最大漲幅
- max_loss_20d：過去 20 天最大跌幅
- vol_breakout_days：近 20 天成交量突破 1.5 倍均量的天數

### 回測命中率

| 日期 | 環境 | 做多漲>10% | 做空跌>10% |
|------|------|-----------|-----------|
| 2025-08-13 | 盤整 | 9/10 | 0/10 |
| 2025-08-18 | 盤整 | 6/10 | 0/10 |
| 2026-05-04 | 多頭 | 6/10 | 7/10 |
| 2026-06-02 | 多頭末 | 3/10 | 8/10 |
| 2026-07-15 | 空頭 | 0/10 | 8/10 |

### 輸出格式

`/predict` 傳回 Excel，兩個 sheet：

**做多 Top10：**
| 代號 | 名稱 | 收盤 | 起漲機率 | 預估最高漲幅 | 預估目標價 |
|------|------|------|---------|-----------|----------|

**做空 Top10：**
| 代號 | 名稱 | 收盤 | 做空機率 | 預估最大跌幅 | 預估目標價 |
|------|------|------|---------|-----------|----------|

### 建議使用方式

1. 每天 20:00 後執行（確保法人/融資數據已更新）
2. 做多 Top 1 > 70% 且做空 < 55% → 🟢 做多
3. 做空 Top 1 > 70% 且做多 < 55% → 🔴 做空
4. 兩邊都 50~65% → ⚠️ 盤整觀望
5. 搭配停損（建議 -5%）

---

## 安裝 & 部署

### 1. 建立 Telegram Bot

1. 在 Telegram 找 `@BotFather`
2. 發送 `/newbot`
3. 取名字和 username
4. 拿到 Bot Token

### 2. 設定環境

```bash
cp .env.example .env
# 編輯 .env，填入 Bot Token 和 FinMind Token
```

### 3. 啟動

```bash
docker-compose up -d
```

### 4. 查看 log / 停止

```bash
docker-compose logs -f
docker-compose down
```

## 設定

在 `.env` 中可調整：

- `BOT_TOKEN` — Telegram Bot Token
- `FINMIND_TOKEN` — FinMind API Token（需 Sponsor 方案）
- `SCAN_HOUR` — 每日掃描小時（預設 14）
- `SCAN_MINUTE` — 每日掃描分鐘（預設 30）

## 檔案結構

```
├── bot.py                  ← Bot 主程式
├── channel_engine.py       ← 通道計算引擎
├── chart_drawer.py         ← 繪圖模組
├── finmind_fetcher.py      ← 資料抓取模組
├── predict_engine.py       ← 選股預測引擎
├── model/
│   ├── stock_model_v4.pkl  ← 做多模型（37 特徵）
│   └── stock_model_v5.pkl  ← 做空模型（41 特徵）
├── data/
│   ├── watchlist.json      ← 觀察清單
│   └── predictions/        ← 預測歷史記錄
├── Dockerfile
├── docker-compose.yml
├── .env
└── README.md
```

## 注意事項

- 電腦需保持開機狀態，Bot 才能持續運作
- 資料來源為 FinMind API，需自行申請 Token
- 圖表已安裝中文字型（Noto Sans CJK）
- Telegram 訊息採用 HTML 格式傳送
- 此工具僅供參考，不構成投資建議

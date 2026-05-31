# Real-time Crypto Trading Advisor — ICT/SMC Edition

Hệ thống tư vấn giao dịch tiền mã hóa thời gian thực, sử dụng **Apache Spark Structured Streaming**, **HDFS**, **XGBoost** và chiến lược **ICT / Smart Money Concept (SMC)**.

## Kiến trúc hệ thống

```
                         ┌────────────────────────────────────────────────────────────┐
                         │                     HDFS (Hadoop)                          │
                         │  /user/hdoop/bigdata/                                     │
                         │  ├── raw/btc_raw.parquet           (50 MB – 1.5M dòng)    │
                         │  ├── features/btc_features_*.parquet  (6 timeframes)      │
                         │  ├── models/thresholds.json                               │
                         │  └── output/signals/               (Spark output)         │
                         └───────────┬──────────────────────────────┬─────────────────┘
                                     │ Spark đọc                    │ Spark ghi
                                     ▼                              ▲
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────────────┐
│  Binance API     │    │ Spark Structured     │    │  Flask API Server        │
│  (REST + WS)     │    │ Streaming            │    │  (api.py :5000)          │
│                  │    │ (spark_stream.py)     │    │  - Đọc features từ HDFS  │
│  download_binance│───▶│                      │    │  - ICT Scoring Engine    │
│  .py (lịch sử)  │    │ TCP:9999 ──▶ Parse   │    │  - XGBoost Predict       │
│                  │    │ ──▶ ICT Features ──▶ │    │  - REST API + Overlays   │
│  binance_socket  │───▶│    XGBoost Predict   │    └─────────┬────────────────┘
│  .py (real-time) │    │ ──▶ HDFS output      │              │
└──────────────────┘    └──────────────────────┘              ▼
                                                    ┌──────────────────────────┐
                                                    │  React Frontend          │
                                                    │  (Vite + TypeScript)     │
                                                    │  - Lightweight Charts    │
                                                    │  - Binance WebSocket     │
                                                    │  - LONG/SHORT signals    │
                                                    │  http://localhost:5173   │
                                                    └──────────────────────────┘
```

## Cấu trúc dự án

```
bigdata/

├── models/                      # Mô hình đã train
│   ├── xgb_ict_*.pkl            # 6 mô hình XGBoost ICT (1 per timeframe) ← MỚI
│   └── thresholds.json          # Ngưỡng LONG/SHORT tối ưu
├── output/                      # Kết quả trên HDFS
│   └── signals/                 # Tín hiệu real-time (lưu trên HDFS)
├── frontend/                    # React + Vite + TypeScript
│   └── src/
│       ├── App.tsx              # Main app (ICT sidebar, websocket, countdown)
│       ├── TradingChart.tsx     # Biểu đồ nến + LONG/SHORT markers
│       └── api.ts               # API client (ICT types)
├── checkpoints/                 # Spark streaming checkpoints
│
│── download_binance.py          # [Bước 1] Thu thập dữ liệu lịch sử
│── ict_features.py              # [CORE] ICT/SMC Engine — 14 features vectorized ← MỚI
│── feature_engineering.py       # [Bước 2] Tính ICT features + sync HDFS
│── train_model.py               # [Bước 3] Train XGBoost ICT (đọc từ HDFS)
│── hdfs_manager.py              # [Bước 4] Quản lý HDFS (upload/đọc/ghi)
│── binance_socket.py            # [Bước 5] WebSocket → TCP socket (OHLCV full)
│── spark_stream.py              # [Bước 6] Spark Streaming + ghi HDFS
│── realtime_features.py         # Rolling window 1000 candles — ICT real-time
│── realtime_predict.py          # Hybrid Scoring: XGBoost 60% + ICT rules 40%
│── api.py                       # [Bước 7] Flask API (HDFS + predict + overlays)
└── requirements.txt             # Dependencies Python
```

## ICT / SMC Features (14 features)

| Feature | Mô tả |
|---------|--------|
| `htf_trend` | Higher Timeframe Trend: 1=Bull, 0=Range, -1=Bear |
| `structure_strength` | Sức mạnh cấu trúc (HH/HL vs LH/LL) |
| `displacement_strength` | Cường độ dịch chuyển giá (body/range × ATR) |
| `mss_strength` | Market Structure Shift strength [-1, 1] |
| `cisd_state` | Change in State of Delivery: 1=Bull, -1=Bear |
| `inside_bisi_mid_zone` | BISI midpoint reaction score (bullish FVG zone) |
| `inside_sibi_mid_zone` | SIBI midpoint reaction score (bearish FVG zone) |
| `fvg_fill_ratio` | Tỷ lệ fill Fair Value Gap [0, 1] |
| `imbalance_reaction_score` | BISI vs SIBI net score |
| `liquidity_sweep_detected` | 0=None, 1=Sweep Low (bullish), 2=Sweep High (bearish) |
| `delivery_strength` | Sức mạnh delivery (efficiency × volume expansion) |
| `momentum_score` | Tổng hợp momentum [-1, 1] |
| `macro_window` | 0=Outside, 1=London Macro, 2=NY Macro ⚡ (xx:50→xx:10) |

### ICT Macro Windows (ưu tiên cao nhất)

| Giờ UTC | Macro Window |
|---------|-------------|
| 13:50–14:10 | **NY Open** ⭐ |
| 17:50–18:10 | **NY PM Open** ⭐ |
| 06:50–07:10 | London Open |
| 18:50–19:10 | NY Power Hour |
| 12:50–13:10 | NY Pre-Market |

## Yêu cầu hệ thống

| Phần mềm | Phiên bản | Mục đích |
|-----------|-----------|----------|
| Java (OpenJDK) | 11 | Spark + Hadoop runtime |
| Hadoop | 3.4.3 | HDFS lưu trữ phân tán |
| Spark | 3.5.x | Structured Streaming |
| Python | 3.12 | Backend |
| Node.js | 18+ | Frontend (React + Vite) |

## Cài đặt

### 1. Cài đặt Python dependencies

```bash
cd ~/bigdata
python -m venv backend/env
source backend/env/bin/activate
pip install -r requirements.txt
```

### 2. Cài đặt Frontend

```bash
cd frontend
npm install
```

---

## Quy trình chạy

> **Lưu ý:** Mọi lệnh Python đều chạy từ thư mục `~/bigdata` với virtualenv đã kích hoạt:
> ```bash
> cd ~/bigdata
> source backend/env/bin/activate
> ```

### Phase 1: Chuẩn bị dữ liệu (chạy 1 lần)

#### Bước 1 – Thu thập dữ liệu lịch sử

```bash
python download_binance.py
```

- Tải 3 năm nến 1 phút BTC/USDT từ Binance REST API
- Output: Ghi trực tiếp vào Data Lake `hdfs://.../raw/btc_raw.parquet`
- Thời gian: ~15–20 phút (rate limit Binance)

#### Bước 2 – ICT Feature Engineering

```bash
python feature_engineering.py
```

- Tính **14 ICT/SMC features** (vectorized, xử lý 1.5M rows trong ~17 giây)
- Tạo features cho 6 timeframes: `1min`, `5min`, `15min`, `1H`, `4H`, `1D`
- Output: Ghi trực tiếp lên HDFS `hdfs://.../features/btc_features_*.parquet`

> ⚠️ **Lưu ý:** Phiên bản cũ dùng RSI/EMA/MACD. Phiên bản ICT mới **không tương thích** với model cũ.

#### Bước 3 – Train mô hình XGBoost ICT

```bash
python train_model.py
```

- Đọc ICT features trực tiếp từ HDFS (Bắt buộc)
- Train 6 mô hình XGBoost (1 per timeframe)
- Tìm ngưỡng LONG/SHORT tối ưu dựa trên precision
- Output: `models/xgb_ict_*.pkl` (6 files) + `models/thresholds.json`

---

### Phase 2: Lưu trữ & Xác nhận HDFS

#### Bước 4 – Khởi động HDFS & Kiểm tra

```bash
# Khởi động HDFS
start-dfs.sh

# Kiểm tra dữ liệu trên HDFS (đã tự động được tải lên từ Bước 1, 2, 3)
python hdfs_manager.py list
```

Cấu trúc HDFS sau khi Pipeline tự động ghi dữ liệu:

```
hdfs://127.0.0.1:9000/user/hdoop/bigdata/
├── raw/btc_raw.parquet               (50.6 MB)
├── features/
│   ├── btc_features_1min.parquet     (175.9 MB – 1,578,235 dòng × 24 cols)
│   ├── btc_features_5min.parquet     (38.1 MB – 315,644 dòng × 24 cols)
│   ├── btc_features_15min.parquet    (14.6 MB – 105,212 dòng × 24 cols)
│   ├── btc_features_1H.parquet       (3.6 MB – 26,300 dòng × 24 cols)
│   ├── btc_features_4H.parquet       (913.7 KB – 6,572 dòng × 24 cols)
│   └── btc_features_1D.parquet       (143.3 KB – 1,092 dòng × 24 cols)
├── models/thresholds.json
└── output/                           (Spark ghi kết quả)
```

#### Bước 4b – Demo đọc/ghi từ HDFS bằng Spark

```bash
python hdfs_manager.py demo
```

Demo 6 phần:
1. Đọc dữ liệu thô từ HDFS bằng Spark
2. Phân tích thống kê trên Spark (phân tán)
3. Đọc features 6 timeframes từ HDFS
4. Phân tích dữ liệu trên Spark (avg, min, max, stddev)
5. Ghi kết quả phân tích lên HDFS
6. Liệt kê cấu trúc dữ liệu trên HDFS

---

### Phase 3: Real-time Streaming (4 terminals)

> **Quan trọng:** Phải chạy đúng thứ tự. Mỗi lệnh chạy ở 1 terminal riêng.

#### Terminal 1 – WebSocket Producer

```bash
cd ~/bigdata && source backend/env/bin/activate
python binance_socket.py
```

- Kết nối Binance WebSocket (`btcusdt@kline_1s`)
- Gửi **OHLCV đầy đủ** (open, high, low, close, volume, timestamp) qua TCP socket (port 9999)
- Cần OHLCV để tính ICT features (MSS, FVG, BISI/SIBI)

#### Terminal 2 – Spark Structured Streaming

```bash
cd ~/bigdata && source backend/env/bin/activate
python spark_stream.py
```

- Đọc stream OHLCV từ TCP:9999
- Rolling window **1000 candles** (~16h40m) → tính ICT features → predict XGBoost
- **Data Lake Branching**: Append nối đuôi trực tiếp dữ liệu nến thô (OHLCV) vào `hdfs://.../raw/btc_raw.parquet`
- Ghi tín hiệu dự đoán ra HDFS `hdfs://.../output/signals/`
- **Lưu ý:** Cần chờ ~1000 giây (warm-up) cho rolling window đầy đủ

#### Terminal 3 – Flask API Server

```bash
cd ~/bigdata && source backend/env/bin/activate
python api.py
```

- Chạy trên `http://localhost:5000`
- Đọc ICT features trực tiếp từ HDFS (không dùng local disk)
- Bổ sung nến thiếu từ Binance REST API (data patching)
- **ICT Scoring Engine**: 60% XGBoost + 40% rule-based ICT confluence
- Nhân hệ số tối đa ×1.30 trong NY Macro windows
- Endpoints:
  - `GET /api/data?timeframe=5min` – Dữ liệu + tín hiệu LONG/SHORT/NEUTRAL + overlays
  - `GET /api/hdfs-status` – Trạng thái HDFS

#### Terminal 4 – React Frontend

```bash
cd ~/bigdata/frontend
npm run dev
```

- Chạy trên `http://localhost:5173`
- Bố cục Resizable (react-resizable-panels): Biểu đồ nến TradingView-style (trên) & Bảng Real-time ICT Features (dưới).
- Múi giờ đồng bộ hoàn toàn theo **UTC-4 (America/New_York)** chuẩn ICT.
- WebSocket Binance trực tiếp → cập nhật nến mỗi giây.
- Sidebar: ICT bias (LONG/SHORT/NEUTRAL), Prob Up/Down, Killzone, HTF Bias.
- 6 timeframes: 1min, 5min, 15min, 1H, 4H, 1D

---

## Luồng dữ liệu chi tiết

```
1. download_binance.py
   Binance REST API ──▶ PySpark ──▶ hdfs://.../raw/btc_raw.parquet

2. feature_engineering.py  [~17 giây cho 1.5M rows]
   HDFS raw data ──▶ ict_features.py (14 ICT features vectorized)
                 ──▶ PySpark ──▶ hdfs://.../features/btc_features_*.parquet

3. train_model.py
   HDFS ICT features ──▶ train XGBoost × 6 ──▶ models/xgb_ict_*.pkl + thresholds.json

4. hdfs_manager.py
   (Chỉ dùng cho các utilities kiểm tra/đọc dữ liệu từ HDFS)

5. binance_socket.py  →  spark_stream.py
   Binance WS (OHLCV) ──▶ TCP:9999 ──▶ Spark
   ──▶ Nhánh 1: Append nến thô vào HDFS raw data
   ──▶ Nhánh 2: Tính ICT features ──▶ predict ──▶ HDFS signals

6. api.py
   HDFS ICT features + Binance REST (patch)
   ──▶ ICT Scoring (60% XGBoost + 40% rules + NY Macro ×1.30)
   ──▶ LONG / SHORT / NEUTRAL + JSON API

7. frontend (React)
   API :5000 + Binance WS ──▶ Lightweight Charts
   ──▶ Candlestick + LONG/SHORT markers + ICT sidebar
```

## Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|------------|-----------|---------| 
| Lưu trữ phân tán | **HDFS (Hadoop 3.4.3)** | Lưu dữ liệu thô, features, kết quả |
| Xử lý streaming | **Spark Structured Streaming 3.5** | Micro-batch từ TCP socket |
| Machine Learning | **XGBoost 2.0** | Phân loại LONG/SHORT/NEUTRAL |
| Trading Strategy | **ICT / Smart Money Concept** | 14 features: MSS, FVG, BISI/SIBI, Liquidity, Macro |
| Nguồn dữ liệu | **Binance API + WebSocket** | REST (lịch sử) + WS (real-time OHLCV) |
| Backend API | **Flask + Flask-CORS** | REST API + ICT Scoring Engine |
| Frontend | **React 19 + TypeScript + Vite** | Dashboard real-time |
| Biểu đồ | **Lightweight Charts 4.1** | TradingView-style candlestick + markers |
| Feature Engineering | **NumPy/Pandas vectorized** | ICT features O(n) — không dùng ta library |
| Định dạng dữ liệu | **Apache Parquet** | Columnar, nén tốt, Spark-native |

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|------------|-----------|
| `Connection refused` (HDFS) | HDFS chưa chạy | `start-dfs.sh` |
| `Connection refused` (port 9999) | `binance_socket.py` chưa chạy | Chạy Terminal 1 trước Terminal 2 |
| `Address already in use` (port 5000) | Flask đang chạy ở process khác | `lsof -i :5000` rồi `kill <PID>` |
| Không có tín hiệu (Spark) | Warm-up period | Chờ ~1000 giây cho rolling window đầy |
| `Missing ICT features` | feature_engineering.py chưa chạy | Chạy Bước 2 + Bước 3 lại |
| `feature_names mismatch` | Model cũ (RSI/EMA) không tương thích | Xóa `models/xgb_model_*.pkl`, chạy lại `train_model.py` |
| `Checkpoint error` (Spark) | Checkpoint cũ không tương thích | Xóa thư mục `checkpoints/` |
| `feature_engineering` quá chậm | Python for-loop cũ | Đã fix: dùng vectorized rolling O(n) — ~17s cho 1.5M rows |

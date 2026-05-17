# Real-time Crypto Trading Advisor

Hệ thống tư vấn giao dịch tiền mã hóa thời gian thực, sử dụng **Apache Spark Structured Streaming**, **HDFS**, **XGBoost** và **Binance WebSocket**.

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
│  download_binance│───▶│                      │    │  - Predict XGBoost       │
│  .py (lịch sử)  │    │ TCP:9999 ──▶ Parse   │    │  - REST API cho frontend │
│                  │    │ ──▶ Features ──▶     │    └─────────┬────────────────┘
│  binance_socket  │───▶│    XGBoost Predict   │              │
│  .py (real-time) │    │ ──▶ HDFS output      │              ▼
└──────────────────┘    └──────────────────────┘    ┌──────────────────────────┐
                                                    │  React Frontend          │
                                                    │  (Vite + TypeScript)     │
                                                    │  - Lightweight Charts    │
                                                    │  - Binance WebSocket     │
                                                    │  - Real-time updates     │
                                                    │  http://localhost:5173   │
                                                    └──────────────────────────┘
```

## Cấu trúc dự án

```
bigdata/
├── data/                        # Dữ liệu Parquet (local + sync HDFS)
│   ├── btc_raw.parquet          # Dữ liệu thô 3 năm BTC/USDT (1.5M nến 1 phút)
│   └── btc_features_*.parquet   # Features đã tính cho 6 timeframes
├── models/                      # Mô hình đã train
│   ├── xgb_model_*.pkl          # 6 mô hình XGBoost (1 per timeframe)
│   └── thresholds.json          # Ngưỡng BUY/SELL tối ưu
├── output/                      # Kết quả
│   ├── signals.jsonl            # Tín hiệu real-time
│   └── dashboard.html           # Dashboard Plotly tĩnh
├── frontend/                    # React + Vite + TypeScript
│   └── src/
│       ├── App.tsx              # Main app (sidebar, websocket, countdown)
│       ├── TradingChart.tsx     # Biểu đồ nến TradingView-style
│       └── api.ts               # API client
├── checkpoints/                 # Spark streaming checkpoints
│
│── download_binance.py          # [Bước 1] Thu thập dữ liệu lịch sử
│── feature_engineering.py       # [Bước 2] Tính features + sync HDFS
│── train_model.py               # [Bước 3] Train XGBoost (đọc từ HDFS)
│── hdfs_manager.py              # [Bước 4] Quản lý HDFS (upload/đọc/ghi)
│── binance_socket.py            # [Bước 5] WebSocket → TCP socket
│── spark_stream.py              # [Bước 6] Spark Streaming + ghi HDFS
│── realtime_features.py         # Rolling window tính features real-time
│── realtime_predict.py          # Inference XGBoost real-time
│── api.py                       # [Bước 7] Flask API (đọc HDFS + predict)
│── visualize.py                 # Tạo dashboard Plotly tĩnh
│── dash_app.py                  # Dashboard Dash interactive
│── scratch_eval.py              # Đánh giá threshold mô hình
└── requirements.txt             # Dependencies Python
```

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
- Output: `data/btc_raw.parquet` (~52 MB, ~1.5 triệu dòng)
- Thời gian: ~15-20 phút (rate limit Binance)

#### Bước 2 – Feature Engineering

```bash
python feature_engineering.py
```

- Tính 11 chỉ báo kỹ thuật (RSI, EMA, MACD, Bollinger Bands, ...)
- Tạo features cho 6 timeframes: `1min`, `5min`, `15min`, `1H`, `4H`, `1D`
- Tự động sync lên HDFS (nếu HDFS đang chạy)
- Output: `data/btc_features_*.parquet` (6 files, tổng ~246 MB)

#### Bước 3 – Train mô hình XGBoost

```bash
python train_model.py
```

- Đọc features từ HDFS (fallback local nếu HDFS chưa chạy)
- Train 6 mô hình XGBoost (1 per timeframe)
- Tìm ngưỡng BUY/SELL tối ưu dựa trên precision
- Đồng bộ `thresholds.json` lên HDFS
- Output: `models/xgb_model_*.pkl` (6 files) + `models/thresholds.json`

---

### Phase 2: Lưu trữ & Xác nhận HDFS

#### Bước 4 – Khởi động HDFS & Upload dữ liệu

```bash
# Khởi động HDFS
start-dfs.sh

# Upload toàn bộ dữ liệu lên HDFS
python hdfs_manager.py upload

# Kiểm tra dữ liệu trên HDFS
python hdfs_manager.py list
```

Cấu trúc HDFS sau khi upload:

```
hdfs://127.0.0.1:9000/user/hdoop/bigdata/
├── raw/btc_raw.parquet               (50.6 MB)
├── features/
│   ├── btc_features_1min.parquet     (175.9 MB – 1,578,202 dòng)
│   ├── btc_features_5min.parquet     (38.1 MB – 315,611 dòng)
│   ├── btc_features_15min.parquet    (14.6 MB – 105,179 dòng)
│   ├── btc_features_1H.parquet      (3.6 MB – 26,251 dòng)
│   ├── btc_features_4H.parquet      (913.7 KB – 6,523 dòng)
│   └── btc_features_1D.parquet      (143.3 KB – 993 dòng)
├── models/thresholds.json            (371 bytes)
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
- Chuyển tiếp dữ liệu qua TCP socket (port 9999)
- Chờ Spark kết nối trước khi nhận dữ liệu

#### Terminal 2 – Spark Structured Streaming

```bash
cd ~/bigdata && source backend/env/bin/activate
python spark_stream.py
```

- Đọc stream từ TCP:9999
- Parse JSON → tính features (rolling window 250 điểm) → predict XGBoost
- Ghi tín hiệu ra `output/signals.jsonl` (local) + HDFS (mỗi 100 signals)
- **Lưu ý:** Cần chờ ~250 giây (warm-up) trước khi có tín hiệu đầu tiên

#### Terminal 3 – Flask API Server

```bash
cd ~/bigdata && source backend/env/bin/activate
python api.py
```

- Chạy trên `http://localhost:5000`
- Đọc features từ HDFS (fallback local)
- Bổ sung nến thiếu từ Binance REST API (data patching)
- Predict XGBoost cho toàn bộ dữ liệu
- Endpoints:
  - `GET /api/data?timeframe=5min` – Dữ liệu + tín hiệu
  - `GET /api/hdfs-status` – Trạng thái HDFS

#### Terminal 4 – React Frontend

```bash
cd ~/bigdata/frontend
npm run dev
```

- Chạy trên `http://localhost:5173`
- Biểu đồ nến TradingView-style (Lightweight Charts)
- WebSocket Binance trực tiếp → cập nhật nến mỗi giây
- Sidebar: AI signal, probability, EMA, countdown timer
- 6 timeframes: 1min, 5min, 15min, 1H, 4H, 1D

---

### Phase 4: Trực quan tĩnh (tùy chọn)

```bash
# Dashboard Plotly (output/dashboard.html)
python visualize.py

# Dashboard Dash interactive (localhost:5000)
python dash_app.py
```

---

## Luồng dữ liệu chi tiết

```
1. download_binance.py
   Binance REST API ──▶ data/btc_raw.parquet (local)

2. feature_engineering.py
   btc_raw.parquet ──▶ tính RSI, EMA, MACD, BB, ... ──▶ btc_features_*.parquet (local + HDFS)

3. train_model.py
   HDFS features ──▶ train XGBoost × 6 ──▶ models/*.pkl + thresholds.json (local + HDFS)

4. hdfs_manager.py
   local data ──▶ HDFS upload ──▶ Spark đọc lại xác nhận

5. binance_socket.py  →  spark_stream.py
   Binance WS ──▶ TCP:9999 ──▶ Spark parse JSON ──▶ features ──▶ predict ──▶ signals (local + HDFS)

6. api.py
   HDFS features + Binance REST (patch) ──▶ XGBoost predict ──▶ JSON API

7. frontend (React)
   API :5000 + Binance WS ──▶ Lightweight Charts ──▶ UI real-time
```

## Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|------------|-----------|---------|
| Lưu trữ phân tán | **HDFS (Hadoop 3.4.3)** | Lưu dữ liệu thô, features, kết quả |
| Xử lý streaming | **Spark Structured Streaming 3.5** | Micro-batch từ TCP socket |
| Machine Learning | **XGBoost 2.0** | Phân loại BUY/SELL/HOLD |
| Nguồn dữ liệu | **Binance API + WebSocket** | REST (lịch sử) + WS (real-time) |
| Backend API | **Flask + Flask-CORS** | REST API cho frontend |
| Frontend | **React 19 + TypeScript + Vite** | Dashboard real-time |
| Biểu đồ | **Lightweight Charts 4.1** | TradingView-style candlestick |
| Feature Engineering | **ta (Technical Analysis)** | RSI, MACD, EMA, Bollinger Bands |
| Định dạng dữ liệu | **Apache Parquet** | Columnar, nén tốt, Spark-native |

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|------------|-----------|
| `Connection refused` (HDFS) | HDFS chưa chạy | `start-dfs.sh` |
| `Connection refused` (port 9999) | `binance_socket.py` chưa chạy | Chạy Terminal 1 trước Terminal 2 |
| `Address already in use` (port 5000) | Flask đang chạy ở process khác | `lsof -i :5000` rồi `kill <PID>` |
| Không có tín hiệu (Spark) | Warm-up period | Chờ ~250 giây cho rolling window đầy |
| `Checkpoint error` (Spark) | Checkpoint cũ không tương thích | Xóa thư mục `checkpoints/` |
| `Model Not Found` | Chưa train mô hình | Chạy `python train_model.py` |
| Parquet timestamp error (Spark) | Pandas dùng nanosecond | Đã xử lý trong `hdfs_manager.py` |

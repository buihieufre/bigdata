import websocket
import json
import socket
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1s"
TCP_IP = '127.0.0.1'
TCP_PORT = 9999

def start_tcp_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_IP, TCP_PORT))
    server.listen(1)
    logger.info(f"Listening for Spark on {TCP_IP}:{TCP_PORT}")
    conn, addr = server.accept()
    logger.info(f"Spark connected from {addr}")
    return server, conn

def on_message(ws, message, conn):
    try:
        data = json.loads(message)
        kline = data['k']
        
        # Parse payload — full OHLCV + timestamp for ICT features
        payload = {
            "symbol":    kline['s'],
            "open":      float(kline['o']),
            "high":      float(kline['h']),
            "low":       float(kline['l']),
            "close":     float(kline['c']),
            "volume":    float(kline['v']),
            "timestamp": int(kline['t']),   # kline open time (ms UTC)
            "is_closed": kline['x'],
        }
        
        # Emit over TCP
        payload_str = json.dumps(payload) + '\n'
        conn.sendall(payload_str.encode('utf-8'))
        # logger.info(f"Sent: {payload}")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info("WebSocket Closed")

def on_open(ws):
    logger.info("WebSocket Opened")

def main():
    while True:
        server, conn = None, None
        try:
            server, conn = start_tcp_server()
            
            # Using lambda to pass conn to on_message
            ws = websocket.WebSocketApp(
                BINANCE_WS_URL,
                on_open=on_open,
                on_message=lambda ws, msg: on_message(ws, msg, conn),
                on_error=on_error,
                on_close=on_close
            )
            
            ws.run_forever()
        except Exception as e:
            logger.error(f"Main loop error: {e}")
        finally:
            if conn:
                conn.close()
            if server:
                server.close()
            logger.info("Reconnecting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()

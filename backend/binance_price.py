import websocket
import json
import socket

HOST = "localhost"
PORT = 9999

def on_message(ws, message):
    data = json.loads(message)

    # lấy kline data
    k = data.get("k", {})
    out = {
        "symbol": data.get("s"),
        "close": float(k.get("c", 0)),
        "volume": float(k.get("v", 0)),
        "is_closed": k.get("x", False)
    }

    conn.send((json.dumps(out) + "\n").encode())

def on_open(ws):
    print("Connected to Binance")

if __name__ == "__main__":
    # socket server
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print("Waiting Spark connection...")
    conn, addr = server.accept()
    print("Spark connected:", addr)

    ws = websocket.WebSocketApp(
        "wss://stream.binance.com:9443/ws/btcusdt@kline_1s",
        on_message=on_message,
        on_open=on_open
    )

    try:
        ws.run_forever()
    finally:
        conn.close()
        server.close()
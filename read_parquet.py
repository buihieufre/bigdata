import pandas as pd
import sys

def read_and_display(file_path):
    try:
        # Dùng fastparquet hoặc pyarrow có sẵn trong môi trường của backend
        df = pd.read_parquet(file_path)

        print(f"\n{'='*50}")
        print(f"Đã tải xong file: {file_path}")
        print(f"{'='*50}")
        print(f"Tổng số dòng: {len(df):,}")
        print(f"Tổng số cột: {len(df.columns)}")
        print("\n[ Danh sách các cột ]")
        print(df.columns.tolist())

        print("\n[ 5 dòng dữ liệu đầu tiên ]")
        print(df.head())
        print(f"{'='*50}\n")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{file_path}'")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    # Lấy đường dẫn file từ tham số dòng lệnh nếu có, mặc định là btc_raw.parquet
    target_file = sys.argv[1] if len(sys.argv) > 1 else 'data/btc_raw.parquet'
    read_and_display(target_file)

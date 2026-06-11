import os
import torch
from pathlib import Path
from ultralytics import YOLO

def train_yolov5(data_yaml, epochs=20, batch_size=16, img_size=640):
    """
    Huấn luyện mô hình YOLOv5 để phát hiện biển số xe.
    
    :param data_yaml: Đường dẫn đến file cấu hình `data.yaml`.
    :param epochs: Số lượng epoch huấn luyện.
    :param batch_size: Kích thước batch.
    :param img_size: Kích thước ảnh đầu vào.
    """
    # Tải mô hình YOLOv5 cơ bản (yolov5s)
    model = YOLO(r'D:\Downloads\BC_MCLN\A_MAIN\yolo11n.pt')  # Bạn có thể thay đổi mô hình từ 'yolov5s' sang các mô hình khác như 'yolov5m', 'yolov5l' tùy vào nhu cầu

    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình...")
    model.train(
        data=data_yaml,  # Tệp cấu hình dữ liệu
        epochs=epochs,   # Số lượng epoch
        batch=batch_size,  # Sửa 'batch_size' thành 'batch'
        imgsz=img_size,  # Kích thước ảnh đầu vào
        workers=4,  # Số lượng luồng xử lý dữ liệu
        cache=True  # Bật cache để tăng tốc độ đọc ảnh
    )
    print("Huấn luyện hoàn thành!")

def main():
    # Đường dẫn đến file cấu hình dữ liệu
    data_yaml = r'D:\Downloads\BC_MCLN\A_MAIN\data_1.yaml'
    
    # Kiểm tra xem có CUDA không (GPU) để huấn luyện nhanh hơn
    if torch.cuda.is_available():
        print(f"CUDA có sẵn. Huấn luyện trên GPU.")
    else:
        print(f"CUDA không có sẵn. Huấn luyện trên CPU.")

    # Huấn luyện mô hình
    train_yolov5(data_yaml, epochs=20, batch_size=16, img_size=640)

if __name__ == '__main__':
    main()

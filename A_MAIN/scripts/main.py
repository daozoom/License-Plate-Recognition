from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load a pretrained YOLO model (recommended for training)
model = YOLO(r"D:/Downloads/BC_MCLN/runs/detect/train7/weights/best.pt")

results = model(r'D:/Downloads/BC_MCLN/A_MAIN/data/anh-13.jpg')

for r in results:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('output.jpg')

# # Đọc ảnh và phát hiện biển số
# image_path = r'D:/Downloads/BC_MCLN/A_MAIN/data/anh-7.jpg'  # Đường dẫn tới ảnh xe
# image = cv2.imread(image_path)

# # Chạy YOLO để phát hiện biển số
# results = model(image_path)

# # Kiểm tra nếu results là danh sách và lấy đối tượng kết quả đầu tiên
# if isinstance(results, list):
#     results = results[0]  # Lấy phần tử đầu tiên từ danh sách

# # Hiển thị ảnh với các bounding boxes
# results.show()  # Dùng phương thức show() để hiển thị ảnh đã được chú thích

# # Trích xuất các bounding box
# for bbox in results.boxes.data:  # Vùng bounding box đầu tiên trong ảnh
#     x1, y1, x2, y2, conf, cls = bbox
#     cropped_plate = image[int(y1):int(y2), int(x1):int(x2)]  # Cắt vùng biển số
#     cv2.imwrite('cropped_plate.jpg', cropped_plate)  # Lưu ảnh biển số đã cắt
#     print(f'Biển số được phát hiện tại tọa độ ({x1}, {y1}, {x2}, {y2}) với độ chính xác {conf}')
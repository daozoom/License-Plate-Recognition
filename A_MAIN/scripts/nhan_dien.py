import easyocr
import cv2
import matplotlib.pyplot as plt

# Khởi tạo đối tượng EasyOCR
reader = easyocr.Reader(['en'])  # Chọn ngôn ngữ là tiếng Anh

# Đọc ảnh biển số đã cắt (từ bước trên)
cropped_plate_path = 'output.jpg'  # Đường dẫn đến ảnh biển số đã cắt
cropped_plate = cv2.imread(cropped_plate_path)

# Nhận diện ký tự trong ảnh biển số
result = reader.readtext(cropped_plate)

# In kết quả nhận diện
for detection in result:
    print(f"Ký tự nhận diện: {detection[1]}")  # detection[1] chứa văn bản nhận diện

# Nếu bạn muốn hiển thị kết quả lên ảnh:
for (bbox, text, prob) in result:
    # bbox là một danh sách 4 điểm
    # Cách giải quyết: tìm điểm min và max của bbox để vẽ rectangle
    points = bbox  # bbox là danh sách các điểm
    x_min = min([point[0] for point in points])  # Lấy giá trị x nhỏ nhất
    y_min = min([point[1] for point in points])  # Lấy giá trị y nhỏ nhất
    x_max = max([point[0] for point in points])  # Lấy giá trị x lớn nhất
    y_max = max([point[1] for point in points])  # Lấy giá trị y lớn nhất

    # Vẽ bounding box và văn bản lên ảnh
    cv2.rectangle(cropped_plate, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(cropped_plate, text, (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Hiển thị ảnh với các ký tự nhận diện bằng matplotlib
# Chuyển đổi màu ảnh từ BGR sang RGB để matplotlib hiển thị đúng màu
cropped_plate_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

# Hiển thị ảnh
plt.imshow(cropped_plate_rgb)
plt.axis('off')  # Tắt trục tọa độ
plt.show()

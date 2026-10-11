"""Thử Roboflow trên 1 ảnh (test.jpg) và lưu ảnh có vẽ box vào output/test_result.jpg.

Chạy:  python test_api.py
Mở output/test_result.jpg để xem box có đúng con cá hay không.
"""
import sys

import cv2

from config import ROOT, load_config
from roboflow_detector import DetectionError, make_detector

try:
    config = load_config()
except ValueError as exc:
    sys.exit(f"LỖI cấu hình: {exc}")

image_path = ROOT / "test.jpg"
frame = cv2.imread(str(image_path))
if frame is None:
    sys.exit(f"Không đọc được {image_path}. Hãy đặt ảnh test.jpg vào thư mục dự án.")

h, w = frame.shape[:2]
if w > config.max_width:
    frame = cv2.resize(frame, (config.max_width, round(h * config.max_width / w)))

detector = make_detector(config)
try:
    detections, latency = detector.predict(frame)
except DetectionError as exc:
    sys.exit(f"LỖI: {exc}")
finally:
    detector.close()

print(f"Chế độ: {config.detector_mode} | Thời gian gọi API: {latency:.0f} ms")
print(f"Số box tìm thấy (confidence >= {config.confidence}): {len(detections)}")
for d in detections:
    x1, y1, x2, y2 = (int(v) for v in d.xyxy)
    print(f"  box=({x1},{y1},{x2},{y2}) confidence={d.confidence:.2f}")
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
    cv2.putText(frame, f"{d.confidence:.2f}", (x1, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

out = ROOT / "output" / "test_result.jpg"
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), frame)
print(f"Đã lưu ảnh kết quả: {out}")

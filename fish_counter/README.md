# Fish Counter

Ứng dụng Windows/Python phân tích video bằng Roboflow Object Detection,
Supervision ByteTrack và OpenCV.

## Chuẩn bị
1. Cài Python 3.11 64-bit.
2. `py -3.11 -m venv .venv`
3. PowerShell: `.\.venv\Scripts\Activate.ps1`
4. `python -m pip install -r requirements.txt`
5. Sao chép `.env.example` thành `.env`; điền API key và model ID thật.
6. Đặt video trong `videos/`, hoặc để trống VIDEO_PATH để chọn bằng hộp thoại.

## Chạy
`python main.py`
Hoặc: `python main.py --video "C:\duong-dan\fish.mp4"`

## Điều khiển
Q: thoát; P/Space: tạm dừng; R: đặt lại tổng lượt;
N: thử lại frame lỗi; E: phát lại sau khi hết video; S: lưu ảnh.

## Kết quả
`output/fish_count_log.csv`: thống kê từng LOG_EVERY_FRAMES frame
và mọi frame có sự kiện vượt đường.
`output/crossing_events.csv`: một dòng cho mỗi lượt vượt hợp lệ.

Số cá hiện diện không phải tổng lượt vượt; lượt vượt không phải số cá
duy nhất trong bể. Mỗi frame gửi lên Cloud API có thể phát sinh mức sử dụng.

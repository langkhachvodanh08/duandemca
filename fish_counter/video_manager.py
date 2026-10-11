from pathlib import Path

import cv2
import numpy as np


class VideoManager:
    def __init__(self, path: Path, max_width: int):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy video: {path}")
        self.path = path
        self.max_width = max_width
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(
                f"OpenCV không mở được video: {path}. Kiểm tra codec/tệp."
            )

        self.source_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_index = 0

        if self.width < 1 or self.height < 1:
            self.release()
            raise RuntimeError("Không đọc được kích thước video.")

    def read(self) -> tuple[np.ndarray, int, float] | None:
        ok, frame = self.cap.read()
        if not ok:
            return None

        self.frame_index += 1
        h, w = frame.shape[:2]
        if w > self.max_width:
            new_h = max(1, round(h * self.max_width / w))
            frame = cv2.resize(
                frame, (self.max_width, new_h), interpolation=cv2.INTER_AREA
            )

        position_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
        if position_ms > 0:
            seconds = position_ms / 1000.0
        elif self.source_fps > 0:
            seconds = (self.frame_index - 1) / self.source_fps
        else:
            seconds = 0.0

        return frame, self.frame_index, seconds

    def rewind(self) -> None:
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            self.cap.release()
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise RuntimeError("Không thể mở lại video để phát lại.")
        self.frame_index = 0

    def release(self) -> None:
        self.cap.release()

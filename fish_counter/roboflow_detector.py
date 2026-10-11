import base64
from dataclasses import dataclass
import json
import math
import re
import time

import cv2
import numpy as np
import requests

from config import Config


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float


class DetectionError(RuntimeError):
    pass


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise DetectionError("Không thể mã hóa frame thành JPEG")
    return base64.b64encode(encoded.tobytes())


def _parse_response(
    response: requests.Response,
    frame: np.ndarray,
    config: Config,
    filter_classes: bool,
) -> list[Detection]:
    """Chuẩn hóa JSON Roboflow (x, y là tâm) thành box xyxy trên frame."""
    if not response.ok:
        raise DetectionError(
            f"Roboflow trả HTTP {response.status_code}. "
            "Kiểm tra API key, model ID, quyền truy cập và hạn mức."
        )
    try:
        result = response.json()
    except ValueError:
        raise DetectionError("Roboflow không trả JSON hợp lệ") from None

    if isinstance(result, list) and result:
        result = result[0]
    if not isinstance(result, dict) or not isinstance(
        result.get("predictions"), list
    ):
        raise DetectionError(
            "Response không có predictions dạng danh sách; "
            "hãy kiểm tra DETECTOR_MODE và mô hình."
        )

    h, w = frame.shape[:2]
    image_info = result.get("image") or {}
    if not isinstance(image_info, dict):
        image_info = {}
    try:
        rw = float(image_info.get("width") or w)
        rh = float(image_info.get("height") or h)
    except (TypeError, ValueError):
        raise DetectionError("Kích thước ảnh trong response không hợp lệ")
    if not math.isfinite(rw) or not math.isfinite(rh) or rw <= 0 or rh <= 0:
        raise DetectionError("Kích thước ảnh trong response không hợp lệ")

    sx, sy = w / rw, h / rh
    detections: list[Detection] = []
    classes_seen: set[str] = set()

    for pred in result["predictions"]:
        if not isinstance(pred, dict):
            continue
        name = str(pred.get("class", pred.get("class_name", ""))).strip()
        classes_seen.add(name)
        if filter_classes and name.casefold() not in config.fish_classes:
            continue
        try:
            conf = float(pred["confidence"])
            x = float(pred["x"])
            y = float(pred["y"])
            bw = float(pred["width"])
            bh = float(pred["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(map(math.isfinite, (conf, x, y, bw, bh))):
            continue
        if conf < config.confidence or conf > 1 or bw <= 0 or bh <= 0:
            continue

        x1 = max(0.0, min(float(w - 1), (x - bw / 2) * sx))
        y1 = max(0.0, min(float(h - 1), (y - bh / 2) * sy))
        x2 = max(0.0, min(float(w - 1), (x + bw / 2) * sx))
        y2 = max(0.0, min(float(h - 1), (y + bh / 2) * sy))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(Detection((x1, y1, x2, y2), conf))

    if filter_classes and classes_seen and not any(
        n.casefold() in config.fish_classes for n in classes_seen
    ):
        safe = [
            re.sub(r"[^a-zA-Z0-9 _-]", "?", n)[:40]
            for n in sorted(classes_seen)
        ]
        raise DetectionError(
            "Mô hình trả lớp không khớp FISH_CLASS_NAMES: "
            + ", ".join(safe[:10])
        )
    return detections


class _BaseDetector:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Accept": "application/json",
        })

    def _post(self, **kwargs) -> tuple[requests.Response, float]:
        started = time.perf_counter()
        try:
            response = self.session.post(
                self.url,
                timeout=(5, self.config.request_timeout_seconds),
                **kwargs,
            )
        except requests.RequestException as exc:
            # Không in URL/exception thô: có thể chứa dữ liệu nhạy cảm.
            raise DetectionError(
                f"Lỗi kết nối Roboflow ({type(exc).__name__}). "
                "Kiểm tra mạng và máy chủ suy luận."
            ) from None
        return response, (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        self.session.close()


class RoboflowDetector(_BaseDetector):
    """Mô hình object detection bạn đã huấn luyện (DETECTOR_MODE=custom_model)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.url = f"{config.api_url}/{config.model_id}"
        self.session.headers["Content-Type"] = "application/x-www-form-urlencoded"

    def predict(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        response, latency_ms = self._post(data=_encode_jpeg(frame))
        return _parse_response(response, frame, self.config, True), latency_ms


class YoloWorldDetector(_BaseDetector):
    """YOLO-World zero-shot trên Roboflow: gõ tên vật thể, không cần huấn luyện."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.url = f"{config.api_url}/yolo_world/infer"
        self.session.headers["Content-Type"] = "application/json"

    def predict(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        body = {
            "image": {
                "type": "base64",
                "value": _encode_jpeg(frame).decode("ascii"),
            },
            "text": list(self.config.yolo_world_prompts),
            "yolo_world_version_id": self.config.yolo_world_version,
            "confidence": self.config.confidence,
        }
        response, latency_ms = self._post(data=json.dumps(body))
        # Chỉ gửi đúng các câu mô tả cần tìm nên không lọc theo tên lớp.
        return _parse_response(response, frame, self.config, False), latency_ms


def make_detector(config: Config):
    if config.detector_mode == "yolo_world":
        return YoloWorldDetector(config)
    return RoboflowDetector(config)

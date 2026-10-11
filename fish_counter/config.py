from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

YOLO_WORLD_VERSIONS = {"v2-s", "v2-m", "v2-l", "v2-x", "s", "m", "l", "x"}


def _positive_int(name: str, default: str) -> int:
    value = int(os.getenv(name, default))
    if value < 1:
        raise ValueError(f"{name} phải >= 1")
    return value


def _unit_float(name: str, default: str) -> float:
    value = float(os.getenv(name, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} phải nằm trong [0, 1]")
    return value


def _point(name: str, default: str) -> tuple[float, float]:
    values = [float(v.strip()) for v in os.getenv(name, default).split(",")]
    if len(values) != 2 or not all(0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"{name} phải gồm x,y chuẩn hóa trong [0,1]")
    return values[0], values[1]


@dataclass(frozen=True)
class Config:
    api_key: str
    detector_mode: str            # "yolo_world" hoặc "custom_model"
    model_id: str                 # chỉ dùng khi custom_model
    yolo_world_prompts: tuple[str, ...]
    yolo_world_version: str
    video_path: Path | None
    api_url: str
    fish_classes: frozenset[str]
    confidence: float
    max_width: int
    line_a: tuple[float, float]
    line_b: tuple[float, float]
    direction: str
    line_margin: int
    crossing_cooldown: int
    lost_track_buffer: int
    log_every_frames: int
    request_timeout_seconds: int


def load_config() -> Config:
    load_dotenv(ROOT / ".env")

    key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not key or key == "YOUR_ROBOFLOW_API_KEY":
        raise ValueError("Thiếu ROBOFLOW_API_KEY thật trong .env")

    mode = os.getenv("DETECTOR_MODE", "yolo_world").strip().lower()
    if mode not in {"yolo_world", "custom_model"}:
        raise ValueError("DETECTOR_MODE phải là yolo_world hoặc custom_model")

    model = os.getenv("ROBOFLOW_MODEL_ID", "").strip()
    if mode == "custom_model":
        if not model or model == "your-fish-model/1":
            raise ValueError("Thiếu ROBOFLOW_MODEL_ID thật trong .env")
        if "/" not in model or model.startswith("/") or model.endswith("/"):
            raise ValueError(
                "ROBOFLOW_MODEL_ID phải là ID từ phần triển khai mô hình"
            )

    prompts = tuple(
        s.strip()
        for s in os.getenv("YOLO_WORLD_CLASSES", "fish").split(",")
        if s.strip()
    )
    if mode == "yolo_world" and not prompts:
        raise ValueError("YOLO_WORLD_CLASSES không được rỗng")

    version = os.getenv("YOLO_WORLD_VERSION", "v2-s").strip()
    if version not in YOLO_WORLD_VERSIONS:
        raise ValueError(
            "YOLO_WORLD_VERSION phải là một trong: "
            + ", ".join(sorted(YOLO_WORLD_VERSIONS))
        )

    raw_path = os.getenv("VIDEO_PATH", "").strip().strip('"')
    path = Path(raw_path).expanduser() if raw_path else None
    if path is not None and not path.is_absolute():
        path = ROOT / path

    a = _point("LINE_A", "0.10,0.50")
    b = _point("LINE_B", "0.90,0.50")
    if a == b:
        raise ValueError("LINE_A và LINE_B không được trùng nhau")

    direction = os.getenv("COUNT_DIRECTION", "both").strip().lower()
    if direction not in {"both", "a_to_b", "b_to_a"}:
        raise ValueError("COUNT_DIRECTION: both, a_to_b hoặc b_to_a")

    classes = frozenset(
        s.strip().casefold()
        for s in os.getenv("FISH_CLASS_NAMES", "fish").split(",")
        if s.strip()
    )
    if mode == "custom_model" and not classes:
        raise ValueError("FISH_CLASS_NAMES không được rỗng")

    return Config(
        api_key=key,
        detector_mode=mode,
        model_id=model,
        yolo_world_prompts=prompts,
        yolo_world_version=version,
        video_path=path,
        api_url=os.getenv(
            "ROBOFLOW_API_URL", "https://serverless.roboflow.com"
        ).rstrip("/"),
        fish_classes=classes,
        confidence=_unit_float("CONFIDENCE_THRESHOLD", "0.10"),
        max_width=_positive_int("MAX_FRAME_WIDTH", "960"),
        line_a=a,
        line_b=b,
        direction=direction,
        line_margin=_positive_int("LINE_MARGIN_PIXELS", "6"),
        crossing_cooldown=_positive_int("CROSSING_COOLDOWN_FRAMES", "12"),
        lost_track_buffer=_positive_int("LOST_TRACK_BUFFER", "30"),
        log_every_frames=_positive_int("LOG_EVERY_FRAMES", "10"),
        request_timeout_seconds=_positive_int("REQUEST_TIMEOUT_SECONDS", "15"),
    )

import csv
from datetime import datetime, timezone
from pathlib import Path

from counter import Crossing


class StatisticsLogger:
    def __init__(self, output_dir: Path):
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.frame_path = output_dir / "fish_count_log.csv"
            self.events_path = output_dir / "crossing_events.csv"
            self.frame_file = self.frame_path.open(
                "w", newline="", encoding="utf-8-sig"
            )
            try:
                self.event_file = self.events_path.open(
                    "w", newline="", encoding="utf-8-sig"
                )
            except OSError:
                self.frame_file.close()
                raise

            self.frames = csv.writer(self.frame_file)
            self.events = csv.writer(self.event_file)
            self.frames.writerow([
                "timestamp_utc", "run_index", "frame_index",
                "video_time_seconds", "fish_present", "total_crossings",
                "inference_latency_ms", "display_fps",
            ])
            self.events.writerow([
                "timestamp_utc", "run_index", "frame_index",
                "video_time_seconds", "track_id", "direction",
            ])
        except OSError as exc:
            raise RuntimeError(
                f"Không thể tạo/ghi CSV trong {output_dir}: {exc}"
            ) from exc

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def log(
        self,
        run_index: int,
        frame_index: int,
        video_time: float,
        present: int,
        total: int,
        latency_ms: float,
        display_fps: float,
        events: list[Crossing],
        write_frame: bool,
    ) -> None:
        stamp = self.now()
        try:
            if write_frame:
                self.frames.writerow([
                    stamp, run_index, frame_index,
                    f"{video_time:.3f}", present, total,
                    f"{latency_ms:.1f}", f"{display_fps:.2f}",
                ])
                self.frame_file.flush()
            for event in events:
                self.events.writerow([
                    stamp, run_index, event.frame_index,
                    f"{event.video_time_seconds:.3f}",
                    event.track_id, event.direction,
                ])
            if events:
                self.event_file.flush()
        except OSError as exc:
            raise RuntimeError(f"Lỗi ghi CSV: {exc}") from exc

    def close(self) -> None:
        self.frame_file.close()
        self.event_file.close()

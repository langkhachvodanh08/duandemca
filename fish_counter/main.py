import argparse
from pathlib import Path
import queue
import sys
import threading
import time
from tkinter import Tk, filedialog

import cv2
import numpy as np
import supervision as sv

from config import ROOT, load_config
from counter import FishCounter
from roboflow_detector import DetectionError, make_detector
from statistics_logger import StatisticsLogger
from tracker import ObjectTracker
from video_manager import VideoManager
from visualizer import Visualizer


def choose_video() -> Path | None:
    root = Tk()
    root.withdraw()
    try:
        filename = filedialog.askopenfilename(
            title="Chọn video có cá",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv"),
                ("Tất cả tệp", "*.*"),
            ],
        )
    finally:
        root.destroy()
    return Path(filename) if filename else None


def _pixel_point(
    normalized: tuple[float, float], width: int, height: int
) -> tuple[int, int]:
    return (
        round(normalized[0] * (width - 1)),
        round(normalized[1] * (height - 1)),
    )


def run() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=Path, help="Ghi đè VIDEO_PATH trong .env"
    )
    args = parser.parse_args()

    path = args.video or config.video_path or choose_video()
    if path is None:
        print("Chưa chọn video; không có công việc nào được chạy.")
        return
    if not path.is_absolute():
        path = ROOT / path

    video = VideoManager(path, config.max_width)
    logger = None
    work: queue.Queue = queue.Queue(maxsize=1)
    results: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()

    def worker() -> None:
        detector = make_detector(config)
        try:
            while not stop.is_set():
                try:
                    packet = work.get(timeout=0.1)
                except queue.Empty:
                    continue
                frame, index, seconds = packet
                try:
                    detections, latency = detector.predict(frame)
                    results.put((frame, index, seconds, detections, latency, None))
                except DetectionError as exc:
                    results.put((frame, index, seconds, [], 0.0, str(exc)))
                except Exception as exc:
                    results.put((
                        frame, index, seconds, [], 0.0,
                        f"Lỗi suy luận không dự kiến: {type(exc).__name__}"
                    ))
        finally:
            detector.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        logger = StatisticsLogger(ROOT / "output")
        tracker = ObjectTracker(
            video.source_fps, config.lost_track_buffer
        )
        visual = Visualizer()
        cv2.namedWindow(visual.WINDOW, cv2.WINDOW_NORMAL)

        pending = False
        paused = False
        ended = False
        failed_packet = None
        run_index = 1
        last_frame = None
        last_tracked: sv.Detections | None = None
        last_latency = 0.0
        display_fps = 0.0
        last_finished_at = None
        status = "STARTING"
        last_index = 0
        line_a = None
        line_b = None
        counter = None
        rendered = None

        while True:
            if not paused and not pending and not ended and failed_packet is None:
                packet = video.read()
                if packet is None:
                    ended = True
                    status = "END - E TO REPLAY"
                else:
                    frame, index, seconds = packet
                    h, w = frame.shape[:2]
                    if counter is None:
                        line_a = _pixel_point(config.line_a, w, h)
                        line_b = _pixel_point(config.line_b, w, h)
                        if line_a == line_b:
                            raise ValueError(
                                "Đường đếm quá ngắn trên kích thước video này"
                            )
                        counter = FishCounter(
                            line_a, line_b, config.line_margin,
                            config.crossing_cooldown, config.direction,
                            max(config.lost_track_buffer + 5, 45),
                        )
                    last_frame = frame
                    last_index = index
                    work.put_nowait(packet)
                    pending = True
                    status = "INFERENCING"

            if pending and not paused:
                try:
                    frame, index, seconds, detections, latency, error = (
                        results.get_nowait()
                    )
                except queue.Empty:
                    pass
                else:
                    pending = False
                    last_frame = frame
                    last_index = index
                    if error:
                        failed_packet = (frame, index, seconds)
                        status = "ERROR - N RETRY / Q QUIT"
                        print(f"Frame {index}: {error}", file=sys.stderr)
                    else:
                        failed_packet = None
                        last_tracked = tracker.update(detections)
                        events = counter.update(last_tracked, index, seconds)
                        last_latency = latency
                        now = time.perf_counter()
                        if last_finished_at is not None:
                            elapsed = now - last_finished_at
                            display_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                        last_finished_at = now
                        status = "OK"
                        logger.log(
                            run_index, index, seconds,
                            counter.fish_present, counter.total_crossings,
                            latency, display_fps, events,
                            write_frame=(
                                index == 1
                                or index % config.log_every_frames == 0
                                or bool(events)
                            ),
                        )

            if last_frame is not None:
                h, w = last_frame.shape[:2]
                if line_a is None:
                    line_a = _pixel_point(config.line_a, w, h)
                    line_b = _pixel_point(config.line_b, w, h)
                shown_status = "PAUSED" if paused else status
                rendered = visual.draw(
                    last_frame,
                    last_tracked if status == "OK" or paused or ended else None,
                    line_a, line_b,
                    counter.fish_present if counter else 0,
                    counter.total_crossings if counter else 0,
                    display_fps, last_latency, last_index,
                    video.total_frames, shown_status,
                )
                cv2.imshow(visual.WINDOW, rendered)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("p"), ord("P"), 32):
                paused = not paused
            elif key in (ord("r"), ord("R")) and counter is not None:
                counter.reset_crossings()
                print("Đã đặt lại tổng lượt qua đường.")
            elif key in (ord("n"), ord("N")) and failed_packet is not None:
                work.put_nowait(failed_packet)
                pending = True
                failed_packet = None
                status = "RETRYING"
            elif key in (ord("e"), ord("E")) and ended and not pending:
                video.rewind()
                tracker.reset()
                counter.reset_all()
                last_tracked = None
                last_frame = None
                last_index = 0
                last_finished_at = None
                display_fps = 0.0
                paused = False
                ended = False
                run_index += 1
                status = "REPLAYING"
            elif key in (ord("s"), ord("S")) and rendered is not None:
                snapshot = ROOT / "output" / f"frame_{run_index}_{last_index}.jpg"
                if not cv2.imwrite(str(snapshot), rendered):
                    print(f"Không lưu được ảnh: {snapshot}", file=sys.stderr)
                else:
                    print(f"Đã lưu ảnh: {snapshot}")

            try:
                if cv2.getWindowProperty(
                    visual.WINDOW, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break
            except cv2.error:
                break
    finally:
        stop.set()
        video.release()
        if logger is not None:
            logger.close()
            print(f"CSV frame: {logger.frame_path}")
            print(f"CSV sự kiện: {logger.events_path}")
        cv2.destroyAllWindows()
        # Worker daemon có timeout mạng; không chặn vô thời hạn khi đóng cửa sổ.
        thread.join(timeout=0.2)


if __name__ == "__main__":
    try:
        run()
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nĐã dừng.")

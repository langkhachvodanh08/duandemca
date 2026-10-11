import numpy as np
import supervision as sv

from roboflow_detector import Detection


class ObjectTracker:
    """ByteTrack qua Supervision.

    ByteTrack ghép box theo (IoU x confidence) nên detector cho confidence
    thấp (như YOLO-World) sẽ làm mất ID. Vì chỉ có một lớp, ta đưa confidence
    về thang 0.5-1.0 cho riêng bước theo dõi, rồi trả lại confidence thật
    cho từng box để hiển thị và ghi nhận.
    """

    def __init__(self, fps: float, lost_track_buffer: int):
        self.fps = fps if fps > 0 else 30.0
        self.lost_track_buffer = lost_track_buffer
        self._create()

    def _create(self) -> None:
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=self.fps,
            minimum_consecutive_frames=1,
        )

    def update(self, items: list[Detection]) -> sv.Detections:
        if items:
            xyxy = np.asarray([d.xyxy for d in items], dtype=np.float32)
            real_conf = np.asarray(
                [d.confidence for d in items], dtype=np.float32
            )
            dets = sv.Detections(
                xyxy=xyxy,
                confidence=0.5 + 0.5 * real_conf,
                class_id=np.zeros(len(items), dtype=np.int32),
            )
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            real_conf = np.empty((0,), dtype=np.float32)
            dets = sv.Detections(
                xyxy=xyxy,
                confidence=real_conf,
                class_id=np.empty((0,), dtype=np.int32),
            )

        tracked = self.tracker.update_with_detections(dets)

        # Trả lại confidence thật: lấy của detection có IoU lớn nhất.
        if len(tracked) and len(items):
            iou = sv.box_iou_batch(
                np.asarray(tracked.xyxy, dtype=np.float32), xyxy
            )
            best = iou.argmax(axis=1)
            restored = np.where(
                iou[np.arange(len(tracked)), best] > 0.1,
                real_conf[best],
                np.asarray(tracked.confidence, dtype=np.float32) * 2 - 1,
            )
            tracked.confidence = np.clip(restored, 0.0, 1.0).astype(np.float32)
        return tracked

    def reset(self) -> None:
        self._create()

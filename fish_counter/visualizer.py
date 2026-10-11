import cv2
import numpy as np
import supervision as sv


class Visualizer:
    WINDOW = "Fish counter - Q quit | P/Space pause | R reset | E replay | S save"

    def draw(
        self,
        frame: np.ndarray,
        tracked: sv.Detections | None,
        a: tuple[int, int],
        b: tuple[int, int],
        present: int,
        crossings: int,
        fps: float,
        latency_ms: float,
        frame_index: int,
        total_frames: int,
        status: str,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.line(out, a, b, (0, 220, 255), 2, cv2.LINE_AA)

        if tracked is not None and tracked.tracker_id is not None:
            for i, raw_id in enumerate(tracked.tracker_id):
                x1, y1, x2, y2 = (
                    int(round(float(v))) for v in tracked.xyxy[i]
                )
                x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
                y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
                confidence = (
                    float(tracked.confidence[i])
                    if tracked.confidence is not None else 0.0
                )
                cv2.rectangle(
                    out, (x1, y1), (x2, y2), (0, 255, 100), 2
                )
                label = f"fish #{int(raw_id)} {confidence:.1%}"
                cv2.putText(
                    out, label, (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA
                )

        lines = [
            f"FISH PRESENT: {present}   TOTAL CROSSINGS: {crossings}",
            f"FPS: {fps:.1f}   API: {latency_ms:.0f} ms",
            f"FRAME: {frame_index}/{total_frames or '?'}   {status}",
        ]
        overlay = out.copy()
        panel_height = min(h, 80)
        cv2.rectangle(overlay, (0, 0), (w - 1, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
        for j, text in enumerate(lines):
            y = 22 + 24 * j
            if y < h:
                cv2.putText(
                    out, text[:100], (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA
                )
        return out

from dataclasses import dataclass

import supervision as sv

Point = tuple[float, float]


@dataclass(frozen=True)
class Crossing:
    track_id: int
    frame_index: int
    video_time_seconds: float
    direction: str


@dataclass
class TrackState:
    stable_side: int
    stable_point: Point
    last_seen: int
    last_counted: int = -1_000_000


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (
        b[1] - a[1]
    ) * (p[0] - a[0])


def _segments_intersect(p: Point, q: Point, a: Point, b: Point) -> bool:
    c1, c2 = _cross(p, q, a), _cross(p, q, b)
    c3, c4 = _cross(a, b, p), _cross(a, b, q)
    return c1 * c2 <= 0 and c3 * c4 <= 0


class FishCounter:
    def __init__(
        self,
        line_a: Point,
        line_b: Point,
        margin_pixels: int,
        cooldown_frames: int,
        direction: str,
        expiry_frames: int,
    ):
        self.a, self.b = line_a, line_b
        self.margin = margin_pixels
        self.cooldown = cooldown_frames
        self.direction = direction
        self.expiry = expiry_frames
        self.total_crossings = 0
        self.fish_present = 0
        self.states: dict[int, TrackState] = {}

    def _side(self, p: Point) -> int:
        line_length = max(
            1.0,
            ((self.b[0] - self.a[0]) ** 2
             + (self.b[1] - self.a[1]) ** 2) ** 0.5,
        )
        signed_distance = _cross(self.a, self.b, p) / line_length
        if signed_distance > self.margin:
            return 1
        if signed_distance < -self.margin:
            return -1
        return 0

    def update(
        self,
        tracked: sv.Detections,
        frame_index: int,
        video_time_seconds: float,
    ) -> list[Crossing]:
        events: list[Crossing] = []
        ids = tracked.tracker_id
        self.fish_present = len(tracked) if ids is not None else 0

        if ids is not None:
            for i, raw_id in enumerate(ids):
                track_id = int(raw_id)
                if track_id < 0:
                    continue
                x1, y1, x2, y2 = map(float, tracked.xyxy[i])
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                side = self._side(center)
                old = self.states.get(track_id)

                if old is None:
                    if side != 0:
                        self.states[track_id] = TrackState(
                            side, center, frame_index
                        )
                    continue

                previous_seen = old.last_seen
                old.last_seen = frame_index

                # Ở trong dải chống rung: giữ điểm ở phía ổn định trước đó.
                if side == 0:
                    continue

                if side != old.stable_side:
                    direction = (
                        "a_to_b" if old.stable_side == -1 else "b_to_a"
                    )
                    valid = (
                        frame_index - previous_seen <= self.expiry
                        and frame_index - old.last_counted >= self.cooldown
                        and _segments_intersect(
                            old.stable_point, center, self.a, self.b
                        )
                        and self.direction in {"both", direction}
                    )
                    if valid:
                        self.total_crossings += 1
                        old.last_counted = frame_index
                        events.append(
                            Crossing(
                                track_id, frame_index,
                                video_time_seconds, direction
                            )
                        )

                old.stable_side = side
                old.stable_point = center

        for track_id, state in list(self.states.items()):
            if frame_index - state.last_seen > self.expiry:
                del self.states[track_id]
        return events

    def reset_crossings(self) -> None:
        self.total_crossings = 0
        # Giữ phía hiện tại của mỗi ID để bấm R không tạo lượt giả.
        for state in self.states.values():
            state.last_counted = -1_000_000

    def reset_all(self) -> None:
        self.total_crossings = 0
        self.fish_present = 0
        self.states.clear()

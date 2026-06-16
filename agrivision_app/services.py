from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass
class TelemetryFrame:
    timestamp: datetime
    link_state: str
    flight_mode: str
    gps_lat: float
    gps_lon: float
    altitude_m: float
    battery_percent: float
    battery_voltage: float
    current_a: float
    speed_mps: float


@dataclass
class VisionFrame:
    timestamp: datetime
    source_state: str
    target_class: str
    confidence: float
    x_m: float
    y_m: float
    z_m: float
    fps: float
    box_x: float
    box_y: float
    box_w: float
    box_h: float


@dataclass
class LogRecord:
    timestamp: datetime
    category: str
    confidence: float
    x_m: float
    y_m: float
    z_m: float
    status: str


class FlightTelemetryPort(QObject):
    telemetry_received = Signal(object)
    status_changed = Signal(str)

    def connect_link(self) -> None:
        raise NotImplementedError

    def disconnect_link(self) -> None:
        raise NotImplementedError

    def send_command(self, command: str, payload: dict | None = None) -> None:
        raise NotImplementedError


class VideoStreamPort(QObject):
    vision_received = Signal(object)
    status_changed = Signal(str)

    def open_stream(self, source: str) -> None:
        raise NotImplementedError

    def close_stream(self) -> None:
        raise NotImplementedError


class DataRepositoryPort:
    def add_detection(self, frame: VisionFrame) -> None:
        raise NotImplementedError

    def search(self, keyword: str = "") -> list[LogRecord]:
        raise NotImplementedError


class MockFlightTelemetry(FlightTelemetryPort):
    def __init__(self) -> None:
        super().__init__()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._emit_frame)
        self._battery = 87.0
        self._lat = 31.2304
        self._lon = 121.4737

    def connect_link(self) -> None:
        self.status_changed.emit("数传在线")
        self._timer.start(900)

    def disconnect_link(self) -> None:
        self._timer.stop()
        self.status_changed.emit("数传离线")

    def send_command(self, command: str, payload: dict | None = None) -> None:
        self.status_changed.emit(f"已下发指令: {command}")

    def _emit_frame(self) -> None:
        self._battery = max(0, self._battery - random.uniform(0.01, 0.05))
        self._lat += random.uniform(-0.00002, 0.00002)
        self._lon += random.uniform(-0.00002, 0.00002)
        self.telemetry_received.emit(
            TelemetryFrame(
                timestamp=datetime.now(),
                link_state="ONLINE",
                flight_mode=random.choice(["AUTO", "GUIDED", "LOITER"]),
                gps_lat=self._lat,
                gps_lon=self._lon,
                altitude_m=random.uniform(0.6, 1.8),
                battery_percent=self._battery,
                battery_voltage=random.uniform(47.6, 49.2),
                current_a=random.uniform(8.0, 16.5),
                speed_mps=random.uniform(0.2, 1.4),
            )
        )


class MockVideoStream(VideoStreamPort):
    def __init__(self) -> None:
        super().__init__()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._emit_frame)

    def open_stream(self, source: str) -> None:
        self.status_changed.emit(f"图传在线: {source}")
        self._timer.start(180)

    def close_stream(self) -> None:
        self._timer.stop()
        self.status_changed.emit("图传离线")

    def _emit_frame(self) -> None:
        target_class = random.choice(["作物苗", "阔叶杂草", "禾本科杂草", "障碍物"])
        self.vision_received.emit(
            VisionFrame(
                timestamp=datetime.now(),
                source_state="CAMERA_ONLINE",
                target_class=target_class,
                confidence=random.uniform(0.61, 0.96),
                x_m=random.uniform(-0.8, 0.8),
                y_m=random.uniform(-0.35, 0.35),
                z_m=random.uniform(0.7, 3.2),
                fps=random.uniform(22.0, 38.0),
                box_x=random.uniform(0.18, 0.66),
                box_y=random.uniform(0.18, 0.54),
                box_w=random.uniform(0.12, 0.22),
                box_h=random.uniform(0.14, 0.26),
            )
        )


class MemoryRepository(DataRepositoryPort):
    def __init__(self) -> None:
        self._records: list[LogRecord] = []

    def add_detection(self, frame: VisionFrame) -> None:
        self._records.insert(
            0,
            LogRecord(
                timestamp=frame.timestamp,
                category=frame.target_class,
                confidence=frame.confidence,
                x_m=frame.x_m,
                y_m=frame.y_m,
                z_m=frame.z_m,
                status="已记录",
            ),
        )
        del self._records[300:]

    def search(self, keyword: str = "") -> list[LogRecord]:
        if not keyword:
            return list(self._records)
        key = keyword.lower()
        return [item for item in self._records if key in item.category.lower() or key in item.status.lower()]

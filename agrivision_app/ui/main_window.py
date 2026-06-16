from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from agrivision_app.services import (
    MemoryRepository,
    MockFlightTelemetry,
    MockVideoStream,
    TelemetryFrame,
    VisionFrame,
)


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, tone: str = "green") -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self.value_label = QLabel(value)
        self.value_label.setObjectName(f"metricValue_{tone}")
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class VideoCanvas(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(500)
        self.setObjectName("videoCanvas")
        self.frame: VisionFrame | None = None

    def set_vision_frame(self, frame: VisionFrame) -> None:
        self.frame = frame
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(18, 18, -18, -18)

        painter.setPen(QPen(QColor(48, 85, 85), 1))
        for x in range(rect.left(), rect.right(), 42):
            painter.drawLine(x, rect.top(), x, rect.bottom())
        for y in range(rect.top(), rect.bottom(), 42):
            painter.drawLine(rect.left(), y, rect.right(), y)

        painter.setPen(QPen(QColor(38, 184, 137), 2))
        painter.drawRect(rect)

        if not self.frame:
            painter.setPen(QColor(157, 174, 184))
            painter.drawText(rect, Qt.AlignCenter, "等待视频流 / 相机端接入")
            return

        f = self.frame
        box = rect.adjusted(
            int(rect.width() * f.box_x),
            int(rect.height() * f.box_y),
            -int(rect.width() * (1 - f.box_x - f.box_w)),
            -int(rect.height() * (1 - f.box_y - f.box_h)),
        )
        painter.setPen(QPen(QColor(35, 229, 157), 3))
        painter.drawRect(box)
        painter.setPen(QColor(255, 209, 102))
        painter.drawText(box.left(), max(rect.top() + 18, box.top() - 8), f"{f.target_class} {f.confidence:.2f}")

        cx, cy = box.center().x(), box.center().y()
        painter.setPen(QPen(QColor(255, 90, 90), 2))
        painter.drawLine(cx - 15, cy, cx + 15, cy)
        painter.drawLine(cx, cy - 15, cx, cy + 15)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AgriVision Edge 农用目标智能管控平台")
        self.resize(1440, 920)

        self.telemetry = MockFlightTelemetry()
        self.video = MockVideoStream()
        self.repository = MemoryRepository()
        self.current_vision: VisionFrame | None = None

        self.telemetry.telemetry_received.connect(self.on_telemetry)
        self.video.vision_received.connect(self.on_vision)
        self.telemetry.status_changed.connect(self.set_backend_status)
        self.video.status_changed.connect(self.set_backend_status)

        self.stack = QStackedWidget()
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.status_label = QLabel("后端接口: 模拟模式")

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_sidebar())
        root_layout.addWidget(self._build_workspace(), 1)
        self.setCentralWidget(root)

        self._apply_style()
        self.telemetry.connect_link()
        self.video.open_stream("mock://camera-01")

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._tick_clock)
        self.clock_timer.start(1000)

    def _build_sidebar(self) -> QWidget:
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(238)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(20, 22, 20, 22)
        layout.setSpacing(14)

        brand = QLabel("AgriVision\nEdge Console")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        for index, text in enumerate(["集群总览", "相机识别", "数据日志", "硬件配置"]):
            button = QPushButton(text)
            button.setCheckable(True)
            button.setObjectName("navButton")
            button.clicked.connect(lambda checked, i=index: self.stack.setCurrentIndex(i))
            self.nav_group.addButton(button, index)
            layout.addWidget(button)
            if index == 0:
                button.setChecked(True)

        layout.addStretch(1)
        layout.addWidget(QLabel("预留接口"))
        for text in ["飞控数传: FlightTelemetryPort", "图传视频: VideoStreamPort", "数据库: DataRepositoryPort"]:
            label = QLabel(text)
            label.setObjectName("portHint")
            label.setWordWrap(True)
            layout.addWidget(label)
        return side

    def _build_workspace(self) -> QWidget:
        wrap = QWidget()
        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(26, 18, 26, 22)
        layout.setSpacing(18)

        top = QHBoxLayout()
        title = QLabel("农用目标智能管控平台")
        title.setObjectName("pageTitle")
        self.clock_label = QLabel()
        self.clock_label.setObjectName("clock")
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.status_label)
        top.addWidget(self.clock_label)
        layout.addLayout(top)

        self.stack.addWidget(self._page_overview())
        self.stack.addWidget(self._page_vision())
        self.stack.addWidget(self._page_logs())
        self.stack.addWidget(self._page_hardware())
        layout.addWidget(self.stack, 1)
        return wrap

    def _page_overview(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        grid = QGridLayout()
        self.card_model = MetricCard("AI 模型", "YOLO / ONNX 待接入", "green")
        self.card_count = MetricCard("今日识别记录", "0 条", "blue")
        self.card_link = MetricCard("飞控链路", "数传在线", "green")
        self.card_fps = MetricCard("推理帧率", "0.0 FPS", "yellow")
        for idx, card in enumerate([self.card_model, self.card_count, self.card_link, self.card_fps]):
            grid.addWidget(card, 0, idx)
        layout.addLayout(grid)

        body = QHBoxLayout()
        self.system_log = QLabel("> [SYS] Qt 桌面端已启动\n> [PORT] 飞控数传/图传/相机端使用接口占位\n> [MODE] 当前为前端优先模拟模式")
        self.system_log.setObjectName("terminal")
        self.map_panel = QLabel("路径规划 / GPS 地图占位\n\n后续接入: 飞控航点、GPS 轨迹、任务线")
        self.map_panel.setAlignment(Qt.AlignCenter)
        self.map_panel.setObjectName("mapPanel")
        body.addWidget(self.map_panel, 2)
        body.addWidget(self.system_log, 1)
        layout.addLayout(body, 1)
        return page

    def _page_vision(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        self.video_canvas = VideoCanvas()
        hud = QFrame()
        hud.setObjectName("hud")
        hud_layout = QVBoxLayout(hud)
        self.hud_class = QLabel("类别: ---")
        self.hud_conf = QLabel("置信度: 0.00")
        self.hud_x = QLabel("相机坐标 X: 0.00 m")
        self.hud_y = QLabel("相机坐标 Y: 0.00 m")
        self.hud_z = QLabel("目标距离 Z: 0.00 m")
        self.hud_fps = QLabel("AI FPS: 0.0")
        for widget in [self.hud_class, self.hud_conf, self.hud_x, self.hud_y, self.hud_z, self.hud_fps]:
            widget.setObjectName("hudRow")
            hud_layout.addWidget(widget)
        hud_layout.addStretch(1)
        self.stream_input = QLineEdit("mock://camera-01")
        open_button = QPushButton("打开视频端")
        open_button.clicked.connect(lambda: self.video.open_stream(self.stream_input.text()))
        close_button = QPushButton("关闭视频端")
        close_button.clicked.connect(self.video.close_stream)
        hud_layout.addWidget(self.stream_input)
        hud_layout.addWidget(open_button)
        hud_layout.addWidget(close_button)
        layout.addWidget(self.video_canvas, 1)
        layout.addWidget(hud)
        return page

    def _page_logs(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("按日期、类别、状态检索，例如: 杂草 / 作物 / 已记录")
        search_button = QPushButton("查询")
        search_button.clicked.connect(self.refresh_logs)
        export_button = QPushButton("导出 CSV 预留")
        controls.addWidget(self.search_input, 1)
        controls.addWidget(search_button)
        controls.addWidget(export_button)
        layout.addLayout(controls)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["时间", "分类", "置信度", "X", "Y", "Z", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)
        return page

    def _page_hardware(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        self.battery_card = MetricCard("动力电池", "87.0%", "green")
        self.gps_card = MetricCard("GPS 坐标", "31.2304, 121.4737", "blue")
        self.mode_card = MetricCard("无人车模式", "AUTO", "yellow")
        self.speed_card = MetricCard("速度/高度", "0.0 m/s | 0.0 m", "green")
        layout.addWidget(self.battery_card, 0, 0)
        layout.addWidget(self.gps_card, 0, 1)
        layout.addWidget(self.mode_card, 1, 0)
        layout.addWidget(self.speed_card, 1, 1)

        config = QFrame()
        config.setObjectName("configPanel")
        config_layout = QVBoxLayout(config)
        config_layout.addWidget(QLabel("算法模型路径"))
        config_layout.addWidget(QLineEdit(r"F:\YOLOv\runs\train\exp55\weights\best.onnx"))
        config_layout.addWidget(QLabel("置信度阈值"))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, 100)
        slider.setValue(25)
        config_layout.addWidget(slider)
        config_layout.addWidget(QLabel("风偏补偿档位"))
        gear_row = QHBoxLayout()
        for i in range(1, 6):
            gear_row.addWidget(QPushButton(f"{i} 档"))
        config_layout.addLayout(gear_row)
        config_layout.addWidget(QPushButton("下发参数到飞控/算法服务"))
        layout.addWidget(config, 2, 0, 1, 2)
        return page

    def on_telemetry(self, frame: TelemetryFrame) -> None:
        self.card_link.set_value(frame.link_state)
        self.battery_card.set_value(f"{frame.battery_percent:.1f}% | {frame.battery_voltage:.1f}V")
        self.gps_card.set_value(f"{frame.gps_lat:.5f}, {frame.gps_lon:.5f}")
        self.mode_card.set_value(frame.flight_mode)
        self.speed_card.set_value(f"{frame.speed_mps:.1f} m/s | {frame.altitude_m:.1f} m")

    def on_vision(self, frame: VisionFrame) -> None:
        self.current_vision = frame
        self.repository.add_detection(frame)
        self.video_canvas.set_vision_frame(frame)
        self.hud_class.setText(f"类别: {frame.target_class}")
        self.hud_conf.setText(f"置信度: {frame.confidence:.2f}")
        self.hud_x.setText(f"相机坐标 X: {frame.x_m:.2f} m")
        self.hud_y.setText(f"相机坐标 Y: {frame.y_m:.2f} m")
        self.hud_z.setText(f"目标距离 Z: {frame.z_m:.2f} m")
        self.hud_fps.setText(f"AI FPS: {frame.fps:.1f}")
        self.card_fps.set_value(f"{frame.fps:.1f} FPS")
        self.card_count.set_value(f"{len(self.repository.search()):d} 条")
        if self.stack.currentIndex() == 2:
            self.refresh_logs()

    def refresh_logs(self) -> None:
        records = self.repository.search(self.search_input.text().strip())
        self.table.setRowCount(len(records))
        for row, item in enumerate(records):
            values = [
                item.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                item.category,
                f"{item.confidence:.2f}",
                f"{item.x_m:.2f}",
                f"{item.y_m:.2f}",
                f"{item.z_m:.2f}",
                item.status,
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

    def set_backend_status(self, text: str) -> None:
        self.status_label.setText(f"后端接口: {text}")

    def _tick_clock(self) -> None:
        self.clock_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #eef3f5; color: #17242b; font-family: "Microsoft YaHei", "Segoe UI"; }
            #sidebar { background: #102027; color: white; }
            #brand { color: #37e2a1; font-size: 24px; font-weight: 800; line-height: 1.2; }
            #navButton { background: transparent; color: #b9c9cf; border: 1px solid #253940; border-radius: 6px; padding: 13px; text-align: left; font-size: 15px; }
            #navButton:checked { background: #1c3a3f; color: #37e2a1; border-color: #37e2a1; }
            #portHint { color: #8da2aa; font-size: 12px; padding: 5px 0; }
            #pageTitle { font-size: 24px; font-weight: 800; }
            #clock, QLabel { font-size: 14px; }
            #metricCard, #hud, #configPanel { background: white; border: 1px solid #d8e1e5; border-radius: 8px; padding: 14px; }
            #metricTitle { color: #61757e; font-size: 13px; }
            #metricValue_green, #metricValue_blue, #metricValue_yellow { font-size: 25px; font-weight: 800; }
            #metricValue_green { color: #13865c; }
            #metricValue_blue { color: #176ab6; }
            #metricValue_yellow { color: #a97700; }
            #terminal { background: #0c1418; color: #56d364; border-radius: 8px; padding: 18px; font-family: Consolas; }
            #mapPanel { background: #dfe8eb; border: 1px dashed #9eb2ba; border-radius: 8px; color: #587078; font-size: 18px; }
            #videoCanvas { background: #071014; border-radius: 8px; border: 1px solid #1d3f43; }
            #hud { min-width: 300px; max-width: 340px; }
            #hudRow { background: #f3f7f8; border-radius: 5px; padding: 10px; font-size: 15px; }
            QPushButton { background: #176ab6; color: white; border: 0; border-radius: 6px; padding: 10px 14px; font-weight: 700; }
            QPushButton:hover { background: #0f5799; }
            QLineEdit { background: white; border: 1px solid #c8d4da; border-radius: 6px; padding: 9px; }
            QTableWidget { background: white; border: 1px solid #d8e1e5; border-radius: 8px; gridline-color: #e7edf0; }
            QHeaderView::section { background: #e7edf0; padding: 9px; border: 0; font-weight: 700; }
            """
        )

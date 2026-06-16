import sys

from PySide6.QtWidgets import QApplication

from agrivision_app.ui.main_window import MainWindow


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AgriVision Edge")
    app.setOrganizationName("AgriVision")
    window = MainWindow()
    window.show()
    return app.exec()

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTableWidget, QTableWidgetItem
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QMutex


class DataFiller(QThread):
    data_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.paused = False
        self.current_step = 1
        self.mutex = QMutex()

    def run(self):
        while self.current_step <= 10:
            self.mutex.lock()
            if self.paused:
                self.mutex.unlock()
                self.sleep(1)
                continue
            self.mutex.unlock()

            self.data_updated.emit(f"Data {self.current_step}")
            self.current_step += 1
            self.sleep(1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Filling App")
        self.setGeometry(100, 100, 400, 300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(1)
        layout.addWidget(self.table_widget)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.start_data_filling)
        layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_data_filling)
        layout.addWidget(self.pause_button)

        self.data_filler = DataFiller()
        self.data_filler.data_updated.connect(self.update_table)

    def start_data_filling(self):
        if not self.data_filler.isRunning():
            self.data_filler.current_step = 1  # Reset the step counter
            self.data_filler.start()
        else:
            self.data_filler.paused = False

    def pause_data_filling(self):
        self.data_filler.paused = not self.data_filler.paused

    def update_table(self, data):
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        self.table_widget.setItem(row_position, 0, QTableWidgetItem(data))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

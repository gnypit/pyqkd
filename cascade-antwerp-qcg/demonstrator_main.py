"""Author: Jakub T. Gnyp; LinkedIn, GitHub @ Instagram: @gnypit; created for Quantum Cybersecurity Group (2023)"""
import sys

from PyQt6.QtWidgets import QApplication

from demonstrator_gui import QKDWindow
from bb84_model_controller import ModelControllerBB84


def main():
    """Demonstrator's main function"""
    demonstrator_app = QApplication(sys.argv)
    demonstrator_window = QKDWindow()
    ModelControllerBB84(view=demonstrator_window)
    demonstrator_window.show()
    sys.exit(demonstrator_app.exec())


if __name__ == "__main__":
    main()

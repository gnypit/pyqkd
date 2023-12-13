"""Author: Jakub T. Gnyp; LinkedIn, GitHub @ Instagram: @gnypit; created for Quantum Cybersecurity Group (2023)"""
import sys

from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTabWidget, QCheckBox, QLabel, QListWidget, QListWidgetItem, QTableWidget, QHBoxLayout
)

from PyQt6.QtCore import QTimer, QThread, pyqtSignal


class DataFiller(QThread):
    data_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.paused = False

    def run(self):
        for i in range(1, 11):
            if self.paused:
                self.sleep(1)  # Pause for 1 second
                continue
            self.data_updated.emit(f"Data {i}")
            self.sleep(1)  # Simulate some data filling time


class QKDWindow(QMainWindow):
    def __init__(self):
        """In this initializer we create all the necessary fields; the whole app will consist of three tabs:
        'input', 'quantum channel' & 'public channel'.

        The input tab has a form with labels to fill in and a message space for the program to respond to the User's
        input, whether it's correct or not. If proper input is submitted, parts of 'quantum channel' tab will be
        updated.

        The quantum channel tab is mainly a table to present a 'path' of each quantum state - what is it once prepared
        by Alice, how she measures it, what the result is, is the transfer successful or what are the events happening
        in the quantum channel, what state did Bob receive (same or changed), how he measured it and what his result
        was. Additionally, on this tab sifting is performed, as a crossing-out of measurements results stemming from
        different basis choices.

        Also, I added the 'Prepare simulation' function, which runs my simulation_bb84() function as a model
        for this app - the rest is only how the results are used and presented.

        The public channel tab primarily serves as the visualisation of the error correction process, focusing on BINARY
        performance on given blocks of bits. Moreover, error estimation and privacy amplification are performed.
        """
        super().__init__()

        """Hiperparameters:"""
        self.setWindowTitle("QCG Demonstrator for Quantum Key Distribution (QKD) - by Jakub T. Gnyp")
        self.resize(1920, 1080)

        """Parameters:"""
        self.no_message = 0
        self.simulation_hiperarameters = {}

        """Main widgets - input section:"""
        self.message_list = QListWidget(self)
        self.input_prompt_label = None
        self.submit_button = QPushButton(f"Submit\ninput", self)

        self.gain_input = QLineEdit()
        self.rect_prob_input = QLineEdit()
        self.no_quantum_states_input = QLineEdit()
        self.rect_pub_prob_input = QLineEdit()
        self.diag_pub_prob_input = QLineEdit()
        self.no_cascade_input = QLineEdit()
        self.dist_prob_input = QLineEdit()

        """Main widgets - quantum channel section:"""
        self.quantum_table = QTableWidget()

        self.simulate_button = QPushButton(f"Prepare\nsimulation", self)
        self.quantum_play_button = QPushButton(f"Play\nsimulation", self)
        self.quantum_pause_button = QPushButton(f"Pause\nsimulation", self)
        self.quantum_fast_forward_button = QPushButton(f"Fast\nForward", self)
        self.quantum_post_processing_button = QPushButton(f'Perform\npost-processing', self)
        self.colors_legend = QTableWidget()

        """Create the tab widget with two tabs:"""
        tabs = QTabWidget()
        tabs.addTab(self.simple_bb84_tab(), f"Simple BB84 demonstrator")
        # tabs.addTab(self.error_correction_tab(), "Error Correction Breakdown")
        self.setCentralWidget(tabs)  # Set the central widget to the tab widget

        """Now let's try to set up timers and all that for playing/pausing simulation:"""
        self.play_timer = QTimer()
        self.play_timer.setInterval(500)
        self.play_timer.timeout.connect(self.play_tick)  # https://stackoverflow.com/questions/35836660/functional-start-and-stop-button-in-a-gui-using-pyqt-or-pyside-for-real-time-dat

    def simple_bb84_tab(self):
        """This tab consists of both the input section and channels: quantum & public. Originally these sections
        were put into separate tabs, but for cosmetic reasons I decided to merge them.

        The channels section consists of a table with summary of states, basis & measurement results encoded as bits,
        together with a commentary on what is happening in the quantum channel."""
        main_tab = QWidget()
        main_layout = QGridLayout()

        """Create a label to prompt the user:"""
        self.input_prompt_label = QLabel(
            "Please provide input data for the BB84 protocol with refined error analysis.",
            self)
        main_layout.addWidget(self.input_prompt_label, 0, 0, 1, 2)

        """Create the fields for input with their respective labels:"""
        main_layout.addWidget(QLabel(f"Quantum Channel gain:"), 1, 0)
        main_layout.addWidget(self.gain_input, 1, 1)
        main_layout.addWidget(QLabel(f"Probability of choosing the rectilinear basis:"), 2, 0)
        main_layout.addWidget(self.rect_prob_input, 2, 1)
        main_layout.addWidget(QLabel(f"Number of quantum states to prepare:"), 3, 0)
        main_layout.addWidget(self.no_quantum_states_input, 3, 1)
        main_layout.addWidget(QLabel(f"Probability of publishing results stemming from measurement in the "
                                     f"rectilinear basis:"), 4, 0)
        main_layout.addWidget(self.rect_pub_prob_input, 4, 1)
        main_layout.addWidget(QLabel(f"Probability of publishing results stemming from measurement in the diagonal "
                                     f"basis:"), 5, 0)
        main_layout.addWidget(self.diag_pub_prob_input, 5, 1)
        main_layout.addWidget(QLabel(f"Number of CASCADE passes to be performed:"), 6, 0)
        main_layout.addWidget(self.no_cascade_input, 6, 1)
        main_layout.addWidget(QLabel(f"Disturbance probability:"), 7, 0)
        main_layout.addWidget(self.dist_prob_input, 7, 1)

        """Experimenting with sizes:"""
        self.gain_input.setFixedWidth(100)
        self.rect_prob_input.setFixedWidth(100)
        self.no_quantum_states_input.setFixedWidth(100)
        self.rect_pub_prob_input.setFixedWidth(100)
        self.diag_pub_prob_input.setFixedWidth(100)
        self.no_cascade_input.setFixedWidth(100)
        self.dist_prob_input.setFixedWidth(100)

        """Create 'submit' button to trigger accepting the input and/or displaying messages whether
        it's correct or not:"""
        self.submit_button.clicked.connect(self.on_submit)
        buttons = QHBoxLayout()
        buttons.addWidget(self.submit_button)

        """To control the flow of the demonstration, we put 'play' & 'pause' push-buttons 
        and include button to perform sifting in here:"""
        buttons.addWidget(self.simulate_button)
        buttons.addWidget(self.quantum_play_button)
        buttons.addWidget(self.quantum_pause_button)
        buttons.addWidget(self.quantum_fast_forward_button)
        buttons.addWidget(self.quantum_post_processing_button)
        # buttons.addStretch()
        main_layout.addLayout(buttons, 8, 0, 1, 2)

        """Create a QListWidget to display messages for either confirmation of correct input provided or
        invalid data commentary & call to action:"""
        main_layout.addWidget(QLabel("Dialog window:"), 9, 0, 1, 2)
        main_layout.addWidget(self.message_list, 10, 0, 1, 2)

        """Add the table to display the states, measurements and comments 
        on how the quantum channel operates:"""
        main_layout.addWidget(self.quantum_table, 0, 2, 11, 4)

        """Play around with stretching:"""
        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)
        main_layout.setColumnStretch(2, 2)

        """I set the main layout to the QWidget, which will be the main one on this particular tab:"""
        main_tab.setLayout(main_layout)
        return main_tab

    def error_correction_tab(self):
        """Create the Quantum Channel page UI."""
        p_channel_tab = QWidget()  # p for public
        layout = QVBoxLayout()

        layout.addWidget(QCheckBox("General Option 1"))
        layout.addWidget(QCheckBox("General Option 2"))

        p_channel_tab.setLayout(layout)
        return p_channel_tab

    def on_submit(self):
        """In this method I verify each and every input: quantum channel gain, probabilities of choosing the
        rectilinear basis and of publication in their respective basis, number of CASCADE passes to be performed
        and the disturbance probability."""

        gain_text = self.gain_input.text()
        try:
            gain = float(gain_text)
            if 0 <= gain <= 1:
                input_info = QListWidgetItem(f"[{self.no_message}] Gain provided by the User ({gain}) is valid.")
                self.message_list.addItem(input_info)
                self.simulation_hiperarameters['gain'] = gain
            else:
                input_info = QListWidgetItem(f"[{self.no_message}] Gain provided by the User ({gain}) is either too "
                                             f"big or too small. Please enter a numerical value between 0 and 1, "
                                             f"e.g. 0.95")
                self.message_list.addItem(input_info)
                self.gain_input.clear()
        except ValueError:
            input_info = QListWidgetItem(f"[{self.no_message}] Invalid input for the quantum gain. Please enter a "
                                         f"numerical value between 0 and 1, e.g. 0.95")
            self.message_list.addItem(input_info)
            self.gain_input.clear()

        self.no_message += 1  # after each variable verification there must've been a message

        rect_prob_text = self.rect_prob_input.text()
        try:
            rect_prob = float(rect_prob_text)
            if 0 <= rect_prob <= 1:
                input_info = QListWidgetItem(f"[{self.no_message}] Probability of using the rectilinear basis for "
                                             f"measurement provided by the User ({rect_prob}) is valid.")
                self.message_list.addItem(input_info)
                self.simulation_hiperarameters['rect_prob'] = rect_prob
            else:
                input_info = QListWidgetItem(f"[{self.no_message}] Probability of using the rectilinear basis for "
                                             f"measurement provided by the User ({rect_prob}) is either too big or too "
                                             f"small. Please enter a numerical value between 0 and 1, e.g. 0.75")
                self.message_list.addItem(input_info)
                self.rect_prob_input.clear()
        except ValueError:
            input_info = QListWidgetItem(f"[{self.no_message}] Invalid input for the probability of choosing the "
                                         f"rectilinear basis for the measurement. Please enter a numerical value "
                                         f"between 0 and 1, e.g. 0.95")
            self.message_list.addItem(input_info)
            self.rect_prob_input.clear()

        self.no_message += 1  # after each variable verification there must've been a message

        no_quantum_states_text = self.no_quantum_states_input.text()
        try:
            no_quantum_states = int(no_quantum_states_text)
            if 8 <= no_quantum_states:
                input_info = QListWidgetItem(f"[{self.no_message}] Number of quantum states provided by the User "
                                             f"({no_quantum_states}) is valid.")
                self.message_list.addItem(input_info)
                self.simulation_hiperarameters['no_quantum_states'] = no_quantum_states

                """As now I have the number of states to be prepared and sent, I can set the size of table on the 
                'Quantum Channel' tab:"""
                self.quantum_table.setRowCount(no_quantum_states)
                self.quantum_table.setColumnCount(7)
                self.quantum_table.setHorizontalHeaderLabels([
                    "States prepared\nby Alice", "Alice's\nBasis", "Alice's\nBits",
                    "Comment",
                    "States received\nby Bob", "Bob's\nBasis", "Bob's\nBits"
                ])

            else:
                input_info = QListWidgetItem(f"[{self.no_message}] Number of quantum states provided by the User "
                                             f"({no_quantum_states}) is too small. Please enter an integer greater"
                                             f"than or equal to 8.")
                self.message_list.addItem(input_info)
                self.no_quantum_states_input.clear()
        except ValueError:
            input_info = QListWidgetItem(f"[{self.no_message}] Invalid input for the number of quantum states to be "
                                         f"prepared. Please enter an integer greater than or equal to 8.")
            self.message_list.addItem(input_info)
            self.no_quantum_states_input.clear()

        self.no_message += 1  # after each variable verification there must've been a message

        diag_pub_prob_text = self.diag_pub_prob_input.text()
        try:
            diag_pub_prob = float(diag_pub_prob_text)
            if 0 <= diag_pub_prob <= 1:
                input_info = QListWidgetItem(f"[{self.no_message}] Probability of publishing measurement results from "
                                             f"the diagonal basis provided by the User ({diag_pub_prob}) is valid.")
                self.message_list.addItem(input_info)
                self.simulation_hiperarameters['diag_pub_prob'] = diag_pub_prob
            else:
                input_info = QListWidgetItem(f"[{self.no_message}] Probability of publishing measurement results from "
                                             f"the diagonal basis provided by the User ({diag_pub_prob}) is either"
                                             f"too big or too small. Please enter a numerical value between 0 and 1, "
                                             f"e.g. 0.25.")
                self.message_list.addItem(input_info)
                self.diag_pub_prob_input.clear()
        except ValueError:
            input_info = QListWidgetItem(f"[{self.no_message}] Invalid input for the probability of publishing "
                                         f"measurement results from the diagonal basis. Please enter a numerical value "
                                         f"between 0 and 1, e.g. 0.25.")
            self.message_list.addItem(input_info)
            self.diag_pub_prob_input.clear()

        self.no_message += 1  # after each variable verification there must've been a message

        rect_pub_prob_text = self.rect_pub_prob_input.text()
        try:
            rect_pub_prob = float(rect_pub_prob_text)
            if 0 <= rect_pub_prob <= 1:
                input_info = QListWidgetItem(f"[{self.no_message}] Probability of publishing measurement results from "
                                             f"the rectilinear basis provided by the User ({rect_pub_prob}) is valid.")
                self.message_list.addItem(input_info)
                self.simulation_hiperarameters['rect_pub_prob'] = rect_pub_prob
            else:
                input_info = QListWidgetItem(f"[{self.no_message}] Probability of publishing measurement results from "
                                             f"the rectilinear basis provided by the User ({rect_pub_prob}) is either"
                                             f"too big or too small. Please enter a numerical value between 0 and 1, "
                                             f"e.g. 0.25.")
                self.message_list.addItem(input_info)
                self.rect_pub_prob_input.clear()
        except ValueError:
            input_info = QListWidgetItem(f"[{self.no_message}] Invalid input for the probability of publishing "
                                         f"measurement results from the rectilinear basis. Please enter a numerical "
                                         f"value between 0 and 1, e.g. 0.25.")
            self.message_list.addItem(input_info)
            self.rect_pub_prob_input.clear()

        self.no_message += 1  # after each variable verification there must've been a message

        no_cascade_text = self.no_cascade_input.text()
        try:
            no_cascade = int(no_cascade_text)
            if 0 <= no_cascade <= 4:
                input_info = QListWidgetItem(f"[{self.no_message}] Number of CASCADE passes to be performed provided "
                                             f"by the User ({no_cascade}) is valid.")
                self.message_list.addItem(input_info)
                self.simulation_hiperarameters['no_cascade'] = no_cascade
            else:
                input_info = QListWidgetItem(f"[{self.no_message}] Number of CASCADE passes to be performed by the "
                                             f"User ({no_cascade}) is either too big or too small. Please enter an "
                                             f"integer between 0 and 4 included, e.g. 4.")
                self.message_list.addItem(input_info)
                self.no_cascade_input.clear()
        except ValueError:
            input_info = QListWidgetItem(f"[{self.no_message}] Invalid input for the number of CASCADE passes to be "
                                         f"performed. Please enter an integer between 0 and 4 included, e.g. 4.")
            self.message_list.addItem(input_info)
            self.no_cascade_input.clear()

        self.no_message += 1  # after each variable verification there must've been a message

        dist_prob_text = self.dist_prob_input.text()
        try:
            dist_prob = float(dist_prob_text)
            if 0 <= dist_prob <= 0.15:
                input_info = QListWidgetItem(f"[{self.no_message}] Probability of disturbance in the quantum channel "
                                             f"provided by the User ({dist_prob}) is valid.")
                self.message_list.addItem(input_info)
                self.simulation_hiperarameters['dist_prob'] = dist_prob
            else:
                input_info = QListWidgetItem(f"[{self.no_message}] Probability of disturbance in the quantum channel "
                                             f" provided by the User ({dist_prob}) is either too big or too small. "
                                             f"Please enter a numerical value between 0 and 0.15, e.g. 0.05.")
                self.message_list.addItem(input_info)
                self.dist_prob_input.clear()
        except ValueError:
            input_info = QListWidgetItem(f"[{self.no_message}] Invalid input for the probability of disturbance "
                                         f"in the quantum channel. Please enter a numerical value between 0 and 1, "
                                         f"e.g. 0.05.")
            self.message_list.addItem(input_info)
            self.dist_prob_input.clear()

    def play_tick(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QKDWindow()
    window.show()
    sys.exit(app.exec())

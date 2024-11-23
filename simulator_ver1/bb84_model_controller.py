from PyQt6.QtCore import pyqtSignal, QThread, QMutex
from PyQt6.QtWidgets import QTableWidgetItem
from demonstrator_gui import QKDWindow
from binary import binary
import bb84
import random
import ast
import numpy as np


possible_states = ['0', '1', '+', '-']


class DataFiller(QThread):
    data_updated = pyqtSignal(str)

    def __init__(self, loop_limit, data: list):
        super().__init__()
        self.fast_forward = None
        self.paused = False
        self.current_step = 0
        self.mutex = QMutex()
        self.loop_limit = loop_limit
        self.data = data

    def run(self):
        while self.current_step < self.loop_limit:
            self.mutex.lock()
            if self.paused:
                self.mutex.unlock()
                self.msleep(1000)
                continue
            self.mutex.unlock()

            data_entry = self.data[self.current_step]
            self.data_updated.emit(data_entry)
            self.current_step += 1

            if self.fast_forward:
                self.msleep(200)
            else:
                self.msleep(1000)


class ModelControllerBB84:
    def __init__(self, view: QKDWindow):
        """Method for initialisation of a single BB84 protocol's logic for demonstration. In internal variables are
        created, but no operations are performed yet, as the User didn't have a chance to act."""
        self.alice_bits_list = None
        self.alice_basis_list = None
        self.sent_states = []
        self.received_states = []
        self.alice_states_list = None
        self.comments = None
        self.data_filler = None
        self.length_of_key_into_cascade = None
        self.status_list = None  # for status updates to the viewer as the simulation unfolds
        self.error_estimation = None
        self.cascade_n_passes = None
        self.publication_probability_diagonal = None
        self.publication_probability_rectilinear = None
        self.disturbance_probability = None
        self.rectilinear_basis_prob = None
        self.gain = None
        self.view = view
        self.no_quantum_states = None  # number of quantum states to be prepared by Alice
        self.data = []
        self.disturbed_states_indexes = []

        """We want to store numbers of bits exchanged in each phase of the simulation: while performing sifting, 
        error estimation and finally error correction (CASCADE)."""
        self.number_of_published_bits = {
            'sifting': 0,
            'error estimation': 0,
            'error correction': 0
        }

        """There's a probability that due to disturbances in quantum channel or eavesdropping some states change while 
        being sent."""
        # self.disturbance_probability = None

        """We need a number of variables to perform the simulation as in the function above:"""
        self.alice_states = ''
        self.bob_states = ''

        self.alice_basis = ''
        self.bob_basis = ''

        self.alice_bits = ''
        self.bob_bits = ''

        self.received_states = ''
        self.change_states = {'0': '0', '1': '1', '2': '+', '3': '-'}  # dictionary for randomizing changed states

        self.bob_measurement_indicators = ''
        self.bob_indicated_basis = ''
        self.bob_indicated_bits = ''
        self.alice_indicated_bits = ''
        self.alice_indicated_basis = ''

        self.alice_sifted_key = ''
        self.alice_sifted_basis = ''
        self.bob_sifted_key = ''
        self.bob_sifted_basis = ''

        self.error_estimation_results = None
        self.error_estimate = None
        self.alice_sifted_key_after_error_estimation = None
        self.bob_sifted_key_after_error_estimation = None

        self.exchanged_bits_counter = 0
        self.alice_cascade = {}
        self.bob_cascade = {}
        self.blocks_sizes = None

        self.history = []
        self.error_rates = []
        self.pass_number = 0

        self.alice_correct_key = ''
        self.bob_correct_key = ''

        self.deleted_bits_counter = 0
        self.final_key_error_rate = 0
        self.key_length = None

        self.results = {
            'error rate': None,
            'key length': None,
            'comp. cost': None,
            'no. del. bits': None,
            'no. cascade pass.': None
        }

        """Next, we can connect the signals & slots:"""
        self._connect_signals_and_slots()
        print("Should be connected")

    def __del__(self):
        if self.data_filler and self.data_filler.isRunning():
            self.data_filler.quit()
            self.data_filler.wait()

    def simulate(self):
        """simulation_bb84() function from above, adjusted to be a method fot this Model class for the QKD
        demonstrator based on MVC design pattern."""
        self.gain = float(self.view.gain_input.text())
        self.no_quantum_states = int(self.view.no_quantum_states_input.text())
        self.rectilinear_basis_prob = float(self.view.rect_prob_input.text())
        self.disturbance_probability = float(self.view.dist_prob_input.text())
        self.publication_probability_rectilinear = float(self.view.rect_pub_prob_input.text())
        self.publication_probability_diagonal = float(self.view.diag_pub_prob_input.text())
        self.cascade_n_passes = int(self.view.no_cascade_input.text())
        self.error_estimation = bb84.refined_average_error

        """Let's inform the user that something is happening:"""
        self.add_status_update("Preparing the simulation - please wait.")
        print("Test")

        """Step 1: Alice prepares her states & performs her measurements."""
        self.start_simulation()

        """Step 2: Bob receives the states & performs his measurements."""
        self.measurement()

        """Step 3: Bob prepares & sends his measurement indicators."""
        self.measure_indicate()

        """Step 4: Sifting."""
        self.sifting()

        """Step 5: Error estimation."""
        self.estimate_error()

        """Step 6: Error correction."""
        self.correct_errors()

        """Step 7: Privacy amplification."""
        self.amplify_privacy()

        """Now we finally have the proper keys"""
        self.number_of_published_bits['error correction'] = self.exchanged_bits_counter + self.deleted_bits_counter

        """Let's calculate key error rate"""
        self.final_key_error_rate = 0
        index = 0
        for bit in self.alice_correct_key:
            if bit != self.bob_correct_key[index]:
                self.final_key_error_rate += 1
            index += 1
        try:
            self.final_key_error_rate = self.final_key_error_rate / len(self.alice_correct_key)
        except ZeroDivisionError:
            error_message = [
                self.blocks_sizes,
                self.pass_number,
                self.no_quantum_states,
                self.gain,
                self.disturbance_probability,
                self.error_estimate,
                self.length_of_key_into_cascade,
                self.rectilinear_basis_prob,
                self.publication_probability_rectilinear,
                self.cascade_n_passes
            ]
            print(error_message)
            self.final_key_error_rate = 1  # we set a max value to punish such a case

        key_length = len(self.alice_correct_key)

        results = {
            'error rate': self.final_key_error_rate,
            'key length': key_length,
            'comp. cost': self.number_of_published_bits,
            'no. del. bits': self.deleted_bits_counter,
            'no. cascade pass.': len(self.error_rates),
            'cascade history': self.history,
            'alice states': self.alice_states,
            'bob states': self.bob_states,
            'alice basis': self.alice_basis,
            'bob basis': self.bob_basis,
            'alice bits': self.alice_bits,
            'bob bits': self.bob_bits,
            'alice sifted key': self.alice_sifted_key,
            'bob sifted key': self.bob_sifted_key,
            'alice sifted key after error estimation': self.alice_sifted_key_after_error_estimation,
            'bob sifted key after error estimation': self.bob_sifted_key_after_error_estimation,
            'error estimate': self.error_estimate,
            'alice correct key': self.alice_correct_key,
            'bob correct_key': self.bob_correct_key
        }

        """It's probably the right moment to prepare data for filling into the table:"""
        self.prepare_data()

        return results

    def start_simulation(self):
        """To set up the quantum channel, Alice and Bob need to establish their basis choices for consecutive
        measurements. As Alice moderates the desired states by a combination of her basis choices AND her bits choices,
        that represent the wave-plates, etc., 'Alice chooses now randomly two strings independent of each other with
        length (...)' provided by the User in the GUI - `self.no_quantum_states`:"""
        self.add_status_update("Simulation preparation initiated.")

        self.alice_basis_list = np.random.binomial(1, 1 - self.rectilinear_basis_prob, self.no_quantum_states)
        for basis in self.alice_basis_list:
            self.alice_basis += str(int(basis))

        self.alice_bits_list = np.random.binomial(1, 0.5, self.no_quantum_states)
        for bit in self.alice_bits_list:
            self.alice_bits += str(int(bit))

        bob_basis_list = np.random.binomial(1, 1 - self.rectilinear_basis_prob, self.no_quantum_states)
        self.bob_basis = ''

        """Additionally, losses in the quantum channel are already simulated in Bob's basis string:"""
        for basis in bob_basis_list:
            if random.uniform(0, 1) <= self.gain:
                self.bob_basis += str(int(basis))
            else:
                self.bob_basis += 'L'  # L for loss

        self.add_status_update("Step 1/7 completed.")

    def measurement(self):
        """In this method both Alice's and Bob's measurements are performed. Additionally, any disturbances
        in the quantum channel are simulated."""
        for index in range(self.no_quantum_states):
            """Firstly, Alice prepares her state based on independent, random choices of basis and bit value:"""
            bit = self.alice_bits[index]
            first_basis = self.alice_basis[index]

            match first_basis + bit:
                case '00':
                    state = '0'
                case '01':
                    state = '1'
                case '10':
                    state = '+'
                case '11':
                    state = '-'
                case _:
                    state = 'loss'

            self.alice_states += state

            """Secondly, the state prepared by Alice is sent through the quantum channel:"""
            if random.uniform(0, 1) <= self.disturbance_probability:
                change_indicator = str(random.randint(0, 3))

                while state == self.change_states.get(change_indicator):
                    """We keep randomising new state, because in this case we have the probability of getting a NEW
                    state due to disturbances:"""
                    change_indicator = str(random.randint(0, 3))

                received_state = self.change_states.get(change_indicator)

                """In the case of disturbance, for the demonstration purposes we want to remember which states were 
                changed:"""
                self.disturbed_states_indexes.append(index)
            else:
                received_state = state

            """We memorise the state received by Bob:"""
            self.bob_states += received_state

            """Thirdly, Bob performs his measurement - we don't care about the states after the measurement, only
            the corresponding bit values:"""
            second_basis = self.bob_basis[index]

            if received_state == '0' or received_state == '1':  # states were generated with the rectilinear basis
                if second_basis == '0':  # rectilinear
                    state_after_second_measurement = received_state
                else:  # diagonal
                    state_after_second_measurement = random.choice(possible_states)
            else:  # states were generated with the diagonal basis
                if second_basis == '0':  # rectilinear
                    state_after_second_measurement = random.choice(possible_states)
                else:  # diagonal
                    state_after_second_measurement = received_state
            match state_after_second_measurement:
                case '0':
                    self.bob_bits += '0'
                case '1':
                    self.bob_bits += '1'
                case '+':
                    self.bob_bits += '0'
                case '-':
                    self.bob_bits += '1'

        self.add_status_update("Step 2/7 completed.")

    def measure_indicate(self):
        for bit in self.bob_bits:
            if bit == '0' or bit == '1':
                self.bob_measurement_indicators += '1'
            else:
                self.bob_measurement_indicators += '0'

        index = 0

        for indicator in self.bob_measurement_indicators:
            if indicator == '1':
                self.bob_indicated_basis += self.bob_basis[index]
                self.bob_indicated_bits += self.bob_bits[index]
                self.alice_indicated_bits += self.alice_bits[index]
                self.alice_indicated_basis += self.alice_basis[index]
            index += 1

        self.add_status_update("Step 3/7 completed.")

    def sifting(self):
        index = 0
        for basis in self.alice_indicated_basis:
            if basis == self.bob_indicated_basis[index]:
                self.alice_sifted_key += self.alice_indicated_bits[index]
                self.alice_sifted_basis += self.alice_basis[index]
            index += 1

        index = 0
        for basis in self.bob_indicated_basis:
            if basis == self.alice_indicated_basis[index]:
                self.bob_sifted_key += self.bob_indicated_bits[index]
                self.bob_sifted_basis += self.bob_indicated_basis[index]
            index += 1

        self.number_of_published_bits['sifting'] = (len(self.bob_measurement_indicators) +
                                                    len(self.alice_indicated_basis))

        self.add_status_update("Step 4/7 completed.")

    def estimate_error(self):
        if self.error_estimation == bb84.refined_average_error:
            error_estimation_results = bb84.refined_average_error(
                rect_prob=self.rectilinear_basis_prob,
                rect_pub_prob=self.publication_probability_rectilinear,
                diag_pub_prob=self.publication_probability_diagonal,
                alice_bits=self.alice_sifted_key,
                bob_bits=self.bob_sifted_key,
                alice_basis=self.alice_sifted_basis,
                bob_basis=self.bob_sifted_basis
            )
        else:
            error_estimation_results = bb84.naive_error(
                alice_key=self.alice_sifted_key,
                bob_key=self.bob_sifted_key,
                publication_probability=self.publication_probability_rectilinear
            )

        self.error_estimate = error_estimation_results.get('error estimator')
        self.alice_sifted_key_after_error_estimation = error_estimation_results.get('alice key')
        self.bob_sifted_key_after_error_estimation = error_estimation_results.get('bob key')
        self.number_of_published_bits['error estimation'] = error_estimation_results.get('number of published bits')

        self.add_status_update("Step 5/7 completed.")

    def correct_errors(self):
        self.length_of_key_into_cascade = len(self.alice_sifted_key_after_error_estimation)

        for i in range(self.length_of_key_into_cascade):
            """I dynamically create dictionaries with indexes as keys and bits as values"""
            self.alice_cascade[str(i)] = self.alice_sifted_key_after_error_estimation[i]
            self.bob_cascade[str(i)] = self.bob_sifted_key_after_error_estimation[i]

        self.blocks_sizes = bb84.cascade_blocks_sizes(
            qber=self.error_estimate,
            key_length=self.length_of_key_into_cascade,
            n_passes=self.cascade_n_passes)

        for size in self.blocks_sizes:
            try:
                """I calculate how many blocks are in total in this pass"""
                pass_number_of_blocks = int(-1 * np.floor(-1 * self.length_of_key_into_cascade // size))
            except ZeroDivisionError:
                error_message = [
                    self.blocks_sizes,
                    self.pass_number,
                    self.no_quantum_states,
                    self.gain,
                    self.disturbance_probability,
                    self.error_estimate,
                    self.length_of_key_into_cascade,
                    self.rectilinear_basis_prob,
                    self.publication_probability_rectilinear,
                    self.cascade_n_passes
                ]
                print(error_message)
                continue

            alice_pass_parity_list = []
            bob_pass_parity_list = []
            alice_blocks = []
            bob_blocks = []

            for block_index in bb84.cascade_blocks_generator(
                    key_length=self.length_of_key_into_cascade, blocks_size=size):

                alice_block = {}  # a dictionary for a single block for Alice
                bob_block = {}  # a dictionary for a single block for Bob

                for index in block_index:  # I add proper bits to these dictionaries
                    alice_block[str(index)] = self.alice_cascade[str(index)]
                    bob_block[str(index)] = self.bob_cascade[str(index)]

                """I append single blocks created for given indexes to lists of block for this particular 
                CASCADE's pass"""
                alice_blocks.append(alice_block)
                bob_blocks.append(bob_block)

            for i in range(pass_number_of_blocks):
                current_indexes = list(alice_blocks[i].keys())  # same as Bob's
                alice_current_bits = list(alice_blocks[i].values())
                bob_current_bits = list(bob_blocks[i].values())
                alice_bit_values = []
                bob_bit_values = []

                for j in range(len(current_indexes)):
                    alice_bit_values.append(int(alice_current_bits[j]))
                    bob_bit_values.append(int(bob_current_bits[j]))

                alice_pass_parity_list.append(sum(alice_bit_values) % 2)
                bob_pass_parity_list.append(sum(bob_bit_values) % 2)

                if alice_pass_parity_list[i] != bob_pass_parity_list[i]:  # we check if we should perform BINARY

                    binary_results = binary(
                        sender_block=alice_blocks[i],
                        receiver_block=bob_blocks[i],
                        indexes=current_indexes
                    )

                    """Firstly we add the number of exchanged bits during this BINARY performance to the general number
                    of bits exchanged via the public channel.
                    """
                    self.exchanged_bits_counter += binary_results[2]

                    """Secondly we change main dictionary with final results and current blocks for history"""
                    self.bob_cascade[binary_results[1]] = binary_results[0]
                    bob_blocks[i][binary_results[1]] = binary_results[0]

                    """Thirdly we change the error bit in blocks' history
                    We need to perform BINARY on all blocks which we correct in history list
                    history[number of pass][owner][number of block]
                    """
                    if self.pass_number > 0:  # in the first pass of CASCADE there are no previous blocks
                        for n_pass in range(self.pass_number):  # we check all previous passes
                            for n_block in range(
                                    len(self.history[0][n_pass][1])):  # we check all Bob's blocks in each previous pass
                                if binary_results[1] in self.history[n_pass][1][n_block]:
                                    self.history[n_pass][1][n_block] = binary_results[0]

                                    try:
                                        if type(self.history[n_pass][1][n_block]) == str:
                                            indexes = ast.literal_eval(self.history[n_pass][1][n_block])
                                            binary_previous = binary(
                                                sender_block=self.history[n_pass][0][n_block],
                                                receiver_block=self.history[n_pass][1][n_block],
                                                indexes=indexes.keys()
                                            )
                                        elif type(self.history[n_pass][1][n_block]) == dict:
                                            binary_previous = binary(
                                                sender_block=self.history[n_pass][0][n_block],
                                                receiver_block=self.history[n_pass][1][n_block],
                                                indexes=self.history[n_pass][1][n_block].keys()
                                            )
                                    except AttributeError:
                                        error_message = [
                                            self.blocks_sizes,
                                            self.no_quantum_states,
                                            self.gain,
                                            self.disturbance_probability,
                                            self.error_estimate,
                                            self.length_of_key_into_cascade,
                                            self.rectilinear_basis_prob,
                                            self.publication_probability_rectilinear,
                                            self.cascade_n_passes,
                                            "AttributeError for binary_previous"
                                        ]
                                        print(error_message)

                                        file = open("error.txt", "w")
                                        file.write(
                                            '\n' + 'type of history: ' +
                                            str(type(self.history)) +
                                            '\n' +
                                            'type of history[n_pass]: ' +
                                            str(type(self.history[n_pass])) +
                                            '\n' + 'type of history[n_pass][1]: ' +
                                            str(type(self.history[n_pass][1])) +
                                            '\n' + 'type of history[n_pass][1][n_block]: ' +
                                            str(type(self.history[n_pass][1][n_block])) +
                                            '\n' +
                                            str(self.history) +
                                            '\n'
                                        )
                                        file.close()
                                        exit()

                                    self.exchanged_bits_counter += binary_previous[2]
                                    self.bob_cascade[binary_previous[1]] = binary_previous[0]
                                    bob_blocks[i][binary_previous[1]] = binary_previous[0]

            self.history.append([alice_blocks, bob_blocks])
            self.pass_number += 1

            """For the purposes of optimizing CASCADE we check the error rate after each pass:"""
            alice_key_error_check = ''.join(list(self.alice_cascade.values()))
            bob_key_error_check = ''.join(list(self.bob_cascade.values()))

            """Poniższe już można by wbić w osobną funkcję"""
            key_error_rate = 0
            index = 0
            for bit in alice_key_error_check:
                if bit != bob_key_error_check[index]:
                    key_error_rate += 1
                index += 1
            try:
                key_error_rate = key_error_rate / len(alice_key_error_check)
                self.error_rates.append(key_error_rate)  # its length is equivalent to no. CASCADE passes performed
                if key_error_rate < 0.01:  # VERY ARBITRARY!!! + ryzyko odejmowania małych liczb? Co z tym?
                    break  # let's not waste time for more CASCADE passes if there are 'no more' errors
            except ZeroDivisionError:
                error_message = [
                    self.blocks_sizes,
                    self.pass_number,
                    self.no_quantum_states,
                    self.gain,
                    self.disturbance_probability,
                    self.error_estimate,
                    self.length_of_key_into_cascade,
                    self.rectilinear_basis_prob,
                    self.publication_probability_rectilinear,
                    self.cascade_n_passes
                ]
                print(error_message)

        """Time to create strings from cascade dictionaries into corrected keys"""
        self.alice_correct_key = self.alice_correct_key.join(list(self.alice_cascade.values()))
        self.bob_correct_key = self.bob_correct_key.join(list(self.bob_cascade.values()))

        self.add_status_update("Step 6/7 completed.")

    def amplify_privacy(self):
        deleted_bits_counter = 0
        try:
            deletion_prob = self.exchanged_bits_counter / len(self.alice_correct_key)
        except ZeroDivisionError:
            error_message = [
                self.blocks_sizes,
                self.pass_number,
                self.no_quantum_states,
                self.gain,
                self.disturbance_probability,
                self.error_estimate,
                self.length_of_key_into_cascade,
                self.rectilinear_basis_prob,
                self.publication_probability_rectilinear,
                self.cascade_n_passes]
            print(error_message)
            deletion_prob = 0  # no idea how to set it better in such a case

        index = 0

        while deleted_bits_counter < self.exchanged_bits_counter:
            if index == len(self.alice_correct_key):  # just in case we won't delete enough bits in the first 'run'
                index = 0
            if random.uniform(0, 1) <= deletion_prob:  # we "increase" the prob. by < OR =
                self.alice_correct_key = self.alice_correct_key[0: index:] + self.alice_correct_key[index + 1::]
                self.bob_correct_key = self.bob_correct_key[0: index] + self.bob_correct_key[index + 1::]
                deleted_bits_counter += 1

            index += 1

        self.add_status_update("Step 7/7 completed. You may proceed to playing the simulation.")

    def prepare_data(self):
        # TODO: add into 1D/2D list states, basis, bits, comments (for now just "") for both Alice & Bob
        created_states = list(self.alice_states)
        alice_basis = list(self.alice_basis)
        alice_bits = list(self.alice_bits)
        comments = []
        received_states = list(self.bob_states)
        bob_basis = list(self.bob_basis)
        bob_bits = list(self.bob_bits)

        for i in range(self.no_quantum_states):
            match created_states[i]:  # denoting states with bra-ket quantum mechanics notation
                case '0':
                    self.data.append('|0>')
                case '1':
                    self.data.append('|1>')
                case '+':
                    self.data.append('|+>')
                case '-':
                    self.data.append('|->')

            match alice_basis[i]:
                case '0':
                    self.data.append('rectilinear')
                case '1':
                    self.data.append('diagonal')

            self.data.append(alice_bits[i])

            if i in self.disturbed_states_indexes:
                self.data.append(f'there was disturbance in the quantum channel, the state was changed')
            elif received_states[i] == 'L':
                self.data.append('loss in quantum channel')
                self.data.append('N/A')
                self.data.append('N/A')
                self.data.append('N/A')
                continue
            elif created_states[i] == '0' or created_states[i] == '1':
                if alice_basis[i] == '0':
                    self.data.append(f'same state sent')
                else:
                    self.data.append(f'random state sent')
            elif created_states[i] == '+' or created_states[i] == '-':
                if alice_basis[i] == '1':
                    self.data.append(f'same state sent')
                else:
                    self.data.append(f'random state sent')

            match received_states[i]:  # denoting states with bra-ket quantum mechanics notation
                case '0':
                    self.data.append('|0>')
                case '1':
                    self.data.append('|1>')
                case '+':
                    self.data.append('|+>')
                case '-':
                    self.data.append('|->')

            match bob_basis[i]:
                case '0':
                    self.data.append('rectilinear')
                case '1':
                    self.data.append('diagonal')

            self.data.append(bob_bits[i])

        """At this point data is ready to be fed to the table filler for the GUI. As all 7 columns are fit into 
        a one-dimensional list, the loop limit for filling in has to be 7 times larger than the size of a single
        column."""
        self.data_filler = DataFiller(loop_limit=self.no_quantum_states * 7, data=self.data)
        self.data_filler.data_updated.connect(self.update_table)
        self.view.quantum_play_button.clicked.connect(lambda: self.start_data_filling())
        self.view.quantum_pause_button.clicked.connect(lambda: self.pause_data_filling())
        self.view.quantum_fast_forward_button.clicked.connect(lambda: self.toggle_fast_forward())

        print(self.data)

    def post_processing(self):
        pass

    def add_status_update(self, message):
        # Add status update to the list/queue
        self.view.message_list.addItem(message)
        pass

    def _connect_signals_and_slots(self):
        self.view.simulate_button.clicked.connect(lambda: self.simulate())
        print("Execution of connection")
        # self.view.status_update.connect(self._handle_status_update)

    def start_data_filling(self):
        if not self.data_filler.isRunning():
            # self.data_filler.current_step = 0  # Reset the step counter
            self.data_filler.start()
        else:
            self.data_filler.paused = False

    def pause_data_filling(self):
        self.data_filler.paused = not self.data_filler.paused
        self.data_filler.fast_forward = not self.data_filler.fast_forward

    def toggle_fast_forward(self):
        self.data_filler.fast_forward = not self.data_filler.fast_forward
        self.start_data_filling()

    def update_table(self, data):
        # TODO: create proper index numeration
        row = self.data_filler.current_step // 7
        column = self.data_filler.current_step % 7

        print("Updating table. Cords = ({},{})".format(row, column))
        self.view.quantum_table.setItem(
            row,
            column,
            QTableWidgetItem(data)
        )
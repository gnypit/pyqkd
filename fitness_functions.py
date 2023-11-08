import numpy as np

from bb84 import simulation_bb84


def fitness_negative_time(self, protocol_results):
    """Function assigning to every chromosome fitness value equal to the time of bb84 simulation with
    hyperparameters given as genes multiplied by -1.
    """
    if self.genes.get('length') == 0 or self.genes.get('publication_prob_rectilinear') == 0.0 or self.genes.get(
            'publication_prob_diagonal'):  # punishment for no key or no error
        # estimation
        return -1000000.0
    else:
        """In the first approach we multiply computational time by -1, thus having higher fitness values for
        protocols which were performed faster; in this particular case one may consider fit_value as loss
        """
        fit_value: float = -1.0 * protocol_results.get('global time')

        if protocol_results.get('key length') < 256:
            fit_value -= 10000  # we punish for too short keys; too long ones will have a long computation time

        return fit_value


def fitness_inv(self, protocol_results):  # we use internal time measurement and error rate of the final key
    if self.genes.get('length') != 0 and self.genes.get('pub_prob_rect') != 0.0 and self.genes.get(
            'pub_prob_diag') != 0:

        error_rate = protocol_results.get('error rate')  # is between 0 and 1
        global_time = protocol_results.get('global time')  # in practice, it's significantly greater then 1
        fit_value = 100 * (1 / global_time - error_rate)  # the faster, the better & we want keys that do match

        key_length = protocol_results.get('key length')  # we want 256 bits
        if key_length != 256:
            fit_value -= 10000  # we punish for both too short and too long keys

        return fit_value
    else:
        return -1000000.0


def evaluation(self, protocol_results):
    if self.genes.get('length') != 0 and self.genes.get('pub_prob_rect') != 0.0 and self.genes.get(
            'pub_prob_diag') != 0:
        """Fitness value without normalisation:"""
        fit_value: float = -1.0 * (
                abs(256 - protocol_results.get('key length')) +
                sum(protocol_results.get('computational cost').values()) + protocol_results.get('error rate'))
        return fit_value
    else:
        return -1000000.0


def factored_fit(self, quantum_gain, disturbance_prob):
    """Method assigning to the chromosome a factor of 1 minus final key error rate and number of CASCADE passes
    performed in the simulation of BB84, divided by the max number of passes to be performed -> (1-e) * n/N if
    the final key length is equal to or greater than 256; otherwise it's 0."""
    if self.genes.get('length') != 0 and self.genes.get('pub_prob_rect') != 0.0 and self.genes.get(
            'pub_prob_diag') != 0:
        protocol_results = simulation_bb84(gain=quantum_gain,
                                           alice_basis_length=self.genes.get('length'),
                                           rectilinear_basis_prob=self.genes.get('rect_basis_prob'),
                                           disturbance_probability=disturbance_prob,
                                           publication_probability_rectilinear=self.genes.get('pub_prob_rect'),
                                           publication_probability_diagonal=self.genes.get('pub_prob_diag'),
                                           cascade_n_passes=self.genes.get('no_pass'))

        if protocol_results.get('key length') >= 0:
            cascade_passes = float(protocol_results.get('no. cascade pass.')) / float(self.genes.get('no_pass'))
            bit_effectivness = protocol_results.get('key length') / self.genes.get('length')
            error_rate = protocol_results.get('error rate')
            fit_value = (1 - error_rate) * cascade_passes * bit_effectivness
            return fit_value
        else:  # week punishment for too short keys
            return 0.0  # tym bardziej musi być losowość w krzyżowaniu, fitness values będą się powtarzać
    else:  # strong punishment for
        # no key or no error estimation
        return -1.0

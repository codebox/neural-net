from __future__ import division

import math
import signal
from network import NeuralNet
from data_reader import DataReader
from persistence import Persist

class Helper:
    def __init__(self, file_name, layer_spec):
        if len(layer_spec) < 3:
            raise ValueError('Must specify at least 3 layers')

        self.file_name = file_name
        self.layer_spec = layer_spec
        self.iterations = 1000
        self.learning_rate = 1
        self.regularisation_coefficient = 0
        self.activation_function = self.__sigmoid
        self.test_data_size = 0.3
        self.persistence_id = None
        self.net = None
        self.interrupted = False
        self.err_threshold = None
        self.last_err = None
        self.report_interval = 1000

    def with_iterations(self, iterations):
        self.iterations = iterations
        return self

    def with_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

    def with_regularisation_coefficient(self, regularisation_coefficient):
        self.regularisation_coefficient = regularisation_coefficient
        return self

    def with_activation_function(self, activation_function):
        self.activation_function = activation_function
        return self

    def with_test_data_size(self, test_data_size):
        self.test_data_size = test_data_size
        return self

    def with_persistence(self, persistence_id):
        self.persistence_id = persistence_id
        return self

    def with_error_checking(self, err_threshold):
        self.err_threshold = err_threshold
        return self

    def __sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            if x < 0:
                return 0
            else:
                return 1

    def __get_error(self):
        total_err = 0
        inputs  = self.reader.training_input_values
        outputs = self.reader.training_output_values
        for i, o in zip(inputs, outputs):
            o_net = self.net.calculate(i)
            err2 = 0
            for o1, o2 in zip(o, o_net):
                err = o1 - o2
                err2 += err * err
            total_err += math.sqrt(err2)

        return total_err / len(inputs)

    def on_interrupt(self, _1, _2):
        print 'Stopping, please wait...'
        self.interrupted = True

    def get_net(self):
        net = NeuralNet(self.layer_spec, self.activation_function, self.regularisation_coefficient)
        if self.persistence_id:
            persist = Persist(self.persistence_id)
            if persist.exists():
                persist.load(net)
                print 'Loaded network from', persist.get_filename()
        return net

    def test(self):
        print 'Running network against test values...'
        for i, o_data in zip(self.reader.testing_input_values, self.reader.testing_output_values):
            o_net = self.net.calculate(i)
            print '{0} .... {1}'.format(o_data, o_net)

        return self

    def report(self, iteration):
        print "Iteration {:>6} = {:>2.10f}".format(iteration, self.__get_error())

    def train(self):
        required_inputs  = self.layer_spec[0]
        required_outputs = self.layer_spec[-1:][0]

        self.reader = reader = DataReader(open(self.file_name).readlines(), self.test_data_size, required_inputs, required_outputs)

        m = len(reader.training_input_values)

        self.net = net = self.get_net()
        signal.signal(signal.SIGINT, self.on_interrupt)

        print 'Press Ctrl-C at any time to stop working and show results'

        for i in range(self.iterations):
            if i % self.report_interval == 0:
                self.report(i)

            for inputs, outputs in zip(reader.training_input_values, reader.training_output_values):
                net.train(inputs, outputs)

            new_weights = []
            for weight, derivative in zip(net.get_weights(), net.get_derivatives(m)):
                new_weights.append(weight + self.learning_rate * derivative)

            net.set_weights(new_weights)

            if self.interrupted:
                break

            if self.err_threshold != None:
                current_error = self.__get_error()
                if self.last_err == None:
                    self.last_err = current_error
                elif current_error - self.last_err > self.err_threshold:
                    raise ValueError('Error increased beyond threshold, try reducing the learning rate')                
                else:
                    self.last_err = current_error

        if self.persistence_id:
            persist = Persist(self.persistence_id)
            persist.save(self.net)
            print 'Saved network to', persist.get_filename()

        return self

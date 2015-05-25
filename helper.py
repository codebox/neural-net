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
        self.continuous_test = False

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

    def with_continuous_test(self):
        self.continuous_test = True
        return self

    def __sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            if x < 0:
                return 0
            else:
                return 1

    def __test(self, net, test_inputs, test_outputs):
        for i, o in zip(test_inputs, test_outputs):
            ao = net.calculate(i)
            #print "{:>8} = {:>16.8f}".format(float(o), ao)
            print o,ao

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

    def train_and_test(self):
        required_inputs  = self.layer_spec[0]
        required_outputs = self.layer_spec[-1:][0]

        reader = DataReader(open(self.file_name).readlines(), self.test_data_size, required_inputs, required_outputs)

        m = len(reader.training_input_values)

        self.net = net = self.get_net()
        signal.signal(signal.SIGINT, self.on_interrupt)

        print 'Press Ctrl-C at any time to stop working and show results'

        for i in range(self.iterations):
            print 'Iteration ', i
            for inputs, outputs in zip(reader.training_input_values, reader.training_output_values):
                net.train(inputs, outputs)

            new_weights = []
            for weight, derivative in zip(net.get_weights(), net.get_derivatives(m)):
                new_weights.append(weight + self.learning_rate * derivative)

            net.set_weights(new_weights)

            if self.interrupted:
                break

            if self.continuous_test:
                self.__test(net, reader.testing_input_values, reader.testing_output_values)

        self.__test(net, reader.testing_input_values, reader.testing_output_values)

        if self.persistence_id:
            persist = Persist(self.persistence_id)
            persist.save(self.net)
            print 'Saved network to', persist.get_filename()

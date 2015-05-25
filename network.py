from axon import Axon
from layer import InputLayer, HiddenLayer, OutputLayer
from node import HasInputAxonsMixin, InputNode, HasOutputAxonsMixin, BiasNode
import random

class NeuralNet:
    def __init__(self, layer_spec, activation_fn, lambda_term):
        if len(layer_spec) < 3:
            raise ValueError("Must specify at least 3 layers")

        for l in layer_spec:
            if l < 1:
                raise ValueError("All layers must have non-zero size")

        self.activation_fn = activation_fn
        self.lambda_term   = lambda_term
        self.__init_layers(layer_spec)
        self.__init_weights()

    def __init_layers(self, layer_spec):
        self.layers = []
        last_index = len(layer_spec) - 1
        for i, size in enumerate(layer_spec):
            if i == 0:
                self.layers.append(InputLayer(size, self.activation_fn))
            elif i == last_index:
                self.layers.append(OutputLayer(size, self.activation_fn))
            else:
                self.layers.append(HiddenLayer(size, self.activation_fn))

        for i in range(len(self.layers) - 1):
            self.__join_layers(self.layers[i], self.layers[i+1])

    def __init_weights(self):
        def set_random_weight(a):
            a.weight = self.__get_random_weight()

        self.__for_each_axon(set_random_weight)

    def __get_random_weight(self):
        return random.uniform(-1, 1)

    def __join_layers(self, layer1, layer2):
        for n1 in layer1.nodes:
            for n2 in layer2.nodes:
                if isinstance(n2, HasInputAxonsMixin):
                    axon = Axon(n1, n2, 1)
                    n1.output_axons.append(axon)
                    n2.input_axons.append(axon)

    def get_input_layer(self):
        return self.layers[0]

    def get_output_layer(self):
        return self.layers[-1:][0]

    def train(self, input_values, output_values):
        self.__set_input_values(input_values)
        self.__set_output_values(output_values)
        self.__update_axon_error_sums()

    def get_weights(self):
        weights = []
        def append_weight(a):
            weights.append(a.weight)

        self.__for_each_axon(append_weight)

        return weights

    def set_weights(self, weights):
        def set_weight(a):
            a.weight = weights.pop(0)
            a.error_sum = 0

        self.__for_each_axon(set_weight)

    def get_derivatives(self, m):
        derivatives = []

        def get_derivative(a):
            value = a.error_sum / m
            if not isinstance(a.input_node, BiasNode):
                value += self.lambda_term * a.weight
            derivatives.append(value)

        self.__for_each_axon(get_derivative)

        return derivatives

    def calculate(self, inputs):
        self.__set_input_values(inputs)

        return map(lambda n : n.get_output_value(), self.get_output_layer().nodes)

    def __set_input_values(self, values):
        self.__for_each_node(lambda n : n.reset())
        input_nodes = filter(lambda n : isinstance(n, InputNode), self.get_input_layer().nodes)

        if len(values) != len(input_nodes):
            raise ValueError("Wrong number of input values, got % but expected %" % (len(values), len(input_nodes)))
        
        for node, value in zip(input_nodes, values):
            node.input_value = value

    def __set_output_values(self, values):
        output_nodes = self.get_output_layer().nodes
        if len(values) != len(output_nodes):
            raise ValueError("Wrong number of output values, got % but expected %" % (len(values), len(output_nodes)))
            
        for node, value in zip(output_nodes, values):
            node.target_value = value

    def __update_axon_error_sums(self):
        def update_error_sum(a):
            a.error_sum += a.input_node.get_activation() * a.output_node.get_error()

        self.__for_each_axon(update_error_sum)

    def __for_each_node(self, fn):
        for l in self.layers:
            for n in l.nodes:
                fn(n)

    def __for_each_axon(self, fn):
        def for_each_axon_on_node(n):
            if isinstance(n, HasOutputAxonsMixin):
                for a in n.output_axons:
                    fn(a)

        self.__for_each_node(for_each_axon_on_node)

    def __str__(self):
        result = []
        indent = '    '
        def print_layer(l):
            result.append(str(l))
            for n in l.nodes:
                print_node(n)

        def print_node(n):
            result.append(indent + str(n))
            if isinstance(n, HasOutputAxonsMixin):
                for a in n.output_axons:
                    result.append(indent * 2 + str(a))

        for l in self.layers:
            print_layer(l)

        return '\n'.join(result)

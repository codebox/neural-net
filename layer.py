from node import BiasNode, InputNode, OutputNode, HiddenNode, HasOutputAxonsMixin

class Layer:
    def __init__(self, size, activation_fn):
        self.size = size
        self.activation_fn = activation_fn
        self.nodes = []

        self._create_bias_unit()
        self._create_nodes()

    def _create_bias_unit(self):
        self.nodes.append(BiasNode())

    def _create_nodes(self):
        for i in range(self.size):
            self.nodes.append(self.get_node_class()(self.activation_fn))

    def get_node_class(self):
        assert False

    def __str__(self):
        return '{}: Node Count = {}'.format(self.__class__.__name__, len(self.nodes))

class InputLayer(Layer):
    def get_node_class(self):
        return InputNode

class HiddenLayer(Layer):
    def get_node_class(self):
        return HiddenNode

class OutputLayer(Layer):
    def _create_bias_unit(self):
        pass

    def get_node_class(self):
        return OutputNode

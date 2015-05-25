class Node:
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn
        self.activation = None
        self.error = None

    def get_activation(self):
        if not self.activation:
            self.activation = self.activation_fn(self._get_sum_of_inputs())
        return self.activation

    def get_error(self):
        if not self.error:
            self.error = self._calculate_error()
        return self.error

    def reset(self):
        self.error = None
        self.activation = None

    def __str__(self):
        return '{}: Activation = {}'.format(self.__class__.__name__, self.get_activation())

class HasOutputAxonsMixin:
    def __init__(self):
        self.output_axons = []

    def _calculate_error(self):
        error = 0
        for output_axon in self.output_axons:
            a = output_axon.input_node.get_activation()
            error += output_axon.output_node.get_error() * output_axon.weight * a * (1-a)
        return error

class HasInputAxonsMixin:
    def __init__(self):
        self.input_axons = []

    def _get_sum_of_inputs(self):
        return sum(map(lambda a : a.get_output(), self.input_axons))

class BiasNode(Node, HasOutputAxonsMixin):
    def __init__(self):
        Node.__init__(self, None)
        HasOutputAxonsMixin.__init__(self)

    def get_activation(self):
        return 1

class InputNode(Node, HasOutputAxonsMixin):
    def __init__(self, activation_fn):
        Node.__init__(self, activation_fn)
        HasOutputAxonsMixin.__init__(self)
        self.input_value = 0

    def get_activation(self):
        return self.input_value

    def __str__(self):
        return 'InputNode: Input Value = {} Activation = {}'.format(self.input_value, self.get_activation())

class HiddenNode(Node, HasInputAxonsMixin, HasOutputAxonsMixin):
    def __init__(self, activation_fn):
        Node.__init__(self, activation_fn)
        HasInputAxonsMixin.__init__(self)
        HasOutputAxonsMixin.__init__(self)

class OutputNode(Node, HasInputAxonsMixin):
    def __init__(self, activation_fn):
        Node.__init__(self, activation_fn)
        HasInputAxonsMixin.__init__(self)
        self.target_value = None

    def get_output_value(self):
        return self.get_activation()

    def _calculate_error(self):
        return self.target_value - self.get_output_value()

    def __str__(self):
        return 'OutputNode: Activation = {} Target Value = {}'.format(self.get_activation(), self.target_value)
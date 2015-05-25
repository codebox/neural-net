class Axon:
    def __init__(self, input_node, output_node, weight):
        self.input_node  = input_node
        self.output_node = output_node
        self.weight      = weight
        self.error_sum   = 0

    def get_output(self):
        return self.input_node.get_activation() * self.weight

    def __str__(self):
        return 'Axon: Weight = {} Error Sum = {}'.format(self.weight, self.error_sum)
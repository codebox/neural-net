import os.path

class Persist:
    SEPARATOR = ','

    def __init__(self, id):
        self.id = id

    def get_filename(self):
        return self.id + '.net'

    def exists(self):
        return os.path.isfile(self.get_filename())

    def load(self, net):
        with open (self.get_filename(), 'r') as f:
            weights = map(float, f.read().split(Persist.SEPARATOR))
            net.set_weights(weights)

    def save(self, net):
        with open (self.get_filename(), 'w') as f:
            weights = net.get_weights()
            f.write(Persist.SEPARATOR.join(map(str, weights)))

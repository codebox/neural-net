class DataReader(object):
    def __init__(self, lines, test_proportion, required_inputs, required_outputs):
        inputs   = []
        outputs  = []
        rejected = []
        ok_count = 0

        def split_to_floats(txt):
            return map(float, txt.split(','))

        def is_comment(line):
            return line[0] == '#'

        for line in map(str.strip, lines):
            if is_comment(line):
                continue

            line_err = False
            parts = line.split(':')

            if len(parts) != 2:
                line_err = True
            else:
                try:
                    _inputs  = split_to_floats(parts[1])
                    _outputs = split_to_floats(parts[0])

                    if required_inputs == len(_inputs) and required_outputs == len(_outputs):
                        inputs.append(_inputs)
                        outputs.append(_outputs)

                    else:
                        line_err = True

                except ValueError:
                    line_err = True

            if line_err:
                rejected.append(line)
            else:
                ok_count += 1

        t = int(round(ok_count * (1 - test_proportion)))

        self.training_input_values  = inputs[:t]
        self.training_output_values = outputs[:t]
        self.testing_input_values  = inputs[t:]
        self.testing_output_values  = outputs[t:]

        self.accepted_count  = ok_count
        self.rejected_lines  = rejected

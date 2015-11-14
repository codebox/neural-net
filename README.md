# neural-net
This Python utility provides a simple implementation of a Neural Network, and was written mostly as a learning exercise. If you need a neural network for an application where performance is important, then you should use one of the well maintained [open source libraries](http://pybrain.org/) that are available for free online. However, if you interested in learning how neural networks work, or want a very simple implementation to adapt for your own purposes, then this utility may be useful.

The network consists of many nodes or _neurons_ arranged in a series of layers, with each neuron in a given layer being connected to all the neurons in the following layer. Neurons accept one or more input values and use those inputs to calculate a single output value, which is then passed on to all the neurons in the next layer. Neurons are connected together by _axons_ which pass values from one neuron to the next. Axons modify the values that pass through them, the degree to which values are modified is determined by the axon's _weight_ - a numeric value that can change over time.

The network analyses a set of data that you supply, known as the _training set_, which consists of multiple data items or _training examples_. Each training example must contain one or more input values, and one or more output values. The input values from the training set are used as inputs to the first layer of neurons in the network, and the outputs from the last layer of neurons are compared with the training example output values.

The network is initialised by setting the weights of all the axons to random values. Using an iterative process, the network gradually alters these weights so that eventually a given input will yield an output consistent with those contained in the training set.

### Training Data File Format

To use the utility with a training set, the data must be saved in a correctly formatted text file, with each line in the file containing the data for a single training example. Each line must consist of a comma-separated list of output values, followed by a ':', followed by a comma-separated list of input values. The number of input and output values in each training example must match the number of input and output neurons in the network.

A data file representing the XOR function might look like this:

<pre># XOR data
# line format is: <output>:<input 1>,<input 2>
0:0,0
1:1,0
1:0,1
0:1,1</pre>

### Helper Configuration

As well as supplying a training set, you will need to write a few lines of Python code to configure how the network will run. It is recommended that you use the Helper class to do this, which will simplify the use of the network by handling the wiring and instantiation of the other classes, and by providing reasonable defaults for many of the required configuration parameters.

The Helper class has many configuration options, which are documented below. A simple invocation might look something like this:

<pre>Helper('data.txt', [10,10,1]) \
  .with_iterations(100000) \
  .with_learning_rate(5) \
  .with_test_data_size(0.1) \
  .with_error_checking(0.01) \
  .train() \
  .test()</pre>

The constructor of the Helper class requires 2 arguments. The first argument is the name/location of the file containing the training set. The second argument is a list of integer values specifying how many neurons are needed in each layer of the network. The network must have at least 3 layers, in the example above the input layer and hidden layer have 10 neurons each, and the output layer has only 1.

Once the Helper object has been created, it is configured using the following methods:

#### with_iterations

An integer value, defaulting to 1000\. This determines the number of training cycles that will be performed. During a training cycle each of the training examples is processed once by the network, and then the axon weights are adjusted based on the differences between the networks outputs, and the output values in the training set. Higher values will yield more accurate results, but will increase the required running time.

#### with_learning_rate

A numeric value, defaulting to 1\. This method sets the learning rate parameter used when updating the axon weights after each training cycle. Up to a point, higher values will cause the network to converge on the optimal configuration more quickly, however if the value is set too high then it will fail to converge at all, yielding successively larger errors on each iteration. Finding a good learning rate value is largely a matter of experimentation.

#### with_regularisation_coefficient

An integer value, defaulting to '0'. Setting a non-zero regularisation coefficient will have a 'smoothing' effect on the network configuration, yielding larger errors on the training set, but making the network less prone to overfitting and possibly providing better results for new data.

#### with_activation_function

The function used to calculate the output value of a neuron from the sum of its inputs, defaulting to the [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).

#### with_test_data_size

Accepts a single floating point value between 0 and 1, indicating the proportion of the training data that should be reserved for testing the network (by default this will be 30%). This test data will not be used while training the network, allowing you to see how well the resulting configuration performs against new data.

#### with_persistence

Accepts any non-empty string. Setting a value using this method will cause the network's configuration to be saved into a file when processing completes, and will cause that configuration to be loaded again the next time the utility is used with the same identifier. By using a persistence identifier lengthy processing sessions can be broken up into small chunks, and the resulting network configuration can be easily saved or shared.

#### with_error_checking

Accepts any numeric value, referred to as the _error threshold_. This method enables error checking while the network is being trained. When error checking is enabled, the network will calculate the current error at the end of each training cycle, and compare it with the error from the previous cycle. In general, if the network is behaving correctly this error should reduce each time, however if the new error is larger than the previous one by more than the specified threshold then training will halt and an error message will be displayed. The most common cause of increasing error values is a learning rate parameter that is too large. The network calculates error values by determining the mean difference between its output values and the outputs specified in the training set - an error value of '0' indicates perfect correspondence between the two.

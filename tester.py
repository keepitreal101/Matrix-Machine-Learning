# ----------------------
# - read the input data:

import loader
import matrix as mtx

training_data, validation_data, test_data = loader.load_data_wrapper()

training_data = list(training_data)

net = mtx.Matrix([784, 100, 10])
net.learn(training_data, 30, 10, 5.0, test_data)


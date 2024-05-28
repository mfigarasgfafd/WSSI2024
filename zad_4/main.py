import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        self.b = bias
        if weights:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

    def _f(self, x):
        return max(x * 0.1, x)

    def __call__(self, xs):
        return self._f(xs @ self.ws + self.b)

n_inputs = 3
n_hidden_neurons_1 = 4
n_hidden_neurons_2 = 4
n_outputs = 1

perceptron = Neuron(n_inputs)

plt.figure(figsize=(8, 6))

# Input layer
input_layer_x = [0] * n_inputs
input_layer_y = np.linspace(-1, 1, n_inputs)
plt.scatter(input_layer_x, input_layer_y, color='red', marker='s', label='Input Layer')

# Hidden Layer#1
hidden_layer_1_x = [1] * n_hidden_neurons_1
hidden_layer_1_y = np.linspace(-1, 1, n_hidden_neurons_1)
plt.scatter(hidden_layer_1_x, hidden_layer_1_y, color='blue', marker='s', label='Hidden Layer 1')

# Hidden Layer#2
hidden_layer_2_x = [2] * n_hidden_neurons_2
hidden_layer_2_y = np.linspace(-1, 1, n_hidden_neurons_2)
plt.scatter(hidden_layer_2_x, hidden_layer_2_y, color='cyan', marker='s', label='Hidden Layer 2')

# Output layer
output_layer_x = [3] * n_outputs
output_layer_y = np.linspace(-1, 1, n_outputs)
plt.scatter(output_layer_x, output_layer_y, color='green', marker='s', label='Output Layer')

for i in range(n_inputs):
    for j in range(n_hidden_neurons_1):
        plt.plot([input_layer_x[i], hidden_layer_1_x[j]], [input_layer_y[i], hidden_layer_1_y[j]], color='gray', linestyle='--')

for i in range(n_hidden_neurons_1):
    for j in range(n_hidden_neurons_2):
        plt.plot([hidden_layer_1_x[i], hidden_layer_2_x[j]], [hidden_layer_1_y[i], hidden_layer_2_y[j]], color='gray', linestyle='--')

for i in range(n_hidden_neurons_2):
    for j in range(n_outputs):
        plt.plot([hidden_layer_2_x[i], output_layer_x[j]], [hidden_layer_2_y[i], output_layer_y[j]], color='gray', linestyle='--')

plt.xlabel('Layer')
plt.ylabel('Node Index')
plt.title('neuron visualization')
plt.legend()
plt.grid(True)
plt.show()

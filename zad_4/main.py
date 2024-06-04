import numpy as np
import matplotlib.pyplot as plt

# zdefiniowane później w neuronie, tutaj tylko dla wizualizacji
n_inputs = 3
n_hidden_neurons_1 = 4
n_hidden_neurons_2 = 4
n_outputs = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron:
    def __init__(self, x, y):
        self.input = x
        # x = vector, vector size = 3, --> to 4 layer1 neurons 
        self.weights1 = np.random.rand(self.input.shape[1], 4) 
        # 4 layer1 neurons --> to 4 layer2 neurons
        self.weights2 = np.random.rand(4, 4)  
        # 4 layer2 neurons --> to 1 output neuron
        self.weights3 = np.random.rand(4, 1)  
        self.y = y
        self.output = np.zeros(y.shape)
        self.learning_rate = 0.5

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))

    def backprop(self):
        d_weights3 = np.dot(self.layer2.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights2 = np.dot(self.layer1.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))
        d_weights1 = np.dot(self.input.T,  (np.dot(np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += self.learning_rate * d_weights1
        self.weights2 += self.learning_rate * d_weights2
        self.weights3 += self.learning_rate * d_weights3

    def train(self, epochs=10000):
        for epoch in range(epochs):
            self.feedforward()
            self.backprop()

def visualNetwork():
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

def plot_training_results(nn, y):
    plt.figure()
    plt.plot(nn.output, label='Prediction')
    plt.plot(y, label='True values', linestyle='--')
    plt.title('Training Results')
    plt.xlabel('Sample')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 0],
                  [0, 0, 0],
                  [1, 1, 0],
                  [1, 0, 0]])
    
    y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

    nn = Neuron(X, y)

    nn.feedforward()
    print("Initial predictions:")
    print(nn.output)

    nn.train(epochs=8000)

    print("Final predictions after training:")
    nn.feedforward()
    print(nn.output)

    #plot_training_results(nn, y) #uncomment to visualize the training results
    
    visualNetwork() #uncomment to visualize the network
    
    

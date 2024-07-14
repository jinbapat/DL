import numpy as np
from matplotlib import pyplot as plt
def sigmoid(x):
    return 1/(1+np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)
def leaky_relu(x):
    return np.maximum(0.01*x, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)
def plot_activation_functions():
    x = np.linspace(-10, 10, 400)
    x_softmax = np.linspace(-2, 2, 400)
    
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 2, 1)
    plt.plot(x, sigmoid(x))
    plt.title("Sigmoid")

    plt.subplot(3, 2, 2)
    plt.plot(x, tanh(x))
    plt.title("Tanh")

    plt.subplot(3, 2, 3)
    plt.plot(x, relu(x))
    plt.title("ReLU")

    plt.subplot(3, 2, 4)
    plt.plot(x, leaky_relu(x))
    plt.title("Leaky ReLU")

    plt.subplot(3, 2, 5)
    plt.plot(x_softmax, softmax(x_softmax))
    plt.title("Softmax")

    plt.tight_layout()
    plt.show()

plot_activation_functions()

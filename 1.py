import numpy as np
np.random.seed(42)
input_size = 3
output_size = 1
weights = np.random.randn(input_size, output_size)
def hebbian(inp, w):
    return w + np.outer(inp,inp)
def perceptron(inp, tar, w, lr = 0.1):
    pred = np.dot(w, inp)
    error = tar-pred
    return w + lr*error*inp
def delta(inp, tar, w, lr= 0.1):
    pred = np.dot(w, inp)
    error = tar - pred
    return w + lr* np.outer(error, inp)
def correlation(inp, w):
    return w + np.outer(inp,inp)
def out_star(inp, w, lr = 0.1):
    return w + lr*np.outer(inp,inpu)
inp = np.random.randn(input_size)
w = np.random.randn(input_size)
tar = 1
print(hebbian(inp, w))
print(perceptron(inp, tar, w, lr=0.01))
print(delta(inp, tar, w, lr=0.01))

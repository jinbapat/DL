import numpy as np

def perceptron(inputs, weights, bias):
    return 1 if np.dot(inputs, weights) + bias >= 0 else 0

weights = np.array([0.2, 0.4, 0.2])
bias = -0.5

inputs = np.array([1, 1, 1])  # Favorite hero, heroine, Climate
expected_output = 1  

output = perceptron(inputs, weights, bias)

print("Output:", output)
print("Expected Output:", expected_output)
print("Accuracy:", "100%" if output == expected_output else "0%")

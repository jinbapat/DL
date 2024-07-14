import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
def train_model(use_batch = False, use_drop = False):
    model=Sequential([
        Flatten(input_shape=(28,28)),
        Dense(128, activation='relu'),
        BatchNormalization() if use_batch else tf.keras.layers.Lambda(lambda x: x),  # Add BatchNorm if specified
        Dropout(0.3) if use_drop else tf.keras.layers.Lambda(lambda x: x),  # Add Dropout if specified
        Dense(64, activation='relu'),
        BatchNormalization() if use_batch else tf.keras.layers.Lambda(lambda x: x),
        Dropout(0.3) if use_drop else tf.keras.layers.Lambda(lambda x: x),
        Dense(10, activation='softmax')
    ])
    return model
models = [train_model(use_batch=bn, use_drop=do) for bn in [False, True] for do in [False, True]]
histories = [model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
             or model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test)).history
             for model in models]
test_results = [model.evaluate(x_test, y_test)[1] for model in models]
model_names = ["Baseline", "Batch Norm", "Dropout", "Batch Norm + Dropout"]
for name, acc in zip(model_names, test_results):
    print(f"Test Accuracy - {name}: {acc:.4f}")

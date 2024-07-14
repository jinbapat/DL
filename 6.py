import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense
model = Sequential([
    Flatten(input_shape = (32,32,3)),
    Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test,y_test))
pred = model.predict(x_test)
pred_labels = np.argmax(pred, axis=1)
num_samples = 5
for i in range(num_samples):
    print(f"Sample {i+1} : True Label : {y_test[i][0]}, Predicted Label : {pred_labels[i]}")

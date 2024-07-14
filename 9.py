import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0  # Normalize to [0, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension

generator = Sequential([
    Dense(7*7*128, input_dim=100),
    Reshape((7, 7, 128)),
    BatchNormalization(),
    Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh')
])

discriminator = Sequential([
    Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
    LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Flatten(),
    Dense(1, activation='sigmoid')
])

gan = Sequential([generator, discriminator])
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

batch_size = 64
for epoch in range(50):  # Adjust epochs as needed
    for i in range(x_train.shape[0] // batch_size):
        real_images = x_train[i * batch_size: (i + 1) * batch_size]
        labels_real = np.ones((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_images, labels_real)

        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        labels_fake = np.zeros((batch_size, 1))
        d_loss_fake = discriminator.train_on_batch(fake_images, labels_fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        labels_gan = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, labels_gan)

    print(f"Epoch {epoch + 1}, Discriminator Loss: {0.5 * np.add(d_loss_real, d_loss_fake)}, Generator Loss: {g_loss}")

def plot_generated_images(generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_generated_images(generator)

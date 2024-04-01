import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

# Define parameters
img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels)
latent_dim = 100

# Generator model
def build_generator():
    generator = Sequential()
    generator.add(Dense(128 * 16 * 16, input_dim=latent_dim))
    generator.add(Reshape((16, 16, 128)))
    generator.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    generator.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    generator.add(Conv2DTranspose(channels, kernel_size=4, strides=2, padding='same', activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer='adam')
    return generator

# Discriminator model
def build_discriminator():
    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

# Combined model (Generator + Discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

# Plot generated images
def plot_generated_images(generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, img_rows, img_cols, channels)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Build and train the GAN
def train_gan(epochs=10000, batch_size=128, save_interval=1000):
    # Load and preprocess data (not required for this simplified example)
    
    # Build models
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    # Training loop
    for epoch in range(epochs):
        # Generate random noise
        noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
        
        # Generate fake images
        generated_images = generator.predict(noise)
        
        # Get a random batch of real images (not required for this simplified example)
        
        # Combine real and fake images into one array
        X = np.concatenate([generated_images])
        
        # Labels for generated and real data
        y_dis = np.zeros(batch_size)
        y_dis[:batch_size] = 1
        
        # Train discriminator
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y_dis)
        
        # Train generator (via combined GAN model)
        noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gen)
        
        # Print progress and save generated images
        if epoch % save_interval == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
            plot_generated_images(generator)

# Train the GAN
train_gan()

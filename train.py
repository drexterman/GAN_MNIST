import numpy as np
from models.gan import build_gan
from utils.data_loader import load_mnist
from utils.image_utils import save_images

# Load data
X_train, y_train = load_mnist()

# Build GAN
latent_dim = 100
img_shape = (28, 28, 1)
generator, discriminator, combined = build_gan(latent_dim, img_shape)

# Training function
def train_gan(epochs, batch_size=128):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        real_labels = y_train[idx].reshape(-1, 1)  # Use real labels as slider values
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_labels = np.random.uniform(0, 9, (batch_size, 1))  # Random slider values
        fake_imgs = generator.predict([noise, fake_labels])

        d_loss_real = discriminator.train_on_batch([real_imgs, real_labels], valid)
        d_loss_fake = discriminator.train_on_batch([fake_imgs, fake_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        slider_values = np.random.uniform(0, 9, (batch_size, 1))  # Random slider values
        g_loss = combined.train_on_batch([noise, slider_values], valid)

        # Print progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

        # Save generated images
        if epoch % 100 == 0:
            save_images(generator, epoch, latent_dim)

# Train the GAN
train_gan(epochs=1000, batch_size=32)
# Save the generator weights
generator.save_weights("generator_weights.h5")
print("Generator weights saved to generator_weights.h5")
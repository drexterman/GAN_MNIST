import matplotlib.pyplot as plt
import numpy as np

def save_images(generator, epoch, latent_dim, r=5, c=5):
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    slider_values = np.linspace(0, 9, r * c).reshape(-1, 1)  # Slider values from 0 to 9
    gen_imgs = generator.predict([noise, slider_values])

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(f"gan_images/mnist_{epoch}.png")
    plt.close()
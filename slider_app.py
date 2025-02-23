import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from models.generator import build_generator

# Load trained generator
latent_dim = 100
generator = build_generator(latent_dim)
generator.load_weights("generator_weights.h5")  # Load trained weights

# Function to generate and display an image based on the slider value
def update_image(val):
    slider_value = slider.val
    noise = np.random.normal(0, 1, (1, latent_dim))
    slider_input = np.array([[slider_value]])
    generated_image = generator.predict([noise, slider_input])
    ax.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.draw()

# Create a slider
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = widgets.Slider(ax_slider, 'Slider', 0, 9, valinit=0)
slider.on_changed(update_image)

# Display initial image
update_image(0)
plt.show()
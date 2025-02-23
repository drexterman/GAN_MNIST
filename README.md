# MNIST GAN with Interactive Slider: Generate Custom Handwritten Digits

This project implements a Generative Adversarial Network (GAN) that generates MNIST-style handwritten digits with an interactive slider control. The system combines deep learning with user interaction to allow real-time generation and manipulation of synthetic handwritten digits.

The GAN architecture consists of a generator and discriminator network trained on the MNIST dataset. What makes this implementation unique is its conditional generation capability - users can control the type of digit generated through a slider interface that accepts values from 0 to 9. The generator takes both random noise and the slider value as input, enabling targeted generation of specific digits while maintaining the natural variation and style of handwritten numbers.

## Repository Structure
```
.
├── gpu.py                  # GPU availability checker utility
├── models/                 # Core model architecture definitions
│   ├── discriminator.py    # Discriminator network implementation
│   ├── gan.py             # GAN model builder and training configuration
│   └── generator.py       # Generator network implementation
├── slider_app.py          # Interactive GUI application for digit generation
├── train.py              # Main training script for the GAN
└── utils/                # Utility functions and helpers
    ├── data_loader.py    # MNIST dataset loading and preprocessing
    └── image_utils.py    # Image generation and saving utilities
```

## Usage Instructions
### Prerequisites
- Python 3.6 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib
- CUDA-capable GPU (recommended for training)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd mnist-slider-gan

# Install required packages
pip install tensorflow numpy matplotlib
```

### Quick Start
1. Train the GAN model:
```bash
python train.py
```

2. Launch the interactive slider application:
```bash
python slider_app.py
```

### More Detailed Examples
#### Training the GAN
```python
from train import train_gan

# Train for 1000 epochs with batch size of 32
train_gan(epochs=1000, batch_size=32)
```

#### Generating Images Programmatically
```python
import numpy as np
from models.generator import build_generator

# Initialize generator
latent_dim = 100
generator = build_generator(latent_dim)
generator.load_weights("generator_weights.h5")

# Generate an image
noise = np.random.normal(0, 1, (1, latent_dim))
slider_value = np.array([[5.0]])  # Generate digit 5
generated_image = generator.predict([noise, slider_value])
```

### Troubleshooting
#### Common Issues
1. GPU Memory Errors
   - Error: `ResourceExhaustedError: OOM when allocating tensor`
   - Solution: Reduce batch size in `train.py`
   ```python
   train_gan(epochs=1000, batch_size=16)  # Reduce from 32 to 16
   ```

2. Image Generation Issues
   - Problem: Generated images are too noisy or unclear
   - Solution: Check if the generator weights are properly loaded
   ```python
   # Verify weights file exists
   import os
   assert os.path.exists("generator_weights.h5")
   ```

#### Debugging
- Enable TensorFlow debug logging:
```python
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
```
- Check GPU availability:
```python
python gpu.py
```

## Data Flow
The system transforms random noise vectors and slider inputs into synthetic MNIST-style digits through a trained GAN architecture.

```
[Noise Vector (100d)] ----\
                          +---> [Generator] ---> [Generated Image (28x28)]
[Slider Value (0-9)] ----/          ^
                                    |
                            [Discriminator Feedback]
```

Key component interactions:
1. Generator receives 100-dimensional noise vector and slider value (0-9)
2. Generator produces 28x28 grayscale images
3. Discriminator evaluates generated images with corresponding slider values
4. Training process alternates between generator and discriminator updates
5. Slider app provides real-time control over generation process
6. Image utils handle saving and visualization of generated samples
7. Data loader provides normalized MNIST data for training
#  X-ray Image Denoising using CNN Autoencoder

##  Overview
This project demonstrates how to use a **Convolutional Autoencoder** (a type of CNN) for **image denoising** on X-ray images.  
The model learns to reconstruct clean images from noisy or corrupted inputs, effectively removing noise and improving medical image quality.
An autoencoder is a neural network that learns to compress and then reconstruct data. 
In this project:
X-ray images are preprocessed (grayscale, resized to 128×128, normalized).
The autoencoder learns to **reconstruct X-ray images** from a compressed representation.

**Reconstruction can be used for anomaly detection, feature extraction, or image compression**
---

## Features
- Uses **Convolutional Neural Networks (CNNs)** for feature extraction.
- Built as an **Autoencoder** for unsupervised learning.
- Automatically reconstructs images to remove noise.
- Visualizes **original vs reconstructed images** using Matplotlib.

---

##  Tech Stack
- **Language:** Python  
- **Libraries:**
  - TensorFlow / Keras
  - NumPy
  - OpenCV
  - Matplotlib
  - Scikit-learn

---

## Autoencoder Architecture
Input: 128x128x1 grayscale image
Encoder:
- Conv2D(32, 3x3, relu, padding='same')
- MaxPooling2D(2x2, padding='same')
- Conv2D(16, 3x3, relu, padding='same')
- MaxPooling2D(2x2, padding='same')

Decoder:
- Conv2D(16, 3x3, relu, padding='same')
- UpSampling2D(2x2)
- Conv2D(32, 3x3, relu, padding='same')
- UpSampling2D(2x2)
- Conv2D(1, 3x3, sigmoid, padding='same')


Loss: binary_crossentropy

Optimizer: Adam

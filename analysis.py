import os
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

#  Set the correct dataset path
dataset_path = " ./chest_xray/train/"

# ✅ Load all images (both NORMAL and PNEUMONIA)
image_paths = glob.glob(os.path.join(dataset_path, "*/*.jpeg")) + \
              glob.glob(os.path.join(dataset_path, "*/*.jpg")) + \
              glob.glob(os.path.join(dataset_path, "*/*.png"))

print(f"Total images found: {len(image_paths)}")

images = []
for img_path in image_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        images.append(img)

x = np.array(images).reshape(-1, 128, 128, 1)
print(f"Shape of dataset: {x.shape}")

#  Split into train/test sets
x_train, x_test = train_test_split(x, test_size=0.2, random_state=42)

#  Build Autoencoder model
input_img = Input(shape=(128, 128, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, output)
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

#  Train
autoencoder.fit(x_train, x_train, epochs=10, batch_size=8, validation_data=(x_test, x_test))

#  Predict and visualize results
decoded = autoencoder.predict(x_test)

for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i].reshape(128, 128), cmap='gray')
    plt.axis('off')
    plt.title("Original")
    plt.subplot(2, 5, i+6)
    plt.imshow(decoded[i].reshape(128, 128), cmap='gray')
    plt.axis('off')
    plt.title("Reconstructed")

plt.show()

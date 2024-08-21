import os
import cv2
import numpy as np
from scipy import signal
from random import choice

def add_random_noise(image):
    # Generate random noise
    noise = np.random.normal(loc=0, scale=25, size=image.shape).astype(np.uint8)
    # Add noise to image
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_pink_noise(image):
    def pink_noise(shape):
        if len(shape) == 2:  # Grayscale image
            rows, cols = shape
            ch = 1
        else:  # Color image
            rows, cols, ch = shape

        # Create pink noise
        noise = np.random.randn(rows, cols)
        noise = np.fft.fft2(noise)
        freq = np.fft.fftfreq(rows) ** 2 + np.fft.fftfreq(cols) ** 2
        freq = np.sqrt(freq)
        pink = np.fft.ifft2(noise / (freq[:, None] + 1e-10)).real
        pink -= pink.min()
        pink /= pink.max()
        pink = (pink * 255).astype(np.uint8)

        # Expand dimensions of noise to match image channels
        if ch > 1:
            noise = np.stack([pink] * ch, axis=-1)
        else:
            noise = np.expand_dims(pink, axis=-1)
        
        return noise

    noise = pink_noise(image.shape)
    # Resize pink noise to match the image dimensions
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_gaussian_noise(image):
    row, col = image.shape[:2]
    ch = 1 if len(image.shape) == 2 else image.shape[2]
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    # Add noise to image
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    noise_type = choice([add_random_noise, add_pink_noise, add_gaussian_noise])
    noisy_image = noise_type(image)
    cv2.imwrite(output_path, noisy_image)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        output_class_path = os.path.join(output_folder, class_folder)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
        for filename in os.listdir(class_path):
            input_image_path = os.path.join(class_path, filename)
            output_image_path = os.path.join(output_class_path, filename)
            process_image(input_image_path, output_image_path)


if __name__ == "__main__":
    input_base_folder = "C:\\Users\\PC\\Desktop\\lisnen_data\\Melspectrograms\\Melspectrograms"  # Update with the actual path
    output_base_folder = "C:\\Users\\PC\\Desktop\\lisnen_data\\noise_melspectrograms" # Update with the desired output path
    process_folder(input_base_folder, output_base_folder)

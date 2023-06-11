from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt


# Function to quantize an image with a given number of bits
def quantize_image(image, bits):
    levels = 2 ** bits
    quantized = np.floor(image / 256 * levels) * (256 / levels)
    return quantized.astype(np.uint8)


# Function to perform DPCM encoding on an image
def dpcm_encode(image):
    width, height = image.shape
    encoded = np.zeros_like(image, dtype=np.int32)

    # Predict the first pixel as is
    encoded[0, 0] = image[0, 0]

    # Perform DPCM encoding
    for i in range(1, width):
        encoded[i, 0] = image[i, 0] - image[i - 1, 0]

    for i in range(width):
        for j in range(1, height):
            encoded[i, j] = image[i, j] - image[i, j - 1]

    return encoded


# Function to perform DPCM decoding on an encoded image
def dpcm_decode(encoded):
    width, height = encoded.shape
    decoded = np.zeros_like(encoded, dtype=np.uint8)

    # Decode the first pixel as is
    decoded[0, 0] = encoded[0, 0]

    # Perform DPCM decoding
    for i in range(1, width):
        decoded[i, 0] = encoded[i, 0] + decoded[i - 1, 0]

    for i in range(width):
        for j in range(1, height):
            decoded[i, j] = encoded[i, j] + decoded[i, j - 1]

    return decoded


# Load the grayscale image
image = Image.open('pic.png').convert('L')
image_array = np.array(image)

# Define the quantization levels
quantizers = [1, 2, 4, 8]

# Perform encoding and decoding for different quantizers
reconstructed_images = []
psnr_values = []
ssim_values = []

for bits in quantizers:
    # Quantize the image
    quantized_image = quantize_image(image_array, bits)

    # Encode the quantized image
    encoded_image = dpcm_encode(quantized_image)

    # Decode the encoded image
    decoded_image = dpcm_decode(encoded_image)

    # Calculate PSNR and SSIM
    psnr = peak_signal_noise_ratio(image_array, decoded_image)
    ssim = structural_similarity(image_array, decoded_image)

    reconstructed_images.append(decoded_image)
    psnr_values.append(psnr)
    ssim_values.append(ssim)

# Display the original image and the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_array, cmap='gray')
plt.title("Original Image")
plt.axis('off')

for i, image in enumerate(reconstructed_images):
    plt.subplot(2, 3, i + 2)
    plt.imshow(image, cmap='gray')
    plt.title(f"{quantizers[i]}-bit Quantizer\nPSNR: {psnr_values[i]:.2f}\nSSIM: {ssim_values[i]:.2f}")
    plt.axis('off')

plt.show()

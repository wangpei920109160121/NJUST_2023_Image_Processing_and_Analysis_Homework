import numpy as np
import cv2
import matplotlib.pyplot as plt


def wiener_filter_unknown_snr(img, noise):
    # Compute noise power spectrum
    noise_spectrum = np.fft.fft2(noise)
    noise_power = np.abs(noise_spectrum) ** 2

    # Compute image power spectrum
    img_spectrum = np.fft.fft2(img)
    img_power = np.abs(img_spectrum) ** 2

    # Estimate signal-to-noise ratio (SNR)
    snr_est = np.mean(img_power) / np.mean(noise_power)

    # Apply Wiener filter
    result_spectrum = (1 / (1 + (1 / snr_est) * (noise_power / img_power))) * img_spectrum
    result = np.real(np.fft.ifft2(result_spectrum))

    return result


def wiener_filter_known_snr(img, noise, snr):
    # Compute noise power spectrum
    noise_spectrum = np.fft.fft2(noise)
    noise_power = np.abs(noise_spectrum) ** 2

    # Compute image power spectrum
    img_spectrum = np.fft.fft2(img)
    img_power = np.abs(img_spectrum) ** 2

    # Apply Wiener filter
    result_spectrum = (1 / (1 + (1 / snr) * (noise_power / img_power))) * img_spectrum
    result = np.real(np.fft.ifft2(result_spectrum))

    return result


def wiener_filter_known_acf(img, noise, img_acf, noise_acf):
    # Compute noise power spectrum
    noise_spectrum = np.fft.fft2(noise)
    noise_power = np.abs(noise_spectrum) ** 2

    # Compute image power spectrum
    img_spectrum = np.fft.fft2(img)
    img_power = np.abs(img_spectrum) ** 2

    # Compute power spectrum of image and noise auto-correlation functions
    img_acf_spectrum = np.fft.fft2(img_acf)
    img_acf_power = np.abs(img_acf_spectrum) ** 2

    noise_acf_spectrum = np.fft.fft2(noise_acf)
    noise_acf_power = np.abs(noise_acf_spectrum) ** 2

    # Apply Wiener filter
    result_spectrum = (1 / (1 + (noise_acf_power / img_acf_power) * (noise_power / img_power))) * img_spectrum
    result = np.real(np.fft.ifft2(result_spectrum))

    return result


def compute_autocorrelation(image):
    # Compute the autocorrelation of the image
    autocorr = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(image)) ** 2))
    return autocorr


# Read the original image
img = cv2.imread('pic.png', 0)  # Grayscale image

# Add Gaussian noise to the original image
mean = 0
stddev = 30  # Adjust the standard deviation to control the noise level
noise = np.random.normal(mean, stddev, img.shape).astype(np.uint8)
noisy_img = cv2.add(img, noise)

# Set the SNR value
snr = 20


# Compute the autocorrelation function of the original image and the noisy image
img_acf = compute_autocorrelation(img)
noisy_img_acf = compute_autocorrelation(noisy_img)


# Apply the Wiener filter with unknown SNR
result_unknown_snr = wiener_filter_unknown_snr(img, noisy_img)

# Apply the Wiener filter with known SNR
result_known_snr = wiener_filter_known_snr(img, noisy_img, snr)

# Apply the Wiener filter with known ACF
result_known_acf = wiener_filter_known_acf(img, noisy_img, img_acf, noisy_img_acf)


# Display the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(result_unknown_snr, cmap='gray')
plt.title('Wiener Filter (Unknown SNR)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(result_known_snr, cmap='gray')
plt.title('Wiener Filter (Known SNR)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(result_known_acf, cmap='gray')
plt.title('Wiener Filter (Known ACF)')
plt.axis('off')

plt.tight_layout()
plt.show()

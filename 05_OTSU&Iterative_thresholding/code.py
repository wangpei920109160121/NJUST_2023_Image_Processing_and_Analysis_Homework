import numpy as np
import cv2
import matplotlib.pyplot as plt


def otsu_thresholding(image):
    # Calculate histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Total number of pixels
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate probabilities
    probabilities = hist / total_pixels

    # Calculate cumulative sums
    cumulative_sum = np.cumsum(probabilities)

    # Calculate cumulative means
    cumulative_mean = np.cumsum(probabilities * np.arange(256))

    # Global mean
    global_mean = cumulative_mean[-1]

    # Calculate between-class variance
    variance_between = cumulative_sum * (1 - cumulative_sum)

    # Find the optimal threshold
    threshold = np.argmax(variance_between)

    # Perform thresholding
    thresholded_image = (image > threshold).astype(np.uint8) * 255

    return thresholded_image


def iterative_thresholding(image, initial_threshold):
    # Initialize threshold
    threshold = initial_threshold

    # Iterate until convergence
    while True:
        # Split the image into two classes based on the threshold
        class1 = image[image <= threshold]
        class2 = image[image > threshold]

        # Calculate the mean values of the two classes
        mean1 = np.mean(class1)
        mean2 = np.mean(class2)

        # Update the threshold as the average of the means
        new_threshold = int((mean1 + mean2) / 2)

        # Check for convergence
        if new_threshold == threshold:
            break

        # Update the threshold
        threshold = new_threshold

    # Perform thresholding
    thresholded_image = (image > threshold).astype(np.uint8) * 255

    return thresholded_image


# Load the image
image = cv2.imread('pic.png', 0)  # Grayscale image

# Apply OTSU's method for thresholding
otsu_result = otsu_thresholding(image)

# Apply iterative thresholding
initial_threshold = 128
iterative_result = iterative_thresholding(image, initial_threshold)

# Display the results
plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(otsu_result, cmap='gray')
plt.title('OTSU Thresholding')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(iterative_result, cmap='gray')
plt.title('Iterative Thresholding')
plt.axis('off')

plt.tight_layout()
plt.show()

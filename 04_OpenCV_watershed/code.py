import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def watershed(image):
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # MorphologyEx Operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(mb, kernel, iterations=3)

    # Distance Transform
    dist = cv.distanceTransform(mb, cv.DIST_L2, 5)
    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)

    # Watershed Transform
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers=markers)
    result = image.copy()
    result[markers == -1] = [0, 0, 255]

    return result


img = cv.imread("pic.png")

# Apply Watershed transformation
result = watershed(img)

# Convert BGR images to RGB for displaying with matplotlib
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

# Create subplots for original image and result
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(result_rgb)
axes[1].set_title('Result')
axes[1].axis('off')

plt.tight_layout()
plt.show()

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("pic.png")
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grayImg_height = grayImg.shape[0]
grayImg_width = grayImg.shape[1]

newImg_move = np.zeros((grayImg_height, grayImg_width), np.uint8)
newImg_reverse = np.zeros((grayImg_height, grayImg_width), np.uint8)
newImg_increase = np.zeros((grayImg_height, grayImg_width), np.uint8)
newImg_decrease = np.zeros((grayImg_height, grayImg_width), np.uint8)
newImg_piecewise = np.zeros((grayImg_height, grayImg_width), np.uint8)
newImg_nonlinear = np.zeros((grayImg_height, grayImg_width), np.uint8)

# Gray Level Shift
for i in range(grayImg_height):
    for j in range(grayImg_width):
        if int(grayImg[i, j] + 50) > 255:
            gray = 255
        else:
            gray = int(grayImg[i, j] + 50)

        newImg_move[i, j] = np.uint8(gray)

# DB=DA*1.5 Contrast Enhancement
for i in range(grayImg_height):
    for j in range(grayImg_width):
        if int(grayImg[i, j] * 1.5) > 255:
            gray = 255
        else:
            gray = int(grayImg[i, j] * 1.5)
        newImg_increase[i, j] = np.uint8(gray)

# DB=DA*0.8 Contrast Reduction
for i in range(grayImg_height):
    for j in range(grayImg_width):
        gray = int(grayImg[i, j]) * 0.8
        newImg_decrease[i, j] = np.uint8(gray)

# DB=255-DA Grayscale Inverse
for i in range(grayImg_height):
    for j in range(grayImg_width):
        gray = 255 - int(grayImg[i, j])
        newImg_reverse[i, j] = np.uint8(gray)


# Piecewise
def piecewise(img, x1, y1, x2, y2):
    lut = np.zeros(256)
    for i in range(256):
        if i < x1:
            lut[i] = (y1 / x1) * i
        else:
            lut[i] = (y2 - y1) / (x2 - x1) * (i - x1) + y1

    newImg_piecewise = cv.LUT(img, lut)
    newImg_piecewise = np.uint8(newImg_piecewise + 0.5)
    return newImg_piecewise


rMin = grayImg.min()
rMax = grayImg.max()
r1, s1 = rMin, 0
r2, s2 = rMax, 255
newImg_piecewise = piecewise(grayImg, r1, s1, r2, s2)


# Nonlinear
def log(c, img):
    newImg_nonlinear = c * np.log(1.0 + img)
    newImg_nonlinear = np.uint8(newImg_nonlinear + 0.5)
    return newImg_nonlinear


newImg_nonlinear = log(30, grayImg)

imgs = [grayImg, newImg_move, newImg_increase, newImg_decrease, newImg_reverse, newImg_piecewise, newImg_nonlinear]
titles = ["Original Image", "Gray Level Shift", "Contrast Enhancement", "Contrast Reduction", "Grayscale Inverse",
          "Piecewise", "Nonlinear"]

plt.figure(figsize=(12, 8))

for i in range(7):
    img0 = plt.subplot(3, 3, i + 1)
    img0.set_title(titles[i])
    plt.imshow(imgs[i], cmap="gray")
    plt.axis('off')

plt.show()

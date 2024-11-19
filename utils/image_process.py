import cv2
import numpy as np
from scipy import ndimage


def detect_edge(img, threshold):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    _, threshold_gradient = cv2.threshold(normalized_gradient, threshold, 1.0, cv2.THRESH_TOZERO)
    return threshold_gradient


def grayscale(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return new_img


def create_gaussian_kernel(size, sigma, direction):
    kernel = cv2.getGaussianKernel(size, sigma, cv2.CV_32F)
    if direction == 0:
        return kernel
    elif direction == 1:
        return kernel.T


def create_dog_kernel(size, sigma_c, sigma_s, rho, direction):
    kernel1 = create_gaussian_kernel(size, sigma_c, direction)
    kernel2 = create_gaussian_kernel(size, sigma_s, direction)
    dog_kernel = kernel1 - rho * kernel2
    if direction == 0:
        return dog_kernel
    elif direction == 1:
        return dog_kernel.T


def trim_image(img, middle_point, width, height):
    new_img = np.zeros((height, width), np.uint8)
    start_point = np.array([middle_point[0] - (width >> 1), middle_point[1] - (height >> 1)])
    for i in range(height):
        for j in range(width):
            if start_point[0] < 0 or start_point[1] < 0 or start_point[0] >= img.shape[1] or start_point[1] >= img.shape[0]:
                new_img[i, j] = 0
            else:
                new_img[i, j] = img[*start_point]
    return new_img




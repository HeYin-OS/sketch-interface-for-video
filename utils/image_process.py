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


def vertical_gaussian(img, sigma):
    new_img = ndimage.gaussian_filter1d(img, sigma=sigma, axis=0)
    return new_img


def DOG_function(img, sigma_c, sigma_s, rho):
    gaussian1 = ndimage.gaussian_filter1d(img, sigma=sigma_c)
    gaussian2 = ndimage.gaussian_filter1d(img, sigma=sigma_s)
    new_img = gaussian1 - rho * gaussian2
    return new_img

import cv2
import numpy
import numpy as np
import taichi
import taichi as ti
import taichi.math as tm
from PIL import Image


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
            if start_point[0] < 0 or start_point[1] < 0 or start_point[0] >= img.shape[1] or start_point[1] >= \
                    img.shape[0]:
                new_img[i, j] = 0
            else:
                new_img[i, j] = img[*start_point]
    return new_img


def getDpi(imgUrl):
    img = Image.open(imgUrl)
    dpi = img.info.get('dpi', (96, 96))
    return dpi


def mm_to_pixels(mm, imgUrl):
    inches = mm / 25.4
    dpi = getDpi(imgUrl)
    pixel_length = dpi[0] * inches
    return pixel_length


def get_total_edge_weight(img_grayscale, x_limit, y_limit, dog_kernel, gaussian_kernel, rs, a, p1, p2, q1, q2):
    # use affine transformation
    m = (q1 + q2) / 2
    modulus = np.linalg.norm(q2 - q1)
    if modulus == 0:
        modulus = 0.00001
    v = (q2 - q1) / modulus  # unit directed vector of p1-p2
    u = np.array([-v[1], v[0]])  # u is perpendicular to v
    affine = np.array([[u[0], v[0], m[0]],
                       [u[1], v[1], m[1]]], dtype=np.float32)
    rows, cols = img_grayscale.shape[:2]
    transformed_image = cv2.warpAffine(img_grayscale, affine, (cols, rows))
    # filter response integral H, use the convolution.
    int_m = np.array(m, dtype=np.int32)
    x_padding = dog_kernel.shape[1] >> 1
    y_padding = gaussian_kernel.shape[1] >> 1
    trimmed_image = trim_image(transformed_image, int_m, x_padding * 2 + x_limit, y_padding * 2 + y_limit)
    conv_x = cv2.filter2D(trimmed_image, -1, dog_kernel)
    conv = cv2.filter2D(conv_x, -1, gaussian_kernel)
    h = np.sum(conv)
    # from H to H~
    if h < 0.0:
        h = 1.0 + np.tanh(h)
    else:
        h = 1.0
    # total edge weight function
    we = np.linalg.norm((p2 - p1) - (q2 - q1)) ** 2 / (rs ** 2) + a * h
    return we


@ti.kernel
def traverse_every_candidates_and_integral_ti(points: ti.template(), candidate_points: ti.template(), gray_image: ti.template(), x_limit: ti.i32, y_limit: ti.i32,
                                              dog_kernel: ti.types.matrix(n=3, m=1, dtype=ti.f64), gaussian_kernel: ti.types.matrix(n=1, m=3, dtype=ti.f64),
                                              r: ti.f64, alpha: ti.f64, weights: ti.template()):
    picking_limit = candidate_points.shape[1]
    for i in range(points.shape[0]):  # for all candidate paths
        for j in range(picking_limit):  # for all candidate points of prev stroke point
            if candidate_points[i, j].x == -1 and candidate_points[i, j].y == -1:  # the end of candidate points
                break
            for k in range(picking_limit):  # for all candidates points of next stroke point
                if candidate_points[i + 1, k].x == -1 and candidate_points[i + 1, k].y == -1:  # the end of candidate points
                    break
                p1 = ti.Vector([0, 0], ti.i32)  # pre-defined
                if i == 0:
                    p1 = candidate_points[i, j]
                else:
                    p1 = points[i - 1]
                p2 = points[i]
                q1 = candidate_points[i, j]
                q2 = candidate_points[i + 1, k]
                weights[i, j, k] = get_total_edge_weight_ti(p1, p2, q1, q2, gray_image, x_limit, y_limit, dog_kernel, gaussian_kernel, r, alpha)


@ti.func
def get_total_edge_weight_ti(p1: ti.types.vector(2, ti.i32), p2: ti.types.vector(2, ti.i32), q1: ti.types.vector(2, ti.i32),
                             q2: ti.types.vector(2, ti.i32), gray_image: ti.template(), x_limit: ti.int32, y_limit: ti.int32,
                             dog_kernel: ti.template(), gaussian_kernel: ti.template(), r: ti.float64, alpha: ti.float64):
    return 0.1

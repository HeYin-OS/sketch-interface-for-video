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
    return dog_kernel


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
    y_padding = gaussian_kernel.shape[0] >> 1
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
def affine_and_integral_ti(points: ti.template(),
                           candidate_points: ti.template(),
                           gray_image: ti.template(),
                           dog_kernel: ti.types.matrix(n=1, m=3, dtype=ti.f64),
                           gaussian_kernel: ti.types.matrix(n=3, m=1, dtype=ti.f64),
                           r: ti.f64, alpha: ti.f64,
                           weights: ti.template(),
                           img_fld: ti.template(),
                           dog_fld: ti.template(),
                           gs_fld: ti.template()):
    picking_limit = candidate_points.shape[1]
    for i in range(points.shape[0]):  # for all candidate paths
        for j in range(picking_limit):  # for all candidate points of prev stroke point
            if candidate_points[i, j].x == -1 or candidate_points[i, j].y == -1:  # the end of candidate points
                break
            for k in range(picking_limit):  # for all candidates points of next stroke point
                if candidate_points[i + 1, k].x == -1 and candidate_points[i + 1, k].y == -1:  # the end of candidate points
                    break
                p1 = ti.Vector([0, 0], ti.f64)  # pre-defined
                if i == 0:
                    p1 = candidate_points[i, j]
                else:
                    p1 = points[i - 1]
                p2 = points[i]
                q1 = candidate_points[i, j]
                q2 = candidate_points[i + 1, k]
                affine_and_trim_ti(i, j, k, q1, q2, gray_image, img_fld)  # get the corresponding color and limit into X Y
                convolution_ti(dog_kernel, gaussian_kernel, dog_fld, gs_fld)  #


@ti.func
def affine_and_trim_ti(i: ti.i32, j: ti.i32, k: ti.i32,
                       q1: ti.types.vector(2, ti.f64),
                       q2: ti.types.vector(2, ti.f64),
                       gray_image: ti.template(),
                       img_fld: ti.template()):
    # middle point
    m = (q1 + q2) / 2
    v = tm.normalize(q2 - q1)
    u = ti.Vector([-v[1], v[0]], dt=ti.f64)
    # affine matrix
    affine = ti.Matrix([[u[0], v[0], m[0]],
                        [u[1], v[1], m[1]],
                        [0.0, 0.0, 1.0]], dt=ti.f64)
    # inverse affine matrix
    affine_inverse = affine.inverse()
    # find the middle point in new coordinate system
    m_new_homo = affine @ ti.Vector([m[0], m[1], 0.0], dt=ti.f64)
    # from the very left top point as initial loop position
    y = img_fld.shape[3]
    x = img_fld.shape[4]
    start_point = ti.Vector([m_new_homo[0] - (x >> 1), m_new_homo[1] - (y >> 1)], dt=ti.i32)
    for ki in range(y):
        for kj in range(x):
            # use inverse affine matrix to find the original coordinate
            org_coord = affine_inverse @ ti.Vector([start_point.x + kj, start_point.y + ki, 0.0], dt=ti.f64)
            # whether the point is in the range of picture
            if 0 <= ti.i32(org_coord.x) < gray_image.shape[1] and 0 <= ti.i32(org_coord.y) < gray_image.shape[0]:
                img_fld[i, j, k, ki, kj] = gray_image[ti.i32(org_coord.y), ti.i32(org_coord.x)]
            else:
                img_fld[i, j, k, ki, kj] = 0.0


@ti.func
def convolution_ti(i: ti.i32, j: ti.i32, k: ti.i32,
                   dog_kernel: ti.types.matrix(n=1, m=3, dtype=ti.f64),
                   gaussian_kernel: ti.types.matrix(n=3, m=1, dtype=ti.f64),
                   dog_fld: ti.template(),
                   gs_fld: ti.template()):
    pass

# @ti.func
# def get_total_edge_weight_ti(p1: ti.types.vector(2, ti.f64), p2: ti.types.vector(2, ti.f64), q1: ti.types.vector(2, ti.f64),
#                              q2: ti.types.vector(2, ti.f64), gray_image: ti.template(), x_limit: ti.int32, y_limit: ti.int32,
#                              dog_kernel: ti.template(), gaussian_kernel: ti.template(), r: ti.float64, alpha: ti.float64):
#     m = (q1 + q2) / 2
#     v = tm.normalize(q2 - q1)
#     u = ti.Vector([-v[1], v[0]])
#     affine = ti.Matrix([[u[0], v[0], m[0]],
#                         [u[1], v[1], m[1]],
#                         [0.0, 0.0, 1.0]], dt=ti.f64)
#     affine_inverse = affine.inverse()
#     int_m = ti.Vector([ti.i32(m[0]), ti.i32(m[1])], dt=ti.i32)
#     convolution_window_width = (dog_kernel.m >> 1) * 2 + x_limit * 2
#     convolution_window_height = (gaussian_kernel.n >> 1) * 2 + y_limit * 2
#     affine_and_convolution_ti(gray_image, affine_inverse, dog_kernel, gaussian_kernel, int_m, convolution_window_width, convolution_window_height)
#     return 0.1
#
#
# @ti.func
# def affine_and_convolution_ti(gray_image: ti.template(), affine_inverse: ti.types.matrix(n=3, m=3, dtype=ti.f64),
#                               dog_kernel: ti.template(), gaussian_kernel: ti.template(),
#                               mid: ti.types.vector(2, ti.i32), width: ti.i32, height: ti.i32):
#     # starting point of the window
#     start_x = mid[0] - width / 2
#     start_y = mid[1] - height / 2
#     # dog filter
#     dog_result = dog_convolution_ti(affine_inverse, dog_kernel, gray_image, height, width, start_x, start_y)
#     print(dog_result)
#     # gaussian
#
#
# @ti.func
# def dog_convolution_ti(affine_inverse, dog_kernel, gray_image, height, width, start_x, start_y):
#     new_height = height - dog_kernel.m + 1
#     new_width = width - dog_kernel.m + 1
#     result = ti.field.zero(dt=ti.f64, n=new_height, m=new_width)
#     for i in range(new_height):
#         for j in range(new_width):
#             small_sum = ti.f64(0.0)
#             for ki in range(dog_kernel.m):
#                 for kj in range(dog_kernel.n):
#                     new_cor_homo = ti.Matrix([[start_x + kj], [start_y + ki], [1.0]], dt=ti.f64)
#                     gray_value = find_corresponding_gray_value_affine_ti(gray_image, affine_inverse, new_cor_homo)
#                     small_sum += gray_value * dog_kernel[kj, ki]
#             result[i, j] = small_sum
#     return result
#
#
# @ti.func
# def find_corresponding_gray_value_affine_ti(gray_image: ti.template(), affine_inverse: ti.types.matrix(n=3, m=3, dtype=ti.f64), cor: ti.types.matrix(n=3, m=1, dtype=ti.f64)):
#     new_cor = affine_inverse @ cor
#     temp = 0.0
#     if new_cor[0, 0] < 0.0 or new_cor[0, 0] > gray_image.shape[1] or new_cor[0, 1] < 0.0 or new_cor[0, 1] > gray_image.shape[0]:
#         temp = 0.0
#     else:
#         temp = gray_image[ti.i32(new_cor[0, 1]), ti.i32(new_cor[0, 0])]
#     return temp

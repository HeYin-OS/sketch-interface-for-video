import numpy as np
import cv2 as cv
import scipy.ndimage as ndi


def test_img_show(window_name, img):
    cv.imshow(window_name, img)


if __name__ == '__main__':
    # image = cv.imread('robot.png')
    # gs_img = cv.imread('robot.png', cv.IMREAD_GRAYSCALE)
    # cv.waitKey(-1)
    # cv.destroyAllWindows()
# 一维数据
    data = np.linspace(-10, 10, 100)

# 应用高斯滤波
    sigma = 2.0
    smoothed_data = ndi.gaussian_filter(6, sigma=sigma)

    print(smoothed_data)

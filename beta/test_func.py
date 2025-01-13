import numpy as np
import cv2
import scipy.ndimage as ndi
import taichi as ti
import taichi.math as tm


def test_img_show(window_name, img):
    cv2.imshow(window_name, img)


def test():
    ti.init(ti.gpu)
    vec1 = ti.Vector([2, 2])
    print(type(vec1))
    vec2 = tm.vec2(2, 2)
    print(type(vec2))
    f1 = ti.Vector.field(2, dtype=ti.f32, shape=(2, 2))
    print(type(f1[0, 0]))


if __name__ == '__main__':
    test()

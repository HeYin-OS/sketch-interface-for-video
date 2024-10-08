import random

import cv2 as cv
import numpy as np
from pypattyrn.creational.singleton import Singleton

from test_func import test_img_show


class LineDrawer(metaclass=Singleton):
    is_LM_holding = False
    img_work_on = None
    img_save_point = None
    windowName = None
    iter = 0
    radius = 0
    b, g, r = 0.0, 0.0, 0.0
    my_lambda = 0.0
    threshold = 0.0
    circle_sampling_time = 0
    circle_sampling_r = 0
    edge_detection_img = None
    coordinate_container = []
    stroke_container = []
    sampling_point_coordinate_container = []  # salient points with maximal gradient magnitude
    candidates_container = []  # candidates set Q a.k.a. the complete bipartite graph

    def __init__(self, radius, b, g, r, iteration, lbd, threshold, circle_sampling_time, circle_sampling_r):
        self.radius = radius
        self.b = b
        self.g = g
        self.r = r
        self.iter = iteration
        self.my_lambda = lbd
        self.threshold = threshold
        self.circle_sampling_time = circle_sampling_time
        self.circle_sampling_r = circle_sampling_r

    def draw_line(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.coordinate_container = []
            self.img_save_point = self.img_work_on.copy()
            self.is_LM_holding = True

        elif event == cv.EVENT_LBUTTONUP:
            self.laplacian_smoothing()  # laplacian optimization
            self.img_work_on = self.img_save_point.copy()  # recover the original canvas
            self.draw_points()  # draw the points
            self.draw_lines()  # draw the connected lines
            cv.imshow(self.windowName, self.img_work_on)
            if len(self.coordinate_container) != 0:  # add this stroke to the container
                self.stroke_container.append(self.coordinate_container)
            self.img_save_point = self.img_work_on.copy()
            self.is_LM_holding = False

        elif event == cv.EVENT_MOUSEMOVE:
            if self.is_LM_holding:
                self.collect_coordinates(x, y)
                self.draw_points()
                self.draw_lines()
                cv.imshow(self.windowName, self.img_work_on)

    def collect_coordinates(self, x, y):
        self.coordinate_container.append([x, y])

    def laplacian_smoothing(self):
        for _ in range(self.iter):
            new_coordinates = self.coordinate_container.copy()

            for i in range(1, len(self.coordinate_container) - 1):
                prev_coordinate = self.coordinate_container[i - 1]
                current_coordinate = self.coordinate_container[i]
                next_coordinate = self.coordinate_container[i + 1]

                new_x = current_coordinate[0] + self.my_lambda * (prev_coordinate[0] + next_coordinate[0] - 2 * current_coordinate[0])
                new_y = current_coordinate[1] + self.my_lambda * (prev_coordinate[1] + next_coordinate[1] - 2 * current_coordinate[1])

                new_coordinates[i] = (int(new_x), int(new_y))

            self.coordinate_container = new_coordinates.copy()

    def insert_points(self):
        pass

    def draw_points(self):
        for point in self.coordinate_container:
            cv.circle(self.img_work_on, point, self.radius, [self.b, self.g, self.r], -1)

    def draw_lines(self):
        for i in range(len(self.coordinate_container) - 1):
            cv.line(self.img_work_on,
                    (self.coordinate_container[i][0], self.coordinate_container[i][1]),
                    (self.coordinate_container[i + 1][0], self.coordinate_container[i + 1][1]),
                    [self.b, self.g, self.r], self.radius, lineType=cv.LINE_AA, shift=0)

    def bind_image(self, img):
        self.img_work_on = img
        return self

    def bind_window(self, window):
        self.windowName = window
        return self

    def pre_processing(self):
        self.__edge_detection()
        self.__sampling_point_calculation()
        self.__local_circle_sampling()
        return self

    def __edge_detection(self):
        gray_img = cv.cvtColor(self.img_work_on, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        normalized_gradient = cv.normalize(gradient_magnitude, None, 0, 1, cv.NORM_MINMAX)
        _, threshold_gradient = cv.threshold(normalized_gradient, self.threshold, 1.0, cv.THRESH_TOZERO)
        self.edge_detection_img = threshold_gradient

    def __sampling_point_calculation(self):
        kernel = np.ones((3, 3), np.float64)
        max_filtered = cv.dilate(self.edge_detection_img, kernel)
        coordinate_bool_map = (max_filtered == self.edge_detection_img) & (self.edge_detection_img != 0.0)
        test_img_show("before_sampling", coordinate_bool_map.astype(np.uint8) * 255)
        self.sampling_point_coordinate_container = np.argwhere(coordinate_bool_map)

    def __local_circle_sampling(self):
        max_y, max_x = self.img_work_on.shape[:2]
        p_v_i_x = np.random.randint(0, max_x)
        p_v_i_y = np.random.randint(0, max_y)
        self.candidates_container.append([p_v_i_x, p_v_i_y])

        for point in self.sampling_point_coordinate_container:
            temp_points_container = []
            for i in range(0, self.circle_sampling_time):
                r = self.circle_sampling_r * np.sqrt(random.random())
                theta = random.random() * 2 * np.pi
                temp_point = [int(point[0] + r * np.cos(theta)), int(point[1] + r * np.sin(theta))]
                temp_points_container.append(temp_point)
            self.candidates_container.append(temp_points_container)

        print(self.candidates_container[0])
        print(self.candidates_container[1])

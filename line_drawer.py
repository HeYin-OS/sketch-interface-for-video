import cv2 as cv
import numpy as np
from pypattyrn.creational.singleton import Singleton


#
# Already converted into Class module.
#
# def draw_line(event, x, y, flags, param):
#     global drawing, lMousePressed
#     if event == cv.EVENT_LBUTTONDOWN:
#         lMousePressed = True
#     elif event == cv.EVENT_MOUSEMOVE and lMousePressed:
#         drawing = True
#         cv.circle(param[0], (x, y), param[1], param[2], -1)
#         cv.imshow(param[3], param[0])
#     elif event == cv.EVENT_LBUTTONUP:
#         lMousePressed = False


# drawer class for openCV event slot function
class LineDrawer(metaclass=Singleton):
    __lMouseHolding = False  # continuously pressing LM?
    __image = None
    __img_save_point = None
    __windowName = None
    __iter = 0
    __lambda = 0.0
    coordinate_container = []
    stroke_container = []

    # bind to mouse event of openCV, draw lines
    def draw_line(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.coordinate_container = []
            self.backup_img()
            self.__lMousePressed()

        elif event == cv.EVENT_LBUTTONUP:
            self.laplacian_smoothing()
            self.draw_points()
            if len(self.coordinate_container) != 0:
                self.stroke_container.append(self.coordinate_container)
            self.backup_img()
            self.__lMouseReleased()

        elif event == cv.EVENT_MOUSEMOVE:
            if self.__is_lMousePressed():
                self.collect_coordinates(x, y)
                self.draw_points()

    def collect_coordinates(self, x, y):
        self.coordinate_container.append([x, y])

    def laplacian_smoothing(self):
        for _ in range(self.get_iter()):
            new_coordinates = self.coordinate_container.copy()

            for i in range(1, len(self.coordinate_container) - 1):
                prev_coordinate = self.coordinate_container[i - 1]
                current_coordinate = self.coordinate_container[i]
                next_coordinate = self.coordinate_container[i + 1]

                new_x = current_coordinate[0] + self.get_lambda() * (prev_coordinate[0] + next_coordinate[0] - 2 * current_coordinate[0])
                new_y = current_coordinate[1] + self.get_lambda() * (prev_coordinate[1] + next_coordinate[1] - 2 * current_coordinate[1])

                new_coordinates[i] = (int(new_x), int(new_y))

            self.coordinate_container = new_coordinates.copy()

    def insert_points(self):

        pass

    def draw_points(self):
        self.restore_img()
        for point in self.coordinate_container:
            cv.circle(self.get_image(), point, self.get_radius(), self.get_bgr(), -1)
        cv.imshow(self.get_window(), self.get_image())

    # creator
    def __init__(self, radius, b, g, r, iteration, lbd):
        self.__radius = radius
        self.__b = b
        self.__g = g
        self.__r = r
        self.__iter = iteration
        self.__lambda = lbd

    # public func
    def set_radius(self, radius):
        self.__radius = radius
        return self

    def set_blue(self, b):
        self.__b = b
        return self

    def set_green(self, g):
        self.__g = g
        return self

    def set_red(self, r):
        self.__r = r
        return self

    def set_color(self, b, g, r):
        self.__r = r
        self.__g = g
        self.__b = b
        return self

    def bind_image(self, img):
        self.__image = img
        return self

    def bind_window(self, window):
        self.__windowName = window
        return self

    def get_image(self):
        return self.__image

    def get_window(self):
        return self.__windowName

    def get_radius(self):
        return self.__radius

    def get_bgr(self):
        return [self.__b, self.__g, self.__r]

    def get_checkpoint(self):
        return self.__img_save_point

    def get_iter(self):
        return self.__iter

    def get_lambda(self):
        return self.__lambda

    def backup_img(self):
        self.__img_save_point = self.get_image().copy()

    def restore_img(self):
        self.__image = self.get_checkpoint().copy()

    # private func

    def __is_lMousePressed(self):
        return self.__lMouseHolding

    def __lMousePressed(self):
        self.__lMouseHolding = True
        return self

    def __lMouseReleased(self):
        self.__lMouseHolding = False
        return self

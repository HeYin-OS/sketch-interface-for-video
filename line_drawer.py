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
    # private variable
    __drawing = False  # is drawing now?
    __lMouseHolding = False  # is continuously pressing L mouse?
    __image = None
    __windowName = None
    coordinate_container = None
    stroke_container = []
    stroke = 0

    # bind to mouse event of openCV, draw lines
    def draw_line(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.stroke += 1
            self.coordinate_container = np.empty((0, 2), dtype=int)
            self.__lMousePressed()
        elif event == cv.EVENT_LBUTTONUP:
            self.stroke_container.append(self.coordinate_container)
            print(self.stroke_container)
            self.__lMouseReleased()
        elif event == cv.EVENT_MOUSEMOVE:
            self.__start_drawing()
            if self.__is_lMousePressed():
                self.collect_coordinates(x, y)
                self.draw_circle()

    def collect_coordinates(self, x, y):
        self.coordinate_container = np.append(self.coordinate_container, np.array([[x, y]]), axis=0)

    def draw_circle(self):
        for x, y in self.coordinate_container:
            cv.circle(self.get_image(), (x, y), self.get_radius(), self.get_bgr(), -1)
            cv.imshow(self.get_window(), self.get_image())

    # creator
    def __init__(self, radius, b, g, r):
        self.__radius = radius
        self.__b = b
        self.__g = g
        self.__r = r

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

    # private func
    def __is_drawing(self):
        return self.__drawing

    def __is_lMousePressed(self):
        return self.__lMouseHolding

    def __start_drawing(self):
        self.__drawing = True
        return self

    def __end_drawing(self):
        self.__drawing = False
        return self

    def __lMousePressed(self):
        self.__lMouseHolding = True
        return self

    def __lMouseReleased(self):
        self.__lMouseHolding = False
        return self

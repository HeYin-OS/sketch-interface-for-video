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
    __gs_img = None
    __ed_img = None
    coordinate_container = []
    stroke_container = []

    # bind to mouse event of openCV, draw lines
    def draw_line(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.coordinate_container = []
            self.backup_img()
            self.__lMousePressed()

        elif event == cv.EVENT_LBUTTONUP:
            self.laplacian_smoothing()  # laplacian optimization
            self.restore_img()  # recover the original canvas
            self.draw_points()  # draw the points
            self.draw_lines()  # draw the connected lines
            cv.imshow(self.get_window(), self.get_image())
            if len(self.coordinate_container) != 0:  # add this stroke to the container
                self.stroke_container.append(self.coordinate_container)
            self.backup_img()  # save current status: strokes + image
            self.__lMouseReleased()

        elif event == cv.EVENT_MOUSEMOVE:
            if self.__is_lMousePressed():
                self.collect_coordinates(x, y)
                self.draw_points()
                self.draw_lines()
                cv.imshow(self.get_window(), self.get_image())

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
        for point in self.coordinate_container:
            cv.circle(self.get_image(), point, self.get_radius(), self.get_bgr(), -1)

    def draw_lines(self):
        for i in range(len(self.coordinate_container) - 1):
            cv.line(self.get_image(),
                    (self.coordinate_container[i][0], self.coordinate_container[i][1]),
                    (self.coordinate_container[i + 1][0], self.coordinate_container[i + 1][1]),
                    self.get_bgr(), self.get_radius(), lineType=cv.LINE_AA, shift=0)

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

    def bind_gs_img(self, gs_img):
        self.__gs_img = gs_img
        return self

    def pre_processing(self):
        self.__generate_gs_img()
        self.__edge_detection()
        return self

    def __edge_detection(self):
        grad_x = cv.Sobel(self.get_gs_img(), cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(self.get_gs_img(), cv.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        normalized_gradient = cv.normalize(gradient_magnitude, None, 0, 1, cv.NORM_MINMAX)
        threshold_value = 0.2
        _, threshold_gradient = cv.threshold(normalized_gradient, threshold_value, 1.0, cv.THRESH_BINARY)
        self.__ed_img = threshold_gradient

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

    def get_gs_img(self):
        return self.__gs_img

    def get_ed_img(self):
        return self.__ed_img

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

    def __generate_gs_img(self):
        self.__gs_img = cv.cvtColor(self.get_image(), cv.COLOR_BGR2GRAY)

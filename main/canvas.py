from dataclasses import dataclass

import numpy as np
import cv2
import utils.image_process as ip


@dataclass
class Brush:
    b: float = 0.0
    g: float = 0.0
    r: float = 0.0
    radius: int = 0


class Canvas:
    """
    Manage canvas of line drawer.

    Attributes:
        original_image (np.ndarray): original image
        current_image (np.ndarray): currently working on image
        save_point_image (np.ndarray): lastly saved image
        gradient_image (np.ndarray): gradient image
        grayscale_image (np.ndarray): grayscale image
        my_brush (Brush): customized brush for drawing

    """
    original_image: np.ndarray = None
    current_image: np.ndarray = None
    save_point_image: np.ndarray = None
    gradient_image: np.ndarray = None
    grayscale_image: np.ndarray = None
    my_brush: Brush = None
    current_stroke: list = []
    all_strokes: list = []

    def __init__(self, img, b, g, r, radius):
        self.original_image = img
        self.current_image = self.original_image.copy()
        self.save_point_image = self.original_image.copy()
        self.grayscale_image = ip.grayscale(self.original_image)
        self.my_brush = Brush(b=b, g=g, r=r, radius=radius)

    def calculate_gradient(self, threshold):
        self.gradient_image = ip.detect_edge(self.original_image, threshold)
        return self

    def save_stroke_track(self):
        self.save_point_image = self.current_image.copy()
        return self

    def load_save_point(self):
        self.current_image = self.save_point_image.copy()
        return self

    def collect_stroke_points(self, x, y):
        self.current_stroke.append(np.array([y, x], dtype=np.int32))
        return self

    def add_stroke_history(self):
        if len(self.current_stroke) != 0:
            self.all_strokes.append(self.current_stroke)
        return self

    def draw_lines_real_time(self):
        if len(self.current_stroke) < 2:
            return self
        last_index = len(self.all_strokes) - 1
        cv2.line(self.current_image,
                 (self.current_stroke[last_index - 1][1], self.current_stroke[last_index - 1][0]),
                 (self.current_stroke[last_index][1], self.current_stroke[last_index][0]),
                 [self.my_brush.b, self.my_brush.g, self.my_brush.r],
                 self.my_brush.radius,
                 lineType=cv2.LINE_AA,
                 shift=0)
        return self

    def draw_lines(self):
        for i in range(len(self.current_stroke) - 1):
            cv2.line(self.current_image,
                     (self.current_stroke[i][1], self.current_stroke[i][0]),
                     (self.current_stroke[i + 1][1], self.current_stroke[i + 1][0]),
                     [self.my_brush.b, self.my_brush.g, self.my_brush.r],
                     self.my_brush.radius,
                     lineType=cv2.LINE_AA,
                     shift=0)
        return self

    def complete_draw(self):
        self.current_stroke = []
        return

    def show_current_image_cv2(self, windowName):
        cv2.imshow(windowName, self.current_image)
        return self

    def then(self):
        return self

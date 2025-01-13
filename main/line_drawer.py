import random
import time
from typing import Callable

import colorama
import cv2
import numpy as np
import scipy
import taichi as ti
from pypattyrn.creational.singleton import Singleton

import utils.image_process as ip
import utils.logging as log
import utils.statistics as st
import utils.yaml_reader as yr
from canvas import Canvas
from beta.test_func import test_img_show


@ti.data_oriented
class LineDrawer(metaclass=Singleton):
    """
    Main control class for drawing lines.

    Attributes:
        windowName (str): The name of the window
        quit_key (str): The hotkey to quit the program
        refresh_key (str): The hotkey to refresh the window
        console_on (bool): A bool value that determine whether to display console output
        canvas (Canvas): A instance of Canvas class to manage all images
        gaussian_kernel_y1d (np.ndarray): A gaussian kernel with 3 rows * 1 column
        dog_kernel_x1d (np.ndarray): A FDoG kernel with 1 row * 3 columns
        radius (int): The radius of brush
        b (int): The blue value of brush
        g (int): The green value of brush
        r (int): The red value of brush
        iter (int): Iteration number for laplacian Smoothing
        my_lambda (float): Lambda value for laplacian Smoothing
        picking_limit (int): Limited number of candidate points for each stroke point (Local)
        picking_radius (float): Limited radius of picking candidates (Local)

    """
    windowName: str = None
    quit_key: str = None
    refresh_key: str = None
    console_on: bool = None
    canvas: Canvas = None
    all_canvas: list = []
    gaussian_kernel_y1d: np.ndarray = None
    dog_kernel_x1d: np.ndarray = None
    radius: int = 0
    b: float = 0.0
    g: float = 0.0
    r: float = 0.0
    iter: int = 0
    my_lambda: float = 0.0
    picking_limit: int = 0
    picking_radius: float = 0
    candidates: np.ndarray = None  # pre-computed all salient points in the image, # shape like: (computed-num-of-all-candidates, 2)
    candidates_kd_tree = None  # kd_tree of all candidates, shape like: tree
    weights: dict = {}  # used for weights storage, key like: (ith-of-1st-point, jth-in-front-point-candidates, kth-in-after-point-candidates), value like: float32 or float64 or any-float
    current_candidate_points: np.ndarray = []  # candidate groups of current stroke, shape like: (num-of-stroke-points + 1, controlled-num-of-candidates, 2) and "+ 1" is due to the virtual point
    all_candidates: list = []  # candidates or candidate groups of all strokes

    kernel_size: int = 0
    sigma_m: float = 0.0
    sigma_c: float = 0.0
    sigma_s: float = 0.0
    rho: float = 0.0
    x_limit: int = 0
    y_limit: int = 0
    alpha: float = 0.0
    edge_weight_limit: float = 0.0

    def __init__(self):
        ti.init(ti.gpu)
        self.read_yaml_file()
        if self.console_on:
            colorama.init(autoreset=True)

    def read_yaml_file(self):
        config = yr.read_yaml()
        self.windowName = config["main_settings"]['windowName']
        image = cv2.imread(config["main_settings"]['testFile'])
        if image is None:
            print("No image")
            exit(4)
        self.quit_key = config["main_settings"]['quitKey']
        self.refresh_key = config["main_settings"]['refreshKey']
        self.console_on = config["main_settings"]['consoleOn']
        self.radius = config['brush']['radius']
        self.b = config['brush']['b']
        self.g = config['brush']['g']
        self.r = config['brush']['r']
        self.canvas = Canvas(image, self.b, self.g, self.r, self.radius)
        self.iter = config['laplacian_smoothing']['iter']
        self.my_lambda = config['laplacian_smoothing']['lambda']
        threshold = config['optimization']['local']['threshold']
        self.canvas.calculate_gradient(threshold)
        self.picking_limit = config['optimization']['local']['candidates_limit']
        mm = config['optimization']['local']['radius']
        self.picking_radius = ip.mm_to_pixels(mm, config["main_settings"]['testFile'])
        self.kernel_size = config['optimization']['local']['kernel_size']
        self.sigma_m = config['optimization']['local']['sigma_m']
        self.sigma_c = config['optimization']['local']['sigma_c']
        self.sigma_s = config['optimization']['local']['sigma_s']
        self.rho = config['optimization']['local']['rho']
        self.x_limit = config['optimization']['local']['x_limit']
        self.y_limit = config['optimization']['local']['y_limit']
        self.alpha = config['optimization']['local']['alpha']
        self.edge_weight_limit = config['optimization']['local']['edge_weight_limit']

    def image_pre_process(self):
        self.candidates_calculation()
        # generate kernels
        self.gaussian_kernel_y1d = ip.create_gaussian_kernel(self.kernel_size, self.sigma_m, 0)
        self.dog_kernel_x1d = ip.create_dog_kernel(self.kernel_size, self.sigma_c, self.sigma_s, self.rho, 1)
        return self

    def reload_img(self):
        self.read_yaml_file()
        self.image_pre_process().setup()
        self.weights = {}
        # no need for all candidates to be re-initialized
        # no need for kd_tree to be re-initialized
        self.current_candidate_points = np.array([], dtype=np.int32)
        self.all_candidates = []

    def setup(self):
        cv2.imshow(self.windowName, self.canvas.current_image)  # show the window first time
        cv2.setMouseCallback(f"{self.windowName}", self.draw_line)  # bind to cv2 mouse event
        return self

    holding: bool = None

    def draw_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # event: mouse down
            self.holding = True

        elif event == cv2.EVENT_LBUTTONUP:  # event: mouse up
            self.optimization()
            self.canvas.load_save_point().draw_lines().show_current_image_cv2(self.windowName)
            self.canvas.then()
            self.canvas.add_stroke_history().save_stroke_track().complete_draw()
            self.holding = False

        elif event == cv2.EVENT_MOUSEMOVE:  # event: mouse move with mouse down
            if self.holding:
                self.canvas.collect_stroke_points(x, y)
                self.canvas.then()
                self.canvas.draw_lines_real_time().show_current_image_cv2(self.windowName)

    def optimization(self):
        self.local_optimization()
        self.semi_global_optimization()
        self.global_optimization()

    def local_optimization(self):
        self.search_candidates()
        self.show_candidates_window()
        return self

    def semi_global_optimization(self):
        pass

    def global_optimization(self):
        pass

    def candidates_calculation(self):
        kernel = np.ones((3, 3), np.float64)
        max_filtered = cv2.dilate(self.canvas.gradient_image, kernel)
        coordinate_bool_map = (max_filtered == self.canvas.gradient_image) & (self.canvas.gradient_image != 0.0)
        test_img_show("all candidates", coordinate_bool_map.astype(np.uint8) * 255)
        self.candidates = np.argwhere(coordinate_bool_map)
        self.candidates_kd_tree = scipy.spatial.cKDTree(self.candidates)
        return self

    def search_candidates(self):
        # expand array with -1
        self.current_candidate_points = np.full((len(self.canvas.current_stroke) + 1, self.picking_limit, 2), -1, dtype=np.int32)
        if not self.add_virtual_point():
            return self
        # initialize taichi vector field container of stroke points
        current_stroke_np = np.array(self.canvas.current_stroke, dtype=np.int32)
        current_stroke_ti = ti.Vector.field(2, dtype=ti.f64, shape=len(current_stroke_np))
        current_stroke_ti.from_numpy(current_stroke_np)  # shape: 2 * num_of_stroke_points
        # initialize taichi vector field container of candidate points
        current_candidates_ti = ti.Vector.field(2, dtype=ti.i32, shape=(1 + len(current_stroke_np), self.picking_limit))
        for i in range(current_stroke_ti.shape[0]):
            indices = self.candidates_kd_tree.query_ball_point(self.canvas.current_stroke[i], self.picking_radius, p=2.0, workers=-1)
            random.shuffle(indices)
            minimum = min(len(indices), self.picking_limit)
            for j in range(minimum):  # all candidates point of one point
                self.current_candidate_points[i + 1][j] = self.candidates[indices[j]]
        current_candidates_ti.from_numpy(self.current_candidate_points)  # shape: 2 * (num_of_stroke_points + 1) * candidates_limit
        # initialize taichi field container of gray scaled image
        gray_image_ti = ti.field(dtype=ti.i32, shape=self.canvas.grayscale_image.shape)
        gray_image_ti.from_numpy(self.canvas.grayscale_image)  # shape: image size
        # initialize taichi matrices of two kernels
        gaussian_kernel_ti = ti.Matrix(self.gaussian_kernel_y1d, dt=ti.f64)  # shape: 3 * 1
        dog_kernel_ti = ti.Matrix(self.dog_kernel_x1d, dt=ti.f64)  # shape: 1 * 3
        # initialize taichi field container of weights
        weights = ti.field(ti.f64, shape=(current_stroke_ti.shape[0], self.picking_limit, self.picking_limit))
        weights.fill(0.0)  # shape: num_of_stroke_points * candidates_limit * candidates_limit
        # fields
        # window_x = (dog_kernel_ti.m >> 1) * 2 + self.x_limit * 2 # 10
        # window_y = (gaussian_kernel_ti.n >> 1) * 2 + self.y_limit * 2 # 16
        # convolution_image_field = ti.field(ti.f64, shape=(len(current_stroke_np), self.picking_limit,
        #                                                   self.picking_limit, window_y, window_x))
        # convolution_image_field.fill(0.0)
        # dog_x = window_x - dog_kernel_ti.m + 1 # 8
        # dog_y = window_y - dog_kernel_ti.n + 1 # 16
        # dog_results_field = ti.field(ti.f64, shape=(len(current_stroke_np), self.picking_limit,
        #                                             self.picking_limit, dog_y, dog_x))
        # dog_results_field.fill(0.0)
        # gs_x = dog_x - gaussian_kernel_ti.m + 1 # 8
        # gs_y = dog_y - gaussian_kernel_ti.n + 1 # 14
        # gs_results_field = ti.field(ti.f64, shape=(len(current_stroke_np), self.picking_limit,
        #                                            self.picking_limit, gs_y, gs_x))
        # gs_results_field.fill(0.0)
        # running taichi kernel function
        start = time.time_ns()
        # ti.sync()
        ip.affine_and_integral_ti(current_stroke_ti, current_candidates_ti, gray_image_ti,
                                  dog_kernel_ti, gaussian_kernel_ti, self.picking_radius, self.alpha, weights)
        # ti.profiler.print_scoped_profiler_info()
        # ti.profiler.print_kernel_profiler_info('trace')
        log.printLog(0, f"Integral costs {(time.time_ns() - start) / 1e9}s.", False)
        # st.plot_taichi_data(weights)
        return self

    def show_candidates_window(self):
        temp_img = np.zeros(self.canvas.current_image.shape[:2], dtype=np.uint8)
        for group in self.current_candidate_points:
            for candidate_point in group:
                temp_img[*candidate_point] = 255
        temp_img[-1][-1] = 0
        test_img_show("candidates of current stroke", temp_img)

    def add_virtual_point(self):
        if len(self.canvas.current_stroke) <= 1:  # mouse down and up without doing anything
            return False
        v_i_point = self.__add_virtual_initial_point(0.5)  # add virtual point
        self.current_candidate_points[0][0] = v_i_point
        return True

    def __add_virtual_initial_point(self, coefficient):
        p_0 = self.canvas.current_stroke[0]
        p_1 = self.canvas.current_stroke[1]
        return np.array([int(p_0[0] + (p_0[0] - p_1[0]) * coefficient), int(p_0[1] + (p_0[1] - p_1[1]) * coefficient)], dtype=np.int32)

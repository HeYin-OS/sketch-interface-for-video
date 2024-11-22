import colorama
import cv2
import numpy as np
import scipy
from pypattyrn.creational.singleton import Singleton

import utils.image_process as ip
import utils.logging as log
import utils.yaml_reader as yr
from data.candidate_point import CandidatePoint
from test_func import test_img_show


class LineDrawer(metaclass=Singleton):
    windowName: str = None
    quit_key: str = None
    refresh_key: str = None
    console_on: bool = None

    img_origin: np.ndarray = None
    img_grayscale: np.ndarray = None
    img_work_on: np.ndarray = None
    img_save_point: np.ndarray = None
    img_edge_detection: np.ndarray = None
    gaussian_kernel_y1d: np.ndarray = None
    dog_kernel_x1d: np.ndarray = None

    iter: int = 0
    radius: int = 0
    b: float = 0.0
    g: float = 0.0
    r: float = 0.0
    my_lambda: float = 0.0
    threshold: float = 0.0

    picking_limit: int = 0
    picking_radius: float = 0

    is_LM_holding: bool = False
    current_stroke: list = []
    all_strokes: list = []
    candidates: np.ndarray = None  # [Pre-calculation] all salient points with maximal gradient magnitude in the image
    weights: dict = {}
    current_candidate: list = []  # candidates or candidate groups of current stroke
    current_candidates_kd_tree = None  # kd_tree of all candidates
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
        self.read_yaml_file()
        self.img_work_on = self.img_origin.copy()
        self.img_save_point = self.img_origin.copy()
        if self.console_on:
            colorama.init(autoreset=True)

    def read_yaml_file(self):
        config = yr.read_yaml()
        # main settings initialization
        self.windowName = config["main_settings"]['windowName']
        image = cv2.imread(config["main_settings"]['testFile'])
        if image is None:
            print("No image")
            exit(4)
        self.img_origin = image
        self.quit_key = config["main_settings"]['quitKey']
        self.refresh_key = config["main_settings"]['refreshKey']
        self.console_on = config["main_settings"]['consoleOn']
        # brush settings initialization
        self.radius = config['brush']['radius']
        self.b = config['brush']['b']
        self.g = config['brush']['g']
        self.r = config['brush']['r']
        # laplacian smoothing settings initialization
        self.iter = config['laplacian_smoothing']['iter']
        self.my_lambda = config['laplacian_smoothing']['lambda']
        # local optimization settings initialization
        self.threshold = config['optimization']['local']['threshold']
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

    def reload_img(self):
        self.read_yaml_file()
        self.img_work_on = self.img_origin.copy()
        self.img_save_point = self.img_origin.copy()
        self.image_pre_process().setup()
        weights = {}
        self.current_stroke = []
        self.all_strokes = []
        # no need for all candidates to be initialized
        # no need for kd_tree to be initialized
        self.current_candidate = []
        self.all_candidates = []

    def setup(self):
        cv2.imshow(self.windowName, self.img_work_on)  # show the window first time
        cv2.setMouseCallback(f"{self.windowName}", self.draw_line)  # bind to cv2 mouse event

    def draw_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # event: mouse down
            self.current_stroke = []
            self.img_save_point = self.img_work_on.copy()
            cv2.imshow(self.windowName, self.img_work_on)
            self.is_LM_holding = True

        elif event == cv2.EVENT_LBUTTONUP:  # event: mouse up
            self.optimization()
            self.img_work_on = self.img_save_point.copy()  # recover the original canvas
            self.draw_points()
            self.draw_lines()
            cv2.imshow(self.windowName, self.img_work_on)
            if len(self.current_stroke) != 0:  # add this stroke to the container
                self.all_strokes.append(self.current_stroke)
            self.img_save_point = self.img_work_on.copy()  # save the current changed canvas
            self.is_LM_holding = False

        elif event == cv2.EVENT_MOUSEMOVE:  # event: mouse move with mouse down
            if self.is_LM_holding:
                self.collect_coordinates(x, y)
                self.draw_points()
                self.draw_lines()
                cv2.imshow(self.windowName, self.img_work_on)

    def optimization(self):
        self.local_optimization()
        self.semi_global_optimization()
        self.global_optimization()

    def local_optimization(self):
        ## self.laplacian_smoothing()
        self.search_candidates()
        return self

    def semi_global_optimization(self):
        pass

    def global_optimization(self):
        pass

    def collect_coordinates(self, x, y):  # Don't modify the code.
        self.current_stroke.append(np.array([y, x]))

    def laplacian_smoothing(self):
        for _ in range(self.iter):
            new_coordinates = self.current_stroke.copy()
            for i in range(1, len(self.current_stroke) - 1):
                prev_coordinate = self.current_stroke[i - 1]
                current_coordinate = self.current_stroke[i]
                next_coordinate = self.current_stroke[i + 1]
                new_x = current_coordinate[0] + self.my_lambda * (prev_coordinate[0] + next_coordinate[0] - 2 * current_coordinate[0])
                new_y = current_coordinate[1] + self.my_lambda * (prev_coordinate[1] + next_coordinate[1] - 2 * current_coordinate[1])
                new_coordinates[i] = np.array([new_x, new_y])
            self.current_stroke = new_coordinates.copy()
        return self

    def insert_points(self):
        pass

    def draw_points(self):  # Don't modify the code.
        for point in self.current_stroke:
            cv2.circle(self.img_work_on, [point[1], point[0]], self.radius, [self.b, self.g, self.r], -1)

    def draw_lines(self):  # Don't modify the code.
        for i in range(len(self.current_stroke) - 1):
            cv2.line(self.img_work_on,
                     (self.current_stroke[i][1], self.current_stroke[i][0]),
                     (self.current_stroke[i + 1][1], self.current_stroke[i + 1][0]),
                     [self.b, self.g, self.r], self.radius, lineType=cv2.LINE_AA, shift=0)

    def image_pre_process(self):
        self.img_edge_detection = ip.detect_edge(self.img_origin, self.threshold)
        # test_img_show("edge_detection", self.img_edge_detection)
        self.candidates_calculation()
        self.img_grayscale = ip.grayscale(self.img_origin)
        # test_img_show("grayscale", self.img_grayscale)
        self.gaussian_kernel_y1d = ip.create_gaussian_kernel(self.kernel_size, self.sigma_m, 1)
        # test_img_show("vertical_gaussian", self.img_v_gaussian)
        self.dog_kernel_x1d = ip.create_dog_kernel(self.kernel_size, self.sigma_c, self.sigma_s, self.rho, 1)
        # test_img_show("dog_function", self.img_dog)
        return self

    def candidates_calculation(self):
        kernel = np.ones((3, 3), np.float64)
        max_filtered = cv2.dilate(self.img_edge_detection, kernel)
        coordinate_bool_map = (max_filtered == self.img_edge_detection) & (self.img_edge_detection != 0.0)
        test_img_show("all candidates", coordinate_bool_map.astype(np.uint8) * 255)
        self.candidates = np.argwhere(coordinate_bool_map)
        self.current_candidates_kd_tree = scipy.spatial.cKDTree(self.candidates)
        return self

    def search_candidates(self):
        if len(self.current_stroke) <= 1:  # mouse down and up without doing anything
            return self
        v_i_point = self.__add_virtual_initial_point(0.5)  # virtual point addition
        self.current_candidate = []
        self.current_candidate.append([v_i_point])
        # search the candidate points of current points
        if self.console_on:
            log.printLog(0, "Logging of candidate searching", True)
        for i in range(len(self.current_stroke)):
            temp_group = []
            indices = self.current_candidates_kd_tree.query_ball_point(self.current_stroke[i], self.picking_radius, p=2.0, workers=-1)
            for j in range(len(indices)):
                p_2 = self.current_stroke[i]
                q_2 = self.candidates[indices[j]]
                temp_group.append(q_2)
                # store the weight information
                if i == 0:
                    p_1 = v_i_point
                    q_1 = v_i_point
                    we = ip.get_total_edge_weight(self.img_grayscale, self.x_limit, self.y_limit, self.dog_kernel_x1d, self.gaussian_kernel_y1d,
                                                  self.picking_radius, self.alpha, p_1, p_2, q_1, q_2)
                    self.weights[(0, 0, j)] = we
                else:
                    for k in range(len(self.current_candidate[i])):
                        p_1 = self.current_stroke[i - 1]
                        q_1 = self.current_candidate[i][k]
                        we = ip.get_total_edge_weight(self.img_grayscale, self.x_limit, self.y_limit, self.dog_kernel_x1d, self.gaussian_kernel_y1d,
                                                      self.picking_radius, self.alpha, p_1, p_2, q_1, q_2)
                        self.weights[(i, k, j)] = we
            self.current_candidate.append(temp_group)
            if self.console_on:
                log.printLog(0, f"Stroke Point No.{i} has {len(temp_group)} candidate(s).", False)
        print(self.weights)
        # for stroke_point in self.current_stroke:
        #     temp_group = []
        #     for candidate_point in self.candidates:
        #         if np.linalg.norm(stroke_point - candidate_point) < self.circle_sampling_r:
        #             temp_group.append(CandidatePoint(coordinate=candidate_point))
        #     self.current_candidate.append(temp_group)
        temp_img = np.zeros(self.img_work_on.shape[:2], dtype=np.uint8)
        for group in self.current_candidate:
            for candidate_point in group:
                temp_img[*candidate_point] = 255
        test_img_show("candidates of current stroke", temp_img)
        # self.edge_weight_calculation()
        return self

    def edge_weight_calculation(self):
        for i in range(len(self.current_candidate) - 1):  # from 1st one to 2nd of the last one
            q1_list = self.current_candidate[i]
            q2_list = self.current_candidate[i + 1]
            for j in range(len(q1_list)):  # q_start
                for k in range(len(q2_list)):  # q_end
                    q1 = q1_list[j]
                    q2 = q2_list[k]
                    if i == 0:
                        p_1 = q1.coordinate
                    else:
                        p_1 = self.current_stroke[i - 1]
                    p_2 = self.current_stroke[i]
                    we = ip.get_total_edge_weight(self.img_grayscale, self.x_limit, self.y_limit, self.dog_kernel_x1d, self.gaussian_kernel_y1d,
                                                  self.picking_radius, self.alpha, p_1, p_2, q1, q2)
                    print(we)
        pass

    def __add_virtual_initial_point(self, coefficient):
        p_0 = self.current_stroke[0]
        p_1 = self.current_stroke[1]
        return np.array([int(p_0[0] + (p_0[0] - p_1[0]) * coefficient), int(p_0[1] + (p_0[1] - p_1[1]) * coefficient)])

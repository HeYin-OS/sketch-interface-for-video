import cv2
import numpy
import numpy as np
import scipy
import colorama
from pypattyrn.creational.singleton import Singleton
from datetime import datetime
import utils.yaml_reader as yr
import utils.image_process as ip
from data.candidate_point import CandidatePoint
from test_func import test_img_show


class LineDrawer(metaclass=Singleton):
    quit_key: str = None
    refresh_key: str = None
    console_on: bool = None
    is_LM_holding: bool = False

    img_origin: np.ndarray = None
    img_grayscale: np.ndarray = None
    img_work_on: np.ndarray = None
    img_save_point: np.ndarray = None
    img_edge_detection: np.ndarray = None
    gaussian_kernel_y1d: np.ndarray = None
    dog_kernel_x1d: np.ndarray = None

    windowName: str = None
    iter: int = 0
    radius: int = 0
    b: float = 0.0
    g: float = 0.0
    r: float = 0.0
    my_lambda: float = 0.0
    threshold: float = 0.0

    circle_sampling_time: int = 0
    circle_sampling_r: float = 0

    current_stroke: list = []
    all_strokes: list = []
    candidates: np.ndarray = None  # [Pre-calculation] all salient points with maximal gradient magnitude in the image
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
        self.windowName = config['windowName']
        image = cv2.imread(config['testFile'])
        if image is None:
            print("No image")
            exit(4)
        self.img_origin = image
        self.quit_key = config['quitKey']
        self.refresh_key = config['refreshKey']
        self.console_on = config['consoleOn']
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
        self.circle_sampling_time = config['optimization']['local']['circle_sampling']
        self.circle_sampling_r = config['optimization']['local']['radius']
        self.kernel_size = config['optimization']['local']['kernel_size']
        self.sigma_m = config['optimization']['local']['sigma_m']
        self.sigma_c = config['optimization']['local']['sigma_c']
        self.sigma_s = config['optimization']['local']['sigma_s']
        self.rho = config['optimization']['local']['rho']
        self.x_limit = config['optimization']['local']['X']
        self.y_limit = config['optimization']['local']['Y']
        self.alpha = config['optimization']['local']['alpha']
        self.edge_weight_limit = config['optimization']['local']['edge_weight_limit']

    def reload_img(self):
        self.read_yaml_file()
        self.img_work_on = self.img_origin.copy()
        self.img_save_point = self.img_origin.copy()
        self.image_pre_process().setup()
        self.current_stroke = []
        self.all_strokes = []
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
        self.current_candidate.append([CandidatePoint(coordinate=v_i_point)])
        # search the candidate points of current points
        k = 0
        if self.console_on:
            print(f"[{datetime.utcnow()}]{colorama.Fore.CYAN}[DEBUG]<<<<<<<<            Logging of candidate searching            >>>>>>>>")
        for stroke_point in self.current_stroke:
            temp_group = []
            indices = self.current_candidates_kd_tree.query_ball_point(stroke_point, self.circle_sampling_r, p=2.0, workers=-1)
            for i in indices:
                temp_group.append(CandidatePoint(coordinate=self.candidates[i]))
            self.current_candidate.append(temp_group)
            k += 1
            if self.console_on:
                print(f"[{datetime.utcnow()}]Stroke Point No.{k} has {len(temp_group)} candidate(s).")
        # for stroke_point in self.current_stroke:
        #     temp_group = []
        #     for candidate_point in self.candidates:
        #         if np.linalg.norm(stroke_point - candidate_point) < self.circle_sampling_r:
        #             temp_group.append(CandidatePoint(coordinate=candidate_point))
        #     self.current_candidate.append(temp_group)
        temp_img = np.zeros(self.img_work_on.shape[:2], dtype=np.uint8)
        for group in self.current_candidate:
            for candidate_point in group:
                temp_img[*candidate_point.coordinate] = 255
        test_img_show("candidates of current stroke", temp_img)
        # self.edge_weight_calculation()
        return self

    def edge_weight_calculation(self):
        for i in range(len(self.current_candidate) - 1):  # from 1st to 2nd of the last
            j = -1
            for q_1 in self.current_candidate[i]:  # q_start
                j += 1
                for q_2 in self.current_candidate[i + 1]:  # q_end
                    # use affine transformation
                    m = (q_1.coordinate + q_2.coordinate) / 2
                    modulus = np.linalg.norm(q_2.coordinate - q_1.coordinate)
                    if modulus == 0:
                        modulus = 0.00001
                    v = (q_2.coordinate - q_1.coordinate) / modulus  # unit directed vector of p1-p2
                    u = np.array([-v[1], v[0]])  # u is perpendicular to v
                    affine = np.array([[u[0], v[0], m[0]],
                                      [u[1], v[1], m[1]]], dtype=np.float32)
                    rows, cols = self.img_origin.shape[:2]
                    transformed_image = cv2.warpAffine(self.img_grayscale, affine, (cols, rows))
                    # filter response integral H
                    # use the convolution
                    int_m = np.array(m, dtype=np.int32)
                    padding = self.kernel_size >> 1
                    trimmed_image = ip.trim_image(self.img_grayscale, int_m, padding * 2 + self.x_limit, padding * 2 + self.y_limit)
                    conv_x = cv2.filter2D(trimmed_image, -1, self.gaussian_kernel_y1d)
                    conv = cv2.filter2D(conv_x, -1, self.dog_kernel_x1d)
                    h = numpy.sum(conv)
                    # from H to H~
                    if h < 0.0:
                        h = 1.0 + np.tanh(h)
                    else:
                        h = 1.0
                    # total edge weight function
                    p_1 = q_1.coordinate
                    p_2 = self.current_stroke[0]
                    if i != 0:
                        p_1 = self.current_stroke[i - 1]
                        p_2 = self.current_stroke[i]
                    we = np.linalg.norm((p_1 - p_2) - (q_2.coordinate - q_1.coordinate)) ** 2 + self.alpha * h
                    if i == 0:
                        self.current_candidate[0][0].next_weight_list.append(we)  # in the 0th point of 0th group, update the weight
                    else:
                        self.current_candidate[i][j].next_weight_list.append(we)  # in the jth point of ith group, update the weight
                    pass

    def __add_virtual_initial_point(self, coefficient):
        p_0 = self.current_stroke[0]
        p_1 = self.current_stroke[1]
        return np.array([int(p_0[0] + (p_0[0] - p_1[0]) * coefficient), int(p_0[1] + (p_0[1] - p_1[1]) * coefficient)])

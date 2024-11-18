import cv2
import numpy as np
import scipy
from pypattyrn.creational.singleton import Singleton
import utils.yaml_reader as yr
import utils.image_process as ip
from data.candidate_point import CandidatePoint
from test_func import test_img_show


class LineDrawer(metaclass=Singleton):
    quit_key: str = None
    is_LM_holding: bool = False

    img_origin: np.ndarray = None
    img_grayscale: np.ndarray = None
    img_work_on: np.ndarray = None
    img_save_point: np.ndarray = None
    img_edge_detection: np.ndarray = None
    img_v_gaussian: np.ndarray = None
    img_dog: np.ndarray = None

    windowName: str = None
    iter: int = 0
    radius: int = 0
    b: float = 0.0
    g: float = 0.0
    r: float = 0.0
    my_lambda: float = 0.0
    threshold: float = 0.0

    circle_sampling_time: int = 0
    circle_sampling_r: int = 0

    current_stroke: list = []
    all_strokes: list = []
    candidates: np.ndarray = None  # [Pre-calculation] all salient points with maximal gradient magnitude in the image
    candidates_kd_tree = None  # kd_tree of all candidates
    current_candidate: list = []  # candidates or candidate groups of current stroke
    all_candidates: list = []  # candidates or candidate groups of all strokes

    sigma_m: float = 0.0
    sigma_c: float = 0.0
    sigma_s: float = 0.0
    rho: float = 0.0
    balancing_weight: float = 0.0
    edge_weight_limit: float = 0.0

    def __init__(self):
        self.read_yaml_file()
        self.img_work_on = self.img_origin.copy()
        self.img_save_point = self.img_origin.copy()

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
        self.sigma_m = config['optimization']['local']['sigma_m']
        self.sigma_c = config['optimization']['local']['sigma_c']
        self.sigma_s = config['optimization']['local']['sigma_s']
        self.rho = config['optimization']['local']['rho']
        self.balancing_weight = config['optimization']['local']['balancing_weight']
        self.edge_weight_limit = config['optimization']['local']['edge_weight_limit']

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
        self.img_v_gaussian = ip.vertical_gaussian(self.img_grayscale, self.sigma_m)
        # test_img_show("vertical_gaussian", self.img_v_gaussian)
        self.img_dog = ip.DOG_function(self.img_grayscale, self.sigma_c, self.sigma_s, self.rho)
        # test_img_show("dog_function", self.img_dog)
        return self

    def candidates_calculation(self):
        kernel = np.ones((3, 3), np.float64)
        max_filtered = cv2.dilate(self.img_edge_detection, kernel)
        coordinate_bool_map = (max_filtered == self.img_edge_detection) & (self.img_edge_detection != 0.0)
        test_img_show("all candidates", coordinate_bool_map.astype(np.uint8) * 255)
        self.candidates = np.argwhere(coordinate_bool_map)
        self.candidates_kd_tree = scipy.spatial.cKDTree(self.candidates)
        return self

    def search_candidates(self):
        if len(self.current_stroke) <= 1:  # mouse down and up without doing anything
            return self
        v_i_point = self.__add_virtual_initial_point(0.5)  # virtual point addition
        self.current_candidate = []
        self.current_candidate.append([CandidatePoint(coordinate=v_i_point)])
        # search the candidate points of current points
        k = 0
        for stroke_point in self.current_stroke:
            temp_group = []
            indices = self.candidates_kd_tree.query_ball_point(stroke_point, self.circle_sampling_r, p=2.0, workers=-1)
            for i in indices:
                temp_group.append(CandidatePoint(coordinate=self.candidates[i]))
            self.current_candidate.append(temp_group)
            k += 1
            print(f"Stroke Point No.{k} has {len(temp_group)} candidates.")
        # for stroke_point in self.current_stroke:
        #     temp_group = []
        #     for candidate_point in self.candidates:
        #         if np.linalg.norm(stroke_point - candidate_point) < self.circle_sampling_r:
        #             temp_group.append(CandidatePoint(coordinate=candidate_point))
        #     self.current_candidate.append(temp_group)
        ##
        ## test area
        ##
        temp_img = np.zeros(self.img_work_on.shape[:2], dtype=np.uint8)
        for group in self.current_candidate:
            for candidate_point in group:
                temp_img[*candidate_point.coordinate] = 255
        test_img_show("candidates of current stroke", temp_img)
        ##
        ## test area
        ##
        # self.edge_weight_calculation()
        return self

    def edge_weight_calculation(self):
        for i in range(len(self.current_candidate) - 1):  # from 1st to 2nd of the last
            j = -1
            for q_1 in self.current_candidate[i]:  # q_start
                j += 1
                for q_2 in self.current_candidate[i + 1]:  # q_end
                    m = (q_1.coordinate + q_2.coordinate) / 2  # m is the midden point
                    modulus = np.linalg.norm(q_2.coordinate - q_1.coordinate)  # length of vec
                    if modulus == 0:
                        modulus = 0.00001
                    v = (q_2.coordinate - q_1.coordinate) / modulus  # unit directed vector of p1 p2
                    u = np.array([-v[1], v[0]])  # u is perpendicular to v
                    height, width = self.img_grayscale.shape[:2]
                    # filter response integral H
                    h = 0.0
                    for y in range(height):
                        for x in range(width):
                            new_coordinate = np.array([int(m[0] + x * u[0] + y * v[0]), int(m[1] + x * u[1] + y * v[1])])
                            gray_value = 0.0
                            if new_coordinate[0] in range(width) and new_coordinate[1] in range(height):
                                gray_value = self.img_grayscale[new_coordinate[1]][new_coordinate[0]]
                            h += self.img_v_gaussian[y][x] * gray_value * self.img_dog[y][x]
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
                    we = np.linalg.norm((p_1 - p_2) - (q_2.coordinate - q_1.coordinate)) ** 2 + self.balancing_weight * h
                    if i == 0:
                        self.current_candidate[0][0].next_weight_list.append(we)  # in the 0th point of 0th group, update the weight
                    else:
                        self.current_candidate[i][j].next_weight_list.append(we)  # in the jth point of ith group, update the weight
                    pass

    def __add_virtual_initial_point(self, coefficient):
        p_0 = self.current_stroke[0]
        p_1 = self.current_stroke[1]
        return np.array([int(p_0[0] + (p_0[0] - p_1[0]) * coefficient), int(p_0[1] + (p_0[1] - p_1[1]) * coefficient)])

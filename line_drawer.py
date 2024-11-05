import cv2
import numpy
import numpy as np
import copy
from pypattyrn.creational.singleton import Singleton
import scipy.ndimage as ndimage
import utils.yaml_reader as yr
from filter_response import FilterResponse
from test_func import test_img_show


class LineDrawer(metaclass=Singleton):
    quit_key = None
    is_LM_holding = False

    img_origin = None
    img_grayscale = None
    img_work_on = None
    img_save_point = None
    img_edge_detection = None
    img_v_gaussian = None
    img_dog = None

    windowName = None
    iter = 0
    radius = 0
    b, g, r = 0.0, 0.0, 0.0
    my_lambda = 0.0
    threshold = 0.0

    circle_sampling_time = 0
    circle_sampling_r = 0

    current_stroke = []
    strokes_container = []
    maximal_points_container = []  # salient points with maximal gradient magnitude
    candidates_container = []  # candidates set Q a.k.a. the complete bipartite graph
    H_response = []  # filtering response

    sigma_m = 0.0
    sigma_c = 0.0
    sigma_s = 0.0
    rho = 0.0
    balancing_weight = 0.0
    edge_weight_limit = 0.0

    def __init__(self):
        self.read_yaml_file()

    def draw_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_stroke = []
            self.img_save_point = self.img_work_on.copy()
            self.is_LM_holding = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.laplacian_smoothing().pick_up_candidates()
            self.img_work_on = self.img_save_point.copy()  # recover the original canvas
            self.draw_points()
            self.draw_lines()
            cv2.imshow(self.windowName, self.img_work_on)
            if len(self.current_stroke) != 0:  # add this stroke to the container
                self.strokes_container.append(self.current_stroke)
            self.img_save_point = self.img_work_on.copy()  # save the current changed canvas
            self.is_LM_holding = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_LM_holding:
                self.collect_coordinates(x, y)
                self.draw_points()
                self.draw_lines()
                cv2.imshow(self.windowName, self.img_work_on)

    def collect_coordinates(self, x, y):
        self.current_stroke.append([x, y])

    def laplacian_smoothing(self):
        for _ in range(self.iter):
            new_coordinates = self.current_stroke.copy()

            for i in range(1, len(self.current_stroke) - 1):
                prev_coordinate = self.current_stroke[i - 1]
                current_coordinate = self.current_stroke[i]
                next_coordinate = self.current_stroke[i + 1]

                new_x = current_coordinate[0] + self.my_lambda * (prev_coordinate[0] + next_coordinate[0] - 2 * current_coordinate[0])
                new_y = current_coordinate[1] + self.my_lambda * (prev_coordinate[1] + next_coordinate[1] - 2 * current_coordinate[1])

                new_coordinates[i] = (int(new_x), int(new_y))

            self.current_stroke = new_coordinates.copy()
        return self

    def insert_points(self):
        pass

    def draw_points(self):
        for point in self.current_stroke:
            cv2.circle(self.img_work_on, point, self.radius, [self.b, self.g, self.r], -1)

    def draw_lines(self):
        for i in range(len(self.current_stroke) - 1):
            cv2.line(self.img_work_on,
                     (self.current_stroke[i][0], self.current_stroke[i][1]),
                     (self.current_stroke[i + 1][0], self.current_stroke[i + 1][1]),
                     [self.b, self.g, self.r], self.radius, lineType=cv2.LINE_AA, shift=0)

    def setup(self):
        cv2.imshow(self.windowName, self.img_work_on)
        cv2.setMouseCallback("%s" % self.windowName, self.draw_line)

    def read_yaml_file(self):
        config = yr.read_yaml()
        # main settings initialization
        self.windowName = config['windowName']
        image = cv2.imread(config['testFile'])
        if image is None:
            print("No image")
            exit(4)
        self.img_origin = image
        self.img_work_on = copy.deepcopy(image)
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

    def image_pre_process(self):
        self.__edge_detection().__candidates_calculation().__grayscale().__vertical_gaussian().__DOG_function()
        return self

    def __edge_detection(self):
        gray_img = cv2.cvtColor(self.img_work_on, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        _, threshold_gradient = cv2.threshold(normalized_gradient, self.threshold, 1.0, cv2.THRESH_TOZERO)
        self.img_edge_detection = threshold_gradient
        return self

    def __candidates_calculation(self):
        kernel = np.ones((3, 3), np.float64)
        max_filtered = cv2.dilate(self.img_edge_detection, kernel)
        coordinate_bool_map = (max_filtered == self.img_edge_detection) & (self.img_edge_detection != 0.0)
        test_img_show("before_sampling", coordinate_bool_map.astype(np.uint8) * 255)
        self.maximal_points_container = np.argwhere(coordinate_bool_map)
        return self

    def __grayscale(self):
        self.img_grayscale = cv2.cvtColor(self.img_origin, cv2.COLOR_BGR2GRAY)
        return self

    def __vertical_gaussian(self):
        self.img_v_gaussian = ndimage.gaussian_filter1d(self.img_origin, sigma=self.sigma_m, axis=0)
        return self

    def __DOG_function(self):
        gaussian1 = ndimage.gaussian_filter1d(self.img_v_gaussian, sigma=self.sigma_c)
        gaussian2 = ndimage.gaussian_filter1d(self.img_v_gaussian, sigma=self.sigma_s)
        self.img_dog = gaussian1 - self.rho * gaussian2
        return self

    def pick_up_candidates(self):
        # max_y, max_x = self.img_work_on.shape[:2]
        # p_v_i_x = np.random.randint(0, max_x)
        # p_v_i_y = np.random.randint(0, max_y)
        # self.candidates_container.append([p_v_i_x, p_v_i_y])  # extend the point along the initial line
        #
        # for point in self.sampling_point_coordinate_container:
        #     temp_points_container = []
        #     for i in range(0, self.circle_sampling_time):
        #         r = self.circle_sampling_r * np.sqrt(random.random())
        #         theta = random.random() * 2 * np.pi
        #         temp_point = [int(point[0] + r * np.cos(theta)), int(point[1] + r * np.sin(theta))]
        #         temp_points_container.append(temp_point)
        #     self.candidates_container.append(temp_points_container)
        v_i_point = self.__add_virtual_initial_point(0.5)
        self.candidates_container = []
        self.candidates_container.append(v_i_point)  # add virtual initial point

        for i in self.current_stroke:  # build the candidates set of the current stroke
            temp_group = []
            for j in self.maximal_points_container:
                v = [i[0] - j[0], i[1] - j[1]]
                if np.linalg.norm(v) < self.circle_sampling_r:
                    temp_group.append(j)
            self.candidates_container.append(temp_group)
        # calculate the filter response
        self.H_response = []
        for i in range(len(self.candidates_container) - 1):  # from 1st to 2nd of the last
            for p_1 in self.candidates_container[i]:  # index of start = i
                for p_2 in self.candidates_container[i + 1]:  # index of end = i + 1
                    m = [(p_1[0] + p_2[0]) / 2, (p_1[1] + p_2[1]) / 2]  # m is the midden point
                    p1p2_len = np.linalg.norm([p_2[0] - p_1[0], p_2[1] - p_1[1]])  # length of vec
                    if p1p2_len == 0:
                        p1p2_len = 0.000000000001
                    v = [(p_2[0] - p_1[0]) / p1p2_len, (p_2[1] - p_1[1]) / p1p2_len]  # unit directed vector of p1 p2
                    u = [-v[1], v[0]]  # u is perpendicular to v
                    height, width = self.img_grayscale.shape[:2]
                    h = 0.0
                    # filter response integral
                    for y in range(height):
                        for x in range(width):
                            new_coordinate = [int(m[0] + x * u[0] + y * v[0]), int(m[1] + x * u[1] + y * v[1])]
                            gray_value = self.img_grayscale[new_coordinate[0], new_coordinate[1]]
                            h += self.img_v_gaussian[x][y] * gray_value * self.img_dog[x][y]
                    # convert h into an edge weight
                    if h < 0:
                        h = 1 + numpy.tanh(h)
                    else:
                        h = 1
                    temp_h = FilterResponse(copy.deepcopy(p_1), copy.deepcopy(p_2), h, i)
                    self.H_response.append(temp_h)

        pass

    def __add_virtual_initial_point(self, coefficient):
        p_0 = self.current_stroke[0]
        p_1 = self.current_stroke[1]
        return [p_0[0] + (p_0[0] - p_1[0]) * coefficient, p_0[1] + (p_0[1] - p_1[1]) * coefficient]

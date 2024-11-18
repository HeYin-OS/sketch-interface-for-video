import numpy as np
import cv2
import scipy.ndimage as ndi


def test_img_show(window_name, img):
    cv2.imshow(window_name, img)


def test():
    points_org = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    points = np.array(points_org, dtype=np.float32)
    stroke_point = [4, 5]
    radius = 3.0
    query_point = np.array(stroke_point, dtype=np.float32).reshape(1, -1)
    flann_params = dict(algorithm=1, trees=5)  # KDTree
    flann = cv2.FlannBasedMatcher(flann_params, {})
    flann.add([points])
    flann.train()
    matches = flann.radiusMatch(query_point, radius**2)
    matched_points = [points[match.trainIdx] for match_list in matches for match in match_list]
    print("查询点：", stroke_point)
    print("半径内的点坐标：\n", np.array(matched_points))


if __name__ == '__main__':
    test()
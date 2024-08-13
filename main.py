import cv2 as cv
import numpy as np
from utils.yamlReader import read_yaml
import line_drawer


# main function

def main():
    cfg = read_yaml()
    drawer = line_drawer.LineDrawer(cfg['brush']['radius'], cfg['brush']['b'], cfg['brush']['g'], cfg['brush']['r'])
    image = cv.imread(cfg['testFile'])
    if image is None:
        print("No image")
        exit(4)
    drawer.bind_image(image).bind_window(cfg['windowName'])
    cv.imshow(cfg['windowName'], image)
    cv.setMouseCallback("%s" % cfg['windowName'], drawer.draw_line)
    while True:
        cv.imshow(cfg['windowName'], image)
        if cv.waitKey(1) & 0xFF == ord(cfg['quitKey']):
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

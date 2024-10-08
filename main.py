import cv2 as cv
from utils.yamlReader import read_yaml
import line_drawer


# main function
# use pytest for testing

def main():
    cfg = read_yaml()
    drawer = line_drawer.LineDrawer(cfg['brush']['radius'],
                                    cfg['brush']['b'],
                                    cfg['brush']['g'],
                                    cfg['brush']['r'],
                                    cfg['laplacian_smoothing']['iter'],
                                    cfg['laplacian_smoothing']['lambda'],
                                    cfg['optimization']['local']['threshold'],
                                    cfg['optimization']['local']['circle_sampling'],
                                    cfg['optimization']['local']['radius'])
    cfg_test_file = cfg['testFile']
    image = cv.imread(cfg_test_file)
    if image is None:
        print("No image")
        exit(4)
    drawer.bind_image(image).bind_window(cfg['windowName']).pre_processing()
    cv.imshow(cfg['windowName'], image)
    cv.setMouseCallback("%s" % cfg['windowName'], drawer.draw_line)
    while True:
        if cv.waitKey(1) & 0xFF == ord(cfg['quitKey']):
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

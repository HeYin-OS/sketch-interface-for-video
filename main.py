import cv2
import line_drawer


# main function
def main():
    drawer = line_drawer.LineDrawer()
    drawer.image_pre_process().setup()
    while True:  # regular events of the main thread
        drawer.read_yaml_file()
        if cv2.waitKey(1) & 0xFF == ord(drawer.quit_key):
            break
    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    main()

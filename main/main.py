import cv2
import line_drawer


# main function
def main():
    drawer = line_drawer.LineDrawer()
    drawer.image_pre_process().setup()
    while True:  # regular events of the main thread
        key = cv2.waitKey(1) & 0xFF
        if key == ord(drawer.quit_key):
            break
        if key == ord(drawer.refresh_key):
            drawer.reload_img()
            continue
    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    main()

import time

import cv2

from backend.core.camera import Camera


def main() -> None:
    camera = Camera()
    window_name = "Camera Test"

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 320, 320)
        while True:
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if frame is None:
                time.sleep(0.01)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


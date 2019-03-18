from TimeRecord import TimeRecord
from pano_obj import *
from ImageLoader import *


def main():
    #pano()
    mtime = TimeRecord()
    stitcher = PanoStitcher(mtime)
    cam = ImageLoader()

    for i in range(cam.length()):
        next = (i + 1) % cam.length()
        image = cam.get(next)

        try:
            stitcher.add_image(image)
            cv.imshow("Canvas", stitcher.canvas)
        except Exception as e:
            print(traceback.format_exc())
            print(e)

        cv.waitKey(0)

    cam.release()
    cv.destroyAllWindows()
    cv.waitKey(1)

if __name__ == '__main__':
    main()

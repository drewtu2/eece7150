import traceback
from pano_obj import *
from ImageLoader import *


def main():
    cam = ImageLoader()

    stitcher = PanoStitcher(cam)
    stitcher.generate_features()


    # First run through...
    for i in range(cam.length()):
        next = (i) % cam.length()
        try:
            stitcher.add_image(next)
            cv.imshow("Canvas", stitcher.canvas)
        except Exception as e:
            print(traceback.format_exc())
            print(e)

        #if cv.waitKey(0) == 27:
        #    break;

    #stitcher.plot_centers()
    cv.destroyAllWindows()
    stitcher.check_cross_matches()
    stitcher.plot_outlines()
    stitcher.export_to_g2o("exported.g2o")
    print("\n\noverlap: \n", stitcher.overlap_thing)
    cv.waitKey(0)
    cam.release()
    cv.destroyAllWindows()
    cv.waitKey(1)

if __name__ == '__main__':
    main()

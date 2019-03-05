import numpy as np
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import math
from TimeRecord import TimeRecord
from TiledFeatureDetector import TiledFeatureDetector

class PanoStitcher():

    def __init__(self, timer: TimeRecord):
        self.feat_d = TiledFeatureDetector(8, 8)
        self.bf = cv.BFMatcher(crossCheck=False)
        self.timer = timer
        self.canvas = None

        # The last image we used
        self.last_image = None

    def add_image(self, image):
        '''
        Add image into the current canvas
        :param image:
        :return:
        '''

        # Intialize with the first image
        if self.canvas is None:
            self.canvas = image.copy()
            self.last_image = image.copy()
            return

        # Feature detection
        self.timer.start("orb")
        kp1, dst1, kp2, dst2 = PanoStitcher.find_features(self.orb, image, self.last_image)
        self.timer.end("orb")

        # Feature matching
        self.timer.start("feature_match")
        lkp1, lkp2 = PanoStitcher.match_features(self.bf, kp1, dst1, kp2, dst2)
        self.timer.end("feature_match")

        # Homography estimation
        self.timer.start("homography")
        h = PanoStitcher.find_h(lkp1, lkp2)
        self.get_odom(h)
        self.timer.end("homography")

        # Blending
        self.timer.start("blend")
        self.canvas = PanoStitcher.make_panorama(image, self.canvas, h)
        self.timer.end("blend")

    def get_odom(self, h):
        dx = h[1][2]
        dy = h[2][2]
        theta_tan = np.rad2deg(math.atan(h[1][0]/h[0][0]))

        print("dx: {} \t dy: {} \t theta: {}".format(dx, dy, theta_tan))


    def get_canvas(self):
        '''
        Returns a copy of the canvas
        :return: copy of the canvas
        '''

        assert(self.canvas is not None)

        return self.canvas.copy()

    @staticmethod
    def find_features(orb, im_src, im_dest):
        '''
        Find features in the given images
        :param orb: an Orb feature detector
        :param im_src: the image source image
        :param im_dest: the destination image
        :return: (key pts src, description, key pts destination, description)
        '''
        plt.figure()
        kp_m1, dst_m1 = orb.detectAndCompute(im_src, None)
        kp_m2, dst_m2 = orb.detectAndCompute(im_dest, None)

        return kp_m1, dst_m1, kp_m2, dst_m2

    @staticmethod
    def match_features(bf, kp_m1, dst_m1, kp_m2, dst_m2):
        '''
        Match the features given out by the find_features method
        :param bf: feature matcher to use
        :param kp_m1: src key points
        :param dst_m1: src descriptions
        :param kp_m2:  dest key points
        :param dst_m2: dest descriptions
        :return: matching points (pts1, pts2)
        '''
        matches = bf.match(dst_m1, dst_m2)
        matches = sorted(matches, key=lambda x: x.distance)

        list_kp1 = np.array([kp_m1[mat.queryIdx].pt for mat in matches])
        list_kp2 = np.array([kp_m2[mat.trainIdx].pt for mat in matches])

        return (list_kp1, list_kp2)

    @staticmethod
    def find_h(pts_src, pts_dest):
        '''
        Find the homography given two sets of matching points. applies RANSAC to clean data
        :param pts_src: list of points [[x, y] ... ]
        :param pts_dest: list of points [[x, y] ...]
        :return: the homography matrix as a [3x3]
        '''
        M, mask = cv.findHomography(pts_src, pts_dest, cv.RANSAC)

        return M

    @staticmethod
    def make_panorama(src, dest, h):
        '''
        Make a panorama given a src and destination image, and a homography. Src is put INSIDE of dest according to h.
        :param src: src image
        :param dest: dest image
        :param h: homography
        :return: combined image
        '''
        min_x, min_y, max_x, max_y = PanoStitcher.get_boundaries(src.shape, h)
        dest, shift = PanoStitcher.dest_shift(dest, min_x, min_y)

        min_y = min(min_y, 0)
        min_x = min(min_x, 0)
        max_y = max(max_y, dest.shape[0])
        max_x = max(max_x, dest.shape[1])
        H = np.matmul(shift, h)
        print("Boundaries: " + str((min_x, min_y, max_x, max_y)))

        if min(min_y, min_x) < -600:
            raise Exception("bad things happening??")

        warped_image = cv.warpPerspective(src, H, (max_x - min_x, max_y - min_y))

        # Put the dest_image ON TOP OF the warped image
        combined = PanoStitcher.combine_images(dest, warped_image, alpha=0);
        return combined


    @staticmethod
    def combine_images(image1, image2, alpha=.5):
        '''
        Combines two images
        :param image1: foreground
        :param image2: background
        :param alpha: alpha factor...
        :return: resulting image
        '''

        foreground, background = image1.copy(), image2.copy()
        # Check if the foreground is inbound with the new coordinates and raise an error if out of bounds

        background_height = background.shape[0]
        background_width = background.shape[1]
        foreground_height = foreground.shape[0]
        foreground_width = foreground.shape[1]

        if foreground_height > background_height or foreground_width > background_width:
            print("bg shape: " + str(background.shape))
            print("fg shape: " + str(foreground.shape))
            raise ValueError("The foreground image exceeds the background boundaries at this location")

        # Calcualte the required sizes
        dy = background_height - foreground_height
        dx = background_width - foreground_width

        # Scale Foreground up to be same size as background
        BLACK = [0, 0, 0]
        foreground = cv.copyMakeBorder(foreground,
                top=0,
                left=0,
                bottom=dy,
                right=dx,
                borderType=cv.BORDER_CONSTANT,
                value=BLACK)

        # Create Masks identifying where the images are black
        fg_mask = cv.inRange(foreground, 0, 0)
        bg_mask = cv.inRange(background, 0, 0)
        combined_mask = fg_mask & bg_mask

        # do composite at specified location
        start_y = 0
        start_x = 0
        end_y = foreground.shape[0]
        end_x = foreground.shape[1]

        blended_portion = cv.addWeighted(foreground,
                alpha,
                background,
                # background[start_y:end_y, start_x:end_x,:],
                1 - alpha,
                0,
                background.shape)
        IS_BLACK = 255
        bg_n_fg = (bg_mask != IS_BLACK) & (fg_mask == IS_BLACK)
        fg_n_bg = (fg_mask != IS_BLACK) & (bg_mask == IS_BLACK)

        blended_portion[bg_n_fg] = background[bg_n_fg]
        blended_portion[fg_n_bg] = foreground[fg_n_bg]

        return blended_portion

    @staticmethod
    def get_boundaries(shape, h):
        '''
        Finds the minimum x and y values of the image of a given shape when using a given homography
        :param shape: the shape of the image
        :param h: the h matrix being applied
        :return: (min_x, min_y)
        '''
        #[x y 1]
        #[col row 1]
        # col -> x -> shape[1]
        # row -> y -> shape[0]
        tl = [0, 0, 1]
        tr = [shape[1], 0, 1]
        bl = [0, shape[0], 1]
        br = [shape[1], shape[0], 1]

        corners = np.hstack(
                [np.array(tl).reshape(3, 1),
                    np.array(tr).reshape(3, 1),
                    np.array(bl).reshape(3, 1),
                    np.array(br).reshape(3, 1)])
                new_corners = np.matmul(h, corners)

        min_x = int(min(new_corners[0, :]))
        min_y = int(min(new_corners[1, :]))
        max_x = int(max(new_corners[0, :]))
        max_y = int(max(new_corners[1, :]))

        return min_x, min_y, max_x, max_y

    @staticmethod
    def dest_shift(dest, min_x, min_y):
        '''
        Shift an image over by a min_x and min_y if they are less than 0. O.w. Leave in place
        :param dest:
        :param min_x:
        :param min_y:
        :return:  resulting image, h matrix of the translation
        '''
        shift_x = min(0, min_x)
        shift_x = abs(shift_x)
        shift_y = min(0, min_y)
        shift_y = abs(shift_y)

        shift = np.eye(3)
        shift[0, 2] = shift_x
        shift[1, 2] = shift_y

        dest_height, dest_width, dest_channels = dest.shape

        warped_image = cv.warpPerspective(dest, shift, (dest_width + shift_x, dest_height + shift_y))
        return (warped_image, shift)

class ImageLoader:
    def __init__(self):
        self.counter = 0
        self.imgs = ["ESC.970622_025500.0621.tif",
                "ESC.970622_025513.0622.tif",
                "ESC.970622_025526.0623.tif",
                "ESC.970622_030140.0651.tif",
                "ESC.970622_030153.0652.tif",
                "ESC.970622_030206.0653.tif"]

        def read(self):
            im = cv.imread("../interest/" + str(self.imgs[self.counter]))
        self.counter = self.counter + 1

        return 0, im

    def length(self):
        return len(self.imgs)

    def release(self):
        pass



def pano(scale = .5):
    cam = ImageLoader()
    cv.startWindowThread()

    #fps = cam.get(cv.CAP_PROP_FPS)  # not quite! figure out real fps
    fps = 30
    print("Frames per second", fps)  # with timers

    frame_rate = 1;

    mtime = TimeRecord()
    stitcher = PanoStitcher(mtime)
    counter = 0
    print(cam.length())

    for i in range(0, cam.length()):

        mtime.start("total")
        mtime.start("read")
        ret_val, img = cam.read()  # read frame
        mtime.end("read")

        #if not ret_val:
        #    break
        if img is None:
            pass

        img = cv.resize(img, (0,0), fx=scale, fy=scale)
        #img = imutils.rotate(img, 270)

        try:
            stitcher.add_image(img)
        except Exception as e:
            print(e)
            pass
        canvas = stitcher.get_canvas()
        img = mtime.add_fps(img)

        mtime.iterate()

        cv.imshow('Feed', img)
        cv.imshow('Canvas', canvas)

        if cv.waitKey(1) == 27:
            break  # esc to quit
        cv.waitKey(0)

    cv.waitKey(1)
    cam.release()
    cv.destroyAllWindows()
    cv.waitKey(1)


def main():
    pano()

if __name__ == '__main__':
    main()

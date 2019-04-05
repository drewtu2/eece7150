import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import plotters
from TiledDetector import TiledDetector
from ImageLoader import ImageLoader
import g2oOptimizer

USE_SIFT = True     # True for SIFT, False for ORB
DEBUG_LEVEL = 3     # 0 off, 1 low, 2 medium, 3 high


class PanoStitcher():

    SAME_IMAGE = 1
    MATCHED_IMAGE = 2
    CROSS_MATCHED = 3
    TOO_FAR = -1
    NO_MATCH = -2
    PROJECTION_TOO_FAR = -3

    def __init__(self, loader: ImageLoader):
        tile_size = 3

        if USE_SIFT:
            feat_d = cv.xfeatures2d.SIFT_create()
            self.bf = cv.BFMatcher(cv.NORM_L2)
        else:
            feat_d = cv.ORB_create()
            self.bf = cv.BFMatcher(cv.NORM_HAMMING)

        # The tiled detector we're using
        self.feat_d = TiledDetector(feat_d, tile_size, tile_size)

        self.canvas = None

        # The last image we used
        self.last_image = None
        self.last_shift = None

        # Tuning Parameters
        self.non_max_radius = 2
        self.lowe_ratio = .7

        # Images & Features
        self.loader = loader
        self.overlap_thing = np.eye(self.loader.length(), self.loader.length())
        self.PROJECTION_THRESHOLD = 500
        self.CROSS_X_THRESHOLD = 500
        self.CROSS_Y_THRESHOLD = 300

        # Homographies
        self.homographies = []
        self.homographies_to_origin = [np.eye(3, 3)]

        # Corner points of each image
        self.corners = {}
        self.relative_odom = {}

    def generate_features(self):
        """
        Goes through the loader and adds all the features and corresponding descriptors
        :return:
        """
        for i in range(self.loader.length()):
            image = self.loader.get(i)
            # Find the initial features
            kp1, dst1 = self.feat_d.detectAndCompute(image)

            # Reduce the original features detected using radial non max supression
            kp1 = TiledDetector.radial_non_max(kp1, self.non_max_radius)

            # Recompute the features based on non the reduced keypoints
            kp1, dst1 = self.feat_d.compute(image, kp1)
            self.loader.add_kpd(kp1, dst1)


    def match_images(self, image1_num: int, image2_num: int):
        """
        Matches two images and prints the homography from image1 to image2
        :param image1:
        :param image2:
        :return: None if a match isn't found, or a 3x3 homography matrix if a match is found
        """
        # Load images and kpd
        image1 = self.loader.get(image1_num)
        kp1, dst1 = self.loader.get_kpd(image1_num)

        image2 = self.loader.get(image2_num)
        kp2, dst2 = self.loader.get_kpd(image2_num)

        # Feature matching
        lkp1, lkp2, matches = self.match_features(kp1, dst1, kp2, dst2)

        # if lkp1 is None, then there isn't a good match...
        if lkp1 is None or len(lkp1) == 0:
            return None

        # Homography estimation
        h, mask = PanoStitcher.find_h(self.keypoints_to_point(lkp1), self.keypoints_to_point(lkp2))

        if DEBUG_LEVEL > 1 and h is not None:
            self.get_odom(h)

            image1_labeled = self.loader.get_labeled(image1_num)
            image2_labeled = self.loader.get_labeled(image2_num)

            matched_image = plotters.displayMatches(image1_labeled, kp1, image2_labeled, kp2, matches, mask, True)
            cv.imshow('Matched', matched_image)

            # Blending
            test_image, _ = PanoStitcher.make_panorama(image1, image2, h)

            # Display some intermediary steps
            cv.imshow("Blended", test_image)

        return h

    def add_image(self, image_num):
        '''
        Add image into the current canvas
        :param image:
        :return:
        '''

        # Intialize with the first image
        if self.canvas is None:
            self.canvas = self.loader.get(image_num)
            self.last_image = image_num
            self.corners[image_num] = (0, 0, 0)
            return

        # Match the single step from last image to this image
        h = self.match_images(image_num, self.last_image)

        if h is None:
            self.set_overlap(image_num, self.last_image, PanoStitcher.NO_MATCH)
        self.set_overlap(image_num, self.last_image, PanoStitcher.MATCHED_IMAGE)

        # Add the h to the list of homographies and add the odometry information too
        self.add_h(h)
        self.add_relative_odom(image_num, self.last_image, self.get_odom(h))

        # Calculate the translations through to this point
        h = self.homographies_to_origin[-1]

        if image_num in self.corners.keys():
            print("WARNING: Corner {} already in self.corners as ({}, {})!!!".format(
                image_num, self.corners[image_num][0], self.corners[image_num][1]))
        else:
            self.corners[image_num] = self.get_odom(h)

        if DEBUG_LEVEL > 0:
            # Get the actual image
            image = self.loader.get(image_num)
            print("making panorama")
            # Make the new image
            self.canvas, shift = self.make_panorama(image, self.canvas, h, self.last_shift)

            self.last_shift = shift
            cv.waitKey()

        # Update the last image
        self.last_image = image_num

    def print_homos(self):
        for index, h in enumerate(self.homographies):
            print("Homography: " + str(index) + "\n" + str(h))

    @staticmethod
    def get_odom(h):
        dx = h[0][2]
        dy = h[1][2]

        R = h[0:2, 0:2]
        u, s, vh = np.linalg.svd(R)

        h[0:2, 0:2] = u @ vh

        theta_tan = math.atan2(h[0][1], h[0][0])

        print("h: \n", h)
        print("dx: {} \t dy: {} \t theta: {} radians".format(dx, dy, theta_tan))
        return dx, dy, theta_tan

    def add_h(self, h):
        self.homographies.append(h)
        self.homographies_to_origin.append(self.homographies_to_origin[-1] @ h)

    def get_h_from_origin(self, n=-1):
        """
        Returns the homography to get form image 0 to the last image.
        :return:
        """

        if n == -1:
            homographies_to_use = self.homographies
        else:
            homographies_to_use = self.homographies[0:n]

        if len(homographies_to_use) == 0:
            return np.eye(3, 3)
        else:
            h = homographies_to_use[-1]
            for index in range(2, len(homographies_to_use) + 1):
                #print("index: -", index, "\n", (homographies_to_use[-index]))
                h = homographies_to_use[-index] @ h
        return h

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
        kp_m1, dst_m1 = orb.detectAndCompute(im_src, None)
        kp_m2, dst_m2 = orb.detectAndCompute(im_dest, None)

        return kp_m1, dst_m1, kp_m2, dst_m2

    def match_features(self, kp_m1, dst_m1, kp_m2, dst_m2):
        """
        Match the features given out by the find_features method
        :param bf: feature matcher to use
        :param kp_m1: src key points
        :param dst_m1: src descriptions
        :param kp_m2:  dest key points
        :param dst_m2: dest descriptions
        :return: matching points (pts1, pts2)
        """

        matches = self.bf.knnMatch(dst_m1, dst_m2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)

        list_kp1 = np.array([kp_m1[mat.queryIdx] for mat in good])
        list_kp2 = np.array([kp_m2[mat.trainIdx] for mat in good])

        print("Matches Pre Ratio Test: " + str(len(matches)))
        print("Matches Post Ratio Test: " + str(len(good)))

        if len(good) < 5:
            print("\n\nONLY {} MATCHES DETECTED\n\n".format(len(good)))

        return list_kp1, list_kp2, good

    def export_to_g2o(self, filename):
        """
        Exports the state of the pano object to g2o format
        :param filename: the file name to export to.
        :return: None
        """
        g2oOptimizer.g2oOptimizer.generate_g2o_file(filename, self.corners, self.relative_odom)


    def plot_centers(self):
        points = [self.get_odom(self.homographies_to_origin[n]) for n in range(len(self.homographies) + 1)]
        x = [points[x][0] for x in range(len(points))]
        y = [points[x][1] for x in range(len(points))]
        t = [points[x][2] for x in range(len(points))]
        labels = ["0621", "0622", "0623", "0651", "0652", "0653"]

        fig, ax = plt.subplots()
        ax.plot(x, y, "*g")
        ax.set_title("Placement of Images")

        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))

        plt.show()
        #plt.title("Movement of Point (0, 0) in each image")

    def plot_outlines(self):
        base = plotters.get_image_corners(self.loader.get(self.last_image).shape)
        corners = [(self.homographies_to_origin[n] @ base).T for n in range(len(self.homographies) + 1)]
        corners = [corner[:, 0:2] for corner in corners]
        #print(corners)

        fig, ax = plt.subplots()
        ax = plotters.plot_image_outlines(ax, corners)
        ax = plotters.plot_links(ax, self.corners, self.relative_odom)
        ax.autoscale()
        #plt.ion()
        plt.show()

    @staticmethod
    def keypoints_to_point(lkp):
        return np.array([kp.pt for kp in lkp])

    @staticmethod
    def find_h(pts_src, pts_dest):
        '''
        Find the homography given two sets of matching points. applies RANSAC to clean data
        :param pts_src: list of points [[x, y] ... ]
        :param pts_dest: list of points [[x, y] ...]
        :return: the homography matrix as a [3x3]
        '''
        M, mask = cv.findHomography(pts_src, pts_dest, cv.RANSAC)
        M = cv.estimateRigidTransform(pts_src, pts_dest, False)

        if M is None:
            return None, mask

        M = np.vstack([M, [0, 0, 1]])

        # Correcting Rotation matrix (doesn't really work though...)
        #R = M[0:2, 0:2]
        #print("R pre-correction\n" + str(R))
        #u, s, vh = np.linalg.svd(R)
        #print(R/(u@vh))

        #M[0:2, 0:2] = u @ vh
        #print("R post-correction\n" + str(u@vh))


        return M, mask

    @staticmethod
    def make_panorama(src, dest, h, last_shift=None):
        '''
        Make a panorama given a src and destination image, and a homography. Src is put INSIDE of dest according to h.
        :param src: src image
        :param dest: dest image
        :param h: homography
        :return: combined image
        '''
        #cv.imshow("original dest", dest)
        #cv.imshow("original src", src)
        min_x, min_y, max_x, max_y = PanoStitcher.get_boundaries(src.shape, h)

        dest_shift_x = min_x
        dest_shift_y = min_y
        if last_shift is not None:
            #inv_last_shift = np.linalg.pinv(last_shift)
            inv_last_shift = last_shift
            print("Last shift\n", last_shift)
            dest_shift_x += int(inv_last_shift[0, 2])
            dest_shift_y += int(inv_last_shift[1, 2])

        #print("MinX: ", min_x, " MinY: ", min_y)
        _, shift = PanoStitcher.dest_shift(dest, dest_shift_x, dest_shift_y)
        #print("shift dest: \n", shift)
        #src, shift = PanoStitcher.dest_shift(src, min_x, min_y)
        #print("shift src: \n", shift)

        #min_x, min_y, max_x, max_y = PanoStitcher.get_boundaries(src.shape, H)
        min_y = min(min_y, 0)
        min_x = min(min_x, 0)
        max_y = max(max_y, dest.shape[0])
        max_x = max(max_x, dest.shape[1])

        #dest, shift = PanoStitcher.dest_shift(dest, min_x, min_y)
        #cv.imshow("shifted dest", dest)

        if min(min_y, min_x) < -800:
            raise Exception("bad things happening??")

        warped_image = cv.warpPerspective(src, h, (max_x, max_y))
        #cv.imshow("warped src", warped_image)

        # Put the dest_image ON TOP OF the warped image
        combined = PanoStitcher.combine_images(dest, warped_image, alpha=0.5);

        return combined, shift


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
    def dest_shift(dest, min_x, min_y, offset=None):
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

        if offset is not None:
            shift = np.linalg.inv(offset) @ shift
            shift_x = int(shift[0, 2])
            shift_y = int(shift[1, 2])

        dest_height, dest_width, dest_channels = dest.shape

        warped_image = cv.warpPerspective(dest, shift, (dest_width + shift_x, dest_height + shift_y))
        return warped_image, shift

    def set_overlap(self, index1: int, index2: int, state: int):
        """
        Sets the state of the overlapping
        :param index1: the first image index
        :param index2: the second image index
        :param state: true if they overlap
        :return: no return
        """

        self.overlap_thing[index1, index2] = state
        self.overlap_thing[index2, index1] = state

    def add_relative_odom(self, src_index, dest_index, odom):
        """
        Adds the relative odometry motion to move from image[src_index] to the image[dest_index]
        :param src_index: the index of the image to come from
        :param dest_index: the index of the image to go to
        :param odom: the odometry motion
        :return: None
        """
        if src_index not in self.relative_odom.keys():
            self.relative_odom[src_index] = {dest_index: odom}
        else:
            self.relative_odom[src_index][dest_index] = odom

    def check_cross_matches(self):
        """
        Runs through all images and looks for potential cross links
        :return:
        """

        # Check half of the corners to every other corner because matching corner N to M is the same
        # as matching corner M to N.
        for index in range(int((len(self.corners) + 1)/2 + 1)):
            base_corner = self.corners[index]

            # Match to every other corner. Offset by two because we should have already matched to corner + 1 before...
            for match_index in range(index + 2, len(self.corners)):
                if DEBUG_LEVEL > 0:
                    cv.waitKey()

                print("Checking images {} and {}...".format(index, match_index))

                test_corner = self.corners[match_index]

                # Calculate the distance between the X and Y component of the two corners
                delta_x = abs(base_corner[0] - test_corner[0])
                delta_y = abs(base_corner[1] - test_corner[1])

                # If the distance exceeds a preset threshold, we're going to assume no match
                if delta_x > self.CROSS_X_THRESHOLD or delta_y > self.CROSS_Y_THRESHOLD:
                    print("\tImage {} and {} are too far apart for overlap...".format(index, match_index))
                    print("\tDelta X: {} \t Delta Y {}".format(delta_x, delta_y))
                    print("\tDelta X Threshold: {} \t Delta Y Threshold: {}".format(
                        self.CROSS_X_THRESHOLD, self.CROSS_Y_THRESHOLD))
                    self.set_overlap(index, match_index, PanoStitcher.TOO_FAR)
                    continue

                # If we've made it here, the images are close enough where a match might exist. Let's attempt to match
                # images
                h = self.match_images(match_index, index)

                # an h may not have been found... if so, mark it as bad...
                if h is None:
                    print("\tNo match between image {} and {}...\n\n".format(index, match_index))
                    self.set_overlap(index, match_index, PanoStitcher.NO_MATCH)

                    continue

                # Check projection distance
                proj_coord = h @ np.array([base_corner[0], base_corner[1], 1])

                # Find the distance between the projected placement and the first past placement
                dist = np.linalg.norm(proj_coord[0:2] - np.array([test_corner[0], test_corner[1]]))

                print("\tThreshold: {} \t Actual: {}...".format(self.PROJECTION_THRESHOLD, dist))
                print("\tFirst Pass: ({}, {}) \t Projected: ({}, {})...".format(
                    test_corner[0], test_corner[1],
                    proj_coord[0], proj_coord[1]))
                # If distance between the projected point is too far, mark as no overlap
                if dist > self.PROJECTION_THRESHOLD:
                    print("\tProjection distance too far between image {} and {}...\n\n".format(index, match_index))
                    self.set_overlap(index, match_index, PanoStitcher.PROJECTION_TOO_FAR)
                    continue

                # If we've gotten to this part, we've got a cross link!
                print("\tCross link between images {} and {}...".format(index, match_index))
                print("Odom: {}".format(self.get_odom(h)))
                self.add_relative_odom(match_index, index, self.get_odom(h))
                self.set_overlap(index, match_index, PanoStitcher.CROSS_MATCHED)



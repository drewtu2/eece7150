import numpy as np
import cv2
from scipy import spatial

SHOW_TILING = True

class TiledDetector:
    """
    This class ensures robust feature detection by using tiling to produce
    well distributed features. 
    """

    def __init__(self, detector, tiles_x, tiles_y):
        """
        tiles_x: number of tiles in x direction
        tiles_y: number of tiles in y direction
        """
        self.detector = detector
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y
        self.features_per_tile = 100

    def compute(self, image, keypoints):
        """
        Computes the descriptors for a given set of keypoints. Piped down to the associated detector.
        :param image: Image
        :param keypoints: list(Keypoints)
        :return:
        """
        return self.detector.compute(image, keypoints)

    def detectAndCompute(self, image, kp = None):
        """
        Detects and computes keypoints and descriptors in a given image
        :param image: the image
        :param kp: UNUSED
        :return: (np.array(keypoints), np.array(descriptors))
        """

        # Get shapes of stuff, make sure that things "fit"...
        HEIGHT, WIDTH, CHANNEL = image.shape
        assert WIDTH%self.tiles_y == 0, "Width is not a multiple of tilex"
        assert HEIGHT%self.tiles_x == 0, "Height is not a multiple of tiley"
        w_width = int(WIDTH/self.tiles_y)
        w_height = int(HEIGHT/self.tiles_x)

        kps, dsc = [], []
        for row in range(0, self.tiles_y):
            # Tile over rows
            origin_row = row * w_height

            for col in range(0, self.tiles_x):
                # Tile of columns
                origin_col = col * w_width

                bit_mask = self.get_mask(image.shape, origin_row, origin_col, w_height, w_width)
                tmp_kp, tmp_dsc = self.detector.detectAndCompute(image, mask=bit_mask)

                if tmp_kp is None or tmp_dsc is None:
                    tmp_kp = []
                    tmp_dsc = []

                inbox = zip(tmp_kp, tmp_dsc)
                inbox_sorted = sorted(inbox, key=lambda x:x[0].response, reverse=True)

                try:
                    kps_out, dsc_out = list(zip(*inbox_sorted))             # Returns as two tuples
                    kps_out, dsc_out = list(kps_out), list(dsc_out)         # Convert back to lists

                    kps.extend(kps_out[:self.features_per_tile])            # Return the best keypoint features
                    dsc.extend(dsc_out[:self.features_per_tile])            # Return the corresponding descriptors

                    if SHOW_TILING:
                        im_out = self.draw_features(image, kps_out)
                        cv2.imshow("mask", cv2.bitwise_and(im_out, im_out, mask=bit_mask))
                        cv2.waitKey(0)
                except Exception as e:
                    print(e)
                    pass
        print("Features returned: " + str(len(kps)))
        return (np.array(kps), np.array(dsc))

    @staticmethod
    def get_mask(img_shape, origin_row, origin_col, num_row, num_col):
        """
        Returns a mask where only the given area is defined as passable
        :param img_shape: image shape (HEIGHT, WIDTH, COLOR)
        :param origin_row: top left corner row
        :param origin_col: top left corner height
        :param num_row: how many rows this spans
        :param num_col: how many cols this spans
        :return: a np array of size image_shape with the appropriate mask
        """
        HEIGHT, WIDTH, CHANNEL = img_shape
        base_mask = np.zeros((HEIGHT, WIDTH, 1), dtype=np.uint8)

        base_mask[origin_row:origin_row + num_row, origin_col:origin_col + num_col, :] = 255

        return base_mask

    @staticmethod
    def draw_features(image, keypoints):
        """
        Visualization of the detected keypoints
        :param image: the image to plot on
        :param keypoints: a list of keypoints to plot
        :return:
        """
        im_out = image.copy()
        for marker in keypoints:
            im_out = cv2.drawMarker(im_out, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
        return im_out

    @staticmethod
    def tiled_features(kp_d, img_shape, tiley, tilex):
        '''
        Given a set of (keypoints, descriptors), this divides the image into a 
        grid and returns len(kp_d)/(tilex*tiley) maximum responses within each 
        cell. If that cell doesn't have enough points it will return all of them.

        Need to give all features in image first, will return a tiled subset of features
        '''

        feat_per_cell = int(len(kp_d)/(tilex*tiley))
        HEIGHT, WIDTH, CHANNEL = img_shape
        assert WIDTH%tiley == 0, "Width is not a multiple of tilex"
        assert HEIGHT%tilex == 0, "Height is not a multiple of tiley"

        w_width = int(WIDTH/tiley)
        w_height = int(HEIGHT/tilex)

        xx = np.linspace(0, HEIGHT-w_height, tilex, dtype='int')
        yy = np.linspace(0, WIDTH-w_width, tiley, dtype='int')

        kps = np.array([])
        pts = np.array([keypoint[0].pt for keypoint in kp_d])
        kp_d = np.array(kp_d)

        for ix in xx:
            for iy in yy:
                inbox_mask = TiledDetector.bounding_box(pts, iy, iy+w_height, ix, ix+w_height)
                inbox = kp_d[inbox_mask]
                inbox_sorted = sorted(inbox, key = lambda x:x[0].response, reverse = True)
                inbox_sorted_out = inbox_sorted[:feat_per_cell]
                kps = np.append(kps,inbox_sorted_out)
        return kps.tolist()

    @staticmethod
    def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
            max_y=np.inf):
        """ Compute a bounding_box filter on the given points

        Parameters
        ----------                        
        points: (n,2) array
            The array containing all the points's coordinates. Expected format:
                array([
                [x1,y1],
                ...,
                [xn,yn]])

        min_i, max_i: float
            The bounding box limits for each coordinate. If some limits are missing,
            the default values are -infinite for the min_i and infinite for the max_i.

        Returns
        -------
        bb_filter : boolean array
            The boolean mask indicating wherever a point should be keept or not.
            The size of the boolean mask will be the same as the number of given points.

        """

        bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
        bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)

        bb_filter = np.logical_and(bound_x, bound_y)

        return bb_filter

    @staticmethod
    def radial_non_max(kp_list, dist):
        '''
        Given a set of Keypoints this finds the maximum response within radial
        distance from each other
        '''
        kp = np.array(kp_list)
        kp_mask = np.ones(len(kp), dtype=bool)
        pts = [k.pt for k in kp]
        tree = spatial.cKDTree(pts)
        #print ("len of kp1:",len(kp))
        for i, k in enumerate(kp):
            if kp_mask[i]:
                pt = tree.data[i]
                idx = tree.query_ball_point(tree.data[i], dist, p=2., eps=0, n_jobs=1)
                resp = [kp[ii].response for ii in idx]
                _, maxi = max([(v,i) for i,v in enumerate(resp)])
                del idx[maxi]
                for kp_i in idx:
                    kp_mask[kp_i] = False
        return kp[kp_mask].tolist()

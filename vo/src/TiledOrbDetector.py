import numpy as np
import cv2

class TiledOrbDetector():
    """
    This class ensures robust feature detection by using tiling to produce
    well distributed features. 
    """

    def __init__(self, tiles_x, tiles_y): 
        """
        tiles_x: number of tiles in x direction
        tiles_y: number of tiles in y direction
        """
        self.orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
        self.tiles_x = tiles_x
        self.tiles_y = tiles_y


    def find_features(self, image):
        kp_m1, dst_m1 = self.orb.detectAndCompute(image, None)
        kp_d = list(zip(kp_m1, dst_m1))

        return TiledOrbDetector.tiled_features(kp_d, image.shape, self.tiles_y, self.tiles_x)
    
    @staticmethod
    def tiled_features(kp_d, img_shape, tiley, tilex):
        '''
        Given a set of (keypoints, descriptors), this divides the image into a 
        grid and returns len(kp_d)/(tilex*tiley) maximum responses within each 
        cell. If that cell doesn't have enough points it will return all of them.
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
                inbox_mask = TiledOrbDetector.bounding_box(pts, iy, iy+w_height, ix, ix+w_height)
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


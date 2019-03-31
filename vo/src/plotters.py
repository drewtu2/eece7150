import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import List, Dict, Tuple

Odom = Tuple[float, float, float]
LinkDict = Dict[int, Dict[int, Odom]]

def get_image_corners(shape):
    """
    Takes in an image shape and returns the corners as an 3x4 matrix
    :param shape:(HEIGHT, WIDTH)
    :return: 3xN np array where each column represents a point
    """
    tl = [0, 0, 1]
    tr = [shape[1], 0, 1]
    br = [shape[1], shape[0], 1]
    bl = [0, shape[0], 1]

    corners = np.hstack(
        [np.array(tl).reshape(3, 1),
         np.array(tr).reshape(3, 1),
         np.array(br).reshape(3, 1),
         np.array(bl).reshape(3, 1)])

    return corners


def plot_image_outlines(ax: Axes, polygons: List[np.ndarray]):
    """
    Plots the image outlines based on sets of points
    :param ax: the axis to plot onto
    :param polygons: a list of Nx2 np.array representing the vertices
    :return: the updated axis
    """
    patches = []
    x = []
    y = []
    num_polygons = len(polygons)

    for i in range(num_polygons):
        patches.append(Polygon(polygons[i]))
        x.append(polygons[i][0, 0])
        y.append(polygons[i][0, 1])

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.4)

    #colors = 100*np.random.rand(len(patches))  # Set to random colors
    colors = [100 for n in range(len(patches))] # Set to blue apparently?
    p.set_array(np.array(colors))

    ax.add_collection(p)
    ax.plot(x, y, "g*", label="Corners")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    return ax


def plot_links(ax: Axes, corners: Dict[int, Tuple[int, int, int]], links: LinkDict):
    """
    Plots the links between every corner
    :param ax: the axis to plot onto
    :param corners: a list of Nx2 np.arrays representing the corners of each image.
    :param links: a dictionary of dictionaries of tuples. The keys are the image numbers. Tuple is an odom tuple.
    :return:
    """
    # For every corner...
    for src_index in range(len(corners)):
        # For every link from this corner
        try:
            for key in links[src_index]:
                corner = corners[key]
                relative_odom = links[src_index][key]
                x = [corner[0], corner[0] + relative_odom[0]]
                y = [corner[1], corner[1] + relative_odom[1]]

                link_color = "r"
                label = "First Pass Link"

                if abs(src_index - key) != 1:
                    link_color = "c"
                    label="Proposed Link"

                ax.plot(x, y, link_color, label=label)
        except Exception as e:
            print(e)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    return ax



def displayMatches(img_left, kp1, img_right, kp2, matches, mask, display_invalid, in_image=None, color=(0, 255, 0)):
    '''
    This function extracts takes a 2 images, set of keypoints and a mask of valid
    (mask as a ndarray) keypoints and plots the valid ones in green and invalid in red.
    The mask should be the same length as matches
    '''

    bool_mask = mask.astype(bool)

    single_point_color = (0, 255, 255)

    if in_image is None:
        mode_flag = 0
    else:
        mode_flag = 1

    img_valid = cv2.drawMatches(img_left, kp1, img_right, kp2, matches, in_image,
                               matchColor=color,
                               singlePointColor=single_point_color,
                               matchesMask=bool_mask.ravel().tolist(), flags=mode_flag)

    if display_invalid:
        img_valid = cv2.drawMatches(img_left, kp1, img_right, kp2, matches, img_valid,
                                   matchColor=(0, 0, 255),
                                   singlePointColor=single_point_color,
                                   matchesMask=np.invert(bool_mask).ravel().tolist(),
                                   flags=1)
    return img_valid

if __name__ == "__main__":
    print("Hello World")
    t = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ])
    t1 = t * .5

    lot = [t, t1]

    fig, ax = plt.subplots()
    ax = plot_image_outlines(ax, lot)
    plt.show()


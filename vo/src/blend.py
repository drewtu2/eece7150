import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

IMG_GROUP={}
img_seq = []

def read_img(group):
    global img_seq
    path ="../"
    if group == "group6":
        path += "group6/*.tif"
    elif group == "group29":
        path += "group29/*.tif"

    files = glob.glob(path)
    for img in files:
        IMG_GROUP[img[-8:-4]] = cv.imread(img,0)
        img_seq.append(img[-8:-4])
    
    img_seq = sorted(img_seq, key=lambda i: int(i))


class blender():
    def __init__(self, graph_file, image_seq, image_group):
        self.graph_file = graph_file
        self.image_seq = image_seq
        self.homographies = []
        self.image_group = image_group
        self.canvas = image_group[image_seq[0]]

    def get_homography(self):
        """Read homography from g2o file"""
        vertices = []
        with open(self.graph_file,'r') as output:
            lines = output.readlines()
            for line in lines:
                if "VERTEX_SE2" in line:
                    temp = line.split(" ")
                    pose = np.array([-float(temp[2]), -float(temp[3]),float(temp[4])])
                    H = np.array([[np.cos(pose[2]), np.sin(pose[2]), -pose[0]],
                                  [-np.sin(pose[2]),  np.cos(pose[2]), -pose[1]],
                                  [ 0., 0., 1.]])
                    vertices.append(pose)       
                    self.homographies.append(H)
    
    def pad_images(self, src, up, bottom, left, right):
        """Pad image for stitching"""
        padded = np.pad(src, ((up, bottom),(left, right)), 'constant')
        # plt.imshow(padded, cmap='gray')
        return padded

    def calculate_corner(self):
        """calculate transformed corners"""
        self.get_homography()
        x_min = 1000000
        x_max = -100000
        y_min = 1000000
        y_max = -100000
        for index, name in enumerate(self.image_seq):
            src = self.image_group[name]
            h,w = src.shape
            canvas_corner = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(canvas_corner, self.homographies[index])
            points = np.vstack((canvas_corner,dst))
            x_min = min(x_min, np.min(points[:,:,0]))
            x_max = max(x_max, np.max(points[:,:,0]))
            y_min = min(y_min, np.min(points[:,:,1]))
            y_max = max(y_max, np.max(points[:,:,1]))

        return [x_min, x_max, y_min, y_max]

    def blend_images(self):
        """stich and blend images"""
        [x_min, x_max, y_min, y_max] = self.calculate_corner()
        canvas = self.image_group[self.image_seq[0]]
        h,w = canvas.shape
        padded_canvas = self.pad_images(canvas, int(abs(y_min)), int(abs(y_max-h+1)), int(abs(x_min)), int(abs(x_max-w+1)))
        for index, name in enumerate(self.image_seq):
            frame = self.image_group[name]
            padded_frame = self.pad_images(frame, int(abs(y_min)), int(abs(y_max-h+1)), int(abs(x_min)), int(abs(x_max-w+1)))
            
            t_frame = cv.warpPerspective(padded_frame, self.homographies[index],(padded_frame.shape[1],padded_frame.shape[0])).astype(np.uint8)
            
            padded_canvas[t_frame > 0] = padded_canvas[t_frame > 0] / 2
            t_frame[padded_canvas > 0] = t_frame[padded_canvas > 0] / 2
            padded_canvas = padded_canvas + t_frame
            plt.imshow(padded_canvas,cmap='gray')
        plt.show()

if __name__ == "__main__":
    group = "group6"
    read_img(group)
    b = blender(f"output.g2o", img_seq, IMG_GROUP)
    b.blend_images()
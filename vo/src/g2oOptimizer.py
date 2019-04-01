from scipy.io import loadmat
import pano_obj
import glob
import numpy as np
import re
import g2o
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

Odom = Tuple[float, float, float]
LinkDict = Dict[int, Dict[int, Odom]]


class g2oOptimizer:

    def __init__(self):
        self.h_list = {}
        self.file_name_regex = r"(?P<id_from>\d+)_(?P<id_to>\d+)"
        self.g2o_input = "input.g2o"
        self.g2o_output = "output.g2o"
        self.order = ["0621", "0622", "0623", "0651", "0652", "0653"]

    @staticmethod
    def generate_g2o_file(filename, vertices: Dict[int, Tuple[int, int, int]], links: LinkDict):
        # Writes all the vertices
        vertex_string = ""
        for src_key in vertices.keys():
            x, y, theta = vertices[src_key]
            vertex_string += g2oOptimizer.write_vertex(src_key, x, y, theta)

        # Writes all the edges
        edge = ""
        for src_key in links.keys():
            # For every link from this corner
            for dest_key in links[src_key]:
                dx, dy, dtheta = links[src_key][dest_key]
                edge += g2oOptimizer.write_edge(dest_key, src_key, dx, dy, dtheta, np.eye(3,3, dtype=np.uint8))

        full_string = vertex_string + "FIX 0\n" + edge
        g2oOptimizer.write_g2o(filename, full_string)

    def g2o_file_from_mats(self):
        mats = glob.glob("../junk/mats/first_six/*.mat")
        edge = ""

        for mat in mats:
            # read match mat file
            match_points = loadmat(mat)
            test = self.extract_image_names(mat)
            id_from = test.group("id_from")
            id_to = test.group("id_to")

            print(mat)
            f1 = mat[10:14]
            f2 = mat[15:19]

            # find homography
            pt1 = match_points['ff']
            pt2 = match_points['gg']
            M, mask = pano_obj.PanoStitcher.find_h(pt2, pt1)
            print(M)
            dx, dy, dtheta = pano_obj.PanoStitcher.get_odom(M)

            edge += self.write_edge(id_from, id_to, dx, dy, dtheta, np.eye(3,3, dtype=np.uint8))

            if id_from not in self.h_list.keys():
                self.h_list[id_from] = {id_to: M}
            else:
                print("From: ", self.h_list[id_from])
                self.h_list[id_from][id_to] = M
            #self.h_list[f"{id_from}_{id_to}"] = M

        vertex_string = self.write_vertices()
        full_string = vertex_string + "FIX " + self.order[0] + "\n" + edge
        self.write_g2o(self.g2o_input, full_string)

        #print(match_points)

    def dict_to_pose(self, outer_image_id):

        def helper(image_id):
            if image_id == self.order[0]:
                return np.eye(3, 3)

            prior_index = self.order.index(image_id) - 1
            prior_image_id = self.order[prior_index]

            h = self.h_list[prior_image_id][image_id]
            return helper(self.order[prior_index]) @ h

        final_h = helper(outer_image_id)
        return pano_obj.PanoStitcher.get_odom(final_h)

    @staticmethod
    def write_g2o(filename, g2o_string):
        with open(filename, "w") as f:
            f.write(g2o_string)

    def extract_image_names(self, filename):
        print(filename)
        p = re.compile(self.file_name_regex);
        result = p.search(filename)

        return result

    def write_vertices(self):
        vertices_string = ""
        for node in self.order:
            x, y, t = self.dict_to_pose(node)
            vertices_string += self.write_vertex(node, x, y, t)
        return vertices_string

    @staticmethod
    def write_vertex(id, x, y, theta):
        return "VERTEX_SE2 {} {} {} {}\n".format(id, x, y, theta)

    @staticmethod
    def write_edge(id_out, id_in, x, y, theta, info):
        return "EDGE_SE2 {} {} {} {} {} {} {} {} {} {} {}\n".format(id_out, id_in,
                                                                    x, y, theta,
                                                                    info[0, 0], info[0, 1], info[0, 2],
                                                                    info[1, 1], info[1, 2], info[2, 2])

    @staticmethod
    def optimize(iterations, input, output):
        solver = g2o.BlockSolverSE2(g2o.LinearSolverEigenSE2())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)

        optimizer = g2o.SparseOptimizer()
        optimizer.set_verbose(True)
        optimizer.set_algorithm(solver)

        optimizer.load(input)
        optimizer.initialize_optimization()
        print("begin optimization")
        optimizer.optimize(iterations)
        print("end optimization")
        optimizer.save(output)

    @staticmethod
    def graph(input_file, output_file):

        fig, ax = plt.subplots()

        input = g2o.SparseOptimizer()
        input.load(input_file)

        output = g2o.SparseOptimizer()
        output.load(output_file)

        g2oOptimizer.plot_g2o_SE2(ax, input, True)
        g2oOptimizer.plot_g2o_SE2(ax, output, False)
        plt.show()


    @staticmethod
    def plot_g2o_SE2(axes, g2o_obj, text=False):
        for key in sorted(g2o_obj.vertices().keys()):
            vert = g2o_obj.vertices()[key]
            print(vert.estimate().to_vector())
            vec = vert.estimate().to_vector()
            R = g2oOptimizer.R2d_from_theta(vec[2])
            t = np.expand_dims(vec[:2], axis=1)
            T = np.vstack((np.hstack((R, t)), np.array([0, 0, 1])))
            g2oOptimizer.plot_pose2_on_axes(axes, T, axis_length=10.0)
            if text:
                axes.text(t[0, 0] + 5, t[1, 0] + 5, str(key))

    @staticmethod
    def plot_pose2_on_axes(axes, pose, axis_length=0.1):
        """
        Plot a 2D pose,  on given axis 'axes' with given 'axis_length'
        is a 2x3 or 3x3 matrix of the form [R | X]
        where R is 2d rotation and X is translation vector.
        """
        # get rotation and translation (center)
        gRp = pose[:2, :2]  # rotation from pose to global
        origin = pose[:2, -1]

        # draw the camera axes
        x_axis = origin + gRp[:, 0] * axis_length
        line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
        axes.plot(line[:, 0], line[:, 1], 'r-')

        y_axis = origin + gRp[:, 1] * axis_length
        line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
        axes.plot(line[:, 0], line[:, 1], 'g-')

    @staticmethod
    def R2d_from_theta(theta):
        return np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])

if __name__ == "__main__":
    opt = g2oOptimizer()
    #opt.g2o_file_from_mats()
    #opt.optimize(20, "vik_solution_before_opt.txt")
    #opt.optimize(20, "exported.g2o")
    g2oOptimizer.optimize(40, "exported.g2o", "output.g2o")
    opt.graph("exported.g2o", "output.g2o")


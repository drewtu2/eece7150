from TiledDetector import TiledOrbDetector
import cv2
import matplotlib.pyplot as plt

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
        im = cv2.imread("../interest/" + str(self.imgs[self.counter]))
        self.counter = self.counter + 1

        return 0, im

    def length(self):
        return len(self.imgs)

    def release(self):
        pass

# This function just makes it really easy to see the matched points. Colors should match up for correspoding 
# points
def test_points(img, points):
    plt.figure()
    plt.imshow(img)
    colors = ['green', 'red', 'blue', 'yellow']
    i = 0;
    for pt in points:
        plt.plot(pt.pt[0], pt.pt[1], color=colors[i%len(colors)], marker='+', linewidth=2, markersize=10)
        i+=1
    plt.show()

tod = TiledOrbDetector(12, 12)
il = ImageLoader()

err, img = il.read()
print("Shape: " + str(img.shape))


orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE, edgeThreshold=5, patchSize=5)
kp, descr = orb.detectAndCompute(img, None)
test_points(img, kp)
print((len(kp)))

kp_d = tod.find_features(img)
kp = kp_d[::2]
desc = kp_d[1::2]
test_points(img, kp)


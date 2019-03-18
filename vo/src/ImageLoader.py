import cv2


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

    def get(self, num):
        return cv2.imread("../interest/" + str(self.imgs[num]))

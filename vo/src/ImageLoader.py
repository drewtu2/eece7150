import cv2
import glob


class ImageLoader:

    def __init__(self, group="group6"):
        self.counter = 0
        #self.imgs = ["ESC.970622_025500.0621.tif",
        #             "ESC.970622_025513.0622.tif",
        #             "ESC.970622_025526.0623.tif",
        #             "ESC.970622_030140.0651.tif",
        #             "ESC.970622_030153.0652.tif",
        #             "ESC.970622_030206.0653.tif"]

        self.features = []
        self.descriptors = []
        self.path = ""

        self.load(group)
        print(self.imgs)

    def load(self, group):

        self.path ="../"
        if group == "group6":
            self.path += "group6"
        elif group == "group29":
            self.path += "group29"

        self.imgs = glob.glob(self.path + "/*.tif")
        self.imgs = sorted(self.imgs, key=lambda x: int(x[-8:-4]))

        #for img in files:
        #    IMG_GROUP[img[-8:-4]] = cv.imread(img,0)
        #    img_seq.append(img[-8:-4])

        #img_seq = sorted(img_seq, key=lambda i: int(i))


    def read(self):
        im = cv2.imread(str(self.imgs[self.counter]))
        self.counter = self.counter + 1

        return 0, im

    def length(self):
        return len(self.imgs)

    def release(self):
        pass

    def get(self, num):
        return cv2.imread(str(self.imgs[num]))

    def get_labeled(self, num):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (0, 0, 255)
        lineType = 2

        img = self.get(num)

        cv2.putText(img, "Image: {}".format(num),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        return img

    def add_kpd(self, kp, d):
        self.features.append(kp)
        self.descriptors.append(d)

    def get_kpd(self, num):
        return self.features[num], self.descriptors[num]

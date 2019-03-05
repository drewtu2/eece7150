import cv2 as cv
import time

class TimeRecord:
    def __init__(self):
        self.times = {
            "total": 0,
            "read": 0,
            "conversion": 0,
            "orb": 0,
            "feature_match": 0,
            "homography": 0,
            "blend": 0,
            "dilation": 0,
            "thresholding": 0,
            "show": 0
        }

        self.scratchpad = {}
        self.frames = 0

    def start(self, segment):
        self.scratchpad[segment] = time.time()

    def end(self, segment):
        self.times[segment] += time.time() - self.scratchpad[segment]

    def iterate(self):
        self.frames += 1

    def get_fps(self, segment):
        if self.times["total"] == 0:
            return float(self.frames)
        return float(self.frames) / self.times[segment]

    def get_time_percentage(self, segment):
        if self.times["total"] == 0:
            return float(self.frames)*100
        return float(self.times[segment]) / self.times["total"] * 100

    def add_fps(self, img):
        font = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (255, 0, 0)
        lineType = 2

        texts = [
            "Total FPS: " + str(self.get_fps("total"))
        ]

        # Create values for all...
        for key in self.times.keys():
            temp_text = str(key) + " Time %: " + str(self.get_time_percentage(key))
            texts.append(temp_text)

        count = 0
        for text in texts:
            if count == 0:
                fontScale = 1
            else:
                fontScale = .5

            bottomLeftCornerOfText = (10, 30 + count * 30)
            cv.putText(img, text,
                   bottomLeftCornerOfText,
                   font,
                   fontScale,
                   fontColor,
                   lineType)

            count += 1
        return img


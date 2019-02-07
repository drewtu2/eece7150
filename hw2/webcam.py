import numpy as np
import cv2 as cv

import time

class TimeRecord:
    
    def __init__(self):
        self.times = {
            "total": 0,
            "read": 0,
            "conversion": 0,
            "harris_corner": 0,
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
        self.frames+=1
    
    def get_fps(self, segment):
        return float(self.frames)/self.times[segment]
    def get_time_percentage(self, segment):
        return float(self.times[segment])/self.times["total"] * 100

def add_fps(img, timer):
  font                   = cv.FONT_HERSHEY_SIMPLEX
  bottomLeftCornerOfText = (10,30)
  fontScale              = 1
  fontColor              = (255,0,0)
  lineType               = 2

  texts = [
    "Total FPS: " + str(timer.get_fps("total")),
    "Read Time %: " + str(timer.get_time_percentage("read")),
    "Conversion Time %: " + str(timer.get_time_percentage("conversion")),
    "Harris Time %: " + str(timer.get_time_percentage("harris_corner")),
    "Dilation Time %: " + str(timer.get_time_percentage("dilation")),
    "Thresholding Time %: " + str(timer.get_time_percentage("thresholding"))
  ]

  count = 0
  for text in texts:
    if count == 0:
      fontScale = 1
    else:
      fontScale = .5

    bottomLeftCornerOfText = (10, 30 + count*30)
    cv.putText(img, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    count += 1
  return img 

def show_webcam_harris():
    cam = cv.VideoCapture(0)
    cv.startWindowThread()
    
    fps = cam.get(cv.CAP_PROP_FPS)   #not quite! figure out real fps
    print("Frames per second", fps)  # with timers
    
    mtime = TimeRecord()
    
    while True:
        mtime.start("total")
        mtime.start("read")
        ret_val, img = cam.read()  # read frame
        mtime.end("read")
        
        if img is None:
            pass
        
        mtime.start("conversion")
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #convert to grayscale
        gray = np.float32(gray)                   # convert to float
        mtime.end("conversion")
        
        mtime.start("harris_corner")
        dst = cv.cornerHarris(gray,2,3,0.04)      # Harris
        mtime.end("harris_corner")
        
        #result is dilated for marking the corners, not important
        mtime.start("dilation")
        dst = cv.dilate(dst,None)
        mtime.end("dilation")
        
        # Threshold for an optimal value, it may vary depending on the image.
        mtime.start("thresholding")
        img[dst>0.005*dst.max()]=[0,0,255]
        mtime.end("thresholding")
        mtime.end("total")
        mtime.iterate()

        img = add_fps(img, mtime)
        
        cv.imshow('dst my webcam',img)
        if cv.waitKey(1) == 27: 
            break  # esc to quit
    cv.waitKey(1)
    cam.release()
    cv.destroyAllWindows()
    cv.waitKey(1)

def main():
    show_webcam_harris()        

if __name__ == '__main__':
    main()

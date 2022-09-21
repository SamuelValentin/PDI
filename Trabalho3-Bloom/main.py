import sys
import cv2
import os 
import numpy as np

INPUT_PATH = "GT2.bmp"
INPUT_PATH = os.path.join(sys.path[0], INPUT_PATH)

img = cv2.imread(INPUT_PATH)
img = img.astype (np.float32) / 255

imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
threshold = 0.5

imgHLS[:,:,1] = np.where(imgHLS[:,:,1] > threshold, imgHLS[:,:,1]/3, 0.0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
h, s, v = cv2.split(hsv)
sv = ((255-s) * v / 255).clip(0,255).astype(np.uint8)
thresh = cv2.threshold(sv, 0.5, 255, cv2.THRESH_BINARY)[1]

im_output = cv2.cvtColor(imgHLS, cv2.COLOR_HLS2BGR)

v = [im_output, im_output, im_output]
vBlur = [im_output, im_output, im_output]

for i in range(3):
    sigma = 5*(3**i)
    v[i] = cv2.GaussianBlur(v[i], (0, 0), sigma)
    
gaussianSum = sum(v, img)

cv2.imshow('original', img)
cv2.waitKey()

cv2.destroyAllWindows()

cv2.imshow('original', im_output)
cv2.waitKey()

cv2.destroyAllWindows()

cv2.imshow('original', gaussianSum)
cv2.waitKey()

cv2.destroyAllWindows()

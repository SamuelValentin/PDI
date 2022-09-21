# Alunos: Samuel Leal Valentin e Yan Pietrzak Pinheiro

import sys
import cv2
import os 
import numpy as np

#=================================================================

INPUT_IMAGE =  "GT2.bmp"
INPUT_IMAGE = os.path.join(sys.path[0], INPUT_IMAGE)

img = cv2.imread(INPUT_IMAGE)
img = img.astype (np.float32) / 255

imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
threshold = 0.5

imgHLS[:,:,1] = np.where(imgHLS[:,:,1] > threshold, imgHLS[:,:,1]/3, 0.0)
im_output = cv2.cvtColor(imgHLS, cv2.COLOR_HLS2BGR)


v = [im_output, im_output, im_output]
vBlur = [im_output, im_output, im_output]


for i in range(3):
    sigma = 5*(2**i)
    v[i] = cv2.GaussianBlur(v[i], (0, 0), sigma)

window_sizes = ((7, 5, 3), (25, 11, 3), (91, 51, 3))


for i in range(3):
    for j in range(3):
        vBlur[i] = cv2.blur(vBlur[i], (window_sizes[j][i], window_sizes[j][i]))


gaussianSum = sum(v, img)
blurSum = sum(vBlur, img)


#====================================================
cv2.imshow('original', img)

cv2.imshow('BlurSum', blurSum)
cv2.imwrite ('BlurSum.png', blurSum*255)

cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('GaussianSum', gaussianSum)
cv2.imwrite ('GaussianSum.png', gaussianSum*255)

cv2.waitKey()
cv2.destroyAllWindows()
# Trabalho 3 Bloom Light 
# Alunos: Samuel Valentin e Yan Pinheiro

import sys
import cv2
import os 
import numpy as np

INPUT_PATH = "wind waker GC.bmp"
INPUT_PATH = "GT2.bmp"
INPUT_PATH = os.path.join(sys.path[0], INPUT_PATH)

img = cv2.imread(INPUT_PATH)
img = img.astype (np.float32)/255

imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
threshold = 0.5

imgHLS[:,:,1] = np.where(imgHLS[:,:,1] > threshold, imgHLS[:,:,1], 0.0)

im_output = cv2.cvtColor(imgHLS, cv2.COLOR_HLS2BGR)

vGau = [im_output, im_output, im_output]
vBlur = [im_output, im_output, im_output]

for i in range(3):
    sigma = 5*(4**i)
    vGau[i] = cv2.GaussianBlur(vGau[i], (0, 0), sigma)
    

for i in range(3):
    for j in range(3):
        window = (i+1)*(j+1)*9
        vBlur[i] = cv2.blur(vBlur[i], (window, window))
        

gauMask= sum(vGau)
blurMask = sum(vBlur)

gaussianSum = cv2.add(gauMask/5,img)
blurSum = cv2.add(blurMask/5,img)


cv2.imshow('original', img)
cv2.waitKey()

cv2.destroyAllWindows()

cv2.imshow('Limiar', im_output)
cv2.waitKey()

cv2.destroyAllWindows()

cv2.imshow('Gaussian', gaussianSum)
cv2.imwrite ('Gaussian.png', gaussianSum*255)
cv2.waitKey()

cv2.destroyAllWindows()

cv2.imshow('Blur', blurSum)
cv2.imwrite ('BlurSum.png', blurSum*255)
cv2.waitKey()

cv2.destroyAllWindows()
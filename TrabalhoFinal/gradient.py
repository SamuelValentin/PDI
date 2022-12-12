import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import cv2
import numpy as np

img = cv.imread('50.png',0)
laplacian = cv.Laplacian(img,cv.CV_64F)
# sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
# sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

# Read the template
template = cv2.imread('50.png', 0)
laplacianT = cv.Laplacian(template,cv.CV_64F)

# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()

 
# Store width and height of template in w and h
w, h = template.shape[::-1]
 
# Perform match operations.
res = cv2.matchTemplate(laplacian, laplacianT, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

print(min_val)
 
# Specify a threshold
threshold = 0.5
 
# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)
 
# Draw a rectangle around the matched region.
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
 
# Show the final image with the matched area.
print(res)
if(max_val > 0.5):
    print("achou")
else:
    print("Nào achou")
    

# cv2.imshow('Detected', img_rgb)
# cv2.waitKey()
# cv2.destroyAllWindows()
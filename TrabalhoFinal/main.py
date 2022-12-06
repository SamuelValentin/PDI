import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from transform import find_circles
from codigo.imagem import encontra_circulos, showimgs, show


# img = cv.imread('20221206_101219_2.jpg',0)
# img2 = img.copy()
# template = cv.imread('20221206_095056_2.jpg',0)
# w, h = template.shape[::-1]
# # All the 6 methods for comparison in a list
# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

# # template.resize(img.shape, )
# img = cv.resize(template, img.shape, interpolation = cv.INTER_AREA)

# # circles = encontra_circulos(img, sigma=3, acc=40, threshold=100)
# # showimgs(circles)

# # for circle in circles:
# #     cv.imshow ('02 - out', circle)
# #     cv.waitKey ()
# #     cv.destroyAllWindows ()

# for meth in methods:
#     img = img2.copy()
#     method = eval(meth)
    
#     # Apply template Matching
#     res = cv.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
        
#     print(res)
#     bottom_right = (top_left[0] + w, top_left[1] + h)
    
#     cv.rectangle(img,top_left, bottom_right, 255, 2)
    
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()



# ---------------

# Python program to illustrate
# template matching
import cv2
import numpy as np
 
# Read the main image
img_rgb = cv2.imread('codigo/imgs/moedas.png')
 
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
 
# Read the template
template = cv2.imread('50.png', 0)
 
# Store width and height of template in w and h
w, h = template.shape[::-1]
 
# Perform match operations.
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
 
# Specify a threshold
threshold = 0.2
 
# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)
 
# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
 
# Show the final image with the matched area.
cv2.imshow('Detected', img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()
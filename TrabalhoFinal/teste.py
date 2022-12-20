import numpy as np
import cv2

threshold = 0.8

##Read Main and Needle Image
imageMainRGB = cv2.imread("imgs/50c.png")
template = cv2.imread("imgs/50c.png")

img_gray = cv2.cvtColor(imageMainRGB, cv2.COLOR_BGR2GRAY)
w, h = img_gray.shape[::-1]

# w, h = imageMainRGB.shape[::2]
dim = (w, h)
# resize image
templateR = cv2.resize(template, dim, interpolation = cv2.INTER_AREA)
        
##Split Both into each R, G, B Channel
imageMainR, imageMainG, imageMainB = cv2.split(imageMainRGB)
imageNeedleR, imageNeedleG, imageNeedleB = cv2.split(template)

##Matching each channel
resultB = cv2.matchTemplate(imageMainR, imageNeedleR, cv2.TM_CCOEFF_NORMED)
resultG = cv2.matchTemplate(imageMainG, imageNeedleG, cv2.TM_CCOEFF_NORMED)
resultR = cv2.matchTemplate(imageMainB, imageNeedleB, cv2.TM_CCOEFF_NORMED)

# res = cv2.matchTemplate(img_gray, templateR, cv2.TM_CCOEFF_NORMED)

min_valB, max_valB, min_loc, max_loc = cv2.minMaxLoc(resultB)
min_valG, max_valG, min_loc, max_loc = cv2.minMaxLoc(resultG)
min_valR, max_valR, min_loc, max_loc = cv2.minMaxLoc(resultR)


print(resultB.min)


##Add together to get the total score
result = resultB + resultG + resultR
loc = np.where(result >= 3 * threshold)
print("loc: ", loc)

pont = max_valR + max_valG + max_valB
print(pont)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from transform import find_circles
from codigo.imagem import encontra_circulos, showimgs, show

# ---------------

# Python program to illustrate
# template matching
import cv2
import numpy as np
 
def templateMatch(img_rgb):
    maxT_val = 0
    maxT_tp = ''


    # Convert it to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    w, h = img_gray.shape[::-1]

    templates = {'1h', '1t', '5h', '5t', '10h', '10t', '25h', '25t', '50h', '50t', '100h', '100t'}
    
    for tp in templates:
        
        # Read the template
        template = cv2.imread('codigo/imgs/'+tp+'.png', 0)
        # template = cv2.imread('50s.png', 0)

        dim = (w, h)
        # resize image
        templateR = cv2.resize(template, dim, interpolation = cv2.INTER_AREA)

        # Store width and height of template in w and h
        # w, h = template.shape[::-1]
    
        # Perform match operations.
        res = cv2.matchTemplate(img_gray, templateR, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    
        # # Specify a threshold
        # threshold = 0.9
    
        # Store the coordinates of matched area in a numpy array
        # loc = np.where(res >= threshold)
    
        # # Draw a rectangle around the matched region.
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    
        # Show the final image with the matched area.
        print(str(tp) + ": Min_val = " + str(max_val))
        print(res)
        
        if(maxT_val < max_val):
            maxT_val = max_val
            maxT_tp = tp
            
    return maxT_tp
        

def main():
    # Read the main image
    img_rgb = cv2.imread('50x.png')
    
    tipo = templateMatch(img_rgb)

    print("Template encontrado :")
    print(tipo) 


main()
# cv2.imshow('Detected', img_rgb)
# cv2.waitKey()
# cv2.destroyAllWindows()
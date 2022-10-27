#=================================================
# Universidade Tecnologica Federal do Parana
# Aluno: Yan Pietrzak Pinheiro
#=================================================

import cv2
import os, sys
import numpy as np
import copy

INPUT_IMAGES =  ['8.bmp']

def remove_bg(img):

    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    green = [62,255,255]

    for x in range(img.shape[0]):      
        for y in range(img.shape[1]):

            diff = green - img[x][y]
            
            dh = diff[0] /22
            ds = diff[1] /255
            dv = diff[2] /255

            distance = np.sqrt(dh*dh+ds+dv*dv)
            dv = abs(dv)

            if distance < 1.03 and (ds <= 0.85 and dv <= 0.85):
                img[x][y][0] = 0
                img[x][y][1] = 0
                img[x][y][2] = 0

            elif distance <= 1.20:
                dist = 1/distance
                img[x][y][1] = img[x][y][1]*abs(1 - dist)
                if ds <= 0.85: 
                    img[x][y][2] = img[x][y][2]*dist
        
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img
 
def main():
    for src in INPUT_IMAGES:
        input_img = os.path.join(sys.path[0], "img/" + src)
        input_img = input_img.replace("\\","/")
        img_src = cv2.imread(input_img, cv2.IMREAD_COLOR)

        img = remove_bg(copy.copy(img_src))

        cv2.imshow(str(src), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img = None
        img_src = None
        input_img = None
    
if __name__ == '__main__':
    main()
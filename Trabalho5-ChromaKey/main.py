#=================================================
# Universidade Tecnologica Federal do Parana
# Alunos: Samuel L. Valentin e Yan P. Pinheiro
#=================================================

import cv2
import os, sys
import numpy as np
import copy
from scipy import stats

INPUT_IMAGES =  ['0.bmp','1.bmp','2.bmp','3.bmp','4.bmp','5.bmp','6.bmp','7.bmp','8.bmp']

INPUT_PATH = "GT2.bmp"
INPUT_PATH = os.path.join(sys.path[0], INPUT_PATH)


def chroma(img, bg):

    print("Adicionando BG")
    
    img_final = copy.copy(img)
    
    dim = (img_final.shape[1], img_final.shape[0])
  
    # resize image
    resized = cv2.resize(bg, dim, interpolation = cv2.INTER_AREA)
    
    img_aux = copy.copy(img)

    img_aux = cv2.cvtColor(img_aux,cv2.COLOR_RGB2HSV)
    resized = cv2.cvtColor(resized,cv2.COLOR_RGB2HSV)
    
    # cor = stats.mode(img)[0]
    # print(cor)
    green = [62,255,255]

    for x in range(img.shape[0]):      
        for y in range(img.shape[1]):

            diff = green - img_aux[x][y]
            
            dh = diff[0] /22
            ds = diff[1] /255
            dv = diff[2] /255

            distance = np.sqrt(dh*dh+ds+dv*dv)
            dv = abs(dv)

            if distance < 1.13 and (ds <= 0.85 and dv <= 0.85):
                img_aux[x][y][0] = resized[x][y][0]
                img_aux[x][y][1] = resized[x][y][1]
                img_aux[x][y][2] = resized[x][y][2]

            # elif distance <= 1.20:
            #     dist = 1/distance
            #     img_aux[x][y][1] = resized[x][y][1]*abs(1 - dist)
            #     if ds <= 0.85: 
            #         img_aux[x][y][2] = resized[x][y][2]*dist
            
    img = cv2.cvtColor(img_aux,cv2.COLOR_HSV2RGB)
    
    return img

 
def main():
    for src in INPUT_IMAGES:
        input_img = os.path.join(sys.path[0], "img/" + src)
        input_img = input_img.replace("\\","/")
        img_src = cv2.imread(input_img, cv2.IMREAD_COLOR)
        
        input_img_bg = os.path.join(sys.path[0], INPUT_PATH)
        input_img_bg = input_img_bg.replace("\\","/")
        bg = cv2.imread(input_img_bg, cv2.IMREAD_COLOR)
        
        img_final = chroma(copy.copy(img_src), bg)
        
        
        print("Mostrando resultados...")

        cv2.imshow(str(src), img_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img = None
        img_src = None
        input_img = None
        img_final = None
    
if __name__ == '__main__':
    main()

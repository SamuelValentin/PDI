#=================================================
# Universidade Tecnologica Federal do Parana
# Aluno: Yan Pietrzak Pinheiro
#=================================================

import cv2
import os
import sys
import numpy as np
import copy
from scipy import stats

sys.setrecursionlimit(3000)

INPUT_IMAGES =  ['60.bmp','82.bmp','114.bmp','150.bmp','205.bmp']
ALTURA_MIN = 5
LARGURA_MIN = 5
N_PIXELS_MIN = 20

#=================================================

def binarize(img):
    #Normalização
    cv2.normalize(img, dst=img, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    #Binarização com threshhold adaptativo
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,71,-45)
    #Operação morfologica para elimar ruido
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 1)
    
    return img

#---------------------------------
#Função flood fill e detect blobs do trabalho 1
def flood_fill(label, img, x, y):
    img[x][y] = label
    if x - 1 >= 0:
        if img[x-1][y] == -1:
            flood_fill(label, img, x-1, y)
    if x + 1 < img.shape[0]:
        if img[x+1][y] == -1:
            flood_fill(label, img, x+1, y)
    if y - 1 >= 0:
        if img[x][y-1] == -1:
            flood_fill(label, img, x, y-1)
    if y + 1 < img.shape[1]:
        if img[x][y+1] == -1:
            flood_fill(label, img, x, y+1)


def detect_blobs(img, largura_min, altura_min, n_pixels_min):
    result = []
    auxRes = []

    cLabel = 1
    nPixels = 0
    topCord =  float('inf')
    leftCord = float('inf')
    botCord = -1 
    rightCord = -1
    
    img = np.where(img == 1.0, -1, 0)
    
    print("Flood Fill..."+'\n')
    i = 0
    j = 0
    for arr in img:
        for pixel in arr:
            if pixel == -1:
                flood_fill(cLabel,img,i,j)
                cLabel = cLabel + 1
            j = j+1
        i = i+1
        j=0

    print("Dict building..."+'\n')
    for lab in range(1,cLabel):
        newDict = {
                "label": lab,
                "n_pixels": nPixels,
                "T": topCord,
                "L": leftCord,
                "B": botCord,
                "R": rightCord,
        }
        auxRes.append(newDict)
   
    i = 0
    j = 0

    for arr in img: 
        for pixel in arr:

            if pixel > 0:

                c = auxRes[int(pixel)-1] 

                c['n_pixels'] = c['n_pixels'] + 1
                if(i < c['T']):
                    c['T'] = i
                if(i > c['B']):
                    c['B'] = i
                if(j < c['L']):
                    c['L'] = j
                if(j > c['R']):
                    c['R'] = j
                auxRes[int(pixel)-1] = c
                
            j = j + 1
        i = i + 1
        j = 0
    
    for comp in auxRes:
        if comp['n_pixels'] >= n_pixels_min:
            if comp['R'] - comp['L'] >= largura_min:
                 if comp['B'] - comp['T'] >= altura_min:
                    result.append(comp)

    return result

#-------------------------------------------

def moda(componentes):

    count = 0
    arr = []
    for c in componentes:
        arr.append(c['n_pixels'])
    arr.sort()

    #step/passo - De quanto em quanto iremos agrupar os valores
    step=35
    size=0
    i =0

    #Agrupa as medidas de tamanho de pixel dos blobs em grupos
    while size < arr[len(arr) - 1 ]:
        i=0
        for c in arr:
            if c - size <= step and c - size > 0 :
                arr[i] = size

            i = i + 1

        size = size + step

    #moda
    moda = stats.mode(arr)[0]

    #desvio padrão
    dp = np.std(arr)
    print("moda: " + str(moda))
    print("desvio padrao: " + str(dp) )

    if moda > dp:
        #Se a moda é maior que desvio padrão, diminuimos meio desvio padrão da moda
        moda = moda - (0.5*dp)

    for c in arr:
        #se o blob é maior que a moda, divide seu tamanho pela moda e pega o resto inteiro
        if c > moda:
            conta = int((c/moda))
            count =  count + conta
        else:
            count = count + 1

    #retorna num. de blobs estimados
    return  count


def main():
    count =0
    for src in INPUT_IMAGES:
        input_img = os.path.join(sys.path[0], src)
        input_img = input_img.replace("\\","/")

        img_src = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)

        img_out = cv2.cvtColor (img_src, cv2.COLOR_GRAY2BGR)

        img = copy.copy(img_src)

       
        img = binarize(img)
        img = img.reshape ((img.shape [0], img.shape [1], 1))
        img = img.astype (np.float32) / 255

        componentes = detect_blobs (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)

        img2 = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

        for c in componentes:
            cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,255))
            cv2.rectangle (img2, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,255))
        count = moda(componentes)
        print ('%d componentes detectados.' % count)
        cv2.imshow('Binarizada',img2)
        cv2.imshow('Original: ' + src.replace(".bmp","") + " Calculado: " + str(count), img_out)
       
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        img = None
        img_src = None
        input_img = None
    
if __name__ == '__main__':
    main()
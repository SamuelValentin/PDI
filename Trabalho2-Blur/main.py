#=================================================================
# Exercício 2: Blur
#-----------------------------------------------------------------
# Universidade Tecnologica Federal do Parana
# Professor: Bogdan T. Nassu
# Aluno: Samuel Leal Valentin e Yan Pietrzak Pinheiro
#=================================================================

from __future__ import division
import sys
import cv2
import os 
import numpy as np

#=================================================================

INPUT_IMAGE =  "Exemplos/a01 - Original.bmp"
INPUT_IMAGE = os.path.join(sys.path[0], INPUT_IMAGE)
print (os.path.isfile(INPUT_IMAGE))
INPUT_IMAGE = INPUT_IMAGE.replace("\\","/")
W_SIZE = 7

#=================================================================

#TODO as funcoes

def blur_ingenuo (img,img_out):
    print("blur_ingenuo")
    img_length = img.shape[0]
    img_width = img.shape[1]
   
    w = 7
    h = 7

    for i in range(1, img_length):
        for j in range (1, img_width):
            soma = 0
            div = 0
            for k in range(i-int(h/2),i+int(h/2)+1):
                for l in range(j-int(w/2),j+int(w/2)+1):
                    if(k > 0 and k < img_length and l > 0 and l< img_width):
                        soma = soma + img[k][l]
                    else:
                        div = div + 1
            img_out[i][j] = soma/((h*w)-div)

            
    return img_out
#-------------------------------

def blur_separable (img,img_out):
    print("blur_separable")

    img_length = img.shape[0]
    img_width = img.shape[1]
   
    w = 11
    h = 1

    img_sep = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    img_sep = img_sep.reshape ((img.shape [0], img.shape [1], 1))
    img_sep = img_sep.astype (np.float32) / 255

    for i in range(1, img_length-1):
        for j in range (1, img_width-1):
            soma = 0
            div = 0
            for l in range(j-int(w/2),j+int(w/2)+1):
                if(l > 0 and l < img_length):
                    soma = soma + img[i][l]
            img_sep[i][j] = soma/(w)

    for i in range(1, img_length-1):
        for j in range (1, img_width-1):
            soma = 0
            div = 0
            for l in range(i-int(h/2),i+int(h/2)+1):
                if(l > 0 and l < img_length):
                    soma = soma + img_sep[l][j]
            img_out[i][j] = soma/(h)
            
    return img_out
    # return

#-------------------------------

def integral (img,img_out):
    print("integral")
    
    img_int = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    img_int = img_int.reshape ((img.shape [0], img.shape [1], 1))
    img_int = img_int.astype (np.float32) / 255

    img_int = img_integral(img_int)

    img_length = img.shape[0]
    img_width = img.shape[1]
   
    w = 3
    h = 3
    
    w2 = int(w/2) 
    h2 = int(h/2) 
    
    for i in range(1, img_length-1):
        for j in range (1, img_width-1):
            soma = 0
            if(i+w < img_length and j+h < img_width):
                if((j-w2+1) < 0 and (i+h2+1) < 0):
                    soma = img_int[i+h2][j+w2] - img_int[i+h2][0] - img_int[0][j+w2] + img_int[0][0]
                elif((i-h2+1) < 0):
                    soma = img_int[i+h2][j+w2] - img_int[i+h2][j-w2-1] - img_int[0][j+w2] + img_int[0][j-w2-1]
                elif((j-w2+1) < 0):
                    soma = img_int[i+h2][j+w2] - img_int[i+h2][0] - img_int[i-h2-1][j+w2] + img_int[i-h2-1][0]
                else:
                    soma = img_int[i+h2][j+w2] - img_int[i+h2][j-w2-1] - img_int[i-h2-1][j+w2] + img_int[i-h2-1][j-w2-1]
                img_out[i][j] = soma / (w*h)

    for i in range(1, img_length):
        img_out[i][0] = img[i][0] 
        
    for j in range (1, img_width):
        img_out[0][j] = img[0][j] 

    return img_out

def img_integral(img_int):
    print("Criando iamgem integral")
    
    img_length = img_int.shape[0]
    img_width = img_int.shape[1]
    
    for i in range(1, img_length):
        img_int[i][0] = img_int[i][0] + img_int[-1][0]
        
    for j in range (1, img_width):
        img_int[0][j] = img_int[0][j] + img_int[0][j-1]
        
    for i in range(1, img_length):
        for j in range (1, img_width):
            img_int[i][j] = img_int[i][j] + img_int[i-1][j] + img_int[i][j-1] - img_int[i-1][j-1]

    return img_int
#================================================================

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro ao abrir a imagem: '+ INPUT_IMAGE +'\n')
        sys.exit ()

    print ('Sucesso ao abrir a imagem: '+ INPUT_IMAGE +'\n')

    # img = img.reshape ((img.shape [0], img.shape [1], img.shape [2]))
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    img_out = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    img_out = img_out.reshape ((img.shape [0], img.shape [1], 1))
    img_out = img_out.astype (np.float32) / 255

    option = -1
    while option != "0":
        option = input("\nDigite o numero de uma das opções abaixo! \n" +
                    "\tAlgoritmo ingenuo - 1 \n" +
                    "\tFiltro separavel - 2 \n" +
                    "\tAlgoritmo Imagens Integrais - 3 \n" +
                    "\tSair - 0 \n"
                ).strip()

        # h = input("Qual a altura da janela?")
        # w = input("Qual a Largura da janela?")
#--------------------------------------------
        if option == "1":

            # img_out = 
            blur_ingenuo(img, img_out)

            cv2.imshow("Blur Ingenuo",img_out)
            cv2.imwrite ('02 - out.png', img_out*255)

            # cv2.waitKey ()
            # cv2.destroyAllWindows ()

#--------------------------------------------
        if option == "2":

            img_out = blur_separable(img,img_out)

            cv2.imshow("Filtro separavel",img_out)
            cv2.imwrite ('02 - sep - out.png', img_out*255)

            cv2.waitKey ()
            cv2.destroyAllWindows ()

#--------------------------------------------
        if option == "3":

            img_out = integral(img,img_out)
            cv2.imshow("Imagem integrais",img_out)
            cv2.imwrite ('02 - int - out.png', img_out*255)

            cv2.waitKey ()
            cv2.destroyAllWindows ()



#--------------------------------------------
if __name__ == '__main__':
    main ()
    




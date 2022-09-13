#=================================================================
# Exercício 2: Blur
#-----------------------------------------------------------------
# Universidade Tecnologica Federal do Parana
# Professor: Bogdan T. Nassu
# Aluno: Samuel Leal Valentin e Yan Pietrzak Pinheiro
#=================================================================

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
   
    w = 3
    h = 3

    for i in range(1, img_length-1):
        for j in range (1, img_width-1):
            soma = 0
            for k in range(i-int(h/2),i+int(h/2)+1):
                for l in range(j-int(w/2),j+int(w/2)+1):
                    soma = soma + img[k][l]
            img_out[i][j] = soma/(h*w)

            
    return img_out
#-------------------------------

def blur_separable (img,img_out):
    print("blur_separable")

    img_length = img.shape[0]
    img_width = img.shape[1]
   
    w = 3
    h = 3

    img_sep = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    img_sep = img_sep.reshape ((img.shape [0], img.shape [1], 1))
    img_sep = img_sep.astype (np.float32) / 255

    for i in range(1, img_length-1):
        for j in range (1, img_width-1):
            soma = 0
            for l in range(j-int(w/2),j+int(w/2)+1):
                soma = soma + img[i][l]
            img_sep[i][j] = soma/(w)

    for i in range(1, img_length-1):
        for j in range (1, img_width-1):
            soma = 0
            for k in range(i-int(h/2),i+int(h/2)+1):
                soma = soma + img_sep[k][j]
            img_out[i][j] = soma/(h)
            
    return img_out
    # return

#-------------------------------

def integral (img,img_out):
    print("integral")

    integral = img_integral()

    img_length = img.shape[0]
    img_width = img.shape[1]
   
    w = 3
    h = 3

    for i in range(1, img_length-1):
        for j in range (1, img_width-1):
            print("IMG")

    return img_out

def img_integral():
    print("Criando iamgem integral")


    return integral
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

            img_out = blur_separable()

            cv2.imshow("Filtro separavel",img_out)

            cv2.waitKey ()
            cv2.destroyAllWindows ()

#--------------------------------------------
        if option == "3":

            img_out = integral()
            cv2.imshow("Imagem integrais",img_out)

            cv2.waitKey ()
            cv2.destroyAllWindows ()



#--------------------------------------------
if __name__ == '__main__':
    main ()
    




#=================================================================
# Exercício 2: Blur
#-----------------------------------------------------------------
# Universidade Tecnologica Federal do Parana
# Professor: Bogdan T. Nassu
# Aluno: Yan Pietrzak Pinheiro
#=================================================================

import sys
import cv2
import os 
import numpy as np

#=================================================================

INPUT_IMAGE =  "Original.bmp"
INPUT_IMAGE = os.path.join(sys.path[0], INPUT_IMAGE)
print (os.path.isfile(INPUT_IMAGE))
INPUT_IMAGE = INPUT_IMAGE.replace("\\","/")
W_SIZE = 7

#=================================================================

#TODO as funcoes

def blur_ingenuo (img):
    img_length = img.shape[0]
    img_width = img.shape[1]

    for i in range(img_length):
        for j in range (img_width):
            print("img")

    # return
#-------------------------------

def blur_separable ():
    print("blur")
    # return

#-------------------------------

def integral ():
    print("integral")
    # return

#================================================================

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro ao abrir a imagem: '+ INPUT_IMAGE +'\n')
        sys.exit ()

    print ('Sucesso ao abrir a imagem: '+ INPUT_IMAGE +'\n')

    img = img.reshape ((img.shape [0], img.shape [1], img.shape [2]))
    img = img.astype (np.float32) / 255


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

            img_out = blur_ingenuo()

            cv2.imshow("Blur Ingenuo",img_out)

            cv2.waitKey ()
            cv2.destroyAllWindows ()

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
    




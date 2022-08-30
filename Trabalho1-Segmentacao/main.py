#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Professor: Bogdan T. Nassu
# Alunos: Samuel Valentin e Yan Pinheiro
# Universidade Tecnológica Federal do Paraná
#===============================================================================

# from asyncio.windows_events import NULL
from email.errors import FirstHeaderLineIsContinuationDefect
from operator import truediv
import sys
import timeit
from tkinter.ttk import LabelFrame
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 10

#===============================================================================

def binariza (img, threshold):
    img = np.where(img < threshold, 0.0, 1.0)
    return img
    
    #     ''' Binarização simples por limiarização.

    # Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
    #               canal independentemente.
    #             threshold: limiar.
                
    # Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):
    componentes = []
    auxRes = []

    label = 1
    nPixels = 0
    topCord =  float('inf')
    leftCord = float('inf')
    botCord = -1 
    rightCord = -1
    
    img = np.where(img == 1.0, -1, 0)
    
    print("Flood Fill")
    height = img.shape[0]
    width = img.shape[1]

    for i in range(height):
        for j in range(width):
            if img[i,j] == -1:
                floodfillRec(img,label,i,j)
                label = label + 1

    print("Dict building")
    for lab in range(1,label):
        newDict = {
                "label": lab,
                "n_pixels": nPixels,
                "T": topCord,
                "L": leftCord,
                "B": botCord,
                "R": rightCord,
        }
        auxRes.append(newDict)

    for i in range(height):
        for j in range(width):
            if img[i,j] > 0:
                c = auxRes[int(img[i,j])-1] 
                c['n_pixels'] = c['n_pixels'] + 1
                if(j < c['L']):
                    c['L'] = j
                if(j > c['R']):
                    c['R'] = j
                if(i < c['T']):
                    c['T'] = i
                if(i > c['B']):
                    c['B'] = i
                auxRes[int(img[i,j])-1] = c
    
    for comp in auxRes:
        if comp['n_pixels'] >= n_pixels_min:
            if comp['R'] - comp['L'] >= largura_min:
                 if comp['B'] - comp['T'] >= altura_min:
                    componentes.append(comp)
                    
    return componentes

    # '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
    # [0.1,0.2,etc].

    # Parâmetros: img: imagem de entrada E saída.
    #             largura_min: descarta componentes com largura menor que esta.
    #             altura_min: descarta componentes com altura menor que esta.
    #             n_pixels_min: descarta componentes com menos pixels que isso.

    # Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
    # com os seguintes campos:

    # 'label': rótulo do componente.
    # 'n_pixels': número de pixels do componente.
    # 'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
    # respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.

#===============================================================================

def floodfillRec(img, label, x, y):
    
    img[x,y] = label

    for i in range(-1, 3, 2):
        if (img[x][y+i] != None):
            if img[x][y+i] == -1:
                floodfillRec(img, label, x, y+i)    
                
    for j in range(-1, 3, 2):
        if (img[x+j][y] != None):
            if img[x+j][y] == -1:
                floodfillRec(img, label, x+j, y)
    
def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================

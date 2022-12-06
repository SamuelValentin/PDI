import cv2 as cv
import numpy as np

from pprint import pprint
from scipy.signal import argrelextrema
from itertools import product

from imagem import encontra_circulos, mapimg, mapimgxy, gradient_hist, show, rotate_image
from tools import dif, pipeC, star


sobelx = lambda imagem: cv.Sobel(imagem, cv.CV_64F, 1, 0, ksize=1)
sobely = lambda imagem: cv.Sobel(imagem, cv.CV_64F, 0, 1, ksize=1)
gradients = lambda imagem: (sobelx(imagem), sobely(imagem))
toPolar = lambda dx, dy, degrees=True: cv.cartToPolar(dx, dy, angleInDegrees=degrees)

gradients_polar = pipeC(gradients, star(toPolar))


EncontraGrandeCirculo = lambda img:\
    encontra_circulos(img,
                      maxrad=int(min(img.shape)/1.9),
                      minrad=int(min(img.shape)/2.2),
                      mindist=int(min(img.shape)/2),
                      acc=20,
                      sigma=3,
                      threshold=100)[0]

'''
  (HOF)
  Funcao que recebe as especificacoes do circulo (px, py, rad)
  e um valor padrao (default),
  retorna uma funcao que pega um valor e um ponto (x, y) e
  retorna o proprio valor se o ponto estiver dentro do circulo
  ou o valor padrao, caso contrario.
'''
preserve_circle = \
    lambda px, py, rad, default:\
    lambda v, x, y:\
    v if (x-px)**2 + (y-py)**2 < rad**2 else default

'''
  Corta o circulo (px, py, rad) na imagem
'''
CortaCirculo = lambda px, py, rad, imagem:\
    mapimgxy(imagem[py-rad:py+rad,px-rad:px+rad],
             preserve_circle(rad, rad, rad, 0))

'''
  Calcula os angulos cujos gradientes foram mais fortes.
'''
def MaximosAngulosGradientes(img):
    mag, ang = gradients_polar(img)

    hist = gradient_hist(mag, ang)
    
    picos = argrelextrema(hist, np.greater, order=18)[0]

    return picos

'''
  Retorna as diferencas de todas as combinacoes entre os
  pontos A e pontos B
'''
def CalcDistances (pontosA, pontosB):
    return map(dif, product(pontosA, pontosB))

'''
  Extrai os circulos na imagem, retornando uma lista com
  as coordenadas dos circulos e outra com as imagens dos
  circulos cortadas.
'''
def ExtraiCirculos (imagem):
    circulos = encontra_circulos(imagem, sigma=3, acc=40, threshold=100)
    imagens_circulos =\
        [ CortaCirculo(*circulo, imagem) for circulo in circulos ]
    
    return circulos, imagens_circulos

'''
  Extrai um circulo de tamanho proximo ao da imagem,
  retornando a imagem do circulo cortada.
'''
def ExtraiGrandeCirculo (imagem):
    circulo = EncontraGrandeCirculo(imagem)
    return CortaCirculo(*circulo, imagem)


'''
  Carrega as imagens das moedas originais, corta as bordas e
  calcula os angulos cujas magnitudes foram maiores e mais
  incidentes.
  Retorna um dicionario { nome : ( imagem, maximos ) }
'''
def CarregaMoedasOriginais():
    nomes_moedas = ['1h.jpg',  '1t.jpg',
                    '5h.png',  '5t.png',
                    '10h.png', '10t.png',
                    '25h.png', '25t.png',
                    '50h.png', '50t.png',
                    '100h.png', '100t.png']

    moedas = { nome : cv.imread('imgs/' + nome, 0) for nome in nomes_moedas }
    moedas = { nome : ExtraiGrandeCirculo(moeda)   for nome, moeda in moedas.items() }
    moedas =\
        { nome : (moeda, MaximosAngulosGradientes(moeda)) for nome, moeda in moedas.items() }
    return moedas


if __name__ == "__main__":
    
    moedas = CarregaMoedasOriginais()

    imagem = cv.imread('imgs/moedas.png', 0)
    # print("a")
    black2white = lambda v: v if v != 0 else 255
    imagem = mapimg (imagem, black2white)
    
    circulos, imagens = ExtraiCirculos (imagem)
    maximos = ( MaximosAngulosGradientes(imagem) for imagem in imagens )

    for imagem, origens in zip(imagens, maximos):
        for nome, (moeda, destinos) in moedas.items():
            for offset in CalcDistances(destinos, origens):
                imagem_girada = rotate_image(imagem, offset)
                show(moeda, imagem_girada, cmap='gray', title=str(offset))

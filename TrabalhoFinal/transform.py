import cv2 as cv
# import pandas as pd
import numpy as np
from tools import *
from matplotlib import pyplot as plt

from numpy import abs, log
from numpy.fft import fft2, fftshift

real = lambda c: c.real
imag = lambda c: c.imag
mag = lambda x, y: (x*x + y*y)**(1/2)
magCplx = lambda c: mag(c.real, c.imag)



get_angulo = lambda x, y: np.arccos( x / np.linalg.norm([x,y]))
get_angulo_deg = lambda x, y: (get_angulo(x, y)/(2*np.pi)) * 360;

def mkhist(f):
    my = int(np.ceil(f.shape[0]/2))
    dx, dy = f.shape[1]/2, f.shape[0]/2
    # Acrescenta o valor val ao histograma hist, retornando-o.
    # O valor é dividido parte para o angulo floor(rad) parte
    # para ceil(rad).
    def acc(hist, val, x, y):
        if x-dx == 0 and y-dy == 0:
            return hist
        rad = get_angulo_deg (x-dx, y-dy);
        hist[int(np.floor(rad)) % 360] += (1 - (rad % 1)) * val
        hist[int(np.ceil(rad)) % 360 ] += (rad % 1) * val
        return hist        
    
    hist = np.ndarray(360)
    foldl_imgxy(f[:my], acc, hist)
    return hist

# Calcula a DFT da imagem GRAYSCALE img
def fourier(img):
    return pipe(img, fft2, fftshift, abs, log)
    


def resize (img, alpha):
    h, w = img.shape[0], img.shape[1]
    return cv.resize(img, (alpha * h, alpha * w), interpolation=cv.INTER_LINEAR)
    
# Cria uma imagem com o plot do histograma
def hist2img(hist, height=360):
    img = np.zeros((360, height))
    m = hist.max()
    if m == 0:
        return img.transpose()
    
    for deg in range(360):
        x = hist[deg]
        qtt = int(np.floor(height * (x/m)))
        qtt = height-qtt
        img[deg][0:qtt] = 0
        img[deg][qtt:] = 1

    return img.transpose()

def find_circles(img):
    blured = cv.GaussianBlur(img, (0,0), 3)
    borders = cv.Canny(blured, 80, 80)
    circles = cv.HoughCircles(borders, cv.HOUGH_GRADIENT, 1, 50,
                              param1=100, param2=20,
                              minRadius=5, maxRadius=100)

    if circles is not None:
        cs = []
        circles = np.uint16(np.around(circles))[0,:]
        for x, y, rad in circles:
            sx = x-rad
            cs.append(img[y-rad:y+rad, x-rad:x+rad])

    return circles, cs
    

def show(img, fourier, hist):
    plt.subplot(131), plt.imshow(img, cmap='gray'),
    plt.subplot(132), plt.imshow(fourier, cmap='gray')
    plt.subplot(133), plt.imshow(hist2img(hist, img.shape[0]), cmap='gray'), plt.axis([0,360,0,hist.max()])
    plt.show()

def main():
    img = cv.imread('teste2.jpg', 0)
    # img = np.ndarray((900, 900))
    # img = mapimgxy(img, lambda v, x, y: 255 if x >100 and x <150 and y >100 and y <150 else 0)

    _, cs = find_circles(img)
    c = cs[2]
    
    f = fourier(c)
    h = mkhist(f)
    show(c, f, h)


    
if __name__ == "__main__":
    main()

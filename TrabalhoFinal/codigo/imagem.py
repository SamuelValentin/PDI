import cv2 as cv
import numpy as np
from tools import *
from matplotlib import pyplot as plt

from numpy import abs, log
from numpy.fft import fft2, fftshift


def mapimg(img, f, dtype=None, channels=None):
    if img is None:
        return img

    if dtype is None:
        dtype = img.dtype

    shape = img.shape
    if channels is not None:
        shape = (shape[0], shape[1], channels)
        
    out = np.ndarray(shape, dtype=dtype)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y][x] = f(img[y][x])
    return out



def mapimgxy(img, f, dtype=None, channels=None):
    if img is None:
        return img

    if dtype is None:
        dtype = img.dtype

    shape = img.shape
    if channels is not None:
        shape = (shape[0], shape[1], channels)
        
    out = np.ndarray(shape, dtype=dtype)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            out[y][x] = f(img[y][x], x, y)
    return out


def foldl_imgxy(img, f, res=0):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            res = f(res, img[y][x], x, y)
    return res


def foldl_img(img, f, res=0):
    for rows in img:
        for pix in rows:
            res = f(res, pix)
    return res



def foldl_i(arr, f, acc=0):
    for i in range(len(arr)):
        acc = f(acc, arr[i], i)
    return acc


def showimgs(*imgs):
    for img in imgs:
        cv.imshow('-', img)
        cv.waitKey()
    cv.destroyAllWindows()


def rotate_image(image, angle):
  center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result


get_angulo = lambda x, y: np.arccos( x / np.linalg.norm([x,y]))
get_angulo_deg = lambda x, y: (get_angulo(x, y)/(2*np.pi)) * 360;

# Calcula a DFT da imagem GRAYSCALE img
def fourier(img):
    return pipe(img, fft2, fftshift, abs, log)
    


def resize (img, alpha):
    h, w = img.shape[0], img.shape[1]
    return cv.resize(img, (alpha * h, alpha * w), interpolation=cv.INTER_LINEAR)
    

def encontra_circulos(img, maxrad=None, minrad=None, mindist=50, acc=20, sigma=5, threshold=80):
    maxrad = maxrad if maxrad is not None else max(img.shape)
    minrad = minrad if minrad is not None else 3

    img_    = cv.GaussianBlur(img, (0,0), sigma)
    circles = cv.HoughCircles(img_, cv.HOUGH_GRADIENT, 1, mindist,
                              param1=threshold, param2=acc,
                              minRadius=minrad, maxRadius=maxrad)

    circles = [] if circles is None else np.uint16(np.around(circles))[0,:]

    return circles


def show(*imgs, cmap=None, title=''):
    if len(imgs) < 1 or len(imgs) > 25:
        raise Exception("show(imgs): imgs must be an array with 0 to 25 images")

    idx = 1
    rows = int(np.ceil(len(imgs) / 5))
    cols = len(imgs) % 5 if len(imgs) < 5 else 5
    for img in imgs:
        if len(img.shape) < 2:
            img = hist2img(img)

        plt.subplot(rows, cols, idx)
        plt.title(title)
        plt.imshow(img, cmap=cmap)
        idx += 1
            
    plt.show()


def fourier_hist(f):
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


'''
  histograma dos angulos
'''
def gradient_hist(magnitudes, angles):
    hist = np.zeros(360)
    for y in range(magnitudes.shape[0]):
        for x in range(magnitudes.shape[1]):
            deg = angles[y][x] % 180
            val = magnitudes[y][x]
            val_up = val * (deg%1)
            val_down = val - val_up
            up, down = int(np.ceil(deg)) % 360, int(np.floor(deg)) % 360
            hist[ up ] += val_up
            hist[down] += val_down
    return hist

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


def draw_circles(circles, img):
    for x, y, rad in circles:
        cv.circle(img, (x, y), rad, (255, 0, 255), 1)
        cv.circle(img, (x, y), 2, (255, 0, 255), 2)
    return img

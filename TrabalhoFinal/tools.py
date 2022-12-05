import cv2 as cv
import numpy as np
from pprint import pprint
# import pandas as pd
from functools import reduce
#------------------------------------------
# Predicates
is_tuple = lambda t: type(t) == tuple
is_str = lambda s: type(s) == str
is_None = lambda x: x == None

isnt_None = lambda x: x is not None

not_ = lambda x: not x

trueP = lambda _: True
falseP = lambda _: False
#-------------------------------------------

#-------------------------------------------
# Tuples
pair = lambda x, y: (x, y)
unit = lambda x: (x,)
fst = lambda t: t[0]
snd = lambda t: t[1]
trd = lambda t: t[2]
yz_ = lambda t: (t[1], t[2])
xy_ = lambda t: (t[0], t[1])
xz_ = lambda t: (t[0], t[2])


# Normaliza 
norm = lambda min, max, alpha=1: lambda x: ((x-min)/(max-min))*alpha


#-------------------------------------------
# Functional

glue = lambda f, g: lambda x: f(g(x))
foreach = lambda xs, f: [f(x) for x in xs]
zipwith = lambda xs, ys, f=pair:  [ f(x, y) for x, y in zip(xs, ys) ]
zipcross = lambda xs, ys, f=pair: [ f(x, y) for x in xs for y in ys ]
join = lambda *ss: ''.join(ss)

dif_ = lambda x: lambda y: x - y
sum_ = lambda x: lambda y: x + y
mul_ = lambda x: lambda y: x * y
div_ = lambda x: lambda y: x / y
divby_ = lambda x: lambda y: y / x

pipe = lambda x,*fs: reduce(lambda x, f: f(x), fs, x)
pipeC = lambda *fs: lambda x: reduce(lambda x, f: f(x), fs, x)
#-------------------------------------------

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



def showimgs(*imgs):
    for img in imgs:
        cv.imshow('-', img)
        cv.waitKey()
    cv.destroyAllWindows()


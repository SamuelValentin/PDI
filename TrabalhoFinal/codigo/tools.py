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

gtC = lambda than: lambda x: x > than
geC = lambda than: lambda x: x >= than
ltC = lambda than: lambda x: x < than
leC = lambda than: lambda x: x <= than
eqC = lambda than: lambda x: x == than

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
flip = lambda t: (t[1], t[0])

star = lambda f: lambda tup: f(*tup)

tup3 = lambda x: (x, x, x)

# Normaliza 
norm = lambda min, max, alpha=1: lambda x: ((x-min)/(max-min))*alpha


def updateArr(arr, n, f):
    arr[n] = f(arr[n])
    return arr
    
updateArrC = lambda arr: lambda n, f: updateArr(arr, n, f)

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

dif = lambda x, *xs: reduce(lambda x, y: x-y, x)
mul = lambda *xs: reduce(lambda x, y: x*y, xs, 1)

pipe = lambda x,*fs: reduce(lambda x, f: f(x), fs, x)
pipeC = lambda *fs: lambda x: reduce(lambda x, f: f(x), fs, x)

mapC = lambda f: lambda x: list(map(f, x))
foldl = lambda arr, f, acc=0: reduce(f, arr, acc)
foldlC = lambda f, acc=0: lambda arr: reduce(f, arr, acc)

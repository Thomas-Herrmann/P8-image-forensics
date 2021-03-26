import cv2 as cv
from functools import partial
import skimage
import skimage.io
import numpy as np
from enum import Enum
import PIL.Image
import PIL.ImageOps
import random

class ImageType(Enum):
    IMG_NONE = -1
    IMG_CV = 0
    IMG_SK = 1
    IMG_PIL = 2

class Image:
    def __init__(self, path, type=ImageType.IMG_NONE, internal=None):
        self.path = path
        self._type = type
        self._internal_img = internal

    def get_cv_image(self):
        if self._type != ImageType.IMG_CV:
            self._type = ImageType.IMG_CV
            self._internal_img = cv.imread(self.path)
        return self._internal_img

    def get_sk_image(self):
        if self._type != ImageType.IMG_SK:
            self._type = ImageType.IMG_SK
            self._internal_img = skimage.io.imread(self.path)
        return self._internal_img

    def get_pil_image(self):
        if self._type != ImageType.IMG_PIL:
            self._type = ImageType.IMG_PIL
            self._internal_img = PIL.Image.open(self.path)
        return self._internal_img

    def save(self, filename=None):
        if filename is None:
            filename = self.path
        
        if self._type == ImageType.IMG_NONE:
            self.get_cv_image()
            self.save()
        if self._type == ImageType.IMG_CV:
            cv.imwrite(filename, self._internal_img)
        elif self._type == ImageType.IMG_SK:
            skimage.io.imsave(filename, self._internal_img)
        elif self._type == ImageType.IMG_PIL:
            self._internal_img.save(filename)

    def get_clone(self, new_internal):
        return Image(self.path, self._type, new_internal)
    


def boxblur(ksize):
    def man_func(img):
        cv_img = img.get_cv_image()
        cv_img = cv.boxFilter(cv_img, -1, (ksize, ksize))
        return img.get_clone(cv_img)

    return (man_func, "BoxBlur-K" + str(ksize), {"ksize": ksize})

def gausblur(ksize):
    def man_func(img):
        cv_img = img.get_cv_image()
        cv_img = cv.GaussianBlur(cv_img, (ksize, ksize), 0)
        return img.get_clone(cv_img)

    return (man_func, "GausBlur-K" + str(ksize), {"ksize": ksize})

def medianblur(ksize):
    def man_func(img):
        cv_img = img.get_cv_image()
        cv_img = cv.medianBlur(cv_img, ksize)
        return img.get_clone(cv_img)

    return (man_func, "MedianBlur-K" + str(ksize), {"ksize": ksize})


def resize_(interpolation, name, f):
    def man_func(img):
        cv_img = img.get_cv_image()

        (orig_width, orig_height) = cv_img.shape[:2]
        cv_img = cv.resize(cv_img, None, fx=f, fy=f, interpolation=interpolation)
        cv_img = cv.resize(cv_img, (orig_height, orig_width), interpolation=interpolation)

        return img.get_clone(cv_img)
    
    return (man_func, name + str(f), {"f": f})

area_resize = partial(resize_, cv.INTER_AREA, "AREAResize-s")
cubic_resize = partial(resize_, cv.INTER_CUBIC, "CUBICResize-s")
lanczos4_resize = partial(resize_, cv.INTER_LANCZOS4, "LANCZOS4Resize-s")
linear_resize = partial(resize_, cv.INTER_LINEAR, "LINEARResize-s")
nearest_resize = partial(resize_, cv.INTER_NEAREST, "NEARESTResize-s")


def jpeg_compress(q):
    def man_func(img):
        cv_img = img.get_cv_image()
        result, encoded = cv.imencode(".jpg", cv_img, [int(cv.IMWRITE_JPEG_QUALITY), q])
        decoded = cv.imdecode(encoded, 1)
        return img.get_clone(decoded)
    
    return (man_func, "JPEGCompress-Q" + str(q), {"q": q})

def jpeg_double_compress(qs):
    (q1, q2) = qs
    def man_func(img):
        cv_img = img.get_cv_image()

        _, encoded1 = cv.imencode(".jpg", cv_img, [int(cv.IMWRITE_JPEG_QUALITY), q1])
        decoded1 = cv.imdecode(encoded1, 1)
        _, encoded2 = cv.imencode(".jpg", decoded1, [int(cv.IMWRITE_JPEG_QUALITY), q2])
        decoded2 = cv.imdecode(encoded2, 1)

        return img.get_clone(decoded2)
    
    return (man_func, f"JPEGDoubleCompress-Q{str(q1)}-{str(q2)}", {"q1": q1, "q2": q2})

def webp_compress(q):
    def man_func(img):
        cv_img = img.get_cv_image()
        result, encoded = cv.imencode(".webp", cv_img, [int(cv.IMWRITE_WEBP_QUALITY), q])
        decoded = cv.imdecode(encoded, 1)
        return img.get_clone(decoded)
    
    return (man_func, "WEBPCompress-Q" + str(q), {"q": q})


def morph_(morph_op, name, s):
    retval = cv.getStructuringElement(cv.MORPH_ELLIPSE, (s,s))

    def man_func(img):
        cv_img = img.get_cv_image()
        cv_img =  cv.morphologyEx(cv_img, morph_op, retval)
        return img.get_clone(cv_img)
    
    return (man_func, name + str(s), {"s": s})

close_morph = partial(morph_, cv.MORPH_CLOSE, "CLOSEMorph-S")
dilate_morph = partial(morph_, cv.MORPH_DILATE, "DILATEMorph-S")
erode_morph = partial(morph_, cv.MORPH_ERODE, "ERODEMorph-S")
open_morph = partial(morph_, cv.MORPH_OPEN, "OPENMorph-S")


def gausnoise(s):
    def man_func(img):
        sk_img = img.get_sk_image()
        sk_img = skimage.util.random_noise(sk_img, mode="gaussian", var=s)
        return img.get_clone(sk_img)

    return (man_func, "GaussianNoise-S" + str(s), {"s": s})

def impulsenoise(p):
    def man_func(img):
        sk_img = img.get_sk_image()
        sk_img = skimage.util.random_noise(sk_img, mode="s&p", salt_vs_pepper=p)
        return img.get_clone(sk_img)

    return (man_func, "ImpulseNoise-P" + str(p), {"p": p})

def poissonnoise():
    def man_func(img):
        sk_img = img.get_sk_image()
        
        out = skimage.util.random_noise(sk_img, mode="poisson")

        return img.get_clone(out)

    return (man_func, "PoissonNoise", {})

def uniformnoise(m):
    def man_func(img):
        sk_img = img.get_sk_image()
        sk_img = skimage.util.random_noise(sk_img, mode="speckle", mean=m)
        sk_img = skimage.util.img_as_uint(sk_img)
        return img.get_clone(sk_img)

    return (man_func, "SpeckleNoise-M" + str(m), {"m": m})


def quantize(c):
    def man_func(img):
        pil_img = img.get_pil_image()
        new = pil_img.quantize(c, dither=PIL.Image.NONE)
        return img.get_clone(new)

    return (man_func, "Quantization-C" + str(c), {"c": c})

def dither():
    def man_func(img):
        pil_img = img.get_pil_image()
        new = pil_img.convert(mode="P", dither=PIL.Image.FLOYDSTEINBERG)
        return img.get_clone(new)

    return (man_func, "Dither", {})

def autocontrast(c):
    def man_func(img):
        pil_img = img.get_pil_image()
        new = pil_img.convert(mode="RGB", dither=PIL.Image.FLOYDSTEINBERG)
        new = PIL.ImageOps.autocontrast(new, c*10)
        return img.get_clone(new)
    
    return (man_func, "AutoContrast-C" + str(c), {"c": c})

def clahe(s):
    def man_func(img):
        cv_img = img.get_cv_image()
        gray = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(s,s))
        new = clahe.apply(gray)
        return img.get_clone(new)
    
    return (man_func, "HistEq-S" + str(s), {"s": s})


boxblur_kernels = [29, 11, 13, 15, 17, 19, 25, 27, 21, 23, 3, 7, 5, 9, 33, 31]
gausblur_kernels = [33, 31, 5, 7, 3, 9, 11, 13, 15, 17, 19, 29, 21, 23, 25, 27]
medianblur_kernels = [21, 23, 25, 27, 29, 19, 15, 17, 11, 13, 9, 3, 7, 5]
#TODO: waveletblur

area_resize_factors = [0.95, 0.73, 0.79, 0.36, 0.31, 0.89, 0.84, 0.63, 0.68, 0.25, 0.20, 0.57, 0.52, 0.15, 0.41, 0.47]
cubic_resize_factors = [0.36, 0.68, 0.63, 0.84, 0.89, 0.31, 0.79, 0.73, 0.95, 0.47, 0.41, 0.15, 0.57, 0.25, 0.20]
lanczos4_resize_factors = [0.63, 0.31, 0.79, 0.73, 0.47, 0.41, 0.89, 0.84, 0.15, 0.52, 0.57, 0.25, 0.20, 0.68, 0.63, 0.95]
linear_resize_factors = [0.41, 0.47, 0.95, 0.73, 0.79, 0.36, 0.31, 0.84, 0.63, 0.68, 0.25, 0.20, 0.57, 0.52, 0.89, 0.15]
nearest_resize_factors = [0.95, 0.20, 0.25, 0.31, 0.79, 0.73, 0.36, 0.41, 0.47, 0.52, 0.57, 0.84, 0.89, 0.15, 0.63, 0.68]

jpeg_compress_qualities = [57, 53, 91, 95, 61, 65, 38, 100, 74, 70, 78, 48, 44, 40, 82, 87]
jpeg_double_compress_qualities = [
    (95, 40), (36, 74), (95, 57), (61, 65), (70, 100), (95, 65), (53, 74), (87, 100), (78, 57), 
    (87, 48), (95, 100), (70, 65), (78, 40), (78, 48), (87, 65), (53, 40), (61, 57), (87, 57), 
    (98, 82), (70, 97), (53, 57), (61, 40), (61, 74), (44, 48), (87, 40), (95, 91), (70, 82), 
    (87, 82), (87, 91), (53, 65), (36, 48), (44, 74), (36, 100), (61, 91), (36, 65), (53, 48), 
    (53, 82), (36, 57), (78, 100), (78, 74), (53, 91), (70, 57), (87, 74), (61, 48), (70, 40), 
    (98, 48), (78, 91), (44, 40), (87, 65), (36, 40), (53, 100), (44, 100), (36, 91), (36, 82), 
    (40, 74), (61, 100), (61, 82), (44, 82), (78, 82), (44, 57), (44, 65), (95, 74), (70, 48), (44, 91)
    ]
webp_compress_qualities = [53, 87, 82, 100, 65, 61, 95, 91, 36, 57, 78, 70, 74, 40, 44, 48]

close_morph_kernels = [19, 16, 14, 12, 10, 27, 23, 21, 25, 29, 34, 31, 6, 4, 2, 8]
dilate_morph_kernels = [21, 23, 25, 27, 19, 14, 16, 10, 12, 29, 31, 34, 8, 2, 4, 6]
erode_morph_kernels = [16, 31, 34, 21, 23, 25, 27, 29, 8, 2, 6, 4, 19, 14, 10, 12]
open_morph_kernels = [10, 12, 14, 16, 19, 21, 23, 25, 27, 2, 6, 4, 8, 31, 34, 29]

gausnoise_vars = [0.03, 0.07, 0.05, 0.09, 0.30, 0.32, 0.35, 0.22, 0.20, 0.26, 0.24, 0.28, 0.17, 0.15, 0.13, 0.11]
impulsenoise_props = [0.11, 0.17, 0.15, 0.09, 0.03, 0.05, 0.07, 0.28, 0.26, 0.24, 0.22, 0.20, 0.30, 0.32, 0.35]
poissonnoise_lams = [24, 26, 20, 22, 28, 3, 7, 5, 9, 35, 32, 30, 15, 17, 11, 13]
uniformnoise_means = [0.35, 0.30, 0.32, 0.07, 0.05, 0.03, 0.09, 0.13, 0.11, 0.17, 0.15, 0.28, 0.26, 0.24, 0.22, 0.20]

quantization_cols = [125, 225, 175, 100, 200, 150, 75]
autocontrast_cutoffs = [2, 3, 0, 1, 6, 7, 4, 5]
clahe_sizes = [7, 6, 5, 4, 3, 2, 1]


boxblur_mans = list(map(boxblur, boxblur_kernels))
gausblur_mans = list(map(gausblur, gausblur_kernels))
medianblur_mans = list(map(medianblur, medianblur_kernels))

area_resize_mans = list(map(area_resize, area_resize_factors))
cubic_resize_mans = list(map(cubic_resize, cubic_resize_factors))
lanczos4_resize_mans = list(map(lanczos4_resize, lanczos4_resize_factors))
linear_resize_mans = list(map(linear_resize, linear_resize_factors))
nearest_resize_mans = list(map(nearest_resize, nearest_resize_factors))

jpeg_compress_mans = list(map(jpeg_compress, jpeg_compress_qualities))
jpeg_double_compress_mans = list(map(jpeg_double_compress, jpeg_double_compress_qualities))
webp_compress_mans = list(map(webp_compress, webp_compress_qualities))

close_morph_mans = list(map(close_morph, close_morph_kernels))
dilate_morph_mans = list(map(dilate_morph, dilate_morph_kernels))
erode_morph_mans = list(map(erode_morph, erode_morph_kernels))
open_morph_mans = list(map(open_morph, open_morph_kernels))

gausnoise_mans = list(map(gausnoise, gausnoise_vars))
impulsenoise_mans = list(map(impulsenoise, impulsenoise_props))
poissonnoise_mans = [poissonnoise()]
uniformnoise_mans = list(map(uniformnoise, uniformnoise_means))

quanti_mans = list(map(quantize, quantization_cols))
dither_mans = [dither()]

autocontrast_mans = list(map(autocontrast, autocontrast_cutoffs))
clahe_mans = list(map(clahe, clahe_sizes))

blur_mans = [boxblur_mans, gausblur_mans, medianblur_mans]
resize_mans = [area_resize_mans, cubic_resize_mans, lanczos4_resize_mans, linear_resize_mans, nearest_resize_mans]
compress_mans = [jpeg_compress_mans, jpeg_double_compress_mans, webp_compress_mans]
morph_mans = [close_morph_mans, dilate_morph_mans, erode_morph_mans, open_morph_mans]
noise_mans = [gausnoise_mans, impulsenoise_mans, poissonnoise_mans, uniformnoise_mans]
quantization_mans = [quanti_mans, dither_mans]
resampling_mans = [autocontrast_mans, clahe_mans]

manipulations_tree = [blur_mans, resize_mans, compress_mans, morph_mans, noise_mans, quantization_mans, resampling_mans]
manipulations_tree = [resize_mans, compress_mans, morph_mans, noise_mans, quantization_mans]

flatten = lambda t: [item for sublist in t for item in sublist]
manipulations_flat = flatten(map(flatten, manipulations_tree))

def get_random_manipulation():
    i = random.randint(0, len(manipulations_tree) - 1)
    j = random.randint(0, len(manipulations_tree[i]) - 1)
    k = random.randint(0, len(manipulations_tree[i][j]) - 1)
    return manipulations_tree[i][j][k]

#for i in range(20):
#    img_man = Image("data/input.jpg")
#    img_man.save("data/temp.png")
#    for _ in range(10):
#        img_man = Image("data/temp.png")
#        (man_func, name, pars) = get_random_manipulation()
#        print(name)
#        img_man = man_func(img_man)
#        img_man.save("data/temp.png")
#    img_man.save(f"data/out_random{i}.png")

img = Image("data/input.jpg")
for (man_func, name, pars) in manipulations_flat:
    img_man = man_func(img)
    img_man.save(f"data/out_{name}.png")
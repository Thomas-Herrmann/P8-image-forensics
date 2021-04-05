import cv2 as cv
from functools import partial
import skimage
import skimage.io
import numpy as np
from enum import Enum
import PIL.Image
import PIL.ImageOps
import random
import os
import os.path
import glob

class ImageType(Enum):
    NONE = -1
    CV = 0
    SK = 1
    PIL = 2

class ManiFamily(Enum):
    BLUR = "blur"
    RESIZE = "resize"
    COMPRESS = "compression"
    MORPH = "morphology"
    NOISE = "noise"
    QUANTIZATION = "quantization"
    RESAMPLING = "resampling"

class Image:
    temp_path = "data/temp.png"

    def __init__(self, path, type=ImageType.NONE, internal=None, modified=False):
        self.path = path
        self._type = type
        self._internal_img = internal
        self._modified = modified

    def crop(self, x, y, width, height):
        cv_img = self.get_cv_image()
        cv_img = cv_img[y:y+height, x:x+width]
        return self.get_clone(cv_img)

    def patches(self, psizex, psizey):
        width, height = self.shape()
        nx = width // psizex
        ny = height // psizey

        return [self.crop(x * psizex, y * psizey, psizex, psizey) for x in range(nx) for y in range(ny)]

    def intensity_deviation(self):
        return np.std(np.sum(self._internal_img, -1) / 3)

    def shape(self):
        cv_img = self.get_cv_image()
        height, width, _ = cv_img.shape
        return (width, height)

    def _get_image(self, img_type, img_func):
        path = self.path
        if self._modified:
            self.save(self.temp_path)
            path = self.temp_path

        if self._type != img_type:
            self._type = img_type
            self._internal_img = img_func(path)
        
        if self._modified:
            try:
                os.remove(self.temp_path)
            except: pass
        return self._internal_img

    def get_cv_image(self):
        return self._get_image(ImageType.CV, cv.imread)

    def get_sk_image(self):
        return self._get_image(ImageType.SK, skimage.io.imread)

    def get_pil_image(self):
        return self._get_image(ImageType.PIL, PIL.Image.open)

    def save(self, filename=None):
        if filename is None:
            filename = self.path
        
        if self._type == ImageType.NONE:
            self.get_cv_image()
            self.save()

        if self._type == ImageType.CV:
            cv.imwrite(filename, self._internal_img)
        elif self._type == ImageType.SK:
            skimage.io.imsave(filename, self._internal_img)
        elif self._type == ImageType.PIL:
            self._internal_img.save(filename)
        else:
            raise Exception("Tried to save invalid file type")

    def get_clone(self, new_internal):
        return Image(self.path, self._type, new_internal, True)
    

def boxblur(ksize):
    def man_func(img):
        cv_img = img.get_cv_image()
        cv_img = cv.boxFilter(cv_img, -1, (ksize, ksize))
        return img.get_clone(cv_img)

    return (man_func, ManiFamily.BLUR, "BoxBlur", {"ksize": ksize})

def gausblur(ksize):
    def man_func(img):
        cv_img = img.get_cv_image()
        cv_img = cv.GaussianBlur(cv_img, (ksize, ksize), 0)
        return img.get_clone(cv_img)

    return (man_func, ManiFamily.BLUR, "GausBlur", {"ksize": ksize})

def medianblur(ksize):
    def man_func(img):
        cv_img = img.get_cv_image()
        cv_img = cv.medianBlur(cv_img, ksize)
        return img.get_clone(cv_img)

    return (man_func, ManiFamily.BLUR, "MedianBlur", {"ksize": ksize})


def resize_(interpolation, name, f):
    def man_func(img):
        cv_img = img.get_cv_image()

        (orig_width, orig_height) = cv_img.shape[:2]
        cv_img = cv.resize(cv_img, None, fx=f, fy=f, interpolation=interpolation)
        cv_img = cv.resize(cv_img, (orig_height, orig_width), interpolation=interpolation)

        return img.get_clone(cv_img)
    
    return (man_func, ManiFamily.RESIZE, name, {"f": f})

area_resize = partial(resize_, cv.INTER_AREA, "AREAResize")
cubic_resize = partial(resize_, cv.INTER_CUBIC, "CUBICResize")
lanczos4_resize = partial(resize_, cv.INTER_LANCZOS4, "LANCZOS4Resize")
linear_resize = partial(resize_, cv.INTER_LINEAR, "LINEARResize")
nearest_resize = partial(resize_, cv.INTER_NEAREST, "NEARESTResize")


def jpeg_compress(q):
    def man_func(img):
        cv_img = img.get_cv_image()
        result, encoded = cv.imencode(".jpg", cv_img, [int(cv.IMWRITE_JPEG_QUALITY), q])
        decoded = cv.imdecode(encoded, 1)
        return img.get_clone(decoded)
    
    return (man_func, ManiFamily.COMPRESS, "JPEGCompress", {"q": q})

def jpeg_double_compress(qs):
    (q1, q2) = qs
    def man_func(img):
        cv_img = img.get_cv_image()

        _, encoded1 = cv.imencode(".jpg", cv_img, [int(cv.IMWRITE_JPEG_QUALITY), q1])
        decoded1 = cv.imdecode(encoded1, 1)
        _, encoded2 = cv.imencode(".jpg", decoded1, [int(cv.IMWRITE_JPEG_QUALITY), q2])
        decoded2 = cv.imdecode(encoded2, 1)

        return img.get_clone(decoded2)
    
    return (man_func, ManiFamily.COMPRESS, "JPEGDoubleCompress", {"q1": q1, "q2": q2})

def webp_compress(q):
    def man_func(img):
        cv_img = img.get_cv_image()
        result, encoded = cv.imencode(".webp", cv_img, [int(cv.IMWRITE_WEBP_QUALITY), q])
        decoded = cv.imdecode(encoded, 1)
        return img.get_clone(decoded)
    
    return (man_func, ManiFamily.COMPRESS, "WEBPCompress", {"q": q})


def morph_(morph_op, name, s):
    retval = cv.getStructuringElement(cv.MORPH_ELLIPSE, (s,s))

    def man_func(img):
        cv_img = img.get_cv_image()
        cv_img =  cv.morphologyEx(cv_img, morph_op, retval)
        return img.get_clone(cv_img)
    
    return (man_func, ManiFamily.MORPH, name, {"s": s})

close_morph = partial(morph_, cv.MORPH_CLOSE, "CLOSEMorph")
dilate_morph = partial(morph_, cv.MORPH_DILATE, "DILATEMorph")
erode_morph = partial(morph_, cv.MORPH_ERODE, "ERODEMorph")
open_morph = partial(morph_, cv.MORPH_OPEN, "OPENMorph")


def gausnoise(s):
    def man_func(img):
        sk_img = img.get_sk_image()
        sk_img = skimage.util.random_noise(sk_img, mode="gaussian", var=s)
        return img.get_clone(sk_img)

    return (man_func, ManiFamily.NOISE, "GaussianNoise", {"s": s})

def impulsenoise(p):
    def man_func(img):
        sk_img = img.get_sk_image()
        sk_img = skimage.util.random_noise(sk_img, mode="s&p", salt_vs_pepper=p)
        return img.get_clone(sk_img)

    return (man_func, ManiFamily.NOISE, "ImpulseNoise", {"p": p})

def poissonnoise():
    def man_func(img):
        sk_img = img.get_sk_image()
        sk_img = skimage.util.random_noise(sk_img, mode="poisson")
        return img.get_clone(sk_img)

    return (man_func, ManiFamily.NOISE, "PoissonNoise", {})

def uniformnoise(m):
    def man_func(img):
        sk_img = img.get_sk_image()
        sk_img = skimage.util.random_noise(sk_img, mode="speckle", mean=m)
        sk_img = skimage.util.img_as_uint(sk_img)
        return img.get_clone(sk_img)

    return (man_func, ManiFamily.NOISE, "SpeckleNoise", {"m": m})


def quantize(c):
    def man_func(img):
        pil_img = img.get_pil_image()
        new = pil_img.quantize(c, dither=PIL.Image.NONE)
        return img.get_clone(new)

    return (man_func, ManiFamily.QUANTIZATION, "Quantization", {"c": c})

def dither():
    def man_func(img):
        pil_img = img.get_pil_image()
        new = pil_img.convert(mode="P", dither=PIL.Image.FLOYDSTEINBERG)
        return img.get_clone(new)

    return (man_func, ManiFamily.QUANTIZATION, "Dither", {})

def autocontrast(c):
    def man_func(img):
        pil_img = img.get_pil_image()
        new = pil_img.convert(mode="RGB", dither=PIL.Image.NONE)
        new = PIL.ImageOps.autocontrast(new, c*10)
        return img.get_clone(new)
    
    return (man_func, ManiFamily.RESAMPLING, "AutoContrast", {"c": c})

def clahe(s):
    def man_func(img):
        cv_img = img.get_cv_image()
        gray = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(s,s))
        new = clahe.apply(gray)
        return img.get_clone(new)
    
    return (man_func, ManiFamily.RESAMPLING, "HistEq", {"s": s})


boxblur_kernels = [29, 11, 13, 15, 17, 19, 25, 27, 21, 23, 3, 7, 5, 9, 33, 31]
gausblur_kernels = [33, 31, 5, 7, 3, 9, 11, 13, 15, 17, 19, 29, 21, 23, 25, 27]
medianblur_kernels = [21, 23, 25, 27, 29, 19, 15, 17, 11, 13, 9, 3, 7, 5]

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

flatten = lambda t: [item for sublist in t for item in sublist]
manipulations_flat = flatten(map(flatten, manipulations_tree))

def get_random_manipulation():
    i = random.randint(0, len(manipulations_tree) - 1)
    j = random.randint(0, len(manipulations_tree[i]) - 1)
    k = random.randint(0, len(manipulations_tree[i][j]) - 1)
    return manipulations_tree[i][j][k]

def generate_patches(input_folder, output_folder, use_manipulations = True):
    files = glob.glob(input_folder + "/*.jpg") + glob.glob(input_folder + "/*.png")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for f in files:
        path, name_ext = os.path.split(f)
        name, _ = os.path.splitext(name_ext)
        
        img = Image(path + "/" + name_ext)

        for i, patch in enumerate(img.patches(256, 256)):
            if patch.intensity_deviation() >= 32:
                folder = f"{output_folder}/"
                if use_manipulations:
                    man_func, man_family, man_name, man_pars = get_random_manipulation()
                    folder += get_path_from_manipulation(man_family, man_name, man_pars)
                    patch = man_func(patch)
                else:
                    folder += "none/"
                folder = folder.replace(".", "")

                if not os.path.exists(folder):
                    os.makedirs(folder)

                patch.save(folder + name + str(i) + ".png")

def generate_manipulated(input_folder, output_folder):
    files = glob.glob(input_folder + "/*.jpg") + glob.glob(input_folder + "/*.png")
    num_files = len(files)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, f in enumerate(files):
        path, name_ext = os.path.split(f)
        name, _ = os.path.splitext(name_ext)
        
        img = Image(path + "/" + name_ext)

        folder = f"{output_folder}/"
            
        man_func, man_family, man_name, man_pars = get_random_manipulation()
        folder += get_path_from_manipulation(man_family, man_name, man_pars)
        img = man_func(img)
        
        folder = folder.replace(".", "")

        if not os.path.exists(folder):
            os.makedirs(folder)

        img.save(folder + "/" + name + ".png")

        if i % 500 == 0:
            print(f"Manipulated {i}/{num_files} ({i/num_files*100}%) images")

def get_path_from_manipulation(family, name, parameters):
    path = str(family.value) + "-" + name

    for k, v in parameters.items():
        path += "-" + str(k) + "-" + str(v)

    return path

def convert_jpg_png(input_path, output_path):
    files = glob.glob(input_path + "/*.jpg")
    num_files = len(files)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, f in enumerate(files):
        path, name_ext = os.path.split(f)
        name, _ = os.path.splitext(name_ext)

        img = Image(path + "/" + name_ext)
        img.save(output_path + "/" + name + ".png")

        if i % 500 == 0:
            print(f"Converted {i}/{num_files} ({i/num_files*100}%) images to png")

    
#convert_jpg_png("data/train2017", "data/train2017_png")
#generate_manipulated("data/train2017", "data/manipulated")
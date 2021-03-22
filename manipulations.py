import cv2 as cv
from functools import partial

def boxblur(ksize):
    def man_func(img):
        return cv.boxFilter(img, -1, (ksize, ksize))

    return (man_func, "BoxBlur-K" + str(ksize), {"ksize": ksize})

def gausblur(ksize):
    def man_func(img):
        return cv.GaussianBlur(img, (ksize, ksize), 0)

    return (man_func, "GausBlur-K" + str(ksize), {"ksize": ksize})

def medianblur(ksize):
    def man_func(img):
        return cv.medianBlur(img, ksize)

    return (man_func, "MedianBlur-K" + str(ksize), {"ksize": ksize})


def resize_(interpolation, name, f):
    def man_func(img):
        (orig_width, orig_height) = img.shape[:2]
        img = cv.resize(img, None, fx=f, fy=f, interpolation=interpolation)
        return cv.resize(img, (orig_height, orig_width), interpolation=interpolation)
    
    return (man_func, name + str(f), {"f": f})

area_resize = partial(resize_, cv.INTER_AREA, "AREAResize-s")
cubic_resize = partial(resize_, cv.INTER_CUBIC, "CUBICResize-s")
lanczos4_resize = partial(resize_, cv.INTER_LANCZOS4, "LANCZOS4Resize-s")
linear_resize = partial(resize_, cv.INTER_LINEAR, "LINEARResize-s")
nearest_resize = partial(resize_, cv.INTER_NEAREST, "NEARESTResize-s")


def jpeg_compress(q):
    def man_func(img):
        result, encoded = cv.imencode(".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), q])
        decoded = cv.imdecode(encoded, 1)
        return decoded
    
    return (man_func, "JPEGCompress-Q" + str(q), {"q": q})

def jpeg_double_compress(qs):
    (q1, q2) = qs
    def man_func(img):
        _, encoded1 = cv.imencode(".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), q1])
        decoded1 = cv.imdecode(encoded1, 1)
        _, encoded2 = cv.imencode(".jpg", decoded1, [int(cv.IMWRITE_JPEG_QUALITY), q2])
        decoded2 = cv.imdecode(encoded2, 1)
        return decoded2
    
    return (man_func, f"JPEGDoubleCompress-Q{str(q1)}-{str(q2)}", {"q1": q1, "q2": q2})

def webp_compress(q):
    def man_func(img):
        result, encoded = cv.imencode(".webp", img, [int(cv.IMWRITE_WEBP_QUALITY), q])
        decoded = cv.imdecode(encoded, 1)
        return decoded
    
    return (man_func, "WEBPCompress-Q" + str(q), {"q": q})


def morph_(morph_op, name, s):
    retval = cv.getStructuringElement(cv.MORPH_ELLIPSE, (s,s))

    def man_func(img):
        return cv.morphologyEx(img, morph_op, retval)
    
    return (man_func, name + str(s), {"s": s})

close_morph = partial(morph_, cv.MORPH_CLOSE, "CLOSEMorph-S")
dilate_morph = partial(morph_, cv.MORPH_DILATE, "DILATEMorph-S")
erode_morph = partial(morph_, cv.MORPH_ERODE, "ERODEMorph-S")
open_morph = partial(morph_, cv.MORPH_OPEN, "OPENMorph-S")

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

mans = boxblur_mans + gausblur_mans + medianblur_mans + \
    area_resize_mans + cubic_resize_mans + lanczos4_resize_mans + linear_resize_mans + nearest_resize_mans + \
    jpeg_compress_mans + jpeg_double_compress_mans + webp_compress_mans + \
    close_morph_mans + dilate_morph_mans + erode_morph_mans + open_morph_mans


img = cv.imread("data/input_.jpg")

for (man_func, name, pars) in mans:
    img_man = man_func(img)

    cv.imwrite(f"data/out_{name}.png", img_man)

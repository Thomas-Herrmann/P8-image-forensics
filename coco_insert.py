import sys
import numpy as np
import tensorflow as tf
import PIL
import tensorflow_datasets as tfds
import random as rand
import math
import scipy
from scipy.ndimage import gaussian_filter
import time

t = time.time()

BANNED_LABELS = [81, 87, 90, 97, 99, 100, 101, 102, 103, 105, 109, 110, 111, 112, 113, 116, 118, 119, 122, 123, 124, 125, 126, 131, 132]
DATASET_DIRECTORY = None
REACH_AROUND_INSERT_BOX = 170
MAX_INSERT_COVER_RATE = 0.30
DUMMY = (np.zeros((1,), dtype="uint8"), np.zeros((1,), dtype="uint8"), False)

def mask_by_color(image, color):
    mask = image.astype("uint32")
    mask = color_encoding(mask)
    mask = np.where(mask == color, np.zeros(mask.shape, "uint32"), np.full_like(mask, 255, "uint32"))
    #mask = gaussian_filter(mask, sigma=rand.uniform(0, 20), truncate=0.5)
    #vectorBlowup = np.vectorize(lambda x: min(x * 2, 255))
    #mask = vectorBlowup(mask)
    #mask = np.where(mask == 0, maskmask, mask)
    mask = mask[:, :, np.newaxis]
    return np.append(mask, np.append(mask, mask, axis=2), axis=2)

def color_encoding(image):
    return image[:,:,0] + image[:,:,1]*256 + image[:,:,2]*256*256

def color_encoding_p(pixel):
    return pixel[0] + pixel[1]*256 + pixel[2]*256*256

# For each image in the dataset
def image_insert(source_image, pan_image, pan_label, pan_id, pan_bbox, target_image):
    source_image = source_image.numpy()
    pan_image = pan_image.numpy()
    target_image = target_image.numpy()
    # pristine_image = target_image

    #PIL.Image.fromarray(source_image).show()
    #PIL.Image.fromarray(target_image).show()
    #PIL.Image.fromarray(pan_image).show()

    pan_object_feats = zip(pan_label.numpy(), pan_id.numpy(), pan_bbox.numpy())
    #pan_object_feats = zip(pan_objects['label'].numpy(), pan_objects['id'].numpy(), pan_objects['bbox'].numpy())
    obj_options = list((label, id, bbox) for (label, id, bbox) in pan_object_feats if label not in BANNED_LABELS)
    if obj_options:
        label, id, bbox = rand.choice(obj_options)
        #label, id, bbox = list(obj_options)[1]
        #print(label)
    else:
        # Skip this image if there is nothing but banned labels in it
        return DUMMY

#   sizes = pan_image.numpy().shape()
    xdim, ydim, _ = source_image.shape
    xdimtarget, ydimtarget, _ = target_image.shape
    if xdimtarget < 256 or ydimtarget < 256:
        return DUMMY

    xmin, ymin, xsize, ysize = bbox
    xmin, ymin, xmax, ymax = (math.floor(xdim * xmin), math.floor(ydim * ymin), math.floor(xdim * (xmin + xsize)), math.floor(ydim * (ymin + ysize)))

    croppedimage = source_image[xmin:xmax, ymin:ymax, :]
    pan_croppedimage = pan_image[xmin:xmax, ymin:ymax, :]
    xcropped, ycropped, _ = croppedimage.shape
    if xcropped >= xdimtarget and ycropped >= ydimtarget:
        return DUMMY
    xoffset, yoffset = (0 if xdimtarget - xcropped < 0 else (rand.randint(0, xdimtarget - xcropped)), 0 if ydimtarget - ycropped < 0 else (rand.randint(0, ydimtarget - ycropped)))

    # Used to be np.zeros, but it needs to be numbers outside the 0-255 range of the elements of RGB
    insert_image = np.full(shape=(xdimtarget+xcropped, ydimtarget+ycropped, 3), dtype=target_image.dtype, fill_value=420)
    insert_image[xoffset:xoffset+xcropped, yoffset:yoffset+ycropped, :] = croppedimage
    insert_image = insert_image[0:xdimtarget, 0:ydimtarget, :]
    pan_insert_image = np.full(shape=(xdimtarget + xcropped, ydimtarget + ycropped, 3), dtype=target_image.dtype, fill_value=420)
    pan_insert_image[xoffset:xoffset + xcropped, yoffset:yoffset + ycropped, :] = pan_croppedimage
    pan_insert_image = pan_insert_image[0:xdimtarget, 0:ydimtarget, :]

    color = id
    # uniques, counts = np.unique(pan_croppedimage.reshape(-1, 3), return_counts=True, axis=0)
    # uniques = map(colorencoding, uniques)
    # most = -1
    # color = 0
    #
    # # Most common color in the panoptic image
    # for unique, count in zip(uniques, counts):
    #     if count > most:
    #         most = count
    #         color = unique

    replacement_mask = mask_by_color(pan_insert_image, color)
    #target_image = np.where(replacement_mask == 0, insert_image, target_image)
    #target_image = np.array(list([((ti*rm)+(ii*(255-rm)))/255 for (ti, ii, rm) in zip(target_image, insert_image, replacement_mask)])).reshape(target_image.shape)
    #replacement_mask = replacement_mask.astype("uint8")

    # x = 0
    # y = 0
    # Pixel by pixel replace
    # for row, insertrow, maskrow in zip(target_image, insertimage, pan_insertimage):
    #     for pixel, insertpixel, maskpixel in zip(row, insertrow, maskrow):
    #         target_image[x][y] = pixelcolormasking(pixel, insertpixel, maskpixel, color)
    #         y = y + 1
    #     x = x + 1
    #     y = 0

    # Final crop
    xoffset = rand.randint(max(0, min(xoffset - REACH_AROUND_INSERT_BOX, xdimtarget - 256)), max(0, min(xdimtarget - 256, xoffset + xcropped + REACH_AROUND_INSERT_BOX - 256)))
    yoffset = rand.randint(max(0, min(yoffset - REACH_AROUND_INSERT_BOX, ydimtarget - 256)), max(0, min(ydimtarget - 256, yoffset + ycropped + REACH_AROUND_INSERT_BOX - 256)))
    final_target = target_image[xoffset:xoffset+256, yoffset:yoffset+256, :]
    final_insert = insert_image[xoffset:xoffset+256, yoffset:yoffset+256, :]
    final_mask = replacement_mask[xoffset:xoffset+256, yoffset:yoffset+256, 0:1]
    # pristine_crop = pristine_image[xoffset:xoffset+256, yoffset:yoffset+256, :]

    if np.average(final_mask)/256 < MAX_INSERT_COVER_RATE:
        return DUMMY

    final_mask = gaussian_filter(final_mask, sigma=rand.uniform(0, 20), truncate=0.5)
    vectorBlowup = np.vectorize(lambda x: min(x * 2, 255))
    final_mask = vectorBlowup(final_mask)
    final_crop = np.array(list([((ti*rm)+(ii*(255-rm)))/255 for (ti, ii, rm) in zip(final_target, final_insert, final_mask)])).reshape(final_target.shape)
    final_mask = final_mask.astype("uint8")

    # PIL.Image.fromarray(final_crop).show()
    # PIL.Image.fromarray(final_mask).show()
    # PIL.Image.fromarray(pristine_crop).show()
    #print("Shapes:", final_crop.shape, final_mask.shape)
    return final_crop, 1-(final_mask//255), True


def get_dataset(split='train'):
    #t = time.time()
    ds = tfds.load('coco/2017_panoptic', data_dir=DATASET_DIRECTORY, download=False, split=split, shuffle_files=True)
    ds = ds.map(lambda x: (x['image'], x['panoptic_image'], x['panoptic_objects']))
    dsz = tf.data.Dataset.zip((ds, ds.map(lambda img, pan_img, pan_obj: img).shuffle(1000))).prefetch(64)
    result = dsz.map(lambda source, target: tf.py_function(image_insert, [source[0], source[1], source[2]['label'], source[2]['id'], source[2]['bbox'], target], Tout=(tf.uint8, tf.int32, tf.bool)), deterministic=False)
    result = result.filter(lambda x, y, bool: bool)
    result = result.map(lambda x, y, bool: (x, y))
    result = result.map(lambda x,y: (tf.reshape(x, (256,256,3)), tf.reshape(y, (256,256,1))))
    return result

i = 0
for image, mask in get_dataset():
    tf.io.write_file("image.png", tf.io.encode_png(image))
    tf.io.write_file("mask.png", tf.io.encode_png(tf.cast(mask*255, tf.uint8)))
    if i > 1000:
        break
    i = i+1
print(time.time()-t)
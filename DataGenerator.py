import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow as tf
import cv2 as cv
import ntpath
from glob import glob
import os
import coco_insert
from manipulations import ManiFamily
from qualitative_test import colored_mask

class DataGenerator():

    def __init__(self, mask_dir_path, man_dir_path, crop_shape=(256, 256), batch_size=1):
        
        self.man_dir_path        = man_dir_path
        self.batch_size          = batch_size
        self.crop_h, self.crop_w = crop_shape

        self.masks = tfds.folder_dataset.ImageFolder(mask_dir_path)              \
                                        .as_dataset(shuffle_files=True)["masks"] \
                                        .repeat()                                \
                                        .as_numpy_iterator()
        
        self.mans = tfds.folder_dataset.ImageFolder(man_dir_path)                     \
                                       .as_dataset(shuffle_files=True)["manipulated"] \
                                       .repeat()                                      \
                                       .as_numpy_iterator()


    def _make_mask(self):

        loaded_masks = []

        for _ in range(self.batch_size):
            
            loaded_mask  = self.masks.next()["image"]
            loaded_masks += [tf.reshape(loaded_mask, (1, loaded_mask.shape[0], loaded_mask.shape[1], 3))]

        mask        = tf.concat(loaded_masks, 0)
        rand_gen    = tf.random.get_global_generator()
        rotated     = tfa.image.rotate(mask, rand_gen.uniform([self.batch_size], 0, 360, tf.float32), fill_mode="reflect")
        cropped     = tf.image.random_crop(rotated, (self.batch_size, self.crop_h, self.crop_w, 3))
        dilated     = tf.nn.dilation2d(cropped, tf.zeros((1, 1, 3), tf.uint8), (1, 1, 1, 1), "SAME", "NHWC", (1, 1, 1, 1))

        return tf.cast(dilated / 255, tf.int32) * 255


    def _get_cropped_img_pair(self):

        man_dic          = self.mans.next()
        width, height, _ = man_dic["image"].shape

        rand_gen     = tf.random.get_global_generator()
        h_crop_start = rand_gen.uniform([], 0, height - self.crop_h, tf.int32)
        w_crop_start = rand_gen.uniform([], 0, width - self.crop_w, tf.int32)
        
        cropf = lambda img: img[w_crop_start:w_crop_start + self.crop_w, h_crop_start:h_crop_start + self.crop_h]

        man_crop = cropf(man_dic["image"])
        basename = ntpath.basename(man_dic["image/filename"].decode("utf-8"))
        pri_path = ntpath.splitext(ntpath.join(ntpath.join(self.man_dir_path, "pristine"), basename))[0] + ".jpg"
        pri_crop = cropf(cv.cvtColor(cv.imread(pri_path), cv.COLOR_BGR2RGB))

        reshapef = lambda img: tf.reshape(img, (1, self.crop_h, self.crop_w, 3))

        return reshapef(pri_crop), reshapef(man_crop)


    def _get_cropped_img_pair_batch(self):

        pristines    = []
        manipulateds = []

        for _ in range(self.batch_size):

            pristine, manipulated = self._get_cropped_img_pair()
            
            pristines    += [pristine]
            manipulateds += [manipulated]

        return tf.concat(pristines, 0), tf.concat(manipulateds, 0)


    def _join_by_mask(self, one, two, mask):

        mapped_neg_mask = tf.cast(mask / 255, tf.int32)
        mapped_mask     = tf.ones((self.batch_size, self.crop_h, self.crop_w, 3), tf.int32) - mapped_neg_mask

        return tf.math.multiply(tf.cast(one, tf.int32), mapped_neg_mask) + \
               tf.math.multiply(tf.cast(two, tf.int32), mapped_mask)


    def next(self):

        mask                  = self._make_mask()
        pristine, manipulated = self._get_cropped_img_pair_batch()
        combined              = self._join_by_mask(pristine, manipulated, mask)

        splitf = lambda batch: list(map(lambda img: tf.reshape(img, (self.crop_h, self.crop_w, 3)), tf.split(batch, self.batch_size, 0)))

        return splitf(combined), splitf(mask)



CROP_SHAPE=(256, 256)
MANIP_DIR = "data/manipulated"
MASK_DIR = "data/masks"
PRISTINE_DIR = glob(os.path.expanduser('~')+"/tensorflow_datasets/downloads/extracted/*/train2017")[0] #"/user/student.aau.dk/slund17/tensorflow_datasets/downloads/extracted/ZIP.images.cocodataset.org_zips_train2017aai7WOpfj5nSSHXyFBbeLp3tMXjpA_H3YD4oO54G2Sk.zip/train2017"

@tf.function
def transform_mask(mask):
    with tf.device("/gpu:0"):
        rand_gen    = tf.random.get_global_generator()
        rotated     = tfa.image.rotate(mask, rand_gen.uniform([tf.shape(mask)[0]], 0, 360, tf.float32), fill_mode="reflect")
        blurred     = gaussian_blur(tf.cast(rotated, tf.float32), rand_gen.uniform([], 1, 13, dtype=tf.int32), rand_gen.uniform([], 1, 10)) 
        cropped     = tf.image.random_crop(blurred, (tf.shape(mask)[0], *CROP_SHAPE, 3))
        dilated     = tf.nn.dilation2d(cropped, tf.zeros((1, 1, 3), tf.float32), (1, 1, 1, 1), "SAME", "NHWC", (1, 3, 3, 1))
        scaled      = tf.cast(dilated / 255, dtype=tf.float32)
        return tf.math.reduce_max(scaled, axis=-1, keepdims=True) # Reduce RGB dimension to 1 dimensional

def gaussian_blur(img, kernel_size=11, sigma=5):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.cast(tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1), tf.float32)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1], padding="VALID", data_format="NHWC")

def get_masks(batch_size):
    mask = tfds.folder_dataset.ImageFolder(MASK_DIR).as_dataset(shuffle_files=True)['train']
    mask = mask.map(lambda x: x['image']).repeat()
    # mask = mask.shuffle(1000) # Do we need to shuffle when shuffle_files=True? 
    mask = mask.batch(batch_size)
    mask = mask.map(transform_mask, num_parallel_calls=tf.data.AUTOTUNE)
    return mask.unbatch()

def to_pristine_path(filename):
    name = tf.strings.split(filename, "/")[-1]
    name = tf.strings.split(name, ".")[0]
    path = PRISTINE_DIR + '/' + name + ".jpg"
    return path

def to_pristine_image(filename):
    path = to_pristine_path(filename)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    # convert? #image = tf.image.convert_image_dtype(image, tf.float32)

    #image = tf.keras.preprocessing.image.load_img(path)
    #arr = tf.keras.preprocessing.image.img_to_array(image)
    return image

def to_random_crop(manipulated, pristine, label):
    stacked = tf.stack([manipulated, pristine])
    cropped = tf.image.random_crop(stacked, (2, *CROP_SHAPE, 3))
    return (*tf.unstack(cropped), label)

def get_manip_pristines(split='train', label_offset=1):
    manips = tfds.folder_dataset.ImageFolder(MANIP_DIR).as_dataset(shuffle_files=True)[split]

    #Filter out small images
    manips = manips.filter(lambda x: tf.shape(x['image'])[0]>CROP_SHAPE[0] and tf.shape(x['image'])[1]>CROP_SHAPE[1])
    
    manips = manips.map(lambda x: (x['image'], x['image/filename'], x['label']))

    #manips = manips.shuffle(10_000) # do we need to shuffle when we have shuffle_files=True?
    ds = manips.map(lambda manip, filename, label: (manip, to_pristine_image(filename), label), deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(to_random_crop, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)

    label_map = get_label_map(split, label_offset)
    ds = ds.map(lambda manip, pristine, label: (manip, pristine, label_map[label]))
    return ds

def apply_mask(manip_pristine_label, mask):
    manip, pristine, label = manip_pristine_label
    #applied = mask*manip + (1-mask)*pristine
    applied = (1-mask)*tf.cast(manip, tf.float32) + mask*tf.cast(pristine, tf.float32)
    return tf.cast(applied, tf.uint8), tf.cast((1-mask)>0, tf.int32) * label

def get_two_class_dataset(batch_size):
    ps = get_manip_pristines(label_offset=2).map(lambda manip, pristine, label: (manip, pristine, 1))
    ds = tf.data.Dataset.zip((ps, get_masks(batch_size)))
    ds = ds.map(apply_mask, num_parallel_calls=tf.data.AUTOTUNE)
    #ds = ds.batch(batch_size)
    return ds

def get_dataset(batch_size):
    ds = tf.data.Dataset.zip((get_manip_pristines(label_offset=2), get_masks(batch_size)))
    ds = ds.map(apply_mask, num_parallel_calls=tf.data.AUTOTUNE)
    #ds = ds.batch(batch_size)
    return ds

def apply_test_mask(manip_pristine_label, mask):
    manip, pristine, label = manip_pristine_label
    applied = (1-mask)*manip + mask*pristine
    return pristine, manip, applied, tf.cast(mask, tf.int32), label

def get_test_dataset():
    ds = tf.data.Dataset.zip((get_manip_pristines(label_offset=0), get_masks(32)))
    ds = ds.map(apply_test_mask, num_parallel_calls=tf.data.AUTOTUNE)
    #ds = ds.batch(batch_size)
    return ds


def get_label_map(split, label_offset=1): # We map the 385 catagory labels to the 7 family labels
    #label_offset is useful when we need a class of no manipulations (offset=1)
    family_id_map = {family.value:i for i, family in enumerate(ManiFamily)}
    path = MANIP_DIR + "/"+split+"/*"
    names = sorted([os.path.basename(x) for x in glob(path)]) # The default labels are the sorted alphanumerical order
    families = [name.split('-')[0] for name in names] # folders are named "FAMILY-TYPE-PARAMS..." so we just take the family
    ids = [family_id_map[family] + label_offset for family in families] # map to family ids + offset
    return tf.constant(ids)


def get_combined_dataset(batch_size):
    coco = coco_insert.get_dataset() #.map(lambda x,y: {'x':x, 'y':y})
    manip = get_dataset(batch_size//2) #.map(lambda x,y: {'x':x, 'y':y})
    # Zip the two datasets and flat map concatenate them to interleave them:
    dataset = tf.data.Dataset.zip((coco, manip.batch(7, drop_remainder=True))).flat_map(
        lambda x0, x1: tf.data.Dataset.from_tensors((x0,)).concatenate(tf.data.Dataset.from_tensors((x1,)).unbatch()))
    dataset = dataset.map(lambda x: (x[0], x[1]))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def get_combined_two_class_dataset(batch_size):
    coco = coco_insert.get_dataset()
    manip = get_two_class_dataset(batch_size//2)
    # Zip the two datasets and flat map concatenate them to interleave them:
    dataset = tf.data.Dataset.zip((coco, manip.batch(7))).flat_map(
        lambda x0, x1: tf.data.Dataset.from_tensors((x0,)).concatenate(tf.data.Dataset.from_tensors((x1,)).unbatch()))
    dataset = dataset.map(lambda x: (x[0], x[1]))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def weighted(ds, class_weights):
    init = tf.lookup.KeyValueTensorInitializer(list(class_weights.keys()), list(class_weights.values()), key_dtype=tf.int32, value_dtype=tf.float32)
    table = tf.lookup.StaticHashTable(init, default_value=-1)
    return ds.map(lambda image,mask: (image, mask, table.lookup(mask)))

def get_weighted_two_class_dataset(batch_size, weights):
    return weighted(get_combined_two_class_dataset(batch_size).map(lambda im, msk: (im, tf.reshape(msk, (batch_size, -1)))), weights)

#for images, masks in get_combined_dataset(2):
#    print(".")
if __name__ == "__main__":
    i = 0
    for images, masks in get_combined_dataset(64):
        for image, mask in zip(images, masks):
            tf.io.write_file(f"samples/image{i}.png", tf.io.encode_png(image))
            c_mask, legends = colored_mask(tf.cast(mask, tf.int64))
            c_mask = tf.squeeze(c_mask)
            tf.io.write_file(f"samples/mask{i}.png", tf.io.encode_png(c_mask))
            tf.io.write_file(f"stitches/stitch{i}.png", tf.io.encode_png(tf.concat([image, c_mask], axis=1)))
            i += 1
        if i > 100: break
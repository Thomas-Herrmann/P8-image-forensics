import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import tensorflow as tf
import cv2 as cv
import ntpath


class DataGenerator():

    def __init__(self, mask_dir_path, man_dir_path, crop_shape=(256, 256)):
        
        self.man_dir_path = man_dir_path

        self.masks = tfds.folder_dataset.ImageFolder(mask_dir_path)              \
                                        .as_dataset(shuffle_files=True)["masks"] \
                                        .repeat()                                \
                                        .as_numpy_iterator()
        
        self.mans = tfds.folder_dataset.ImageFolder(man_dir_path)                     \
                                       .as_dataset(shuffle_files=True)["manipulated"] \
                                       .repeat()                                      \
                                       .as_numpy_iterator()
        
        self.crop_h, self.crop_w = crop_shape


    def _make_mask(self):

        rand_gen    = tf.random.get_global_generator()
        mask        = self.masks.next()["image"]
        reshaped    = tf.reshape(mask, (1, mask.shape[0], mask.shape[1], 3))
        rotated     = tfa.image.rotate(reshaped, rand_gen.uniform([], 0, 360, tf.float32), fill_mode="reflect")
        cropped     = tf.image.random_crop(rotated, (1, self.crop_h, self.crop_w, 3))
        dilated     = tf.nn.dilation2d(cropped, tf.zeros((1, 1, 3), tf.uint8), (1, 1, 1, 1), "SAME", "NHWC", (1, 1, 1, 1))
        regularized = tf.cast(dilated / 255, tf.int32) * 255

        return tf.reshape(regularized, (self.crop_h, self.crop_w, 3))


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

        return pri_crop, man_crop


    def _join_by_mask(self, one, two, mask):

        mapped_neg_mask = tf.cast(mask / 255, tf.int32)
        mapped_mask     = tf.ones((self.crop_h, self.crop_w, 3), tf.int32) - mapped_neg_mask

        return tf.einsum("ijk, ijk -> ijk", one, mapped_neg_mask) \
             + tf.einsum("ijk, ijk -> ijk", two, mapped_mask)


    def next(self):

        mask                  = self._make_mask()
        pristine, manipulated = self._get_cropped_img_pair()
        combined              = self._join_by_mask(pristine, manipulated, mask)

        return combined, mask



    





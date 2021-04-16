import tensorflow as tf
import random
import cv2 as cv
import PIL
from manipulations import ManiFamily
import numpy as np

    
def colored_mask(mask, seed=13):

    make_table = lambda ch_dict: tf.lookup.StaticVocabularyTable(tf.lookup.KeyValueTensorInitializer(
        list(ch_dict.keys()),
        list(ch_dict.values()),
        key_dtype=tf.int64,
        value_dtype=tf.int64),
        num_oov_buckets=1)

    random.seed(seed)

    color_dict_r = {n: random.randrange(255) for n in range(9)}
    color_dict_g = {n: random.randrange(255) for n in range(9)}
    color_dict_b = {n: random.randrange(255) for n in range(9)}

    mask_exp    = tf.expand_dims(mask, 3)
    c_mask      = tf.concat([make_table(color_dict_r).lookup(mask_exp), make_table(color_dict_g).lookup(mask_exp), make_table(color_dict_b).lookup(mask_exp)], 3)
    legend_dict = {n: [color_dict_r[n], color_dict_g[n], color_dict_b[n]] for n in range(9)}

    return tf.cast(c_mask, tf.uint8), legend_dict


def make_legends_image(legends):

    family_id_map    = {i + 2: family.value for i, family in enumerate(ManiFamily)}
    family_id_map[0] = "nothing"
    family_id_map[1] = "splice"  # TODO

    one_ch_square   = lambda ch: tf.ones([80, 80]) * ch
    make_col_square = lambda n: tf.stack([one_ch_square(legends[n][0]), 
                                          one_ch_square(legends[n][1]), 
                                          one_ch_square(legends[n][2])], 2)

    return tf.cast(tf.concat([make_col_square(n) for n in range(9)], 0), tf.uint8)


def f1(y_true, y_pred):
    return 1


if __name__ == "__main__":
    model  = tf.keras.models.load_model("pixel_aaconv_save_at_2.tf", custom_objects={'f1':lambda x,y:1})
    for i in range(10):
        image  = tf.image.decode_png(tf.io.read_file(f"samples/image{i}.png"), channels=3)

        mask = tf.image.decode_png(tf.io.read_file(f"samples/mask{i}.png"), channels=3)

        #tf.image.random_crop(cv.cvtColor(cv.imread("test_img.png"), cv.COLOR_BGR2RGB), [256, 256, 3])
        out = model(tf.expand_dims(image, 0))
        
        mean_logits = tf.math.reduce_max(out, [0,1,2])
        print(np.around(tf.nn.softmax(mean_logits).numpy(), 2))
        #_, top = tf.math.top_k(out, k=8)
        #print(top)
        c_mask, legends = colored_mask(tf.math.argmax(out, axis=-1))
        #c_mask, legends = colored_mask(tf.gather(top, [1], axis=-1))

        tf.io.write_file(f"test_out{i}.png", tf.io.encode_png(tf.concat([image, mask, tf.reshape(c_mask, [256, 256, 3])], axis=1))) 
        #tf.io.write_file("test_in.png", tf.io.encode_png(image))
    tf.io.write_file("test_legends.png", tf.io.encode_png(make_legends_image(legends)))

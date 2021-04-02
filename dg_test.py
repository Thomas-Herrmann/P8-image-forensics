import DataGenerator as dg
import tensorflow as tf


ndg = dg.DataGenerator("C:\\Users\\tomas\\Desktop\\irregular_mask", "C:\\Users\\tomas\\Desktop\\manipulated\\dataset", batch_size=10)


images, masks = ndg.next()

for i in range(len(images)):

    tf.io.write_file(f"t_mask{i}.jpg", tf.io.encode_jpeg(tf.cast(masks[i], tf.uint8)))
    tf.io.write_file(f"t_imag{i}.jpg", tf.io.encode_jpeg(tf.cast(images[i], tf.uint8)))

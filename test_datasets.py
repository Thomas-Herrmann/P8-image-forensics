import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.data as tfd
import glob
import os.path


CG_1050_PATH = "test/CG-1050/"

def get_CG_1050_dataset():
    tampered_dataset = glob.glob(CG_1050_PATH + "TAMPERED/T_*/Im*_*.jpg")[0:15]

    def get_generator():
        for tampered in tampered_dataset:
            (mask_r_path, mask_g_path, mask_b_path) = get_mask_paths(tampered)

            mask_r = 255 - tf.image.decode_png(tf.io.read_file(mask_r_path), channels=1)
            mask_g = 255 - tf.image.decode_png(tf.io.read_file(mask_g_path), channels=1)
            mask_b = 255 - tf.image.decode_png(tf.io.read_file(mask_b_path), channels=1)

            mask = tf.reduce_mean(tf.stack([mask_r, mask_g, mask_b], axis=0), axis=0)
            mask = tf.repeat(mask, 3, 2)
            pristine = tf.image.decode_png(tf.io.read_file(tampered), channels=3)
            
            mask = tf.RaggedTensor.from_tensor(mask)
            pristine = tf.RaggedTensor.from_tensor(pristine)

            yield (pristine, mask)

    return tf.data.Dataset.from_generator(
        get_generator,
        output_signature=(
            tf.RaggedTensorSpec(shape=(None,None,3), dtype=tf.uint8, ragged_rank=1),
            tf.RaggedTensorSpec(shape=(None,None,3), dtype=tf.uint8, ragged_rank=1))
    )


def get_mask_paths(image_path):
    # Example image name: "Im1_cmfr1.jpg"
    path, image_name = os.path.split(image_path)

    name, _ = image_name.split(".")
    image_num, man_type_num = name.split("_")

    image_num = image_num[2:] # remove "Im" prefix

    dir = f"{CG_1050_PATH}MASK/Mask{image_num}/Mask{image_num}_{man_type_num}/Mask{image_num}_{man_type_num}_Band_"
    return (dir + "R(1).png", dir + "G(2).png", dir + "B(3).png")


if __name__ == "__main__":
    dataset = get_CG_1050_dataset()

    for (pristine, mask) in dataset:
        print("_")
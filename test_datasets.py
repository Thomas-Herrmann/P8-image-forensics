import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.data as tfd
import tensorflow_io as tfio
import glob
import os.path

# https://data.mendeley.com/datasets/dk84bmnyw9/2
CG_1050_ORIG_PATH     = "test/CG-1050/ORIGINAL/"
CG_1050_TAMP_PATH     = "test/CG-1050/TAMPERED/"
CG_1050_GEN_MASK_PATH = "test/CG-1050/GEN_MASK/"


# https://github.com/namtpham/casia2groundtruth
CASIA2_GT_PATH       = "test/CASIA2.0_Groundtruth/"
CASIA2_TP_PATH       = "test/CASIA2.0_revised/Tp/"
CASIA2_AU_PATH       = "test/CASIA2.0_revised/Au/"
CASIA2_GEN_MASK_PATH = "test/CASIA2.0_GEN_MASK/" # Unused

def _read_tf_image(path):
    imagefile = tf.io.read_file(path)

    if path[-4:] == ".tif":
        return tfio.experimental.image.decode_tiff(imagefile)[:,:,:3]
    else:
        return tf.image.decode_image(imagefile, channels=3)

def _get_CG_1050_original_path(tampered_image_path):
    path, image_name = os.path.split(tampered_image_path)
    name = image_name.split("_")[0][2:]

    return CG_1050_ORIG_PATH + "Im_" + name + ".jpg"

def _get_CASIA2_original_path(tampered_image_path):
    path, image_name = os.path.split(tampered_image_path)
    name = image_name.split("_")[5]
    typ = name[:3]
    num = name[3:]

    path = CASIA2_AU_PATH + "Au_" + typ + "_" + num

    if os.path.exists(path + ".jpg"):
        return path + ".jpg"
    else:
        return path + ".bmp"

def generate_masks(path_pairs, out_dir, threshold):
    for tampered_path, pristine_path in path_pairs:
        print(tampered_path, pristine_path)
        tampered = _read_tf_image(tampered_path)
        pristine = _read_tf_image(pristine_path)

        if tampered.shape != pristine.shape:
            print(f"{tampered_path} and {pristine_path} do not have the same dimensions ({tampered.shape} vs {pristine.shape}). Ignoring")
            continue

        tampered = tf.cast(tampered, tf.int32)
        pristine = tf.cast(pristine, tf.int32)
        mask = tf.cast(tf.math.reduce_mean(tf.math.squared_difference(tampered, pristine),axis=-1, keepdims=True)>threshold, tf.uint8)*255

        filename_ext = os.path.basename(tampered_path)
        name, _ = os.path.splitext(filename_ext)

        tf.io.write_file(out_dir + name + ".png", tf.io.encode_png(mask))

def generate_CG_1050_masks():
    tampered_path_dataset = glob.glob(CG_1050_TAMP_PATH + "T_*/Im*_*.jpg")
    pairs = [(path, _get_CG_1050_original_path(path)) for path in tampered_path_dataset]
    generate_masks(pairs, CG_1050_GEN_MASK_PATH, 32)

def generate_CASIA2_masks():
    tampered_path_dataset = glob.glob(CASIA2_TP_PATH + "Tp_*")
    pairs = [(path, _get_CASIA2_original_path(path)) for path in tampered_path_dataset]
    generate_masks(pairs, CASIA2_GEN_MASK_PATH, 32)


# Uses generated masks
def get_CG_1050_dataset():
    tampered_dataset = glob.glob(CG_1050_TAMP_PATH + "T_*/Im*_*.jpg")

    def get_generator():
        for tampered_path in tampered_dataset:
            filename_ext = os.path.basename(tampered_path)
            name, _ = os.path.splitext(filename_ext)

            mask = _read_tf_image(CG_1050_GEN_MASK_PATH + name + ".png")
            tampered = tf.image.decode_png(tf.io.read_file(tampered_path), channels=3)

            yield (tampered, mask)

    return tf.data.Dataset.from_generator(
        get_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8))
    )


 # Uses stock masks
def get_CASIA2_dataset():
    tampered_dataset = glob.glob(CASIA2_TP_PATH + "Tp_*")

    def get_generator():
        for tampered_path in tampered_dataset:
            filename_ext = os.path.basename(tampered_path)
            name, _ = os.path.splitext(filename_ext)

            tampered = _read_tf_image(tampered_path)
            mask = _read_tf_image(CASIA2_GT_PATH + name + "_gt.png")

            if tampered.shape[0] < 256 or tampered.shape[1] < 256:
                continue
            
            if tampered.shape != mask.shape:
                print(f"{tampered_path}: Tampered and mask dimensions mismatch ({tampered.shape[:2]} and {mask.shape[:2]})")
                continue

            yield (tampered, mask)

    return tf.data.Dataset.from_generator(
        get_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8))
    )



if __name__ == "__main__":
    #generate_CG_1050_masks()

    #dataset = get_CG_1050_dataset() 
    dataset = get_CASIA2_dataset()

    for i, (tampered, mask) in enumerate(dataset):
        comb = tf.concat([tampered, mask], axis=1)
        tf.io.write_file(f"out/img{i}.png", tf.io.encode_png(comb))
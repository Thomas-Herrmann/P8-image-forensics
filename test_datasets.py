import tensorflow as tf
import tensorflow_io as tfio
import glob
import os.path
import qualitative_test
import tensorflow_addons as tfa

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


def test_metric(model, dataset, metrics, patch_multiplier=1):
    for i, (tampered, mask) in enumerate(dataset):
        if i % 5 == 0:
            print(i)

        patches = qualitative_test.split_patches(tampered, 256, 256, patch_multiplier)
        pred_patches = qualitative_test.predict_patches(model, patches)
        combined = qualitative_test.combine_patches(pred_patches)

        certainties = tf.expand_dims(tf.gather(tf.nn.softmax(combined), 1, axis=-1), axis=-1)

        mask = tf.expand_dims(tf.gather(mask, 0, axis=-1), axis=-1)
        mask = tf.cast(mask>0, tf.int32)

        for metric in metrics:
            metric.update_state(mask, certainties)


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def run_tests(dataset, model, patch_multiplier=1):
        acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
        auc = tf.keras.metrics.AUC(num_thresholds=10)
        f1 = tfa.metrics.F1Score(num_classes=2, average="micro", threshold=0.5)
        ce = tf.keras.metrics.BinaryCrossentropy(from_logits=False)

        test_metric(model, dataset, [acc, auc, f1, ce], patch_multiplier)

        print(f"Acc: {acc.result()}")
        print(f"AUC: {auc.result()}")
        print(f"F1:  {f1.result()}")
        print(f"CrE: {ce.result()}")

    #generate_CG_1050_masks()

    #model  = tf.keras.models.load_model("models/2_class_pixel_conv_save_at_100.tf", custom_objects={'f1':lambda x,y:1})
    #model  = tf.keras.models.load_model("models/2class_blr_aaconv_save_at_52.tf", custom_objects={'f1':lambda x,y:1})
    #model  = tf.keras.models.load_model("models/2_class_pixel_conv_save_at_97_w_blur.tf", custom_objects={'f1':lambda x,y:1})
    model  = tf.keras.models.load_model("models/2class_aaconv_no_sblur_save_at_100.tf", custom_objects={'f1':lambda x,y:1})
    
    

    #dataset = get_CG_1050_dataset().take(10)
    dataset = get_CASIA2_dataset()

    run_tests(dataset, model, 1)
    
import tensorflow as tf
import tensorflow_io as tfio
import glob
import os
import os.path
from qualitative_test import split_patches, predict_patches, combine_patches
import tensorflow_addons as tfa
import numpy as np
from DataGenerator import get_two_class_valid_dataset

# https://data.mendeley.com/datasets/dk84bmnyw9/2
CG_1050_ORIG_PATH     = "test/CG-1050/ORIGINAL/"
CG_1050_TAMP_PATH     = "test/CG-1050/TAMPERED/"
CG_1050_GEN_MASK_PATH = "test/CG-1050/GEN_MASK/"


# https://github.com/namtpham/casia2groundtruth
CASIA2_GT_PATH       = "test/CASIA2.0_Groundtruth/"
CASIA2_TP_PATH       = "test/CASIA2.0_revised/Tp/"
CASIA2_AU_PATH       = "test/CASIA2.0_revised/Au/"
CASIA2_GEN_MASK_PATH = "test/CASIA2.0_GEN_MASK/" # Unused

# https://github.com/wenbihan/coverage
COVERAGE_TAMP_PATH = "test/COVERAGE/image/"
COVERAGE_MASK_PATH = "test/COVERAGE/mask/"

FAU_PATH = "test/benchmark_data/"

RESULT_CACHE_PATH    = "test/results_saved/"

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
            tampered = tf.image.decode_image(tf.io.read_file(tampered_path), channels=3)

            yield (tampered, mask, name, "cg_1050")

    return tf.data.Dataset.from_generator(
        get_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string))
    )


# Uses stock masks
def get_CASIA2_dataset(pattern="Tp_?_???_?_?_*"):
    tampered_dataset = glob.glob(CASIA2_TP_PATH + pattern)

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

            yield (tampered, mask, name, "casia2")

    return tf.data.Dataset.from_generator(
        get_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string))
    )

def get_COVERAGE_dataset():
    tampered_dataset = glob.glob(COVERAGE_TAMP_PATH + "*t.tif")

    def get_generator():
        for tampered_path in tampered_dataset:
            filename_ext = os.path.basename(tampered_path)
            name, _ = os.path.splitext(filename_ext)

            tampered = _read_tf_image(tampered_path)
            mask = _read_tf_image(COVERAGE_MASK_PATH + name[:-1] + "forged.tif")

            if tampered.shape[0] < 256 or tampered.shape[1] < 256:
                continue

            if tampered.shape != mask.shape:
                print(f"{tampered_path}: Tampered and mask dimensions mismatch ({tampered.shape[:2]} and {mask.shape[:2]})")
                continue

            yield (tampered, mask, name, "coverage")

    return tf.data.Dataset.from_generator(
        get_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string))
    )

def get_validation_dataset():
    dataset = get_two_class_valid_dataset(batch_size=1).unbatch()

    def get_generator():
        counter = -1
        for tampered, mask in dataset:
            counter += 1
            mask = tf.repeat(mask, 3, axis=-1)
            yield (tampered, mask, f"img{counter}", "validation")
    
    return tf.data.Dataset.from_generator(
        get_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string))
    )

def get_FAU_image_manipulation_dataset():
    tampered_paths = glob.glob(FAU_PATH + "*/*_copy.png")

    def mask_paths(tampered_path):
        return glob.glob(tampered_path[:-9] + "_?_alpha.png")

    tampered_masks_paths = [(tamp, mask_paths(tamp)) for tamp in tampered_paths]

    def get_generator():
        for tampered_path, mask_paths in tampered_masks_paths:
            filename_ext = os.path.basename(tampered_path)
            name, _ = os.path.splitext(filename_ext)

            tampered = _read_tf_image(tampered_path)
            masks = tf.data.Dataset.from_tensor_slices([_read_tf_image(p) for p in mask_paths])
            mask = masks.reduce(iter(masks).next(), tf.bitwise.bitwise_or)

            yield (tampered, mask, name, "fau_image_manipulation")

    return tf.data.Dataset.from_generator(
        get_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string))
    )


def calculate_certainties(model, tampered, patch_multiplier, model_name, tampered_name, dataset_name):
    cache_dir = f"{RESULT_CACHE_PATH}/{dataset_name}/pmul{patch_multiplier}/{model_name}/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_path = f"{cache_dir}{tampered_name}.npy"
    # Try to load from cache
    if os.path.exists(cache_path):
        certainties = np.load(cache_path)
    else:
        patches = split_patches(tampered, 256, 256, patch_multiplier)
        pred_patches = predict_patches(model, patches)
        combined = combine_patches(pred_patches)

        certainties = tf.expand_dims(tf.gather(tf.nn.softmax(combined), 1, axis=-1), axis=-1)
        certainties = tf.cast(certainties, tf.float16)
        np.save(cache_path, certainties)
    
    return certainties

def test_metric(model, dataset, metrics, model_name, patch_multiplier=1):
    counter = 0
    for i, (tampered, mask, img_name, dataset_name) in enumerate(dataset):
        if i % 5 == 0:
            print(i)

        certainties = calculate_certainties(model, tampered, patch_multiplier, model_name, img_name, dataset_name)

        mask = tf.expand_dims(tf.gather(mask, 0, axis=-1), axis=-1)
        mask = tf.cast(mask>0, tf.int32)

        for metric in metrics:
            metric.update_state(mask, certainties)
        
        counter += 1

    return counter


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def run_tests(dataset, model, model_name, patch_multiplier=1):
        threshold = 0.5
        acc = tf.keras.metrics.BinaryAccuracy(threshold=threshold)
        auc = tf.keras.metrics.AUC(num_thresholds=10)
        f1 = tfa.metrics.F1Score(num_classes=2, average="micro", threshold=threshold)
        ce = tf.keras.metrics.BinaryCrossentropy(from_logits=False)

        num = test_metric(model, dataset, [acc, auc, f1, ce], model_name, patch_multiplier)

        return ({"acc": acc.result().numpy(), "auc": auc.result().numpy(), "f1": f1.result().numpy(), "cre": ce.result().numpy()}, num)

    #generate_CG_1050_masks()

    model_name = "2class_aaconv_save_at_93_final.tf"
    #model_name = "2_class_pixel_conv_save_at_89_final.tf"

    model  = tf.keras.models.load_model("models/" + model_name, custom_objects={'f1':lambda x,y:1})
    
    #dataset = get_CG_1050_dataset()
    #dataset = get_CASIA2_dataset()
    #dataset = get_CASIA2_dataset(pattern="Tp_D_???_?_?_*")
    #dataset = get_CASIA2_dataset(pattern="Tp_S_???_?_?_*")
    dataset = get_COVERAGE_dataset()
    #dataset = get_validation_dataset()
    #dataset = get_FAU_image_manipulation_dataset()

    #for i, (tampered, mask, name, dataset_name) in enumerate(dataset):
        #print(f"out{i}")
        #tf.io.write_file(f"out/out{i}.png", tf.io.encode_png(tf.concat([tampered, mask], axis=1)))
        #tf.io.write_file(f"test_out{i}.png", tf.io.encode_png(tf.concat([image, mask, tf.reshape(c_mask, [256, 256, 3])]))) 

    results, _ = run_tests(dataset, model, model_name, 5)

    print(f"Acc: {results['acc']}")
    print(f"AUC: {results['auc']}")
    print(f"F1 : {results['f1']}")
    print(f"CrE: {results['cre']}")

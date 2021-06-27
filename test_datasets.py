from math import nan
import tensorflow as tf
import tensorflow_io as tfio
import glob
import os
import os.path
from qualitative_test import split_patches, predict_patches, combine_patches
import tensorflow_addons as tfa
import numpy as np
from DataGenerator import get_two_class_valid_dataset
import matplotlib.pyplot as plt

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

def calculate_certainties(model, tampered, patch_multiplier, model_name, tampered_name, dataset_name, use_caching=True):
    cache_dir = f"{RESULT_CACHE_PATH}/{dataset_name}/pmul{patch_multiplier}/{model_name}/"
    if use_caching and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_path = f"{cache_dir}{tampered_name}.npy"
    # Try to load from cache
    if use_caching and os.path.exists(cache_path):
        certainties = np.load(cache_path)
    else:
        patches = split_patches(tampered, 256, 256, patch_multiplier)
        pred_patches = predict_patches(model, patches)
        combined = combine_patches(pred_patches)

        certainties = tf.expand_dims(tf.gather(tf.nn.softmax(combined), 1, axis=-1), axis=-1)
        
        if use_caching:
            certainties = tf.cast(certainties, tf.float16)
            np.save(cache_path, certainties)
    
    return certainties

def test_metric(model, dataset, metrics, model_name, patch_multiplier=1, use_caching=True):
    counter = 0
    for i, (tampered, mask, img_name, dataset_name) in enumerate(dataset):
        if i % 5 == 0:
            print(i)

        certainties = calculate_certainties(model, tampered, patch_multiplier, model_name, img_name, dataset_name, use_caching)

        mask = tf.expand_dims(tf.gather(mask, 0, axis=-1), axis=-1)
        mask = tf.cast(mask>0, tf.int32)

        for metric in metrics:
            metric.update_state(mask, certainties)
        
        counter += 1

    return counter

def test_validation(model, metrics, manipulation_func, manipulation_parameters):
    batch_size = 32
    dataset_original = get_two_class_valid_dataset(batch_size=batch_size)

    results_all = []

    for man_parameter in manipulation_parameters:
        print(f"Parameter {man_parameter}")
        dataset = dataset_original.unbatch()
        dataset = dataset.map(manipulation_func(man_parameter), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)

        for i, (images, masks) in enumerate(dataset):
            if i % 5 == 0:
                print(i * batch_size)
            
            prediction_b = model(images)
            certainties_b = tf.gather(tf.nn.softmax(prediction_b), 1, axis=-1)

            for certainties, mask in zip(certainties_b, masks):
                mask = tf.expand_dims(tf.gather(mask, 0, axis=-1), axis=-1)
                mask = tf.cast(mask>0, tf.int32)

                for metric in metrics:
                    metric.update_state(mask, certainties)

        results = []
        for metric in metrics:
            results.append(metric.result().numpy())
            metric.reset_states()
        print(results)

        results_all.append(results)
    
    return results_all

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model_name = "2class_aaconv_save_at_93_final.tf"
    #model_name = "2_class_pixel_conv_save_at_89_final.tf"

    model  = tf.keras.models.load_model("models/" + model_name, custom_objects={'f1':lambda x,y:1})
    
    #dataset = get_CG_1050_dataset()
    #dataset = get_CASIA2_dataset()
    #dataset = get_CASIA2_dataset(pattern="Tp_D_???_?_?_*")
    #dataset = get_CASIA2_dataset(pattern="Tp_S_???_?_?_*")
    #dataset = get_COVERAGE_dataset().take(1)
    dataset = get_validation_dataset().prefetch(16)
    #dataset = get_FAU_image_manipulation_dataset()

    def run_tests(patch_multiplier=1):
        threshold = 0.5
        acc = tf.keras.metrics.BinaryAccuracy(threshold=threshold)
        auc = tf.keras.metrics.AUC(num_thresholds=10)
        f1 = tfa.metrics.F1Score(num_classes=2, average="micro", threshold=threshold)
        ce = tf.keras.metrics.BinaryCrossentropy(from_logits=False)

        num = test_metric(model, dataset, [acc, auc, f1, ce], model_name, patch_multiplier, False)

        return ({"acc": acc.result().numpy(), "auc": auc.result().numpy(), "f1": f1.result().numpy(), "cre": ce.result().numpy()}, num)

    def run_robustness_tests(parameters, man_func):
        threshold = 0.5
        acc = tf.keras.metrics.BinaryAccuracy(threshold=threshold)
        auc = tf.keras.metrics.AUC(num_thresholds=10)
        f1 = tfa.metrics.F1Score(num_classes=2, average="micro", threshold=threshold)
        ce = tf.keras.metrics.BinaryCrossentropy(from_logits=False)

        results = test_validation(model, [acc, auc, f1, ce], man_func, parameters)
        print(f"Model name: {model_name}")
        print(f"Parameters: {parameters}")
        print(f"Results (acc, auc, f1, ce): {results}")
        return results

    def plot_results_resize(Y_conv, Y_aaconv, X):
        Y_conv = list(map(lambda y: y[1] * 100, Y_conv))
        Y_aaconv = list(map(lambda y: y[1] * 100, Y_aaconv))

        plt.title("Linear Interpolation Resize")

        plt.xticks(X[1::4], X[1::4])
        plt.plot(X, Y_conv)
        plt.scatter(X, Y_conv, marker="^")

        plt.plot(X, Y_aaconv)
        plt.scatter(X, Y_aaconv, marker="D")

        plt.xlabel("Resize Factor %")
        plt.ylabel("AUC %")
        plt.legend(["Conv", "AAConv"])

        plt.show()

    def plot_results_jpeg(Y_conv, Y_aaconv, X):
        Y_conv = list(map(lambda y: y[1] * 100, Y_conv))
        Y_aaconv = list(map(lambda y: y[1] * 100, Y_aaconv))

        plt.title("JPEG Compression")

        plt.xticks(X[1::2] + [104], X[1::2] + ["N/A"])
        plt.plot(X + [104], Y_conv)
        plt.scatter(X + [104], Y_conv, marker="^")

        plt.plot(X + [104], Y_aaconv)
        plt.scatter(X + [104], Y_aaconv, marker="D")

        plt.xlabel("Compression Factor %")
        plt.ylabel("AUC %")
        plt.legend(["Conv", "AAConv"])

        plt.show()

    #generate_CG_1050_masks()

    def man_func_jpeg(par):
        return lambda img, mask : (tf.image.adjust_jpeg_quality(img, par), mask)
    
    def man_func_resize(par):
        down_size = round(256 * (par/100))
        return lambda img, mask : (tf.image.resize(tf.image.resize(img, (down_size, down_size)), (256, 256)), mask)

    #run_robustness_tests(list(range(50, 101, 2)), man_func_jpeg)
    #run_robustness_tests(list(range(10, 101, 2)), man_func_resize)

    # Resize results
    results_conv = [[0.33533344, 0.53471166, 0.19265687, 2.14708], [0.36906654, 0.53609884, 0.19144283, 1.731245], [0.33392966, 0.53745985, 0.19281745, 2.0772269], [0.34863016, 0.5404514, 0.19223796, 1.9373512], [0.35281146, 0.5422297, 0.19241148, 1.8977821], [0.3605734, 0.54374003, 0.19228087, 1.8522769], [0.37592098, 0.54492134, 0.19186112, 1.7579124], [0.3598024, 0.54232705, 0.1924129, 1.921569], [0.35607055, 0.54162097, 0.19260672, 1.9730724], [0.34359714, 0.53978986, 0.19299826, 2.106272], [0.34150776, 0.5391389, 0.19297458, 2.1113045], [0.33140072, 0.5370698, 0.19321382, 2.2958112], [0.32509333, 0.53599936, 0.1934174, 2.4304278], [0.31139943, 0.53293085, 0.19358681, 2.5870123], [0.38048136, 0.54153746, 0.19125748, 1.7355385], [0.31085858, 0.53265095, 0.19354971, 2.5234733], [0.30055162, 0.53170276, 0.19387734, 2.758474], [0.3057069, 0.53123045, 0.19384107, 2.6682036], [0.2972936, 0.52981496, 0.19417503, 2.8439403], [0.2935173, 0.52866393, 0.19420747, 2.9722211], [0.318168, 0.530879, 0.19321944, 2.7239816], [0.28803107, 0.52606475, 0.1942627, 3.0790803], [0.28927228, 0.5244535, 0.1938912, 2.9713945], [0.28880456, 0.52271503, 0.19377775, 2.9838426], [0.28083146, 0.51946175, 0.19389996, 3.0603685], [0.2891563, 0.5174826, 0.19353805, 2.875969], [0.35581017, 0.5244943, 0.19105645, 2.078591], [0.27896214, 0.5120358, 0.19340028, 3.1344244], [0.28409174, 0.51325727, 0.19342595, 3.0975738], [0.2865429, 0.51314783, 0.19329311, 3.0141466], [0.28930143, 0.5136489, 0.1930715, 2.9552698], [0.2954388, 0.5141473, 0.19280452, 2.8532314], [0.30186653, 0.5168437, 0.19252416, 2.744891], [0.3070102, 0.51950073, 0.19240615, 2.682449], [0.3038356, 0.5181603, 0.19236995, 2.7571354], [0.3017383, 0.5174386, 0.19244619, 2.7858846], [0.3022532, 0.51726556, 0.19237517, 2.7824366], [0.2967081, 0.51384187, 0.1925182, 2.8772469], [0.3141933, 0.51547945, 0.19138308, 2.4718041], [0.5119211, 0.55709255, 0.18381988, 1.34757], [0.31271842, 0.52104414, 0.1920303, 2.3076484], [0.29450557, 0.5095077, 0.19166574, 2.494348], [0.30794367, 0.51607, 0.19126724, 2.4739583], [0.33339146, 0.52968264, 0.19106133, 2.3118916], [0.34129697, 0.5402729, 0.19189532, 2.5009863], [0.94107765, 0.8517723, 0.18126671, 0.18161024]]
    results_aaconv = [[0.406167, 0.53707886, 0.19004633, 1.8931779], [0.42838517, 0.5389364, 0.18873513, 1.6742066], [0.4106865, 0.5401092, 0.18976633, 1.829362], [0.3901984, 0.54040647, 0.19091356, 1.9974651], [0.3796089, 0.54044586, 0.1912568, 2.089286], [0.37264898, 0.54131866, 0.19187003, 2.1918142], [0.3586923, 0.5408333, 0.19264698, 2.3487952], [0.34383956, 0.5389574, 0.19287156, 2.5725095], [0.3419388, 0.53840816, 0.19282232, 2.6413145], [0.33368143, 0.53716564, 0.19300188, 2.79635], [0.33032525, 0.53691816, 0.19337747, 2.8384206], [0.3273811, 0.53610873, 0.19328131, 2.9099786], [0.3185713, 0.53574324, 0.19356291, 3.0588899], [0.31917715, 0.5340724, 0.19319966, 3.0405076], [0.32148638, 0.53358036, 0.19271015, 2.9822273], [0.30875465, 0.5332562, 0.19321676, 3.2305472], [0.29847935, 0.5336637, 0.1936823, 3.4221532], [0.30018622, 0.53348154, 0.19360127, 3.3592417], [0.30010533, 0.5337612, 0.1936003, 3.373342], [0.29469994, 0.53275186, 0.19343889, 3.5168664], [0.3209975, 0.53512883, 0.19313653, 3.1811152], [0.2793442, 0.5308454, 0.19400904, 3.734955], [0.28861645, 0.53144985, 0.19369699, 3.4807122], [0.31057045, 0.53330284, 0.19280088, 3.0825384], [0.30977187, 0.5313362, 0.1926381, 3.069214], [0.319888, 0.5310686, 0.19240005, 2.8741019], [0.34188887, 0.53242236, 0.1913357, 2.5503066], [0.3327069, 0.53054875, 0.19141442, 2.6298583], [0.333142, 0.5312096, 0.19137278, 2.6332161], [0.3364431, 0.5324985, 0.19143535, 2.5690224], [0.34594047, 0.5345195, 0.19080763, 2.4661286], [0.35204828, 0.53681993, 0.19076988, 2.4104369], [0.35924014, 0.5383876, 0.19038975, 2.3535678], [0.35760453, 0.5393493, 0.19043945, 2.378742], [0.36354864, 0.539627, 0.19022645, 2.3220708], [0.37197483, 0.54294205, 0.18999399, 2.2405112], [0.37194338, 0.5411827, 0.18968989, 2.2591202], [0.370226, 0.53791463, 0.18907608, 2.2698967], [0.398999, 0.53659654, 0.18725674, 2.0654478], [0.4264286, 0.5390289, 0.18584959, 1.8762609], [0.3793186, 0.528661, 0.18726006, 2.1499918], [0.34980226, 0.52443796, 0.1874985, 2.2781782], [0.3555791, 0.5240855, 0.18707013, 2.2234392], [0.34618223, 0.5255939, 0.18783739, 2.4208314], [0.31489155, 0.52767044, 0.19029458, 2.9227295], [0.9325136, 0.8206522, 0.1706419, 0.20666979]]
    plot_results_resize(results_conv, results_aaconv, list(range(10, 101, 2)))

    # JPEG results
    results_conv = [[0.3256288, 0.5195724, 0.19088036, 1.9979612], [0.3080951, 0.5190341, 0.19116636, 2.0740914], [0.31249732, 0.5194013, 0.19091788, 2.0272148], [0.33778462, 0.52144796, 0.19046536, 1.8723682], [0.36287075, 0.5218159, 0.18954386, 1.7217563], [0.3722026, 0.5227339, 0.18912537, 1.6682433], [0.39132273, 0.5239247, 0.18809514, 1.5729057], [0.4331189, 0.52672213, 0.1864535, 1.4006066], [0.45011055, 0.5275746, 0.18549123, 1.3342245], [0.46576625, 0.5300284, 0.18504553, 1.2774959], [0.49401176, 0.5329385, 0.18349122, 1.1872529], [0.5238026, 0.53492534, 0.18173398, 1.0950644], [0.5505752, 0.5382489, 0.1798011, 1.0234705], [0.57828385, 0.5414938, 0.17850244, 0.95572263], [0.59672827, 0.5455027, 0.1771264, 0.90381294], [0.6168893, 0.5488937, 0.17547794, 0.8559702], [0.6348818, 0.5513464, 0.17291194, 0.8142473], [0.6503648, 0.55419695, 0.17182231, 0.7801133], [0.67185915, 0.55978256, 0.16989996, 0.7352507], [0.69119376, 0.56392485, 0.16741343, 0.70438933], [0.6888173, 0.5685227, 0.16846307, 0.7194359], [0.68182635, 0.57245004, 0.17051393, 0.7530623], [0.6734451, 0.5755605, 0.17230028, 0.7880216], [0.632633, 0.57921576, 0.17782927, 0.9188977], [0.6006274, 0.587008, 0.18231402, 1.0070901], [0.6132408, 0.5985377, 0.18248226, 0.95174557], [0.94107836, 0.85177153, 0.18126646, 0.18161029]]
    results_aaconv = [[0.8218939, 0.55485314, 0.115180776, 0.45313784], [0.824332, 0.55425173, 0.11316119, 0.44966602], [0.8254194, 0.5553507, 0.11295095, 0.44798037], [0.8283941, 0.55704933, 0.110843025, 0.44392785], [0.8316387, 0.55733234, 0.10908659, 0.43961066], [0.83430344, 0.5594416, 0.10741011, 0.435612], [0.8364313, 0.55884516, 0.10476868, 0.4328346], [0.83942187, 0.5597152, 0.102502175, 0.42912453], [0.8417637, 0.5614086, 0.10148378, 0.42540672], [0.843288, 0.56315875, 0.10041461, 0.4235142], [0.84495044, 0.563975, 0.09895619, 0.4210889], [0.8470263, 0.56602, 0.09745706, 0.41851094], [0.8498531, 0.5676328, 0.09697471, 0.41560534], [0.8516523, 0.56955314, 0.09515092, 0.41413707], [0.8536044, 0.57154244, 0.09235327, 0.41041157], [0.8551479, 0.57375294, 0.0931682, 0.40849057], [0.8561762, 0.57547635, 0.09408996, 0.40722558], [0.8572144, 0.5793206, 0.09375165, 0.40556237], [0.85779357, 0.5834164, 0.0951673, 0.40452954], [0.8584661, 0.5879238, 0.09729171, 0.40468404], [0.8545752, 0.5920308, 0.102891065, 0.41035974], [0.8481662, 0.5979524, 0.11019958, 0.4212055], [0.8436563, 0.60295635, 0.115961984, 0.42947868], [0.83046794, 0.6117933, 0.1286624, 0.45086476], [0.81507736, 0.6195784, 0.1390224, 0.47287244], [0.80687374, 0.626762, 0.14537896, 0.4799317], [0.9325132, 0.82065, 0.17064159, 0.20667014]]
    plot_results_jpeg(results_conv, results_aaconv, list(range(50, 101, 2)))

    #for i, (tampered, mask, name, dataset_name) in enumerate(dataset):
        #print(f"out{i}")
        #tf.io.write_file(f"out/out{i}.png", tf.io.encode_png(tf.concat([tampered, mask], axis=1)))
        #tf.io.write_file(f"test_out{i}.png", tf.io.encode_png(tf.concat([image, mask, tf.reshape(c_mask, [256, 256, 3])]))) 

    #results, _ = run_tests(1)
    #print(f"Acc: {results['acc']}")
    #print(f"AUC: {results['auc']}")
    #print(f"F1 : {results['f1']}")
    #print(f"CrE: {results['cre']}")
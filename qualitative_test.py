import tensorflow as tf
import random
from manipulations import ManiFamily
import numpy as np
import math
import tensorflow_io as tfio
import cv2

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

def split_patches(image, patch_width, patch_height, patch_multiplier = 1):
    height, width, _ = image.shape

    num_patches_x = math.ceil(width / patch_width * patch_multiplier)
    num_patches_y = math.ceil(height / patch_height * patch_multiplier)

    if num_patches_x - 1 == 0:
        delta_x = 0
    else:
        delta_x = (width - patch_width) / (num_patches_x - 1)

    if num_patches_y - 1 == 0:
        delta_y = 0
    else:
        delta_y = (height - patch_height) / (num_patches_y - 1)

    patches = []
    for x in range(num_patches_x):
        for y in range(num_patches_y):
            nx = math.floor(x * delta_x)
            ny = math.floor(y * delta_y)

            patches.append((nx, ny, image[ny:ny+patch_height, nx:nx+patch_width, :]))

    return patches

def predict_patches(model, patches, num_runs=1):
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    patch_images = [img for (x, y, img) in patches]
    out = []
    
    for imgs in chunks(patch_images, num_runs):
        patches_stack = tf.stack(imgs)
        out.append(model(patches_stack).numpy())

    out = np.concatenate(out)

    return [(x, y, img) for ((x, y, _), img) in zip(patches, out)]

def combine_patches(patches):
    patch_height, patch_width, patch_depth = patches[0][2].shape
    final_width = max([x + patch_width for (x, _, img) in patches])
    final_height = max([y + patch_height for (_, y, img) in patches])

    img_sum = np.zeros([final_height, final_width, patch_depth])
    img_num_added = np.zeros([final_height, final_width, patch_depth])

    for (x, y, img) in patches:
        img_sum[y:y+patch_height,x:x+patch_width,:] += img
        img_num_added[y:y+patch_height,x:x+patch_width,:] += np.ones(img.shape)

    return img_sum / img_num_added

def patch_and_combine(model, image, patch_multiplier = 1):
    patches = split_patches(image, 256, 256, patch_multiplier)
    pred_patches = predict_patches(model, patches)
    combined = combine_patches(pred_patches)
    height, width = combined.shape[:2]

    certainties = tf.gather(tf.nn.softmax(combined), 1, axis=-1)
    out_mask = tf.cast(255*tf.repeat(tf.reshape(certainties, [height, width, 1]),3,axis=-1), tf.uint8)
    return out_mask


def _read_tf_image(path):
    imagefile = tf.io.read_file(path)

    if path[-4:] == ".tif":
        return tfio.experimental.image.decode_tiff(imagefile)[:,:,:3]
    else:
        return tf.image.decode_image(imagefile, channels=3)

def get_heatmap(image):
    gray_img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2GRAY)
    gray_img = 255 - gray_img
    heatmap = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)

    return heatmap

image_paths = ["mik.png", "mik2.png", "mik3_v3.png", "mik3_spl1.png", "mik3_spl2.png", "fireworks.png", "20190814_224442.jpg", "F65gsSs.jpg", "6FjFj80.jpg", "P30Idhx.jpg", "drake_mod.jpg"]
image_paths = ["manip_nos_103b.png"]

if __name__ == "__main__":
    import test_datasets

    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    #model_name = "2class_aaconv_save_at_93_final.tf"
    model_name = "2_class_pixel_conv_save_at_89_final.tf"

    model  = tf.keras.models.load_model("models/" + model_name, custom_objects={'f1':lambda x,y:1})

    def process_data_paths(paths):
        for i, path in enumerate(paths):
            print(f"{i}: {path}")
            image = _read_tf_image(f"data/{path}")
            out_mask = patch_and_combine(model, image, 5)
            height, width = out_mask.shape[:2]

            out_heat_mask = get_heatmap(out_mask)

            tf.io.write_file(f"out/out{i}.png", tf.io.encode_png(tf.reshape(out_heat_mask, [height, width, 3])))

    def process_test_dataset(dataset):
        for (tampered, mask, name, dataset_name) in dataset:
            certainties = test_datasets.calculate_certainties(model, tampered, 4, model_name, name, dataset_name)
            height, width = certainties.shape[:2]

            out_mask = tf.cast(255*tf.repeat(tf.reshape(certainties, [height, width, 1]),3,axis=-1), tf.uint8)
            out_heat_mask = get_heatmap(out_mask)

            tf.io.write_file(f"out/{dataset_name}/{name}_tamp.png", tf.io.encode_png(tampered))
            tf.io.write_file(f"out/{dataset_name}/{name}_mask.png", tf.io.encode_png(mask))
            tf.io.write_file(f"out/{dataset_name}/{name}_pred.png", tf.io.encode_png(tf.reshape(out_heat_mask, [height, width, 3])))


    process_data_paths(image_paths)
    
    #process_test_dataset(test_datasets.get_CASIA2_dataset(pattern="Tp_D_???_?_?_*").take(10))
    #process_test_dataset(test_datasets.get_CASIA2_dataset(pattern="Tp_S_???_?_?_*").take(10))
    #process_test_dataset(test_datasets.get_COVERAGE_dataset())


#    for i in range(10):
#        image  = tf.image.decode_png(tf.io.read_file(f"samples/image{i}.png"), channels=3)
#
#        mask = tf.image.decode_png(tf.io.read_file(f"samples/mask{i}.png"), channels=3)
#
#        #tf.image.random_crop(cv.cvtColor(cv.imread("test_img.png"), cv.COLOR_BGR2RGB), [256, 256, 3])
#        out = model(tf.expand_dims(image, 0))
#        
#        mean_logits = tf.math.reduce_max(out, [0,1,2])
#        print(np.around(tf.nn.softmax(mean_logits).numpy(), 2))
#        #_, top = tf.math.top_k(out, k=8)
#        #print(top)
#        c_mask, legends = colored_mask(tf.math.argmax(out, axis=-1))
#        #c_mask, legends = colored_mask(tf.gather(top, [1], axis=-1))
#
#        tf.io.write_file(f"test_out{i}.png", tf.io.encode_png(tf.concat([image, mask, tf.reshape(c_mask, [256, 256, 3])], axis=1))) 
#        #tf.io.write_file("test_in.png", tf.io.encode_png(image))
#    tf.io.write_file("test_legends.png", tf.io.encode_png(make_legends_image(legends)))

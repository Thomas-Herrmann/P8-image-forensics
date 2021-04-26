import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import AAConv2D as AA
import metrics
from DataGenerator import get_combined_two_class_dataset, get_weighted_two_class_dataset
epochs = 100

image_size = (256, 256)
batch_size = 128


train_ds = get_combined_two_class_dataset(batch_size).repeat()
#train_ds = get_weighted_two_class_dataset(batch_size, {0:0.1, 1:1.}).repeat()
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

#sizes =             [128,64, 32, 16,  8,  4]
down_stack_filters = [64,128,256,512]
up_stack_filters   = reversed(down_stack_filters[:-1])
KERNEL_SIZE = 3

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    # Down stack
    skips = []
    for filter_size in down_stack_filters:
        initializer = tf.random_normal_initializer(0., 0.02)
        x = tf.keras.layers.Conv2D(filter_size, KERNEL_SIZE, strides=2, kernel_initializer=initializer, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        skips.append(x)

    skips = list(reversed(skips))

    for size in [1024]:
        x = AA.AAConv2D(size, KERNEL_SIZE, size//2, size//2, 4, True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        #x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Concatenate()([x, skips[0]])

    # Up stack
    for filter_size, skip in zip(up_stack_filters, skips[1:]):
        initializer = tf.random_normal_initializer(0., 0.02)
        x = tf.keras.layers.Conv2DTranspose(filter_size, KERNEL_SIZE, strides=2, kernel_initializer=initializer, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        # Add skip
        x = tf.keras.layers.Concatenate()([x, skip])

    # Classification layer
    initializer = tf.random_normal_initializer(0., 0.02)
    outputs = tf.keras.layers.Conv2DTranspose(num_classes, KERNEL_SIZE, strides=2, kernel_initializer=initializer, padding='same')(x)
    
    #outputs = tf.keras.layers.Reshape((image_size[0] * image_size[1], num_classes))(outputs)
    
    return keras.Model(inputs, outputs)

load_from_epoch = None

if load_from_epoch is not None:
    model = keras.models.load_model(f"2class_aaconv_save_at_{load_from_epoch}.tf")
else:
    model = make_model(input_shape=image_size + (3,), num_classes=2)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
model.summary()
#exit()
callbacks = [
    keras.callbacks.ModelCheckpoint("2class_aaconv_save_at_{epoch}.tf")
]

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, steps_per_epoch=200000//batch_size,
    #validation_data=val_ds, 
    #class_weight=class_weight, 
    initial_epoch=load_from_epoch or 0,
)
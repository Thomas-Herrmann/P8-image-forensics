import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import AAConv2D as AA
import metrics
from DataGenerator import get_combined_two_class_dataset
epochs = 100

image_size = (256, 256)
batch_size = 32

train_ds = get_combined_two_class_dataset(batch_size).repeat()
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

#val_ds = val_ds.prefetch(buffer_size=batch_size)

#sizes =             [128,64, 32, 16,      8]
down_stack_filters = [64,128,256,512,512+128]
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

    skips = reversed(skips[:-1])

    # Up stack
    for filter_size, skip in zip(up_stack_filters, skips):
        initializer = tf.random_normal_initializer(0., 0.02)
        x = tf.keras.layers.Conv2DTranspose(filter_size, KERNEL_SIZE, strides=2, kernel_initializer=initializer, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        # Add skip
        x = tf.keras.layers.Concatenate()([x, skip])

    # Classification layer
    initializer = tf.random_normal_initializer(0., 0.02)
    outputs = tf.keras.layers.Conv2DTranspose(num_classes, KERNEL_SIZE, strides=2, kernel_initializer=initializer, padding='same')(x)
    
    return keras.Model(inputs, outputs)

load_from_epoch = None

if load_from_epoch is not None:
    model = keras.models.load_model(f"2_class_pixel_conv_save_at_{load_from_epoch}.tf")
else:
    model = make_model(input_shape=image_size + (3,), num_classes=2)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", metrics.f1],
    )
model.summary()

#exit()
callbacks = [
    keras.callbacks.ModelCheckpoint("2_class_pixel_conv_save_at_{epoch}.tf")
]

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, steps_per_epoch=200000//batch_size,
    #validation_data=val_ds, class_weight=class_weight, 
    initial_epoch=load_from_epoch or 0,
)
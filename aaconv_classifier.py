import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import AAConv2D as AA
import metrics
from DataGenerator import get_combined_classification_dataset, get_classification_valid_dataset
epochs = 100

image_size = (256, 256)
batch_size = 128

train_ds = get_combined_classification_dataset(batch_size).repeat()
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = get_classification_valid_dataset(batch_size).prefetch(buffer_size=batch_size)

#sizes =             [128,64, 32, 16,      8]
down_stack_filters = [64,128,256,512]
KERNEL_SIZE = 3

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, KERNEL_SIZE, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, KERNEL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    
    # Down stack
    for filter_size in down_stack_filters:
        initializer1 = tf.random_normal_initializer(0., 0.02)
        x = layers.Conv2D(filter_size, KERNEL_SIZE, strides=1, kernel_initializer=initializer1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        initializer2 = tf.random_normal_initializer(0., 0.02)
        x = layers.Conv2D(filter_size, KERNEL_SIZE, strides=2, kernel_initializer=initializer2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
    
    residual = layers.Conv2D(512,1, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(x)
    residual = layers.LeakyReLU()(x)
    
    for size in [512,512,512]:
        x = AA.AAConv2D(size, KERNEL_SIZE, size//2, size//2, 4, True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    x = layers.Add()([x, residual])

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)
    # Classification layer
    #initializer = tf.random_normal_initializer(0., 0.02)
    #outputs = tf.keras.layers.Conv2DTranspose(num_classes, KERNEL_SIZE, strides=2, kernel_initializer=initializer, padding='same')(x)
    
    return keras.Model(inputs, outputs)

load_from_epoch = None

if load_from_epoch is not None:
    model = keras.models.load_model(f"aaconv_classifier_save_at_{load_from_epoch}.tf")
else:
    model = make_model(input_shape=image_size + (3,), num_classes=9)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
model.summary()

#exit()
callbacks = [
    keras.callbacks.ModelCheckpoint("aaconv_classifier_save_at_{epoch}.tf")
]

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, steps_per_epoch=200000//batch_size,
    validation_data=val_ds, 
    #class_weight=class_weight, 
    initial_epoch=load_from_epoch or 0,
)
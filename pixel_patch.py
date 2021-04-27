import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import AAConv2D as AA
import tensorflow_addons as tfa
import metrics
from DataGenerator import get_combined_dataset

epochs = 100

batch_size = 32
num_epochs = 100
image_size = 256 
input_shape = (image_size, image_size, 3)
patch_size = 8 
patch_dim = image_size // patch_size
num_patches = (image_size // patch_size) ** 2
projection_dim = 256
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers

output_units = [
    projection_dim * 2,
    9*patch_size*patch_size
]
transformer_layers = 7
#mlp_head_units = [1024, 512]  # Size of the dense layers of the final classifier

train_ds = get_combined_dataset(batch_size).repeat()
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size
        })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    # Entry block
    augmented = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    '''
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.BatchNormalization()(encoded_patches)
    x = tf.keras.layers.Reshape((patch_dim, patch_dim, projection_dim))(representation)
    # Classification layer
    initializer = tf.random_normal_initializer(0., 0.02)
    outputs = tf.keras.layers.Conv2DTranspose(num_classes, patch_size, strides=patch_size, kernel_initializer=initializer, padding='same')(x)
    '''

    representation = layers.BatchNormalization()(encoded_patches)
    representation = mlp(representation,  hidden_units=output_units, dropout_rate=0)
    outputs = tf.keras.layers.Reshape((image_size, image_size, num_classes))(representation)
    
    return keras.Model(inputs, outputs)

load_from_epoch = None

if load_from_epoch is not None:
    model = keras.models.load_model(f"pixel_patch_save_at_{load_from_epoch}.tf")
else:
    model = make_model(input_shape=(image_size, image_size, 3), num_classes=9)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", metrics.f1],
    )
model.summary()

#exit()

callbacks = [
    keras.callbacks.ModelCheckpoint("pixel_patch_save_at_{epoch}.tf")
]

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, steps_per_epoch=200000//batch_size,
    #validation_data=val_ds, class_weight=class_weight, 
    initial_epoch=load_from_epoch or 0,
)
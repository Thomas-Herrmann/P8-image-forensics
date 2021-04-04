import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import config
import AAConv2D as AA
from Copied import augmented_conv2d
from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



gpus = config.list_physical_devices('GPU')
for gpu in gpus:
    config.experimental.set_memory_growth(gpu, True)

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = keras.Input(shape=input_shape)
#x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
x=AA.AAConv2D(16 * 3, 3, 16 * 2, 16 * 2, 4, False)(inputs)
#augmented_conv2d(inputs, 16 * 3, 3, 16 * 2, 16 * 2, 2, False) #
x=layers.ReLU()(x)
x=layers.MaxPooling2D(pool_size=(2, 2))(x)
#x=layers.Conv2D(32*3, kernel_size=(3, 3), activation="relu", padding='same')(x)
x=AA.AAConv2D(32 * 3, 3, 32 * 2, 32 * 2, 4, False)(x)
#augmented_conv2d(x, 32 * 3, 3, 32 * 2, 32 * 2, 4, False)#
x=layers.ReLU()(x)                                   # layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
x=layers.MaxPooling2D(pool_size=(2, 2))(x)
x=layers.Flatten()(x)
x=layers.Dropout(0.5)(x)
outputs=layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
'''
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(34, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32*3, kernel_size=(3, 3), activation="relu", padding='same'),
        #AA.AAConv2D(32 * 3, 3, 32 * 2, 32 * 2, 4, True), #
        layers.ReLU(),                                   # layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
'''

model.summary()

batch_size = 64
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[f1_m])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
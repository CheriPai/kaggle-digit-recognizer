from keras.callbacks import EarlyStopping
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model, model_from_json
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from utils import process_data
from utils import MODEL_PATH, WEIGHTS_PATH, TRAIN_PATH, BATCH_SIZE, VAL_PROP
import argparse
import numpy as np
import pandas as pd


def build_model():
    img_input = Input(shape=(1,28,28))

    # Block 1
    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same", name="block1_conv1")(img_input)
    x = Convolution2D(64, 3, 3, activation="relu", border_mode="same", name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation="relu", border_mode="same", name="block2_conv1")(x)
    x = Convolution2D(128, 3, 3, activation="relu", border_mode="same", name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same", name="block3_conv1")(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same", name="block3_conv2")(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same", name="block3_conv3")(x)
    x = Convolution2D(256, 3, 3, activation="relu", border_mode="same", name="block3_conv4")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

    # Classification block
    x = Flatten(name="flatten")(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu", name="fc1")(x)
    x = Dense(10, activation="softmax", name="predictions")(x)

    # Create model
    model = Model(img_input, x)
    model.summary()
    return model


def train_model(pretrain):
    X, y = process_data(TRAIN_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_PROP)

    if pretrain:
        model = model_from_json(open(MODEL_PATH).read())
        model.load_weights(WEIGHTS_PATH)
    else:
        model = build_model()

    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_acc", patience=8, mode="max")
    model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=100, 
        validation_data=(X_val, y_val), callbacks=[early_stop])

    print("Saving model to ", MODEL_PATH)
    print("Saving weights to ", WEIGHTS_PATH)
    open(MODEL_PATH, "w").write(model.to_json())
    model.save_weights(WEIGHTS_PATH, overwrite=True)

    accuracy = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE)
    print("Accuracy: ", accuracy)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", action="store_true")
    args = parser.parse_args()

    train_model(args.p)

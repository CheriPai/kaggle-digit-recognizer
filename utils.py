import keras.utils.np_utils as kutils
import pandas as pd

MODEL_PATH = "data/model.csv"
WEIGHTS_PATH = "data/weights.h5"
TRAIN_PATH = "data/train.csv"
BATCH_SIZE = 64
VAL_PROP = 0.10

def process_data(fname):
    df = pd.read_csv(fname)
    y = df["label"].as_matrix()
    X = df.drop("label", 1).as_matrix()
    X = X.reshape((X.shape[0],1,28,28)).astype(float)
    X /= 255
    y = kutils.to_categorical(y)
    return X, y

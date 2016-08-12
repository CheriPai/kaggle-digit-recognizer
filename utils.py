import keras.utils.np_utils as kutils
import pandas as pd


MODEL_PATH = "data/model.csv"
WEIGHTS_PATH = "data/weights.h5"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
SUBMISSION_PATH = "data/submission.csv"
BATCH_SIZE = 64
VAL_PROP = 0.10


def process_data(fname, mode="TRAIN"):
    """ Reads data from CSV and converts to 
        a format digestable to the algorithm
    """
    df = pd.read_csv(fname)

    if mode == "TRAIN":
        X = preprocess_X(df.drop("label", 1))
        y = df["label"].as_matrix()
        y = kutils.to_categorical(y)
        return X, y
    else:
        X = preprocess_X(df)
        return X


def preprocess_X(X):
    """ Preprocessing steps for input data.
        Reshapes matrix to (1, 28, 28) and 
        scales values to range 0 to 1.
    """
    X = X.as_matrix()
    X = X.reshape((X.shape[0],1,28,28)).astype(float)
    X /= 255
    return X

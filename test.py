from keras.models import model_from_json
import numpy as np
import pandas as pd
from utils import process_data
from utils import TEST_PATH, WEIGHTS_PATH, MODEL_PATH, SUBMISSION_PATH, BATCH_SIZE


if __name__ == "__main__":
    model = model_from_json(open(MODEL_PATH).read())
    model.load_weights(WEIGHTS_PATH)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    X = process_data(TEST_PATH, mode="TEST")
    predictions = model.predict(X, batch_size=BATCH_SIZE)

    predictions = np.argmax(predictions, axis=1)

    submission = pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions})
    submission = submission[["ImageId", "Label"]]
    submission.to_csv(SUBMISSION_PATH, index=False)

import numpy as np
import pandas as pd
from sklearn.externals import joblib


df = pd.read_csv('data/test.csv')

# Load classifier and run test
clf = joblib.load('data/output/clf.pkl')
predictions = clf.predict(df.values)

# Output results
predictions = np.transpose(predictions)[0]
predictions = pd.DataFrame({'ImageId': np.arange(1, len(df.values)+1), 'Label': predictions})
predictions.to_csv('data/output/predictions.csv', index=False)

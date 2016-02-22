import pandas as pd
from sklearn import cross_validation as cv
from sklearn.externals import joblib
from sknn.mlp import Classifier, Convolution, Layer

df = pd.read_csv('data/train.csv')

X = df.drop('label', axis=1).values
y = df.label.values

# Set up train and cv sets
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.1, random_state=0)

# Create neural network
clf = Classifier(
    layers=[
        Convolution('Rectifier', channels=20, kernel_shape=(5, 5), pool_shape=(2, 2)),
        Layer('Rectifier', units=20*12*12),
        Layer('Softmax')
    ],
    learning_rate=0.00002,
    n_iter=10,
    verbose=True
)
clf.fit(X_train, y_train)

# Test on cross validation set
print('Accuracy: %.4f' % clf.score(X_test, y_test))

# Export model
joblib.dump(clf, 'data/output/clf.pkl')

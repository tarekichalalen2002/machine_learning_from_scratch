from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest

data = datasets.load_breast_cancer()
X,y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

# clf = DecisionTree(max_depth=15)
clf = RandomForest()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

print("Decision Tree Classification Accuracy:", accuracy(y_test, predictions))
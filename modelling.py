#IMPORTING SKLEARN FUNCTIONS AND CLASSES
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#IMPORTING PANDAS FOR PROCESSING DATA
import pandas as pd

#LOAD DATA AND DROP INDEX
text_data = pd.read_csv('examples.csv')
del text_data['DOC_INDEX']

#SPLIT DATA INTO TRAIN AND TEST SET
X_train, X_test = train_test_split(text_data, test_size = 0.25, shuffle = True)

#GET X AND Y VALUES FROM DATA FRAME
Y_train = X_train['STEM_DESIGNATION']
Y_test = X_test['STEM_DESIGNATION']

del X_train['STEM_DESIGNATION']
del X_test['STEM_DESIGNATION']



#TRAIN NAIVE BAYES MODEL
bernoulli = BernoulliNB()
bernoulli.fit(X_train, Y_train)

training_predictions = bernoulli.predict(X_train)
training_accuracy = accuracy_score(Y_train, training_predictions)
print("Training Accuracy of Bernoulli Naive Bayes model: {}".format(training_accuracy))
testing_predictions = bernoulli.predict(X_test)
testing_accuracy = accuracy_score(Y_test, testing_predictions)
print("Testing Accuracy of Bernoulli Naive Bayes model: {}".format(testing_accuracy))

#TRAIN LOGISTIC REGRESSION MODEL
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
training_accuracy = logreg.score(X_train, Y_train)
print("Training Accuracy of Logistic Regression model: {}".format(training_accuracy))
testing_accuracy = logreg.score(X_test, Y_test)
print("Testing Accuracy of Logistic Regression model: {}".format(testing_accuracy))

#TRAIN PERCEPTRON MODEL
percep = Perceptron()
percep.fit(X_train, Y_train)
training_accuracy = percep.score(X_train, Y_train)
print("Training Accuracy of Perceptron model: {}".format(training_accuracy))
testing_accuracy = percep.score(X_test, Y_test)
print("Testing Accuracy of Perceptron model: {}".format(testing_accuracy))

#TRAIN KNN MODEL
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
training_accuracy = knn.score(X_train, Y_train)
print("Training Accuracy of KNN model: {}".format(training_accuracy))
testing_accuracy = knn.score(X_test, Y_test)
print("Testing Accuracy of KNN model: {}".format(testing_accuracy))


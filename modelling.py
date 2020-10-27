#IMPORTING SKLEARN FUNCTIONS AND CLASSES
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#IMPORTING PANDAS FOR PROCESSING DATA
import pandas as pd

#IMPORTING JOBLIB TO SAVE MODEL 
from joblib import dump, load 

#LOAD DATA AND DROP INDEX
text_data = pd.read_csv('data/examples.csv')
del text_data['DOC_INDEX']

#SPLIT DATA INTO TRAIN AND TEST SET
X_train, X_test = train_test_split(text_data, test_size = 0.25, shuffle = True)

#GET X AND Y VALUES FROM DATA FRAME
Y_train = X_train['STEM_DESIGNATION']
Y_test = X_test['STEM_DESIGNATION']

del X_train['STEM_DESIGNATION']
del X_test['STEM_DESIGNATION']

list_of_models = {
	'Bernoulli Naive Bayes' : BernoulliNB(), 
	'Logistic Regression' : LogisticRegression(), 
	'Perceptron' : Perceptron(), 
	'K Nearest Neighbors' : KNeighborsClassifier()
}

for model_name in list_of_models.keys():
	print("Model being used: {}".format(model_name))
	model = list_of_models[model_name]

	#TRAIN THE MODEL
	model.fit(X_train, Y_train)

	#GENERATE PREDICTIONS AND CALCULATE IMPACT METRICS ON TRAINING DATA
	training_predictions = model.predict(X_train)
	training_accuracy = accuracy_score(Y_train, training_predictions)
	training_precision, training_recall, training_fscore, training_support = precision_recall_fscore_support(Y_train, training_predictions)
	print("Training Accuracy of {}: {}".format(model_name, training_accuracy))
	print("Training Precision of {}: {}".format(model_name, training_precision))
	print("Training Recall of {}: {}".format(model_name, training_recall))
	print("Training F Score of {}: {}".format(model_name, training_fscore))
	print("Training Support of {}: {}".format(model_name, training_support))
	print("____________________________________________________________________")

	#GENERATE PREDICTIONS AND CALCULATE IMPACT METRICS ON TESTING DATA
	testing_predictions = model.predict(X_test)
	testing_accuracy = accuracy_score(Y_test, testing_predictions)
	testing_precision, testing_recall, testing_fscore, testing_support = precision_recall_fscore_support(Y_test, testing_predictions)
	print("Testing Accuracy of {}: {}".format(model_name, testing_accuracy))
	print("Testing Precision of {}: {}".format(model_name, testing_precision))
	print("Testing Recall of {}: {}".format(model_name, testing_recall))
	print("Testing F Score of {}: {}".format(model_name, testing_fscore))
	print("Testing Support of {}: {}".format(model_name, testing_support))
	print("____________________________________________________________________")

	#SAVE MODEL TO FILE
	model_file_name = "{}.joblib".format(model_name.replace(" ","_"))
	dump(model, "models/{}".format(model_file_name))
	
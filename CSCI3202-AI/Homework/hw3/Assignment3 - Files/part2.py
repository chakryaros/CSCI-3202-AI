import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score






Label = "Credit"
Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

def saveBestModel(clf):
    pickle.dump(clf, open("bestModel.model", 'wb'))

def readData(file):
    df = pd.read_csv(file)
    return df

def trainOnAllData(df, clf):
    #Use this function for part 4, once you have selected the best model
   
    x_train = df[Features]
    y_train = df[Label]
    clf.fit(x_train, y_train)
    
    saveBestModel(clf)

# read data
df = readData("credit_train.csv")

dictLabel = {"Credit":{'good':1,'bad':0}}
df.replace(dictLabel,inplace = True)

#use best model to train data 
trainOnAllData(df, LinearRegression())



X, y = df[["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]], df["Credit"]


def cross_validation(classifier, x, y, k=10):
	k_fold = KFold(n_splits=k)
	AUROC = []
	
	for train_index, test_index in k_fold.split(x,y):
		x_train, x_test = x.loc[train_index], x.loc[test_index]
		y_train, y_test = y[train_index], y[test_index]
	
		#create and fit classifier
		model = classifier
		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		AUROC.append(roc_auc_score(y_test, y_pred))

	
	mean_Auroc = np.mean(AUROC)
	std_Auroc = np.std(AUROC)

	return mean_Auroc, std_Auroc



classifiers = {"Logistic Regression": LogisticRegression(),
               "Naive Bayes" : GaussianNB(),
               "SVC": svm.SVC(C=1000, gamma= 0.0001), 
               "Decision Tree": tree.DecisionTreeClassifier(),
               "Random Forest": RandomForestClassifier(max_depth=6, n_estimators=64),
               "Linear Regression" : LinearRegression(),
               "Gradient Boosting Classifier" : GradientBoostingClassifier(),
               }

for name, clf in zip(classifiers.keys(), classifiers.values()):
	mean, std = cross_validation(clf, X, y)
	print("The mean AUROC of {} : {:.04f}".format(name, mean))
	print("The Standard Deviation of {} : {:.04f}".format(name, std))


 #techinique used for hyperparameter section

 #test on SVM
# c = [1, 10, 100, 1000]
# gamma = [0.001, 0.0001]
# for g in gamma:
# 	for index in c:
# 		svc = svm.SVC(C=index, gamma=g)
# 		mean_auroc, std_auroc = cross_validation(svc, X, y)
# 		print("c = {}, gamma = {},  The mean = {:.04f}".format(index, g,  mean_auroc))
# 		# print("The Standard Deviation = {:.04f}".format( std))

# max_depths = np.linspace(1, 32, 32, endpoint=True)
# n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
# print("hyperparameter on max_depths")
# for depth in max_depths:
# 	randomForest = RandomForestClassifier(max_depth=depth, n_estimators=10)
# 	mean_auroc, std_auroc = cross_validation(randomForest, X, y)
# 	print("max_depth = {}, The mean = {:.04f}".format(depth,  mean_auroc))
# print("hyperparameter on n_estimators")
# for n in n_estimators:
# 	randomForest = RandomForestClassifier(max_depth=9, n_estimators=n)
# 	mean_auroc, std_auroc = cross_validation(randomForest, X, y)
# 	print("n_estimators = {}, The mean = {:.04f}".format(n,  mean_auroc))

def SVC_parameter(x, y, k=10):
	k_fold = KFold(n_splits=k)
	AUROC = []
	param_grid = [
	  {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.005]},
	 ]
	
	#create and fit classifier
	svc = svm.SVC()
	grid = GridSearchCV(svc, param_grid, cv=10)

	
	grid.fit(x, y)
	print("SVM: ",grid.best_params_)
	
SVC_parameter(X, y)

def randomForest_parameter(x, y, k=10):
	k_fold = KFold(n_splits=k)
	AUROC = []


	param_grid = [
	  {'max_depth': np.linspace(1, 32, 32, endpoint=True), 'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200]},
	 ]
	
	#create and fit classifier
	randomForest = RandomForestClassifier()
	grid = GridSearchCV(randomForest, param_grid, cv=10)

	
	grid.fit(x, y)
	print("Random Forest: ",grid.best_params_)
	
randomForest_parameter(X, y)

# analysis best model and write it to file
def analysisBestModel(clf, x, y):

	bestModel = open('bestModel.output', "w")
	df = pd.read_csv("credit_train.csv")
	
	# newdf = df.copy()
	
	conf_Matrix = []
	acc = []
	precision = []
	recall = []
	
	
	clf.fit(x, y)

	y_pred = clf.predict(X)
	y_prediction = abs(y_pred.round())

	
	df["Predicted"] = y_prediction
	
	dictLabel = {"Predicted":{1:'good',0:'bad'}}
	df.replace(dictLabel,inplace = True)
	

	#write data into file
	result = df.to_csv(header = True, index = False)
	bestModel.write(result)
	
	
	AUROC = roc_auc_score(y, y_prediction)
	conf_mat = confusion_matrix(y,  y_prediction )
	print("confusion_matrix :")
	print(conf_mat)
	acc = accuracy_score(y,  y_prediction )
	print("Accuracy :",acc*100)
	precision = precision_score(y, y_prediction)
	print("Precision :",precision*100)
	recall = recall_score(y, y_prediction)
	print("Recall :",recall*100)
	print("AUROC Score :{:.04f} ".format(AUROC*100))

analysisBestModel(LinearRegression(),X, y)
	

	
	



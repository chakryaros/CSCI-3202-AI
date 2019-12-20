import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
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
    print("TODO")

    saveBestModel(clf)

df = readData("credit_train.csv")
dictLabel = {"Credit":{'good':1,'bad':0}}
df.replace(dictLabel,inplace = True) 

X, y = df[["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]], df["Credit"]

def cross_validation(classifier, x, y, k=10):
	k_fold = KFold(n_splits=k)
	score_test = list()
	mean_auroc = []
	std_auroc = []
	AUROC = []
	mean_fpr = np.linspace(0, 1, 100)
	for train_index, test_index in k_fold.split(x,y):
		
		x_train, x_test = x.loc[train_index], x.loc[test_index]
		y_train, y_test = y[train_index], y[test_index]
		



		#create and fit classifier

		model = classifier
		model.fit(x_train, y_train)
		y_pred = model.predict(x_test)
		# print(model.score(x_test, y_test))
		# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
		# tprs.append(np.interp(mean_fpr, fpr, tpr))

		# roc_auc = auc(fpr, tpr)
		AUROC.append(roc_auc_score(y_test, y_pred))

	# mean_tpr = np.mean(tprs, axis=0)
	# mean_auc = auc(mean_fpr, mean_tpr)
	# variance = sum([(score - mean_auc)**2 for score in AUROC])/k
	# std_auc = variance**.5
	# 	score_test.append(model.score(x_test, y_test))
	# mean = sum(score_test)/k
	# variance = sum([(score - mean)**2 for score in score_test])/k
	# std = variance**0.5
	# return score_test, mean, std
	mean_auroc.append(np.mean(AUROC))
	std_auroc.append(np.std(AUROC))
	
	return mean_auroc, std_auroc
# a = cross_validation(LogisticRegression(), X, y)



# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC(gamma="scale")
# clf = GridSearchCV(svc, parameters, cv=5)

		#create and fit classifier
# param_grid = [
#   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]},
#  ]
# svc = svm.SVC()
# grid = GridSearchCV(svc, param_grid, cv=10)
# grid.fit(X, y)

# y_pred = grid.predict(X)
# # print(model.score(x_test, y_test))
# fpr, tpr, thresholds = roc_curve(y, y_pred)

# print(grid)
# roc_auc = auc(fpr, tpr)
	
# print(roc_auc)
# print(std_auc)
classifiers = {"Logistic Regression": LogisticRegression(),
               "Naive Bayes" : GaussianNB(),
               "SVC": svm.SVC(C=1000, gamma=.001), 
               "Decision Tree": tree.DecisionTreeClassifier(),
               "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=100),
               "Linear Regression" : LinearRegression(),
               "Gradient Boosting Classifier" : GradientBoostingClassifier(),
               }
for name, clf in zip(classifiers.keys(), classifiers.values()):
	mean_auroc, std_auroc = cross_validation(clf, X, y)
	print(mean_auroc)
	# print("The mean of {} : {:.04f}".format(name, mean_auroc))
	# print("The Standard Deviation of {} : {:.04f}".format(name, std_auroc))


 #techinique used for hyperparameter section

#  #test on SVM
# c = [1, 10, 100, 1000]
# gamma = [0.001, 0.0001]
# for g in gamma:
# 	for index in c:
# 		svc = svm.SVC(C=index, gamma=g)
# 		mean_auroc, std_auroc = cross_validation(svc, X, y)
# 		print("c = {}, gamma = {},  The mean = {:.04f}".format(index, g,  mean))
# 		# print("The Standard Deviation = {:.04f}".format( std))

# max_depths = np.linspace(1, 32, 32, endpoint=True)
# n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
# print("hyperparameter on max_depths")
# for depth in max_depths:
# 	randomForest = RandomForestClassifier(max_depth=depth, n_estimators=10)
# 	mean_auroc, std_auroc = cross_validation(randomForest, X, y)
# 	print("max_depth = {}, The mean = {:.04f}".format(depth,  mean))
# print("hyperparameter on n_estimators")
# for n in n_estimators:
# 	randomForest = RandomForestClassifier(n_estimators=n)
# 	mean_auroc, std_auroc = cross_validation(randomForest, X, y)
# 	print("n_estimators = {}, The mean = {:.04f}".format(n,  mean))





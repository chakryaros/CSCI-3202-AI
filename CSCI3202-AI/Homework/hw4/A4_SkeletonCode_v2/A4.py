import numpy as np
import utils

import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt



#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData(file):
	
	label = "label"
	feature = ["r00","r01","r02","r03","r04","r10","r11","r12","r13","r14","r20",
			"r21","r22","r23","r24","r30","r31","r32","r33","r34","r40","r41","r42","r43","r44"]

	df = pd.read_csv(file)

	x, y = df[feature], df[label]

	return x, y

def distort_input(instance, percent_distortion):

    #percent distortion should be a float from 0-1
    #Should return a distorted version of the instance, relative to distoriton Rate
    distorted_instance = []
    pattern = list(instance)
    for i in pattern:
    	if random.random() < percent_distortion:
    		if i == 0 :
    			distorted_instance.append(1)
    		else:
    			distorted_instance.append(0)
    	else:
    		distorted_instance.append(i)

    return distorted_instance




class HopfieldNetwork:
	def __init__(self, size):
		self.h = np.zeros([size,size])

	def addSinglePattern(self, p):
		#Update the hopfield matrix using the passed pattern
		# print("TODO")
		weight = np.zeros([len(p), len(p)])
		for i in range(len(p)):
			for j in range(len(p)):
				if i == j:
					weight[i, j] = 0
				else:
					if p[i]==p[j]:
						weight[i, j] = 1
					else :
						weight[i, j] = -1
					weight[j, i] = weight[i, j]
		
		self.h = weight
		return self.h

		

	def fit(self, patterns):
		# for each pattern
		# Use your addSinglePattern function to learn the final h matrix
		# print("TODO")
	
		for p in patterns:
			self.h += self.addSinglePattern(p)
		
		return self.h


	def retrieve(self, inputPattern):
		#Use your trained hopfield network to retrieve and return a pattern based on the
		#input pattern.
		#HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
		#has generally better convergence properties than synchronous updating.
		updateOrder = np.arange(25)

		


		pattern = np.array(inputPattern)
		result = pattern.copy()
		step = len(inputPattern)*len(inputPattern)
		for _ in range(step):
			np.random.shuffle(updateOrder)

			for i in updateOrder:
				pattern[i] = 1 if np.dot(self.h[i, :], np.transpose(pattern)) >= 0 else 0
				
			if np.array_equal(result ,pattern):
				
				return result
			else:
				result = pattern

		return None


	def classify(self, inputPattern):
		#Classify should consider the input and classify as either, five or two
		#You will call your retrieve function passing the input
		#Compare the returned pattern to the 'perfect' instances
		#return a string classification 'five', 'two' or 'unknown'
		
		pattern = self.retrieve(inputPattern)

		if list(pattern) == list(five) :
			return "five"
		elif list(pattern) == list(two) :
			return "two"
		else:
			return "unknown"

#using distorted rate for MLP
def distortInput_MLP(x_test, y_test):

	clf = MLPClassifier()
	x_train = patterns
	y_train = ["five", "two"]
	clf.fit(x_train, y_train)
	distorted_rate = np.arange(0.0, 0.5, 0.01)
	acc_MPL = []
	
	x_distorted = x_test
	
	for rate in distorted_rate:
		for i in range(len(x_test)):
			x_distorted.iloc[i] = distort_input(x_distorted.iloc[i], rate)
	
		y_dist = clf.predict(x_distorted)

		acc_MPL.append(accuracy_score(y_test, y_dist))
	# print("{}, acc = {}".format(rate, accuracy_score(y_test, y_dist)))
	return acc_MPL

#using distorted rate for hopfield network
def distortInput_Hopfield(x_test, y_test, hopfieldNet):

	distorted_rate = np.arange(0.0, 0.5, 0.01)
	
	acc = []
	x_distorted = x_test

	for rate in distorted_rate:
		y_pred = []
		for i in range(len(x_test)):
			x_distorted.iloc[i] = distort_input(x_distorted.iloc[i], rate)

	
			y_pred.append(hopfieldNet.classify(x_distorted.iloc[i]))

		acc.append(accuracy_score(y_test, y_pred))
	
	return acc

if __name__ == "__main__":
	hopfieldNet = HopfieldNetwork(25)
	# utils.visualize(five)
	# utils.visualize(two)
	
	#part 2 hopfield network
	hopfieldNet.fit(patterns)
	x_test, y_test = loadGeneratedData("chro0474_TrainingData.csv")
	# print(x_test)
	y_pred = []
	for i in range(len(x_test)):
		y_pred.append(hopfieldNet.classify(x_test.iloc[i]))
	print("y_pred using hopfield Network : ", y_pred)
	acc_hopfield = accuracy_score(y_test, y_pred)
	print("Accuracy_score using Hopfield Network : ", acc_hopfield*100)

	


	# Part 3 train a MLP
	clf = MLPClassifier()
	x_train = patterns
	y_train = ["five", "two"]
	clf.fit(x_train, y_train)
	y_pred_MLP = clf.predict(x_test)
	print("y_pred using MLP: ",y_pred_MLP)
	
	acc_MPL = accuracy_score(y_test, y_pred_MLP)
	print("Accuracy_score using MLPClassifier : ", acc_MPL*100)
	
	# part 4 distorted input
	distorted_rate = np.arange(0.0, 0.5, 0.01)
	acc_MPL = distortInput_MLP(x_test, y_test)
	acc_hopfield = distortInput_Hopfield(x_test, y_test, hopfieldNet)


	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
	ax.plot(distorted_rate, acc_MPL, color='red', label="MLPClassifier")
	ax.plot(distorted_rate, acc_hopfield, color='green', label="Hopfield")


	#part 5
	x, y = loadGeneratedData("NewInput.csv")
	xTrain_c = x_test.append(x, ignore_index=True)
	yTrain_c = y_test.append(y, ignore_index=True)

	acc_array = []
	acc_hidden = []
	x_distorted = xTrain_c.copy()
	
	hiddenSize = [(50, 30, 10)]
	# colors = ["red", "green", "blue"]
	# hiddenSize=[30, 40, 50]
	x_distorted = xTrain_c.copy()
	for idx , h in enumerate(hiddenSize):
		mlp = MLPClassifier(hidden_layer_sizes= h)
		mlp.fit(xTrain_c , yTrain_c)
		acc_array = []
		for rate in distorted_rate:
			for i in range(len(xTrain_c)):
				x_distorted.iloc[i] = distort_input(x_distorted.iloc[i], rate)
			y_pred = mlp.predict(x_distorted)
			acc_array.append(accuracy_score(yTrain_c, y_pred))
		
		ax.plot(distorted_rate, acc_array, color ="blue", label="hiddenSize")
		     
	ax.set_xlabel("Distored Rate", fontsize=16)
	ax.set_ylabel("Accuracy Score.", fontsize=16)
	ax.legend()
	plt.show()

	
	
	




















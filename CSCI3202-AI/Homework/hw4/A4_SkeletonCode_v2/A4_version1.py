import numpy as np
import csv
import utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Data Preparation
five_data =[(0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0,"five"),
			(0,1,1,1,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,0,0,0,0,"five"),
			(0,1,1,1,1, 0,1,0,0,0, 0,1,1,1,1, 0,0,0,0,1, 0,1,1,1,1,"five"),
			(0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0,"five")]

two_data = [(0,1,1,1,0, 0,1,0,1,0, 0,0,1,0,0, 0,1,0,0,0, 0,1,1,1,0,"two"),
			(0,1,1,1,0, 0,1,0,1,0, 0,0,1,1,0, 0,1,1,0,0, 0,1,1,1,0,"two"),
			(0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0,"two"),
			(1,1,1,0,0, 0,0,1,0,0, 1,1,1,0,0, 1,0,0,0,0, 1,1,1,0,0,"two")]




#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData():
	print("TODO")
	input = open('chro0474_TrainingData.csv', 'r')
	
	

def distort_input(instance, percent_distortion):

    #percent distortion should be a float from 0-1
    #Should return a distorted version of the instance, relative to distoriton Rate
    print("TODO")
    utils.raiseNotDefined()


class HopfieldNetwork:
	def __init__(self, size):
		self.h = np.zeros([size,size])

	def addSinglePattern(self, p):
		#Update the hopfield matrix using the passed pattern
		print("TODO")
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
		# print(weight)
		return weight

		

	def fit(self, patterns):
		# for each pattern
		# Use your addSinglePattern function to learn the final h matrix
		print("TODO")
	
		for p in patterns:
			self.h += self.addSinglePattern(p)
		print(self.h)
		return self.h


	def retrieve(input):
		#Use your trained hopfield network to retrieve and return a pattern based on the
		#input pattern.
		#HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
		#has generally better convergence properties than synchronous updating.
		# pattern = inputPattern

		# num_iterate = len(inputPattern)*len(inputPattern)
		# for _ in range(num_iterate) :
		# 	np.random.shuffle(v_in)

		# 	step = 0
		# 	for index in v_in:
		# 		if np.dot(self.h[index], pattern) < 0:
		# 			if pattern[index] == 1:
		# 				step += 1
		# 			pattern[index] = 0
		# 		else :
		# 			if pattern[index] == 0:
		# 				step += 1
		# 			pattern[index] = 1

		# 	if step == 0:
		# 		return pattern
		# return None



		print("TODO")
		utils.raiseNotDefined()

	def classify(inputPattern):
		#Classify should consider the input and classify as either, five or two
		#You will call your retrieve function passing the input
		#Compare the returned pattern to the 'perfect' instances
		#return a string classification 'five', 'two' or 'unknown'

		print("TODO")
		utils.raiseNotDefined()


#write training data into file
def writeTofile():
	trainFile = open('chro0474_TrainingData.csv', 'w', newline='')
	file = csv.writer(trainFile)
	header = [('r00','r01','r02','r03','r04','r10','r11','r12','r13','r14',
			  'r20','r21','r22','r23','r24','r30','r31','r32','r33','r34',
			  'r40','r41','r42','r43','r44','label')]
	for h in header:
		file.writerow(h)
	for idx, two in enumerate(two_data):
		file.writerow(two)
		file.writerow(five_data[idx])

	trainFile.close()

#Part 3 train a MLP
def trainMLP():
	clf = MLPClassifier()
	x_train = patterns
	y_train = ['five', 'two']
	clf.fit(x_train, y_train)
	y_pred = clf.prdict(x_test)
	y_pred = abs(y_pred.round())
	acc = accuracy_score(y_test, y_pred)

if __name__ == "__main__":
	hopfieldNet = HopfieldNetwork(25)
	writeTofile()
	# utils.visualize(five)
	# utils.visualize(two)
	hopfieldNet.addSinglePattern(five)
	

	hopfieldNet.fit(patterns)


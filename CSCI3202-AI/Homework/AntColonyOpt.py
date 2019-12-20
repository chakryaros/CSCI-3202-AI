import numpy as np
from numpy import inf

d = np.array([[0, 10, 12 ,11, 14],
			[10,0,13,15,0],
			[12,13,0,9,14],
			[11,15,9,0,16],
			[14,8,14,16,0]])

iteration = 100
n_ants = 5
n_citi = 5


m = n_ants
n = n_citi
e = .5
alpha = 1
beta = 2 #visibiltiy factor


#calculate the visibility of next city visibility(i, j) = 1/d(i,j)

visibility = 1/d
visibility[visibility==inf] = 0

#intializing pherome present at the path to the cities

pherome = .1*np.ones((m,n))

#initializing the rute of the ants with size rute(N_ant, n_city+1)
#note add one because we want to come back to source

rute = np.ones((m, n+1))
print(rute)
for ite in range(iteration):

	#initial starting and ending position of every ant 1 and city 1
	rute[:,0] = 1

	for i in range(m):
		temp_visibility = np.array(visibility)
		# print(temp_visibility)

		for j in range(n):
			combine_feature = np.zeros(5)
			cum_prob = np.zeros(5)

			#current city of ant
			cur_location = int(rute[i,j]-1)

			#making visibility of current city as zero
			temp_visibility[:, cur_location] =0

			p_feature = np.power(pherome[cur_location,:], beta)
			v_feature = np.power(temp_visibility[cur_location,:], alpha)

			#adding axis to make a size[5,1]
			p_feature = p_feature[:, np.newaxis]
			v_feature = v_feature[:, np.newaxis]

			combine_feature = np.multiply(p_feature, v_feature)

			total = np.sum((combine_feature))


			












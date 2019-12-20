import robby
import numpy as np
from utils import *
import random, time, os
import datetime

import matplotlib.pyplot as plt


# PAUSE = 0.08
CYCLE_LIMIT = 3
FAST_STEPS = 100

POSSIBLE_ACTIONS = ["MoveNorth", "MoveSouth", "MoveEast", "MoveWest", "StayPut", "PickUpCan", "MoveRandom"]
rw = robby.World(10, 10)
rw.graphicsOff()

def sortByFitness(genomes):
    tuples = [(fitness(g), g) for g in genomes]
    # print("tuples", tuples)
    tuples.sort()
    # sortedFitnessValues = [f for (f, g) in tuples]
    sortedGenomes = [g for (f, g) in tuples]
    # return sortedGenomes, sortedFitnessValues
    return sortedGenomes


def randomGenome(length):
    """
    :param length:
    :return: string, random integers between 0 and 6 inclusive
    """

    """Your Code Here"""
    numStr = ""
    for i in range(length):
        numStr += str(random.randint(0, 6))
    return numStr

# print(randomGenome(6))


def makePopulation(size, length):
    """
    :param size - of population:
    :param length - of genome
    :return: list of length size containing genomes of length length
    """
    


    """Your Code Here"""
    listPop = []
    for _ in range(size):
        listPop.append(randomGenome(length))
    
    return listPop




def fitness(genome, steps=200, init=0.50):
    """

    :param genome: to test
    :param steps: number of steps in the cleaning session
    :param init: amount of cans
    :return:
    """
    

    
    if type(genome) is not str or len(genome) != 243:
        raise Exception("strategy is not a string of length 243")
    for char in genome:
        if char not in "0123456":
            raise Exception("strategy contains a bad character: '%s'" % char)
    if type(steps) is not int or steps < 1:
        raise Exception("steps must be an integer > 0")
    if type(init) is str:
        # init is a config file
        rw.load(init)
    elif type(init) in [int, float] and 0 <= init <= 1:
        # init is a can density
        rw.goto(0, 0)
        rw.distributeCans(init)
    else:
        raise Exception("invalid initial configuration")

    total_reward = 0
    geneome_list = list(genome)
    
    for session in range(25):
        row = random.randint(0,9)
        col = random.randint(0,9)
        rw.goto(row,col)
        # rw.goto(0,0)
        rw.distributeCans(init)
        current_fit = 0
        for i in range(steps):
            p = rw.getPerceptCode()
            action = POSSIBLE_ACTIONS[int(geneome_list[p])]
            current_fit += rw.performAction(action)
        # print("current_fit", current_fit)

        total_reward += current_fit

    return total_reward/25


   


def evaluateFitness(population):
    """
    :param population:
    :return: a pair of values: the average fitness of the population as a whole and the fitness of the best individual
    in the population.
    """
    fit_list =[]
    best_reward = -1000.00
    index = 0
    for individual in population:
        fit_list.append(fitness(individual))
    for reward in fit_list:
        if  best_reward < reward:
            best_reward = float(reward)
            index +=1
    return [Average(fit_list), best_reward, population[index]]




def crossover(genome1, genome2):
    """
    :param genome1:
    :param genome2:
    :return: two new genomes produced by crossing over the given genomes at a random crossover point.
    """

    i  = np.random.randint(1, len(genome1)-1)
    newGen1 = genome1[:i] + genome2[i:] 
    newGen2 = genome2[:i] + genome1[i:]
    result = [''.join(newGen1), ''.join(newGen2)]

    return result

def mutate(genome, mutationRate):
    """
    :param genome:
    :param mutationRate:
    :return: a new mutated version of the given genome.
    """
    genList = list(genome)
    # print("mutate", genList)
    
    list_ = ['0','1','2','3','4', '5','6']
    
    child = []
    for i in genList:
        if np.random.random() < mutationRate:

            child.append(np.random.choice(['0','1','2','3','4', '5','6']))
        else:
            child.append(i)

    return ''.join(child)
    

    # p = rw.getPerceptCode()
    # action = POSSIBLE_ACTIONS[int(genome[p])]

    # fit += rw.performAction(action)
    

def selectPair(population):
    """

    :param population:
    :return: two genomes from the given population using fitness-proportionate selection.
    This function should use RankSelection,
    """

    gen1 = weightedChoice(population, [i for i in range(1, len(population)+1)])
    gen2 = weightedChoice(population, [i for i in range(1,len(population)+1)])

    return [gen1, gen2]
    
    
# selectPair(population)
def runGA(populationSize, crossoverRate, mutationRate, logFile=""):
    """

    :param populationSize: :param crossoverRate: :param mutationRate: :param logFile: :return: xt file in which to
    store the data generated by the GA, for plotting purposes. When the GA terminates, this function should return
    the generation at which the string of all ones was found.is the main GA program, which takes the population size,
    crossover rate (pc), and mutation rate (pm) as parameters. The optional logFile parameter is a string specifying
    the name of a te
    """

    population = makePopulation(populationSize, 243)
    index = 1

    file = open(logFile, "w")

    
    

    
    while index <= 300:
        # print("Run index: {}".format(index))
       

        sortedGenomes = sortByFitness(population)
        # sortedGenomes, sortedFitnessValues = sortByFitness(population)
        newPopulation = []
        
        for i in range(len(sortedGenomes)):
            sortedGenomes[i] = mutate(sortedGenomes[i], mutationRate)
        
        while len(newPopulation) != populationSize:

            '''select the pair of geneome from population'''
            # now = datetime.datetime.now()
            genome_select = selectPair(sortedGenomes)
            # print("microseconds overlap , selectPair", (datetime.datetime.now()-now).microseconds)
            

            '''rate dependent crossover of selected genome'''
            genome_cross = []
    
            if random.random() < crossoverRate:
               
                # now1 = datetime.datetime.now()
                genome_cross = crossover(genome_select[0], genome_select[1])
                # print("microseconds overlap , crossover", (datetime.datetime.now()-now1).microseconds)
    
            else:
                genome_cross = genome_select

    
            '''mutate crossoeverd genome'''
            

           
            '''add new genomee to new population after doing crossover and mutate'''
            newPopulation.append(genome_cross[0])
            newPopulation.append(genome_cross[1])
            # newPopulation.append(newGen1)
            # newPopulation.append(newGen2)
        population = newPopulation
            
        best_fit = evaluateFitness(newPopulation)
       

        # print("Generation  {}: average fittness {:.02f}, best fitness {:.02f}, best strategy {}".format(index, best_fit[0], best_fit[1], f))
        # print("Generation  {}: average fittness {:.02f}, best fitness {:.02f}".format(index, best_fit[0], best_fit[1]))
        if index%10==0:

            # f = fitness(rw.strategyM)
            file.write("{} {:.02f} {:.02f} {}\n".format(index, best_fit[0], best_fit[1], str(best_fit[2])))
            
            print("Generation  {}: average fittness {:.02f}, best fitness {:.02f}".format(index, best_fit[0], best_fit[1]))
            print("best strategyM", fitness(rw.strategyM))
       
            # rw.graphicsOn()
            # rw.demo(best_fit[2])
            
            # rw.graphicsOff()

        '''write best fit to the file'''

        
        index += 1
            
    file.close()

    # return None
            
            
    



def test_FitnessFunction():
    f = fitness(rw.strategyM)
    print("Fitness for StrategyM : {0}".format(f))



test_FitnessFunction()

runGA(100, 1, 0.005, "GAoutput.txt")


file = open('GAoutput.txt', "r")
beststrategyFile = open('bestStrategy.txt', "w")


generation = []
average = []
for line in file:
    gen, avg, fitness, b_t = line.split()
   
    beststrategyFile.write("{}\n".format(b_t))
           
    gen = float(gen)
    avg = float(avg)
    generation.append(gen)
    average.append(avg)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
ax.plot(generation, average, color='red')

        
ax.set_xlabel("generation", fontsize=16)
ax.set_ylabel("Average of fitness.", fontsize=16)
ax.set_title("300 generation with PopulationSize = 300", fontsize=16)
plt.show()
file.close()
beststrategyFile.close()



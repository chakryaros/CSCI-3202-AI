# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Name : Chakrya Ros

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class GraphSearch:
    ''' Create graph to connect node by edgs. Each edge has a length.
    The Grape will be direct and undirect.

    '''
    #create constructor
    def __init__(self, Graphdict=None, directedGraph=True):

        self.__graph_dict = Graphdict
        self.directed = directedGraph

        if not directedGraph:
            self.create_undirected()
    def add_vertice(self, vertex):
        '''
        connect the graph with distance from A to B
        '''
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex]



def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


class Node:

    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.stack = util.Stack()

    def expendNode(self, state, action, cost):

        newNode = Node(state, action, cost)

    def getState():
        return self.state

    def getAction():

        return self.action


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    #LIFO, Stack
    nodeStacK = util.Stack()
    actions_ = []
    visited = []
    actionsList = []
    '''add state, action, visited to stack'''
    nodeStacK.push((problem.getStartState(), actions_, visited) )
    
    while not nodeStacK.isEmpty():
        node, actions_ , visited = nodeStacK.pop()
        # print("node", node)
        # print("acation",actions_)
        # print("visited", visited)
        if problem.isGoalState(node):
            return actionsList

        ''' extend the state'''
        for (nextState, action, stepsCost) in problem.getSuccessors(node):
            if not nextState in visited:

                updateAction = actions_ + [action]

                updateVisited = visited + [node]

                ''' add next state, update action to node that visited to stack'''
                nodeStacK.push((nextState, updateAction , updateVisited))
                actionsList = actions_ + [action]
                         
    return None
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # "*** YOUR CODE HERE ***"

    nodeQueue = util.Queue()
    actions = []
    visited = []

    '''add it to queue'''
    nodeQueue.push((problem.getStartState(), actions))

    while not nodeQueue.isEmpty():
        node, actions = nodeQueue.pop()
        # print(node)
        # print(actions_)
        

        if not node in visited:
            ''' if the node not visit, add it to list'''
            visited.append(node)

            if problem.isGoalState(node):
                ''' if the state is goal, return action'''
                return actions

            ''' extend the state'''
            for (nextState, nextAction, stepCost) in problem.getSuccessors(node):
                '''update the action'''
                updateAction = actions + [nextAction]
                
                ''' add next state, update action to queue'''
                nodeQueue.push((nextState, updateAction))
    return None



    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # "*** YOUR CODE HERE ***"
    nodeQueue = util.PriorityQueue()
    actions_ = []
    cost = 0
    visited = []

    '''add it to PriorityQueue'''
    nodeQueue.push((problem.getStartState(), actions_, cost), cost)

    while not nodeQueue.isEmpty():
        node, actions_ , cost= nodeQueue.pop()
        if not node in visited:
            visited.append(node)

            if problem.isGoalState(node):
                return actions_

            ''' extend the state'''
            for (nextState, action, stepCost) in problem.getSuccessors(node):

                #update action
                updateAction = actions_ + [action]

                #update cost
                updateCost = cost + stepCost

                nodeQueue.push((nextState, updateAction, updateCost), updateCost)

    return None
    
    
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    node = util.PriorityQueue()
    #get the initial state
    init_state = problem.getStartState()
    Heuristic = heuristic(init_state, problem)
    actions = []
    cost = 0
    visited = []
    
    #add it to priorityQueue
    node.push((init_state, actions, cost), Heuristic)
    while not node.isEmpty():
        state, actions, cost = node.pop()
        if not state in visited:
            visited.append(state)

            if problem.isGoalState(state):
                return actions

            ''' extend the state'''
            for (nextState, action, stepCost) in problem.getSuccessors(state):

                #update action
                updateAction = actions + [action]
                
                #update cost
                gn = cost + stepCost

                #update heuristicCost

                hn = heuristic(nextState, problem)
                # print("hn", hn)

                #calculate the cost from state and cost from goal
                fn = gn + hn
                # print("fn", fn)
                node.push((nextState, updateAction, gn), fn)

    return None
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

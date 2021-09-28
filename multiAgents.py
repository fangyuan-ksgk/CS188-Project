# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()
        

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostPositions = [ghoststate.getPosition() for ghoststate in newGhostStates]
        foodPositions = newFood.asList()
        
        food_score = 0
        if foodPositions:
            closestFooddist = min([abs(newPos[0]-foodpos[0])+abs(newPos[1]-foodpos[1]) for foodpos in foodPositions])
            food_score = -closestFooddist
            
        closestGhostdist = min([abs(newPos[0]-ghostPosition[0]) + abs(newPos[1]-ghostPosition[1]) for ghostPosition in ghostPositions])
        if closestGhostdist == 1:
            ghost_score = -20
        else:
            ghost_score = closestGhostdist
        
        return childGameState.getScore() + 0.6*food_score + 0.4*ghost_score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num = gameState.getNumAgents()
        terminate_depth = [self.depth]*num
        def minimax(state, agent, action, searching_depth):
            next_state = state.getNextState(agent, action)
            searching_depth[agent] += 1
            if next_state.isWin() or next_state.isLose() or searching_depth==terminate_depth:
                return self.evaluationFunction(next_state)
            
            searched_layer = min(searching_depth)
            for next_agent in range(num):
                if searching_depth[next_agent]==searched_layer:
                    val = []
                    for act in next_state.getLegalActions(next_agent):
                        next_sd = [searching_depth[i] for i in range(num)]
                        val.append(minimax(next_state, next_agent, act, next_sd))
                    
                    if next_agent!=0:
                        return min(val)
                    else:
                        return max(val)
        
        actscore = {}
        for action in gameState.getLegalActions(0):
            actscore[action] = minimax(gameState, 0, action, [0]*num)
        return max(actscore, key=actscore.get)
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def pruneminmax(state, searching_depth, A, B):
            
            if state.isWin() or state.isLose() or searching_depth==terminate_depth:
                return self.evaluationFunction(state)
            
            searched_layer = min(searching_depth)
            for agent in range(num):
                if searching_depth[agent]==searched_layer:
                    store = [-float('inf'), float('inf')]
                    for act in state.getLegalActions(agent):
                        next_state = state.getNextState(agent,act)
                        next_sd = [searching_depth[i] for i in range(num)]
                        next_sd[agent] += 1
                        val = pruneminmax(next_state, next_sd, A, B)
                        if agent==0:
                            store[0] = max(val, store[0])
                            A = max(A, store[0])
                            if store[0]>B:
                                return store[0]
                        if agent!=0:
                            store[1] = min(val, store[1])
                            B = min(B, store[1])
                            if store[1]<A:
                                return store[1]
                    if agent==0:
                        return store[0]
                    else:
                        return store[1]
           
        inf = float('inf')
        num = gameState.getNumAgents()
        terminate_depth = [self.depth]*num
        v = -inf
        act = None
        A = -inf
        B = inf
        for action in gameState.getLegalActions(0):
            state = gameState.getNextState(0,action)
            tmp = pruneminmax(state, [1]+[0]*(num-1), A, B)
            if tmp>v:
                act = action
                v = tmp
            A = v
        return act
        
        
    def internetversion(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        GhostIndex = [i for i in range(1, gameState.getNumAgents())]
        inf = float('inf')

        def term(state, d):
            return state.isWin() or state.isLose() or d == self.depth
        
        
        def min_value(state, d, ghost, A, B):
            if term(state, d):
                return self.evaluationFunction(state)
            v = inf
            for action in state.getLegalActions(ghost):
                if ghost == GhostIndex[-1]:
                    v = min(v, max_value(state.getNextState(ghost, action), d+1, A, B))
                else:
                    v = min(v, min_value(state.getNextState(ghost, action), d, ghost+1, A, B))
                if v<A:
                    return v
                B = min(v,B)
            return v
        
        def max_value(state, d, A, B):
            if term(state, d):
                return self.evaluationFunction(state)
            v = -inf
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.getNextState(0,action), d, GhostIndex[0], A, B))
                if v>B:
                    return v
                A = max(v,A)
            return v
            
            
        A = -inf
        B = inf
        act = None
        v = -inf
        for action in gameState.getLegalActions(0):
            tmp = min_value(gameState.getNextState(0,action), 0, GhostIndex[0], A, B)
            if tmp>v:
                v = tmp
                act = action
            A = max(A, tmp)
        
        return act
    
                
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        num = gameState.getNumAgents()
        terminate_depth = [self.depth]*num
        
        def expectimax(state, agent, action, searching_depth):
            next_state = state.getNextState(agent, action)
            searching_depth[agent] += 1
            if next_state.isWin() or next_state.isLose() or searching_depth==terminate_depth:
                return self.evaluationFunction(next_state)
            
            searched_layer = min(searching_depth)
            for next_agent in range(num):
                if searching_depth[next_agent]==searched_layer:
                    val = []
                    for act in next_state.getLegalActions(next_agent):
                        next_sd = [searching_depth[i] for i in range(num)]
                        val.append(expectimax(next_state, next_agent, act, next_sd))
                    
                    if next_agent!=0:
                        return sum(val)/len(val)
                    else:
                        return max(val)
        
        d = {}
        for action in gameState.getLegalActions(0):
            d[action] = expectimax(gameState, 0, action, [0]*num)
        act = max(d, key=d.get)
        return act
        
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    scared = min(newScaredTimes) > 0
    
    ghostPositions = [ghoststate.getPosition() for ghoststate in newGhostStates]
    foodPositions = newFood.asList()
    capPositions = currentGameState.getCapsules()
        
    food_score = 0
    if foodPositions:
        closestFooddist = min([abs(newPos[0]-foodpos[0])+abs(newPos[1]-foodpos[1]) for foodpos in foodPositions])
        food_score = -closestFooddist

    closestGhostdist = min([abs(newPos[0]-ghostp[0])+abs(newPos[1]-ghostp[1]) for ghostp in ghostPositions])
    if closestGhostdist == 1:
        ghost_score = -20
    else:
        ghost_score = closestGhostdist
        
    
    capscore = 0
    if capPositions:
        cloestCapdist =  min([abs(newPos[0]-cappos[0])+abs(newPos[1]-cappos[1]) for cappos in capPositions])
        if cloestCapdist == 0:
            capscore = 6
        
    if not scared:
        return currentGameState.getScore() + 0.6*food_score + 0.4*ghost_score + capscore
    else:
        return currentGameState.getScore() + 0.6*food_score
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

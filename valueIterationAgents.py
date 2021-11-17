# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # One Iteration: In-place replacement on self.values
        def vval_iter(s):
            if not self.mdp.isTerminal(s):
                maxv = -99999
                for a in self.mdp.getPossibleActions(s):
                    temp = 0
                    for ns,prob in self.mdp.getTransitionStatesAndProbs(s,a):
                        temp += prob * (self.mdp.getReward(s,a,ns) + self.discount*self.values[ns])
                    maxv = max(temp,maxv)  
                tempvals[s] = maxv
            else:
                tempvals[s] = 0
        
        states = self.mdp.getStates()
        for i in range(self.iterations):
            tempvals = util.Counter()
            # update on the tempvals (off-line)
            for s in states:
                vval_iter(s)
            # update on the values
            for s in states:
                self.values[s] = tempvals[s]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeValue(self,s):
        if not self.mdp.isTerminal(s):
            maxv = -99999
            for a in self.mdp.getPossibleActions(s):
                temp = 0
                for ns,prob in self.mdp.getTransitionStatesAndProbs(s,a):
                    temp += prob * (self.mdp.getReward(s,a,ns) + self.discount*self.values[ns])
                maxv = max(temp,maxv)  
            return maxv
        else:
            return 0

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        tmp = 0
        for ns,prob in self.mdp.getTransitionStatesAndProbs(state,action):
            reward = self.mdp.getReward(state,action,ns)
            tmp += prob * (self.discount * self.values[ns] + reward)
        return tmp
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        
        maxtemp = -99999
        for a in self.mdp.getPossibleActions(state):
            temp = self.computeQValueFromValues(state, a)
            if temp>maxtemp:
                action = a
                maxtemp = temp
        return action
        #util.raiseNotDefined()

    def getNextValue(self, state):
        return self.computeValue(state)
    
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        n = len(states)
        for i in range(self.iterations):
            s = states[i%n]
            if not self.mdp.isTerminal(s):
                # perform value update
                act = self.getAction(s)
                self.values[s] = self.getQValue(s,act)
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # predecessors dictionary: record set of predecessors of any state
        predecessors = collections.defaultdict(set)
        states = self.mdp.getStates()
        for s in states:
            for a in self.mdp.getPossibleActions(s):
                for ns,prob in self.mdp.getTransitionStatesAndProbs(s,a):
                    if prob>0:
                        # we can reach ns from s with positive prob through action a
                        predecessors[ns].add(s)
                
        # initialize priority queue
        queue = util.PriorityQueue()
        
        # push all state into queue according to their priority
        for s in states:
            if not self.mdp.isTerminal(s):
                diff = abs(self.values[s] - self.getNextValue(s))
                queue.push(s,-diff)
                
        # iteration
        for i in range(self.iterations):
            if queue.isEmpty():
                break
                
            s = queue.pop()
            if not self.mdp.isTerminal(s):
                # update v value of s
                self.values[s] = self.getNextValue(s)
            
            for p in predecessors[s]:
                act = self.getAction(p)
                highest_qval = self.getQValue(p,act)
                diff = abs(self.values[p] - highest_qval)
                if diff>self.theta:
                    queue.update(p,-diff)
                
                
                
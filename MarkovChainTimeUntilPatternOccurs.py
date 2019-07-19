# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:37:29 2019

@author: pbraz
"""

import numpy as np

#Defining the Markov chain transition matrix; needs to be a n by n matrix (array) with all positive entries whose rows sum to one#
transitionMatrix = np.array([[.3, .5, .2], [.5, .3, .2], [.5, .2, .3]]);
transitionMatrix = np.transpose(transitionMatrix)


#Obtaining the number of states in the Markov chain#
numStates = len(transitionMatrix[0]);


#Solving for long-run proportions of the given transition matrix. I.E. the unique vector such that P \pi = \pi, or more simply,# 
#(P - Id_{n}) \pi = 0 where Id_{n} is the n by n identity matrix and the \pi vector is all positive numbers whoses entries sum to one#
longrunMatrix = np.array(transitionMatrix, copy = True)
identityMatrixNumStates = np.identity(numStates)

longrunMatrix = longrunMatrix - identityMatrixNumStates

#The system of linear equations, (P - Id_{n}) \pi = 0 has no nonzero solutions. To fix this we change one row (one equation) to 1 = \sum_{i \in \{0, ..., n-1\} \pi_i#
equationSwap = np.full(shape = numStates, fill_value = 1)
longrunMatrix[numStates-1] = equationSwap

#Defining the vector to be solved for, i.e. the b in Ax = b#
solutionVec = equationSwap - equationSwap
solutionVec[numStates-1] = 1

#Obtaining solution#
longrunVec = np.linalg.solve(longrunMatrix, solutionVec)

#On page 215 in Ross's "Introduction to Probability Models" 11th Edition we can find the motivation for the equation#
#\mu(i,j) = 1 + \sum_{k=\{0,...,n-1 \},k \neq j}   (P_{i,k})(\mu(k,j)) where \mu(i, j) is the mean time stating from state i to reach#
#state j in an irredicible finite Markov chain#

#The equation above is not easily simplified. Let P_{0,i} be the transition matrix with column i replaced with all zeroes and 1_{n} be a n-tuple#
#in which every entry is 1 and let \mu_{i} be a vector whose jth entry is \mu(j,i). Then we can write (Id_{n} - P_{0,i}) \mu_{i} = 1_{n}#

#Obtaining avg return time from state j to state j i.e. avg return time from j to j = 1/\pi_{j}. This can be used as a check since \mu(i,i) = 1/(\pi_{i})#
avgReturnTime = 1/longrunVec

#Defining 1_{n}#
oneVector = np.full(shape = numStates, fill_value = 1.0)

#Looping the solution of the linear equations P_{0,i} \mu_{i} = 1_{n} and combining them into a single matrix#
zeroVector = np.full(shape = numStates, fill_value = 0.0)
meanTransitionsBetweenStates = np.copy(identityMatrixNumStates)

for i in range(0, numStates):
    #Making a copy of transition matrix so the transition matrix isn't mutated#
    systemOfEquationsToStatei = np.array(transitionMatrix, copy = True)
    #Replcing ith row with all zeroes as per the linear equations above#
    systemOfEquationsToStatei[i] = zeroVector
    #Transposing matrix so the zeroes are all in the ith column. I made sure this was the "right" column i.e.# 
    #I didn't end up changing a row/column when compared to the system of equations#
    systemOfEquationsToStatei = np.transpose(systemOfEquationsToStatei)
    systemOfEquationsToStatei = (identityMatrixNumStates - systemOfEquationsToStatei)
    #Using a numpy built in to linearly solve for \mu_{i}#
    meanTransitionsToStatei = np.linalg.solve(systemOfEquationsToStatei, oneVector)
    #Adding \mu_{i} to a matrix to house all the \mu_{i,j} values#
    meanTransitionsBetweenStates[i] = meanTransitionsToStatei
    
#Fixing the orientation of the matrix so \mu_{i,j} = \mu(i,j) i.e the mean time from state i to state j value is
#located in the ith row and jth column#
meanTransitionsBetweenStates = np.transpose(meanTransitionsBetweenStates)

#Defining the pattern (or sequence of states) of interest#
pattern = ["0", "1", "0", "1", "0"]

#Let s_{1} be the size of of the largest cycle in the pattern and 
#i_1, ... i_{s_{1}} be the first s_{1} elements of the pattern. We wish to determine
#the value of E[A(i_1, ..., i_{s_{1}})], the mean time until the pattern occurs
#given that last observed states are i_1, ..., i_{s_{1}}. Ross's book tells us
#that this value is equal to 1/(\pi_{i_1} P_{i_1, i_2}* ... * P_{i_{s_{1}} -1, i_{s_{1}}}. 
#The reason for this is described in Ross's book. 

#Once E[A(i_1, ..., i_{s_{1}})] is determined we record it and find the value of s_{2}
#the size of the largest cycle in i_1, ..., i_{s_1}. We again wish to find the 
#value of E[A(i_1, ..., i_{s_{2}})], the mean time until the pattern occurs 
#given that last observed states are i_1, ..., i_{s_{2}}. This process is repeated
#until we the value of the largest cycle, s_{n}, is zero. Finally we record the
#value of \sum_{j=1}^{j=n} E[A(i_1, ..., i_{s_{j}}]#  
def meanTimeForPatternWithCycles(pattern, longrunVec, transitionMatrix, runningSum = 0):
    patternLength = len(pattern)
    cycleLength = 0
    startingStateOfPattern = int(pattern[0])
    meanTimeToPatternGivenCycle = longrunVec[startingStateOfPattern]
    for i in range(0, patternLength - 1):
        currentState = int(pattern[i])
        futureState = int(pattern[i+1])
        meanTimeToPatternGivenCycle = (meanTimeToPatternGivenCycle)*(transitionMatrix[futureState][currentState])
    try:
        meanTimeToPatternGivenCycle = 1/(meanTimeToPatternGivenCycle)
    except:
        print("The pattern is impossible")
        return None
        
    runningSum = runningSum + meanTimeToPatternGivenCycle
#boolean to make sure we check if the current position in the list is#
#is equal to the last item in the list. The boolean becomes false if and only if#
#there exist a cycle (the largest cycle)#
    stillChecking = True

    for i in range(0, patternLength-1):
    #Creating mutable variable keeping track of current position in list. 
    #Starting from the second to last element and moving down will ensure that 
    #the first cycle found is the largest cycle#
        current = patternLength - 2 - i
    
    #Creating a variable. This variable will keep track how many letters in 
    #the pattern are the same starting from the "current" variable and compared
    #to the last element#
        currentPatternLength = 0

        while pattern[current] == pattern[-1 - (currentPatternLength)] and stillChecking:
        #Setting up the check to see if the letter before the currrent letter 
        #is the same as the current letters to the left of the end of the word#
            current = current - 1
            currentPatternLength = currentPatternLength + 1
        #Checking to see if we found the largest cycle#
            if current == -1:
            #Recording length of largest cycle#
                cycleLength = currentPatternLength
            #Breaking out of both loops#
                i = patternLength - 1
                stillChecking = False
    
    if cycleLength == 0:
        return [pattern, runningSum]
    else:
        cyclePattern = pattern[:cycleLength]
        return meanTimeForPatternWithCycles(cyclePattern, longrunVec, transitionMatrix, runningSum)

def meanTimeForPattern(pattern, longrunVec, transitionMatrix, meanTransitionsBetweenStates, startingState):
    [cyclelessPattern, runningSum] = meanTimeForPatternWithCycles(pattern, longrunVec, transitionMatrix)
    patternStart = int(cyclelessPattern[0]);
    patternEnd = int(cyclelessPattern[-1])
    startingState = int(startingState);
    return meanTransitionsBetweenStates[startingState, patternStart] + runningSum - meanTransitionsBetweenStates[patternEnd, patternStart]


print(meanTimeForPattern(pattern, longrunVec, transitionMatrix, meanTransitionsBetweenStates, startingState = "0"))
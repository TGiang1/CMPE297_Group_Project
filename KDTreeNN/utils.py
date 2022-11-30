import numpy as np

# generate numQueries of arrays of size equal to numDimensions. Provide upperLimit for max value of randomly generated values
def generateQueries(numQueries, upperLimit=5, numDimensions=10):
    queriesArray = np.random.uniform(0, upperLimit, size=(numQueries, numDimensions))
    queriesArray = np.round(queriesArray, 2)
    
    return queriesArray

def saveQueries(fileName, queriesArray):
    np.savetxt(fileName, queriesArray, fmt='%1.3f')

def importQueries(fileName):
    queriesArray = np.loadtxt(fileName)
    return queriesArray
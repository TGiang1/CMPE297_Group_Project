import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import utils

# 10 dimensions hardcoded
def calcEuclideanDistance(x, y):
    return ((x[0] - y[0])**2 + 
            (x[1] - y[1])**2 + 
            (x[2] - y[2])**2 + 
            (x[3] - y[3])**2 + 
            (x[4] - y[4])**2 + 
            (x[5] - y[5])**2 + 
            (x[6] - y[6])**2 + 
            (x[7] - y[7])**2 + 
            (x[8] - y[8])**2 + 
            (x[9] - y[9])**2
            )**(0.5)

# Test procedure to brute force nearest neighbor search (k=1) and time the procedure, and save the index to file for checking
# Also,  for k=40 and time the procedure
def bruteForce(fileName, k, saveToFile):
    #preprocessing the dataset into numpy array
    df = pd.read_csv("tripadvisor_review.csv")
    data = df.iloc[:,1:11].to_numpy()

    #import the queries array
    queriesArray = utils.importQueries("queries1.txt")

    nearestNeighborIndexes = []

    start = time.time()
    # For each query
    for i in range(50000):
        distancesList = []

        # For each data point, calculate distance between query and data point. Append to distances list
        for j in range(data.shape[0]):
            distancesList.append(calcEuclideanDistance(queriesArray[i], data[j]))
            
        if (k == 1):
            nearestNeighborIndexes.append(np.array(distancesList).argmin())
        else:
            nearestNeighborIndexes.append(np.argpartition(np.array(distancesList), k))

    end = time.time()
    totalTime = end - start
    if(saveToFile):
        print("For brute force method and data of 10 dimensions, K=1 nearest neighbors for queries1.txt 50K queries are outputted to " + fileName)
        np.savetxt(fileName, nearestNeighborIndexes, fmt='%d')
    
    print("For brute force method, searching for k =" + str(k) + " nearest neighbors for each of 50K queries took", totalTime)

    print("\n")

def main():
    # use queries1.txt

    bruteForce("BruteForce_NN_indexes.txt", 1, True)
    bruteForce("", 40, False)

if __name__=="__main__":
    main()
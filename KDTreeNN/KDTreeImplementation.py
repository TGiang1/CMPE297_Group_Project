import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import random
import utils
import time
import matplotlib.pyplot as plt

"""
Main class for KDTree multidimensional data structure indexing on our travel reviews dataset for nearest neighbor cost/complexity exploration
With the scikiit-learn package, KDTree data structure only supports nearest neighbor queries. For KDTree data structure, inserts and deletes
often causes large reformatting changes to the tree structure that is not ideal; thus for KDTree, inserts and deletes are not recommended
and if there are big changes to the dataset, then just rebuilding of the tree is recommended.

For our purpose, with the KDTree, we will explore how the number of dimensions of the dataset, and parameter values (such as leaf size and distance metric) of
the KDTree affect the time of nearest neighbor queries, and compare with the brute force approach and other multidimensional index structures explored.
"""
class KDTreeExplore:
    #numDimensions between 1 and 10 for our travelreview dataset
    def readDataset(self, fileName, numDimensions):
        df = pd.read_csv(fileName)
        data = df.iloc[:,1:numDimensions+1].to_numpy()
        self.data = data
        self.numDimensions = numDimensions

    def createKDTree(self, leaf_size, metric):
        kdt = KDTree(self.data, leaf_size=leaf_size, metric=metric)
        return kdt

    def nnSearch(self, queriesArray, kdt, k):
        distList = []
        indList = []

        start = time.time()
        distList, indList  = kdt.query(queriesArray, k=k, return_distance=True) 
        end = time.time()

        timeTaken = end - start

        return distList, indList, timeTaken

# Test procedure to output k=1 nearest neighbor for queries1.txt (50k queries) on kd tree of leaf size 10 built from the dataset
# save to "KDTreeNN_Indexes"
def checkNearestNeighbors(fileName):
    kdClass = KDTreeExplore()
    kdClass.readDataset("tripadvisor_review.csv", 10) #10 dimensions

    kd1 = kdClass.createKDTree(10, "euclidean") #Initialize kd tree with the dataset, leaf size of 10

    queriesArray = utils.importQueries("queries1.txt")

    distList, indList, timeTaken = kdClass.nnSearch(queriesArray, kd1, 1)

    print("For kd tree of leaf size 10 built from the dataset (10 dimensions), K=1 nearest neighbors for queries1.txt 50k queries are outputted to " + fileName)
    
    np.savetxt(fileName, indList, fmt='%d')

    print("\n")

# Procedure to iterate through number of leafs for kd Tree and plot how number of leafs affect nn search time for 50k queries (queries1.txt)
def numLeaves(plotName):
    kdClass = KDTreeExplore()
    kdClass.readDataset("tripadvisor_review.csv", 10) #10 dimensions

    numLeafsList = [2 , 4, 8, 16, 32, 64, 128, 256]

    timeTakenList = []
    for x in numLeafsList:
        kd = kdClass.createKDTree(x, "euclidean") #Initialize kd tree with the dataset, leaf size varying

        queriesArray = utils.importQueries("queries1.txt")

        distList, indList, timeTaken = kdClass.nnSearch(queriesArray, kd, 1)

        print("For 50k queries and 10 dimensions and kd tree with leaf size of " + str(x) + ", the time taken for nearest neighbor search (k=1) was ", timeTaken)
        timeTakenList.append(timeTaken)
    
    plt.figure(figsize=(10,5))
    plt.title('KDTree: Leaf Size vs. NN (k=1) search time taken (seconds) for 50k queries (10 dimensions)')
    plt.xlabel('Leaf Size')
    plt.ylabel('NN Search Time (seconds)')
    plt.plot(numLeafsList, timeTakenList)

    plt.savefig(plotName)

    print("\n")

# Taking leaf size of 64, procedure to iterate through number of nearest neighbors to search for (k=2 to k=10) and plot how 
# the number of nearest neighbors to search for for each of the 50k queries (queries1.txt) affects NN search time
def numNearestNeighbor(plotName):
    kdClass = KDTreeExplore()
    kdClass.readDataset("tripadvisor_review.csv", 10) #10 dimensions

    numNeighborsList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100]

    timeTakenList = []
    for x in numNeighborsList:
        kd = kdClass.createKDTree(64, "euclidean") #Initialize kd tree with the dataset, leaf size of 64

        queriesArray = utils.importQueries("queries1.txt")

        distList, indList, timeTaken = kdClass.nnSearch(queriesArray, kd, x)

        print("For 50k queries and 10 dimensions and kd tree with leaf size of 64, the time taken for nearest neighbor search (k=" + str(x) + ") was ", timeTaken)
        timeTakenList.append(timeTaken)
    
    plt.figure(figsize=(10,5))
    plt.title('KDTree: K (num neighbors to search for) vs. NN search time taken (seconds) for 50k queries (10 dimensions, leaf 64)')
    plt.xlabel('K (num neighbors to search for)')
    plt.ylabel('NN Search Time (seconds)')
    plt.plot(numNeighborsList, timeTakenList)

    plt.savefig(plotName)

    print("\n")

# Procedure to iterate through different number of dimensions (2 to 10) and see how this affects NN search time.
# Leaf size of KDTree is set to 64, and we will search for 40 nearest neighbors for each query (out of 50k)
def numDimensions(plotName):
    kdClass = KDTreeExplore()

    numDimensionList = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    timeTakenList = []

    for x in numDimensionList:
        kdClass.readDataset("tripadvisor_review.csv", x) #varying # of dimensions (2 to 10)
        
        kd = kdClass.createKDTree(64, "euclidean") #Initialize kd tree with the dataset, leaf size of 64

        queriesArray = utils.generateQueries(50000, numDimensions=x) #generate 50k random queries (each of size x)

        distList, indList, timeTaken = kdClass.nnSearch(queriesArray, kd, 40) #search for 40 nearest neighbors per query

        print("For 50k queries and " + str(x) + " dimensions and kd tree with leaf size of 64, the time taken for nearest neighbor search (k=40) was ", timeTaken)
        timeTakenList.append(timeTaken)
    
    plt.figure(figsize=(10,5))
    plt.title('KDTree: Num Dimensions vs. NN search time taken (seconds) for 50k queries (leaf size 64, k=40)')
    plt.xlabel('Num Dimensions')
    plt.ylabel('NN Search Time (seconds)')
    plt.plot(numDimensionList, timeTakenList)

    plt.savefig(plotName)

    print("\n")

def main():
    # Generate and save queries. 1 time.
    # queriesArray = utils.generateQueries(50000)
    # utils.saveQueries("queries1.txt", queriesArray)

    checkNearestNeighbors("KDTree_NN_indexes.txt")
    numLeaves("images/numLeafsVsSearchTime.png")
    numNearestNeighbor("images/KVsSearchTime.png")
    numDimensions("images/NumDimensionsSearchTime.png")

if __name__=="__main__":
    main()



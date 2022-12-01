import mtree
import numpy as np

def d_int(x, y):      # define a distance function for numbers. euclidean for 3 dimensions
    return ((x[0] - y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)**(0.5)

queriesArray = np.random.uniform(0, 5, size=(3, 3))
queriesArray = np.round(queriesArray, 2)

tree = mtree.MTree(d_int, max_node_size=4)  

tree.add(list(queriesArray[0]))
tree.add_all([[1,2,3],[4,5,6],[1,1,1]])
a = tree.search([1,1, 1])


print(list(a))
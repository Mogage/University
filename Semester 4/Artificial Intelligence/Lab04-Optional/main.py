import networkx as nx
from dynamicGraph import DynamicGraph
from aco import ACO
from random import seed


def readGraph(fileName):
    graphToRead = nx.Graph()
    with open(fileName, "r") as file:
        size = int(file.readline().removesuffix('\n'))
        for count in range(size):
            graphToRead.add_node(count)
        for row in range(size):
            line = file.readline().split(',')
            for column in range(size):
                if column == row:
                    continue
                graphToRead.add_edge(row, column, weight=int(line[column]), pheromone=1.0)
            graphToRead.add_edge(row, size - 1, weight=int(line[size - 1].removesuffix('\n')), pheromone=1.0)

    return graphToRead


def readTsp(fileName):
    graphToRead = nx.Graph()
    with open(fileName, "r") as inFile:
        for i in range(100000):
            node, longitude, latitude = inFile.readline().strip().split()
            graphToRead.add_node(int(node) - 1, x=int(longitude), y=int(latitude))

    return graphToRead


def readDynamic(fileName):
    graphToRead = DynamicGraph()
    with open(fileName, "r") as file:
        size = int(file.readline().strip())
        for line in file:
            if size == 3:
                source, target, time = line.strip().split(' ')
                weight = 1
            else:
                source, target, weight, time = line.strip().split(' ')
            source = int(source) - 1
            target = int(target) - 1
            graphToRead.add_edge(source, target, time=int(time), weight=float(weight), pheromone=1.0)
            graphToRead.add_node(source)
            graphToRead.add_node(target)

    return graphToRead


def readDynamicGraphFromToWeight(fileName):
    graphToRead = nx.Graph()
    maxNode = 0
    with open(fileName, "r") as file:
        # read while the file is not empty
        while True:
            line = file.readline()
            if not line:
                break
            line = line.split(' ')
            graphToRead.add_edge(int(line[0]), int(line[1]), weight=float(line[2]), pheromone=1.0)
            # if node is not in the graph, add it
            if not graphToRead.has_node(int(line[0])):
                graphToRead.add_node(int(line[0]))
            if not graphToRead.has_node(int(line[1])):
                graphToRead.add_node(int(line[1]))
            # remember the max node
            if int(line[0]) > maxNode:
                maxNode = int(line[0])
            if int(line[1]) > maxNode:
                maxNode = int(line[1])
    # add missing edges with max weight
    for i in range(1, maxNode+1):
        for j in range(1, maxNode+1):
            if not graphToRead.has_edge(i, j):
                graphToRead.add_edge(i, j, weight=float(999), pheromone=1.0)

    # remove node 0
    return graphToRead


if __name__ == '__main__':
    #graph = readGraph('data/easy_tsp.txt')
    #graph = readTsp('data/mona-lisa100K.tsp')
    #graph = readDynamic('data/aves-weaver-social.edges')
    graph = readDynamicGraphFromToWeight('data/aves-sparrow-social.edges')

    #seed(1)

    acoInfo = {'graph': graph, 'alpha': 1.0, 'beta': 4.0, 'colonySize': 100, 'numberOfSteps': 50, 'rho': 0.3}
    aco = ACO(acoInfo)
    output = aco.run()
    print(output)

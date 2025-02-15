import networkx as nx
from population import Population
from random import seed
from math import sqrt
import tsplib95


def readGraph(fileName):
    graphToRead = nx.Graph()
    with open(fileName, "r") as file:
        size = int(file.readline().removesuffix('\n'))
        matrixGraph = [[0 for _ in range(size)] for _ in range(size)]
        for count in range(size):
            graphToRead.add_node(count)
        for row in range(size):
            line = file.readline().split(',')
            for column in range(size):
                if column == row:
                    continue
                graphToRead.add_edge(row, column, weight=int(line[column]))
                matrixGraph[row][column] = int(line[column])
            graphToRead.add_edge(row, size - 1, weight=int(line[size - 1].removesuffix('\n')))
            matrixGraph[row][column] = int(line[size - 1].removesuffix('\n'))

    return graphToRead, matrixGraph


def fitnessFunction(graph, chromosome):
    weightSum = 0
    size = len(chromosome)
    for count in range(size - 1):
        weightSum += graph[chromosome[count]][chromosome[count + 1]]['weight']

    return weightSum + graph[chromosome[0]][chromosome[size - 1]]['weight']


def runGenerationsAlgorithm(numberOfGenerations, population, graph):
    population.initialisation()
    population.evaluation()

    for _ in range(numberOfGenerations):
        population.oneGeneration()

    solutions = set()
    fitness = 0

    for chromo in population.bestChromosomes:
        solutions.add(tuple(chromo.representation))
        fitness = chromo.fitness

    print(fitness)
    for sol in sorted(solutions):
        print(sol)


import itertools


def held_karp(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        dists: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))


def readTsp(fileName):
    graph = nx.Graph()
    with open(fileName, "r") as inFile:
        for i in range(100000):
            node, longitude, latitude = inFile.readline().strip().split()
            graph.add_node(int(node) - 1, x=int(longitude), y=int(latitude))

    return graph


def euclidianDistance(node1, node2):
    return sqrt((node1['x'] - node2['x']) ** 2 + (node1['y'] - node2['y']) ** 2)


def fitnessFunction2(graph, chromosome):
    weightSum = 0
    size = len(chromosome)
    for count in range(size - 1):
        weightSum += euclidianDistance(graph.nodes[chromosome[count]], graph.nodes[chromosome[count + 1]])

    return weightSum + euclidianDistance(graph.nodes[chromosome[0]], graph.nodes[chromosome[size - 1]])


def run():
    graph, matrixGraph = readGraph("data/medium.txt")
    #graph = readTsp("data/mona-lisa100K.tsp")
    #print(held_karp(matrixGraph))

    params = {"graph": graph, "startNode": 0, "populationSize": 250, "fitnessFunction": fitnessFunction}
    pop = Population(params)

    seed(1)

    runGenerationsAlgorithm(7500, pop, graph)


if __name__ == "__main__":
    run()

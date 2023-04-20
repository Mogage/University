import os 
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt 
import warnings 
import time

from networkx.algorithms import community

warnings.simplefilter('ignore')

def plotNetwork(network, communities):
    np.random.seed(1000)
    pos = nx.spring_layout(network) 
    plt.figure(figsize=(16, 12))
    #nx.draw(network, pos, cmap=plt.get_cmap('Set2'), node_color=communities, with_labels=True)
    nx.draw_networkx_nodes(G, pos, node_size=150, cmap=plt.cm.RdYlBu, node_color = communities)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show()

def deltaQ(G, node, community):    
    """
    Calculated the changes in modularity if the node would be moved in the community

    Complexity: O(n) - n number of nodes in community

    Parameters:
        G (networkx.Graph): The input graph.
        node (int): The node to check modularity for
        community(list): The communities of each node in the graph

    Returns:
        float: the change in modularity
    """

    # Sum of links incident to i
    k_i = G.degree(node)
    # Sum of links from node i to nodes in the community
    k_i_in = sum([G.has_edge(node, neighbor) and community[node] == community[neighbor] for neighbor in G.neighbors(node)])
    # Sum of links to nodes in community
    k_tot = sum([G.degree(neighbor) for neighbor in G.nodes() if community[node] == community[neighbor]])
    # Sum of links in the graph
    m = G.number_of_edges()

    return k_i_in - k_tot*k_i/(2*m)

def modularity(G, community):
    """
    Calculated the modularity of the community

    Complexity: O(n^2) - n number of nodes in the graph

    Parameters:
        G (networkx.Graph): The input graph.
        community(list): The communities of each node in the graph

    Returns:
        float: The modularity
    """
    # Sum of links in the graph
    m = G.number_of_edges()
    Q = 0

    for node_i in G.nodes:
        for node_j in G.nodes:
            # We only add to the modularity if the nodes are in different communities
            if node_i == node_j or community[node_i] != community[node_j]:
                continue
            
            # Degree of node i
            k_i = G.degree(node_i)
            # Degree of node j
            k_j = G.degree(node_j)

            Q += G.has_edge(node_i, node_j) - k_i*k_j/(2*m)

    return Q / (2 * m)

def findCommunities(G):
    """
    Implements the Louvain algortihm for community finding

    Complexity: O(nlogn) - n number of nodes in community

    Parameters:
        G (networkx.Graph): The input graph.

    Returns:
        int, dict: Number of communities & a dictionary with pairs (node, communityItBelongs)
    """
    currCommunities = list(G.nodes)
    currModularity = -1
    
    while True:
        nodes = list(G.nodes)
        np.random.shuffle(nodes)
        prev_modularity = currModularity

        # We iterate through each of the nodes
        for node in nodes:
            bestDeltaQ = 0
            bestCommunity = currCommunities[node]

            # Then we try to move the node the the best neighbor's community
            for neighbor in G.neighbors(node):
                if currCommunities[neighbor] != currCommunities[node]:
                    newCommunities = currCommunities
                    newCommunities[node] = currCommunities[neighbor]

                    deltaModularity = deltaQ(G, node, newCommunities)

                    # If the modularity of the community in which we move the node is better
                    # we save the best community to move the node to
                    if deltaModularity > bestDeltaQ:
                        bestCommunity = newCommunities[neighbor]
                        bestDeltaQ = deltaModularity
            
            currCommunities[node] = bestCommunity

        communities = []
        for com in set(currCommunities):
            members = np.where(np.array(currCommunities) == com)[0].tolist()
            communities.append(members)
        partition = {i: com + 1 for com, nodes in enumerate(communities) for i in nodes}
        currModularity = modularity(G, partition)

        if currModularity - prev_modularity < 1e-8:
            break
    
    return len(communities), partition


def run(G, plot=False):
    st = time.time()

    numberOfCommunities, communities = findCommunities(G)
    with open("output.txt", "w") as out:
        out.write(str(numberOfCommunities) + '\n')
        for key in sorted(communities.keys()):
            out.write(f'{key} {communities[key]}\n')

    et = time.time()

    print(f'Seconds spent: {et - st}')
    if plot:
        new_list = []
        for key in sorted(communities.keys()):
            new_list.append(communities[key])
        plotNetwork(G, new_list)

if __name__ == '__main__':
    G = nx.read_gml(f'data_sets/custom_06.gml', label='id')
    run(G)
    exit()

    directory = os.fsencode('data_sets')
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        G = nx.read_gml(f'data_sets/{filename}', label='id')
        if filename == 'custom_06.gml':
            continue
        run(G, True)
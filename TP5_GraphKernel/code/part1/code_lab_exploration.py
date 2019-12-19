"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1  ###DONE###

G = nx.read_edgelist('CA-HepTH.txt')
print("The number  of nodes of the graph is : " + str(G.number_of_nodes()))
print("The number  of edges of the graph is : " + str(G.number_of_edges()))




############## Task 2   ###DONE


K = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
print("the number of connected subcomponents is : " + str(len(K)))
U = max(nx.connected_component_subgraphs(G), key=len)
print("The number  of nodes of the largest connected component is : " + str(U.number_of_nodes()))
print("The number  of edges of the largest connected component is : " + str(U.number_of_edges()))
print("The fraction  of nodes of the largest connected component is : " + str(U.number_of_nodes()/G.number_of_nodes()))
print("The fraction  of edges of the largest connected component is : " + str(U.number_of_edges()/G.number_of_edges()))

############## Task 3   ###DONE
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
print("the lowest degree is : " + str(np.min(degree_sequence)))
print("the highest degree is : " + str(np.max(degree_sequence)))
print("the mean of the degrees in the graph is : " + str(np.mean(degree_sequence)))




############## Task 4

Degree_Histogram = nx.degree_histogram(G)
fig, ax = plt.subplots()
plt.bar(range(len(Degree_Histogram)),Degree_Histogram,color='b')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.savefig("Distribution.jpg")
plt.show()
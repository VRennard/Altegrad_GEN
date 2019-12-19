"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
import scipy
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans


############## Task 5
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    # your code here #
    ##################
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    D = np.zeros((n,n))
    for i, node in enumerate(G.nodes()):
        D[i, i] = G.degree(node)

    L = D - A

    eigvals, eigvecs = eigs(L, k=100, which='SR')
    eigvecs = eigvecs.real

    km = KMeans(n_clusters=k)
    km.fit(eigvecs)
    clustering = dict()
    for i, node in enumerate(G.nodes()):
        clustering[node] = km.labels_[i]

    return clustering



############## Task 6

##################
# your code here #
##################
G = nx.read_edgelist('../datasets/CA-HepTh.txt', comments='#', delimiter='\t', create_using=nx.Graph())
gcc_nodes = max(nx.connected_components(G), key=len)
gcc = G.subgraph(gcc_nodes)

clustering = spectral_clustering(gcc, 50)

############## Task 7
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
   ##################
    modularity=0
    m=G.number_of_edges()

    for k in range(min(clustering.values()),max(clustering.values())+1):
      d_c=0
      communitie=set()
      for node in G.nodes()  :
        
        if clustering[node] == k:
          d_c+=G.degree(node)
          communitie.add(node)
      l_c=G.subgraph(communitie).number_of_edges()
      print([l_c,d_c,m])
      modularity+=(l_c/m) - (d_c/(2*m))**2
   
    # your code here #
    ##################
    
    return  modularity


############## Task 8

##################
# your code here #
##################

print('Modularity spectral clustering:', modularity(gcc, clustering))

random_clustering = dict()
for node in G.nodes():
    random_clustering[node] = randint(0, 49)

print('Modularity random clustering:', modularity(gcc, random_clustering))

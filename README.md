# REFLAG
Representation Learning for Labeled Attributed Graphs is a framework for learning a represenation using multiple sources of information.


Data Format:
  REFLAG reads network in adjacency list. It needs two types of files:
 1. <network name>_graph.adjlist : This adjacency list represents the structural graph (directed or undirected).
 2. <network name>_na.adjlist: This adjacency list is a an undirected bipartite graph. The structural vertices are numbered from to 1 to num. of nodes, and 
		the attribute vertices are numbered after structural vertices. This bipartite graph doesn't contain labels as attributes. 

  The file <network_name>_label_10_na.adjlist is a bipartite graph in which labels of 10% of nodes are incorporated as attributes.  

usage: 
To learn a representation without using label information. 

1. #python __main__.py --data M10

To learn a representation using labels.

2. #python __main__.py --data M10 --label True



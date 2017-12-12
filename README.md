# GAT2VEC
Representation Learning for Attributed Graphs is a framework for learning a represenation
using multiple sources of information.

DIRECTORIES:
  1. src :- has the source code for GAT2VEC, and evaluation
  2. data :- it contains the input networks in respective directories, along with labels for classification task
		the files adjedges.txt, labels.txt, and docs.txt are the orignal files of the datasets. 
		we preprocess and generate the files for GAT2VEC processing and to bring the uniformity in the for vertex id's
		We uniformly start vertex id's from 1. 
  3. embeddings: the embeddings learned are stored in this directory
 
DATA FORMAT
  GAT2VEC reads network in adjacency list. It needs two types of files:
 1. <network name>_graph.adjlist : This adjacency list represents the structural graph (directed or undirected).
 2. <network name>_na.adjlist: This adjacency list is an undirected bipartite graph. The structural vertices are numbered from to 1 to num. of structural nodes, and 
		the attribute vertices are numbered after structural vertices. This bipartite graph doesn't contain labels as attributes. 

  The file <network_name>_label_10_na.adjlist is a bipartite graph in which labels of 10% of nodes are incorporated as attributes.  

USAGE:
To learn a representation without using label information. 

1. #python __main__.py --data M10

To learn a representation using labels.

2. #python __main__.py --data M10 --label True


PS: The pre-processing code for generating structural and bipartite graph will be uploaded soon
    My python code has influence of Java :)

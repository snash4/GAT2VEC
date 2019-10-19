import csv
import logging

import numpy as np
import pandas as pd
from deepwalk import graph

from GAT2VEC import paths

logger = logging.getLogger(__name__)


def get_graph(dataset_dir, gtype='graph'):
    """ load the adjacency list.
    """
    fname_struct = paths.get_adjlist_path(dataset_dir, gtype)
    logger.debug(fname_struct)
    G = graph.load_adjacencylist(fname_struct)
    logger.debug("Number of nodes: {}".format(len(G.nodes())))
    return G


def get_embeddingDF(fname):
    """returns the embeddings read from file fname."""
    df = pd.read_csv(fname, header=None, skiprows=1, delimiter=' ')
    df.sort_values(by=[0], inplace=True)
    df = df.set_index(0)
    return df.as_matrix(columns=df.columns[0:])


def get_labels(dataset_dir):
    """ returns list of labels ordered by the node id's """
    lblmap = {}
    fname = paths.get_labels_path(dataset_dir)
    with open(fname, 'r') as freader:
        lines = csv.reader(freader, delimiter='\t')
        for row in lines:
            lblmap[int(row[0])] = int(row[1])

    node_list = list(lblmap.keys())
    node_list.sort()
    labels = [lblmap[vid] for vid in node_list]
    return np.array(labels), node_list, len(set(labels))


def get_multilabels(dataset_dir, delim='\t'):
    """ returns the multibinarized object for multilabel datasets."""
    lblmap = {}
    fname = paths.get_labels_path(dataset_dir)
    unique_labels = set()
    with open(fname, 'r') as freader:
        lines = csv.reader(freader, delimiter=delim)
        for row in lines:
            lbls = str(row[1]).split(',')
            vid = int(row[0])
            lblmap[vid] = tuple(lbls)
            unique_labels.update(set(lbls))

    nlist = list(lblmap.keys())
    nlist.sort()
    labels = [lblmap[vid] for vid in nlist]
    return labels, nlist, len(unique_labels)

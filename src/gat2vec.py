from deepwalk import graph
from gensim.models import Word2Vec
import os
import psutil
import Evaluation
import pandas as pd
from multiprocessing import cpu_count
import random

p = psutil.Process(os.getpid())

'''
REFLAG learns an embedding jointly from structural contexts and attribute contexts
employing a single layer of neural network.
'''


class gat2vec(object):
    def __init__(self, dataset):
        print "Initializing reflag"
        self.dataset = dataset
        self._seed = 1
        self.dataset_dir = os.getcwd() + "/data/" + dataset + "/"
        self.TR = [0.1, 0.3, 0.5]

    '''return random walks '''
    def _get_random_walks(self, num_walks, wlength, label=False, type='graph'):
        if label:
            fname = self.dataset_dir + self.dataset + type + '.adjlist'
        else:
            fname = self.dataset_dir + self.dataset + '_' + type + '.adjlist'
        print fname

        # type = graph refers to structural graph
        if type == 'graph':
            G = graph.load_adjacencylist(fname)
        else:
            G = graph.load_adjacencylist(fname, undirected=True)

        print("Number of nodes: {}".format(len(G.nodes())))
        total_walks = len(G.nodes()) * num_walks
        print("Number of walks: {}".format(total_walks))
        walks = graph.build_deepwalk_corpus(G, num_paths=num_walks, path_length=wlength, alpha=0,
                                        rand=random.Random(self._seed))
        return walks, len(G.nodes())

    ''' filter attribute nodes from walks in attributed graph'''

    def _filter_walks(self,walks, node_num):
        filter_walks = []
        for walk in walks:
            if walk[0] <= node_num:
                fwalks = [nid for nid in walk if int(nid) <= node_num]
                filter_walks.append(fwalks)
        return filter_walks


    ''' Trains jointly attribute contexts and structural contexts'''
    def _train_word2Vec(self, walks, dimension_size, window_size, cores, output, fname):
        print "Learning Representation"
        model = Word2Vec(walks, size=dimension_size, window=window_size, min_count=0, sg=1,
                         workers=cores)
        if output is True:
            model.wv.save_word2vec_format(fname)
            print "Learned Represenation Saved"
            return fname
        return model


    def train_gat2vec(self, data, label, nwalks, wlength, dsize, wsize, output):
        print "Random Walks on Structural Graph"
        walks_structure, num_str_nodes = self._get_random_walks(nwalks, wlength)
        if label:
            print "Training on Labelled Data"
            reflag_model = self.train_labelled_reflag(data, walks_structure, num_str_nodes, nwalks, wlength, dsize, wsize, output)
        else:
            print "------------ATTRIBUTE walk--- "
            fname = "./embeddings/" + self.dataset + "_reflag.emb"
            walks_attribute, num_atr_nodes = self._get_random_walks(nwalks, wlength * 2, False, 'na')
            filter_walks = self._filter_walks(walks_attribute, num_str_nodes)
            walks = walks_structure + filter_walks
            reflag_model = self._train_word2Vec(walks, dsize, wsize, 8, output, fname)
        return reflag_model


    ''' Trains on labelled dataset, i.e class labels are used as an attribute '''

    def train_labelled_reflag(self, data, walks_structure, num_str_nodes, nwalks, wlength, dsize, wsize, output,
                              evaluate=True):
        alloutput = pd.DataFrame()
        for tr in self.TR:
            f_ext = "_label_" + str(int(tr * 100)) + '_na'
            walks_attribute, num_atr_nodes = self._get_random_walks(nwalks, wlength * 2, True, f_ext)
            filter_walks = self._filter_walks(walks_attribute, num_str_nodes)
            walks = walks_structure + filter_walks
            fname = "./embeddings/" + self.dataset + "_reflag_label_" + str(int(tr * 100)) + ".emb"
            reflag_model = self._train_word2Vec(walks, dsize, wsize, 8, output, fname)
        return reflag_model

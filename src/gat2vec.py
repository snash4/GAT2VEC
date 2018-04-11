from deepwalk import graph
from gensim.models import Word2Vec
import os
import psutil
import Evaluation
import pandas as pd
from multiprocessing import cpu_count
import random
from Evaluation.Classification import Classification
p = psutil.Process(os.getpid())

'''
GAT2VEC learns an embedding jointly from structural contexts and attribute contexts
employing a single layer of neural network.
'''


class gat2vec(object):
    def __init__(self, dataset, label):
        print "Initializing gat2vec"
        self.dataset = dataset
        self._seed = 1
        self.dataset_dir = os.getcwd() + "/data/" + dataset + "/"
        self.TR = [0.1, 0.3, 0.5]
        self.label = label
        print "loading structural graph"
        self.Gs = self._get_graph()
        if self.label == False:
            print "loading attribute graph"
            self.Ga = self._get_graph('na')

    '''
    load the adjacency list
    '''
    def _get_graph(self, gtype='graph'):
        fname_struct = self.dataset_dir + self.dataset + '_'+ gtype + '.adjlist'
        print fname_struct
        G = graph.load_adjacencylist(fname_struct)
        print("Number of nodes: {}".format(len(G.nodes())))
        return G

    '''return random walks '''
    def _get_random_walks(self, G, num_walks, wlength, gtype='graph'):
        walks = graph.build_deepwalk_corpus(G, num_paths=num_walks, path_length=wlength, alpha=0,
                                        rand=random.Random(self._seed))
        return walks

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


    def train_gat2vec(self, data, nwalks, wlength, dsize, wsize, output):
        print "Random Walks on Structural Graph"
        walks_structure = self._get_random_walks(self.Gs, nwalks, wlength)
        num_str_nodes = len(self.Gs.nodes())
        if self.label:
            print "Training on Labelled Data"
            gat2vec_model = self.train_labelled_gat2vec(data, walks_structure, num_str_nodes, nwalks, wlength, dsize, wsize, output)
        else:
            print "------------ATTRIBUTE walk--- "
            fname = "./embeddings/" + self.dataset + "_gat2vec.emb"
            walks_attribute  = self._get_random_walks(self.Ga, nwalks, wlength * 2)
            filter_walks = self._filter_walks(walks_attribute, num_str_nodes)
            walks = walks_structure + filter_walks
            gat2vec_model = self._train_word2Vec(walks, dsize, wsize, 8, output, fname)
        return gat2vec_model


    ''' Trains on labelled dataset, i.e class labels are used as an attribute '''

    def train_labelled_gat2vec(self, data, walks_structure, num_str_nodes, nwalks, wlength, dsize, wsize, output,
                               evaluate=True):
        alloutput = pd.DataFrame()
        for tr in self.TR:
            f_ext = "label_" + str(int(tr * 100)) + '_na'
            # walks_attribute, num_atr_nodes = self._get_random_walks(nwalks, wlength * 2, f_ext)
            self.Ga = self._get_graph(f_ext)
            walks_attribute = self._get_random_walks(self.Ga, nwalks, wlength * 2)
            filter_walks = self._filter_walks(walks_attribute, num_str_nodes)
            walks = walks_structure + filter_walks
            fname = "./embeddings/" + self.dataset + "_gat2vec_label_" + str(int(tr * 100)) + ".emb"
            gat2vec_model = self._train_word2Vec(walks, dsize, wsize, 8, output, fname)
        return gat2vec_model

    ''' Trains on the bipartite graph only'''
    def train_gat2vec_bip(self, data, nwalks, wlength, dsize, wsize, output):
        print "Learning Representation on Bipartite Graph"
        num_str_nodes = len(self.Gs.nodes())
        print "Random Walking..."
        walks_attribute = self._get_random_walks(self.Ga, nwalks, wlength * 2)
        filter_walks = self._filter_walks(walks_attribute, num_str_nodes)
        fname = "./embeddings/" + self.dataset + "_gat2vec_bip.emb"
        gat2vec_model = self._train_word2Vec(filter_walks, dsize, wsize, 8, output, fname)
        return gat2vec_model


    def param_walklen_nwalks(self, param, data, nwalks=10, wlength=80, dsize=128, wsize=5, output=True):
        print "PARAMETER SENSITIVTY ON " + param
        alloutput = pd.DataFrame()
        # p_value = [40,80,120,160,200]
        # p_value = [5, 10, 15, 20, 25]
        p_value = [1, 5,10,15,20,25]
        walks_st = []
        walks_at = []
        wlength = 80
        # nwalks = 10
        num_str_nodes = len(self.Gs.nodes())
        print "performing joint on both graphs..."
        for nwalks in p_value:
            print nwalks
            walks_st.append(self._get_random_walks(self.Gs, nwalks, wlength))
            walks_attribute = self._get_random_walks(self.Ga, nwalks, wlength * 2)
            walks_at.append(self._filter_walks(walks_attribute, num_str_nodes))

        print "Walks finished.... "
        for i, walks_structure in enumerate(walks_st):
            for j, filter_walks in enumerate(walks_at):
                ps = p_value[i]
                pa = p_value[j]
                print "parameters.... ", ps, pa
                walks = walks_structure + filter_walks
                fname = "./embeddings/" + self.dataset + "_gat2vec_" + param + "_nwalks_"+ str(ps)+str(pa) + ".emb"
                gat2vec_model = self._train_word2Vec(walks, dsize, wsize, 8, output, fname)
                p = (ps, pa)
                alloutput = self._param_evaluation(data, alloutput, p, param, gat2vec_model)

        print alloutput
        alloutput.to_csv(data + "_paramsens_" + param +"_nwalks_" +".csv", index=False)
        return gat2vec_model

    def _param_evaluation(self, data, alloutput, param_val, param_name, model):
        if data == 'blogcatalog':
            multilabel = True
        else:
            multilabel = False
        eval = Classification(data, multilabel)
        outDf = eval.evaluate(model, False)
        outDf['ps'] = param_val[0]
        outDf['pa'] = param_val[1]
        return alloutput.append(outDf)

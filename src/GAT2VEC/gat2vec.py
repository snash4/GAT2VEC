from __future__ import print_function

from deepwalk import graph
from gensim.models import Word2Vec
import pandas as pd
import random
from GAT2VEC.evaluation.classification import Classification
from GAT2VEC import parsers, paths


class Gat2Vec(object):
    """
    GAT2VEC learns an embedding jointly from structural contexts and attribute contexts
    employing a single layer of neural network.
    """

    def __init__(self, input_dir, output_dir, label, tr):
        print("Initializing gat2vec")
        self.dataset = paths.get_dataset_name(input_dir)
        self._seed = 1
        self.dataset_dir = input_dir
        self.output_dir = output_dir
        self.TR = tr
        self.label = label
        print("loading structural graph")
        self.Gs = parsers.get_graph(self.dataset_dir)
        if not self.label:
            print("loading attribute graph")
            self.Ga = parsers.get_graph(self.dataset_dir, 'na')

    def _get_random_walks(self, G, num_walks, wlength):
        """return random walks."""
        walks = graph.build_deepwalk_corpus(G, num_paths=num_walks, path_length=wlength, alpha=0,
                                            rand=random.Random(self._seed))
        return walks

    def _filter_walks(self, walks, node_num):
        """ filter attribute nodes from walks in attributed graph."""
        filter_walks = []
        for walk in walks:
            if int(walk[0]) <= node_num:
                fwalks = [nid for nid in walk if int(nid) <= node_num]
                filter_walks.append(fwalks)
        return filter_walks

    def _train_word2Vec(self, walks, dimension_size, window_size, cores, output, fname):
        """ Trains jointly attribute contexts and structural contexts."""
        print("Learning Representation")
        model = Word2Vec([list(map(str, walk)) for walk in walks],
                         size=dimension_size, window=window_size, min_count=0, sg=1,
                         workers=cores)
        if output is True:
            model.wv.save_word2vec_format(fname)
            print("Learned Representation Saved")
            return fname
        return model

    def train_gat2vec(self, nwalks, wlength, dsize, wsize, output):
        print("Random Walks on Structural Graph")
        walks_structure = self._get_random_walks(self.Gs, nwalks, wlength)
        if self.label:
            print("Training on Labelled Data")
            gat2vec_model = self.train_labelled_gat2vec(walks_structure, nwalks, wlength,
                                                        dsize, wsize, output)
        else:
            print("Random Walks on Attribute Graph")
            fname = paths.get_embedding_path(self.dataset_dir, self.output_dir)
            gat2vec_model = self._train_gat2vec(dsize, fname, nwalks, output, walks_structure,
                                                wlength, wsize)
        return gat2vec_model

    def train_labelled_gat2vec(self, walks_structure, nwalks, wlength, dsize, wsize, output):
        """ Trains on labelled dataset, i.e class labels are used as an attribute """
        for tr in self.TR:
            f_ext = "label_" + str(int(tr * 100)) + '_na'
            self.Ga = parsers.get_graph(self.dataset_dir, f_ext)
            fname = paths.get_embedding_path_wl(self.dataset_dir, self.output_dir, tr)
            gat2vec_model = self._train_gat2vec(dsize, fname, nwalks, output, walks_structure,
                                                wlength, wsize)
        return gat2vec_model  # TODO: can also return None

    def train_gat2vec_bip(self, nwalks, wlength, dsize, wsize, output):
        """ Trains on the bipartite graph only"""
        print("Learning Representation on Bipartite Graph")
        fname = paths.get_embedding_path_bip(self.dataset_dir, self.output_dir)
        gat2vec_model = self._train_gat2vec(dsize, fname, nwalks, output, None, wlength, wsize,
                                            add_structure=False)
        return gat2vec_model

    def _train_gat2vec(self, dsize, fname, nwalks, output, walks_structure, wlength, wsize,
                       add_structure=True):
        walks_attribute = self._get_random_walks(self.Ga, nwalks, wlength * 2)
        walks = self._filter_walks(walks_attribute, len(self.Gs.nodes()))
        if add_structure:
            walks = walks_structure + walks
        gat2vec_model = self._train_word2Vec(walks, dsize, wsize, 4, output, fname)
        return gat2vec_model

    def param_walklen_nwalks(self, param, data, tr, nwalks=10, wlength=80, dsize=128, wsize=5,
                             output=True, is_multilabel=False):
        print("PARAMETER SENSITIVTY ON " + param)
        alloutput = pd.DataFrame()
        p_value = [1, 5, 10, 15, 20, 25]
        walks_st = []
        walks_at = []
        wlength = 80
        # nwalks = 10
        num_str_nodes = len(self.Gs.nodes())
        print("performing joint on both graphs...")
        for nwalks in p_value:
            print(nwalks)
            walks_st.append(self._get_random_walks(self.Gs, nwalks, wlength))
            walks_attribute = self._get_random_walks(self.Ga, nwalks, wlength * 2)
            walks_at.append(self._filter_walks(walks_attribute, num_str_nodes))

        print("Walks finished.... ")
        for i, walks_structure in enumerate(walks_st):
            for j, filter_walks in enumerate(walks_at):
                ps = p_value[i]
                pa = p_value[j]
                print("parameters.... ", ps, pa)
                walks = walks_structure + filter_walks
                fname = paths.get_embedding_path_param(self.dataset_dir, self.output_dir, param, ps,
                                                       pa)
                gat2vec_model = self._train_word2Vec(walks, dsize, wsize, 8, output, fname)
                p = (ps, pa)
                alloutput = self._param_evaluation(data, self.output_dir, alloutput, p, param,
                                                   gat2vec_model, is_multilabel, tr)

        print(alloutput)
        alloutput.to_csv(paths.get_param_csv_path(self.dataset, param), index=False)

    def _param_evaluation(self, data, output_dir, alloutput, param_val, param_name, model,
                          is_multilabel, tr):
        eval = Classification(data, output_dir, tr, is_multilabel)
        outDf = eval.evaluate(model, False)
        outDf['ps'] = param_val[0]
        outDf['pa'] = param_val[1]
        return alloutput.append(outDf)

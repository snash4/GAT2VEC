import logging

import pandas as pd

from GAT2VEC.gat2vec import Gat2Vec
from GAT2VEC.evaluation.classification import Classification
from GAT2VEC import paths

logger = logging.getLogger(__name__)

def param_walklen_nwalks(param, input_dir, output_dir, tr, dsize=128, wsize=5, output=True,
                         is_multilabel=False):
    g2v = Gat2Vec(input_dir, output_dir, label=False, tr=tr)

    logger.debug("PARAMETER SENSITIVTY ON " + param)
    alloutput = pd.DataFrame()
    p_value = [1, 5, 10, 15, 20, 25]
    walks_st = []
    walks_at = []
    wlength = 80
    num_str_nodes = len(g2v.Gs.nodes())

    logger.debug("performing joint on both graphs...")
    for nwalks in p_value:
        logger.debug(nwalks)
        walks_st.append(g2v._get_random_walks(g2v.Gs, nwalks, wlength))
        walks_attribute = g2v._get_random_walks(g2v.Ga, nwalks, wlength * 2)
        walks_at.append(g2v._filter_walks(walks_attribute, num_str_nodes))
    logger.debug("Walks finished.... ")

    for i, walks_structure in enumerate(walks_st):
        for j, filter_walks in enumerate(walks_at):
            ps = p_value[i]
            pa = p_value[j]
            logger.debug("parameters.... %s %s", ps, pa)
            walks = walks_structure + filter_walks
            fname = paths.get_embedding_path_param(g2v.dataset_dir, g2v.output_dir, param, ps, pa)
            model = g2v._train_word2Vec(walks, dsize, wsize, 8, output, fname)
            alloutput = _param_evaluation(g2v.dataset_dir, g2v.output_dir, alloutput, (ps, pa),
                                          model, is_multilabel, tr)

    logger.debug(alloutput)
    alloutput.to_csv(paths.get_param_csv_path(g2v.dataset, param), index=False)


def _param_evaluation(data, output_dir, alloutput, param_val, model, is_multilabel, tr):
    eval = Classification(data, output_dir, tr, is_multilabel)
    outDf = eval.evaluate(model, False)
    outDf['ps'], outDf['pa'] = param_val[0], param_val[1]
    return alloutput.append(outDf)

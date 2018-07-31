import os
import unittest

from GAT2VEC.evaluation.classification import Classification
from GAT2VEC.gat2vec import Gat2Vec
from GAT2VEC import paths
import numpy as np
from tests import constants


class Gat2VecTest(unittest.TestCase):

    def test_train_gat2vec(self):
        if not os.path.isdir(constants.OUTPUT_DIR):
            os.makedirs(constants.OUTPUT_DIR)

        g2v = Gat2Vec(constants.DATASET_DIR, constants.OUTPUT_DIR, label=False, tr=constants.TR)
        model = g2v.train_gat2vec(constants.NUM_WALKS, constants.WALK_LENGTH, constants.DIMENSION,
                                  constants.WINDOW_SIZE, output=constants.SAVE_OUTPUT)

        clf_model = Classification(constants.DATASET_DIR, constants.OUTPUT_DIR, tr=constants.TR)
        results_model = clf_model.evaluate(model, label=False, evaluation_scheme="tr")

        ground_truth = paths.get_embedding_path(constants.DATASET_DIR, constants.RESOURCE_DIR)
        results_ground = clf_model.evaluate(ground_truth, label=False, evaluation_scheme="tr")

        np.testing.assert_almost_equal(results_model.values, results_ground.values, decimal=2)

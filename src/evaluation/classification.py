from collections import defaultdict

import pandas as pd
import numpy as np
import csv
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn import linear_model
from sklearn import preprocessing
from src import paths


class Classification:
    """This class performs multi-class/multi-label classification tasks."""

    def __init__(self, dataset, tr, multilabel=False):
        self.dataset = dataset
        self.output = {"TR": [], "accuracy": [], "f1micro": [], "f1macro": [], "auc": []}
        self.TR = tr  # the training ratio for classifier
        self.dataset_dir = paths.get_dataset_dir(dataset)
        self.multi_label = multilabel
        if self.multi_label:
            self.labels, self.label_ind, self.label_count = self.get_multilabels()
        else:
            self.labels, self.label_ind, self.label_count = self.get_labels()

    def get_labels(self):
        """ returns list of labels ordered by the node id's """
        lblmap = {}
        fname = paths.get_labels_path(self.dataset_dir)
        with open(fname, 'r') as freader:
            lines = csv.reader(freader, delimiter='\t')
            for row in lines:
                lblmap[int(row[0])] = int(row[1])

        node_list = lblmap.keys()
        node_list.sort()
        labels = [lblmap[vid] for vid in node_list]
        return np.array(labels), node_list, len(set(labels))

    def get_multilabels(self, delim='\t'):
        """ returns the multibinarized object for multilabel datasets."""
        lblmap = {}
        fname = paths.get_labels_path(self.dataset_dir)
        unique_labels = set()
        with open(fname, 'r') as freader:
            lines = csv.reader(freader, delimiter=delim)
            for row in lines:
                lbls = str(row[1]).split(',')
                vid = int(row[0])
                lblmap[vid] = tuple(lbls)
                unique_labels.update(set(lbls))

        nlist = lblmap.keys()
        nlist.sort()
        labels = [lblmap[vid] for vid in nlist]
        return self.binarize_labels(labels), nlist, len(unique_labels)

    def binarize_labels(self, labels, nclasses=None):
        """ returns the multilabelbinarizer object"""
        if nclasses == None:
            mlb = preprocessing.MultiLabelBinarizer()
            return mlb.fit_transform(labels)
        # for fit_and_predict to return binarized object of predicted classes
        mlb = preprocessing.MultiLabelBinarizer(classes=range(nclasses))
        return mlb.fit_transform(labels)

    def evaluate(self, model, label=False, evaluation_scheme="tr"):
        embedding = 0
        clf = self.get_classifier()

        if not label:
            embedding = self.get_embeddingDF(model)

        if evaluation_scheme == "cv":
            results = self.evaluate_cv(clf, embedding, 5)
        elif evaluation_scheme == "tr" or label:
            results = defaultdict(list)
            for tr in self.TR:
                print("TR ... ", tr)
                if label:
                    model = paths.get_embedding_path_wl(self.dataset, tr)
                    if isinstance(model, str):
                        embedding = self.get_embeddingDF(model)
                results.update(self.evaluate_tr(clf, embedding, tr))

        print("Training Finished")

        df = pd.DataFrame(results)
        return df.groupby(axis=0, by="TR").mean()

    def get_embeddingDF(self, fname):
        """returns the embeddings read from file fname."""
        df = pd.read_csv(fname, header=None, skiprows=1, delimiter=' ')
        df.sort_values(by=[0], inplace=True)
        df = df.set_index(0)
        return df.as_matrix(columns=df.columns[0:])

    def get_classifier(self):
        """ returns the classifier"""
        log_reg = linear_model.LogisticRegression(n_jobs=8)
        ors = OneVsRestClassifier(log_reg)
        return ors

    def evaluate_tr(self, clf, embedding, tr):
        """ evaluates an embedding for classification on training ration of tr."""
        ss = ShuffleSplit(n_splits=10, train_size=tr, random_state=2)
        for train_idx, test_idx in ss.split(self.labels):
            X_train, X_test, Y_train, Y_test = self._get_split(embedding, test_idx, train_idx)
            pred, probs = self.get_predictions(clf, X_train, X_test, Y_train, Y_test)
            self.output["TR"].append(tr)
            self.output["accuracy"].append(accuracy_score(Y_test, pred))
            self.output["f1micro"].append(f1_score(Y_test, pred, average='micro'))
            self.output["f1macro"].append(f1_score(Y_test, pred, average='macro'))
            if self.label_count == 2:
                self.output["auc"].append(roc_auc_score(Y_test, probs[:, 1]))
            else:
                self.output["auc"].append(0)
        return self.output

    def evaluate_cv(self, clf, embedding, n_splits):
        """Do a repeated stratified cross validation.
        :param clf: Classifier object.
        :param embedding: The feature matrix.
        :param n_splits: Number of folds.
        :return: Dictionary containing numerical results of the classification.
        """
        embedding  = embedding[self.label_ind, :]
        results = defaultdict(list)
        for i in range(10):
            rskf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            for train_idx, test_idx in rskf.split(embedding, self.labels):
                X_train, X_test, Y_train, Y_test = self._get_split(embedding, test_idx, train_idx)
                pred, probs = self.get_predictions(clf, X_train, X_test, Y_train, Y_test)
                results["TR"].append(n_splits)
                results["accuracy"].append(accuracy_score(Y_test, pred))
                results["f1micro"].append(f1_score(Y_test, pred, average='micro'))
                results["f1macro"].append(f1_score(Y_test, pred, average='macro'))
                if self.label_count == 2:
                    results["auc"].append(roc_auc_score(Y_test, probs[:, 1]))
                else:
                    results["auc"].append(0)
        return results


    def _get_split(self, embedding, test_id, train_id):
        return embedding[train_id], embedding[test_id], self.labels[train_id], self.labels[test_id]

    def get_predictions(self, clf, X_train, X_test, Y_train, Y_test):
        if self.multi_label:
            return self.fit_and_predict_multilabel(clf, X_train, X_test, Y_train, Y_test)
        else:
            clf.fit(X_train, Y_train)  # for multi-class classification
            return clf.predict(X_test), clf.predict_proba(X_test)

    def fit_and_predict_multilabel(self, clf, X_train, X_test, y_train, y_test):
        """ predicts and returns the top k labels for multi-label classification
        k depends on the number of labels in y_test."""
        clf.fit(X_train, y_train)
        y_pred_probs = clf.predict_proba(X_test)

        pred_labels = []
        nclasses = y_test.shape[1]
        top_k_labels = [np.nonzero(label)[0].tolist() for label in y_test]
        for i in range(len(y_test)):
            k = len(top_k_labels[i])
            probs_ = y_pred_probs[i, :]
            labels_ = tuple(np.argsort(probs_).tolist()[-k:])
            pred_labels.append(labels_)
        return self.binarize_labels(pred_labels, nclasses), y_pred_probs

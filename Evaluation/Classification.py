import pandas as pd
import numpy as np
import csv
import os
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
from sklearn import preprocessing


class Classification:
    """This class performs multi-class/multi-label classification tasks."""

    def __init__(self, dataset, multilabel=False):
        self.dataset = dataset
        self.output = {"TR": [], "accuracy": [], "f1micro": [], "f1macro": [], "auc": []}
        self.dataset_dir = os.getcwd() + "/data/" + dataset + "/"
        self.multi_label = multilabel
        if self.multi_label:
            self.labels = self.get_multilabels()
        else:
            self.labels = self.get_labels()

    def get_labels(self):
        """ returns list of labels ordered by the node id's """
        lblmap = {}
        fname = self.dataset_dir + 'labels_maped.txt'
        with open(fname, 'r') as freader:
            lines = csv.reader(freader, delimiter='\t')
            for row in lines:
                lblmap[int(row[0])] = int(row[1])

        node_list = lblmap.keys()
        node_list.sort()
        labels = [lblmap[vid] for vid in node_list]
        return np.array(labels)

    def get_multilabels(self, delim='\t'):
        """ returns the multibinarized object for multilabel datasets."""
        lblmap = {}
        fname = self.dataset_dir + 'labels_maped.txt'
        with open(fname, 'r') as freader:
            lines = csv.reader(freader, delimiter=delim)
            for row in lines:
                lbls = str(row[1]).split(',')
                vid = int(row[0])
                lblmap[vid] = tuple(lbls)

        nlist = lblmap.keys()
        nlist.sort()
        labels = [lblmap[vid] for vid in nlist]
        return self.binarize_labels(labels)

    def binarize_labels(self, labels, nclasses=None):
        """ returns the multilabelbinarizer object"""
        if nclasses == None:
            mlb = preprocessing.MultiLabelBinarizer()
            return mlb.fit_transform(labels)
        # for fit_and_predict to return binarized object of predicted classes
        mlb = preprocessing.MultiLabelBinarizer(classes=range(nclasses))
        return mlb.fit_transform(labels)

    def evaluate(self, model, label=False):
        embedding = 0
        if not label:
            embedding = self.get_embeddingDF(model)

        clf = self.get_classifier()
        TR = [0.1, 0.3, 0.5]  # the training ratio for classifier
        for tr in TR:
            print("TR ... ", tr)
            if label:
                model = "./embeddings/" + self.dataset + "_gat2vec_label_" + str(
                    int(tr * 100)) + ".emb"
                if isinstance(model, str):
                    embedding = self.get_embeddingDF(model)

            self.evaluate_tr(clf, embedding, tr)
        print("Training Finished")

        df = pd.DataFrame(self.output)
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
            if len(np.unique(self.labels)) == 2:
                self.output["auc"].append(roc_auc_score(Y_test, probs[:, 1]))
            else:
                self.output["auc"].append(0)

    def _get_split(self, embedding, test_id, train_id):
        return embedding[train_id], embedding[test_id], self.labels[train_id], self.labels[test_id]

    def get_predictions(self, clf, X_train, X_test, Y_train, Y_test):
        if self.multi_label:
            return self.fit_and_predict_multilabel(clf, X_train, X_test, Y_train, Y_test)
        else:
            clf.fit(X_train, Y_train)  # for multi-class classification
            # TODO: return values of multi-label and binary label aren't the same anymore.
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
        return self.binarize_labels(pred_labels, nclasses)

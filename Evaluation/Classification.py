import pandas as pd
import numpy as np
import csv
import random, os
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
from sklearn import preprocessing

''' This class performs multi-class/multi-label classification tasks'''

class Classification:

    def __init__(self, dataset, multilabel=False):
        self.dataset = dataset
        self.output = {"DATASET": [], "TR": [], "accuracy": [], "f1micro": [], "f1macro": []}
        self.dataset_dir = os.getcwd() + "/data/" + dataset + "/"
        self.multi_label = multilabel
        if self.multi_label:
            self.labels = self.getMultiLabels()
        else:
            self.labels = self.getLabels()

    """ returns the embeddings read from file fname"""
    def get_embeddingDF(self, fname):
        df = pd.read_csv(fname, header=None, skiprows=1, delimiter=' ')
        df.sort_values(by=[0], inplace=True)
        # dfs = dfs[:num_nodes]
        df = df.set_index(0)
        return df.as_matrix(columns=df.columns[0:])

    ''' returns list of labels ordered by the node id's '''
    def getLabels(self):
        lblmap = {}
        fname = self.dataset_dir+'labels_maped.txt'
        with open(fname, 'r') as freader:
            lines = csv.reader(freader, delimiter='\t')
            for row in lines:
                lblmap[int(row[0])] = int(row[1])

        node_list = lblmap.keys()
        node_list.sort()
        labels = [lblmap[vid] for vid in node_list]
        return np.array(labels)

    ''' returns the multibinarized object for multilabel datasets'''
    def getMultiLabels(self,delim='\t'):
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
        return self.binarizelabels(labels)

    ''' returns the multilabelbinarizer object'''
    def binarizelabels(self, labels, nclasses=None):
        if nclasses == None:
            mlb = preprocessing.MultiLabelBinarizer()
            return mlb.fit_transform(labels)
        # for fit_and_predict to return binarized object of predicted classes
        mlb = preprocessing.MultiLabelBinarizer(classes=range(nclasses))
        return mlb.fit_transform(labels)

    ''' returns the classifier'''
    def getclassifier(self):
        log_reg = linear_model.LogisticRegression(n_jobs=8)
        ors = OneVsRestClassifier(log_reg)
        return ors

    ''' predicts and returns the top k labels for multi-label classification
    k depends on the number of labels in y_test'''
    def fit_and_predict_multilabel(self, clf, X_train, X_test, y_train, y_test):
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
        return self. binarizelabels(pred_labels, nclasses)

    def getPredictions(self, clf, X_train, X_test, Y_train, Y_test):
        if self.multi_label:
            return self.fit_and_predict_multilabel(clf, X_train, X_test, Y_train, Y_test)
        else:
            clf.fit(X_train, Y_train) #for multi-class classification
            return clf.predict(X_test)

    def _get_accuracy(self, tlabels, plabels):
        return accuracy_score(tlabels, plabels)

    def _get_f1micro(self, tlabels, plabels):
        return f1_score(tlabels, plabels, average='micro')

    def getF1macro(self, tlabels, plabels):
        return f1_score(tlabels, plabels, average='macro')

    def _add_rows(self, data, output, tr, acc, f1micro, f1macro):
        output['DATASET'].append(data)
        output["TR"].append(tr)
        output["accuracy"].append(np.mean(np.array(acc)))
        output["f1micro"].append(np.mean(np.array(f1micro)))
        output["f1macro"].append(np.mean(np.array(f1macro)))

    ''' evaluates an embedding for classification on training ration of tr'''
    def evaluate_tr(self, clf, embedding, tr):
        num_nodes = self.labels.size
        ss = ShuffleSplit(n_splits=10, train_size=tr, random_state=2)
        gat2vecAcc = []
        gat2vecF1macro = []
        gat2vecF1micro = []
        for train_idx, test_idx in ss.split(self.labels):
            X_train, X_test, Y_train, Y_test = embedding[train_idx], embedding[test_idx], \
                                               self.labels[train_idx], self.labels[test_idx]
            pred = self.getPredictions(clf, X_train, X_test, Y_train, Y_test)
            gat2vecAcc.append(self._get_accuracy(Y_test, pred))
            gat2vecF1micro.append(self._get_f1micro(Y_test, pred))
            gat2vecF1macro.append(self.getF1macro(Y_test, pred))

        # self.addRows(self.dataset, self.output, tr, gat2vecAcc, gat2vecF1micro, gat2vecF1macro)
        # outDf = pd.DataFrame(self.output)
        return gat2vecAcc, gat2vecF1micro, gat2vecF1macro

    def evaluate(self, model, label = False):
        output = {"DATASET": [], "TR": [], "accuracy": [], "f1micro": [], "f1macro":[]}
        embedding = 0
        if label == False:
            if isinstance(model, str):
                embedding = self.get_embeddingDF(model)
            else:
                embedding = self.get_embeddingDF(model)

        clf = self.getclassifier()
        TR = [0.1, 0.3, 0.5] # the training ratio for classifier
        for tr in TR:
            print "TR ... ", tr
            if label == True:
                model = "./embeddings/" + self.dataset + "_gat2vec_label_" + str(int(tr * 100)) + ".emb"
                if isinstance(model, str):
                    embedding = self.get_embeddingDF(model)

            gat2vecAcc, gat2vecF1micro, gat2vecF1macro = self.evaluate_tr(clf, embedding, tr)
            self._add_rows(self.dataset, output, tr, gat2vecAcc, gat2vecF1micro, gat2vecF1macro)
        print "SVC Training Finished"

        print "results"
        outDf = pd.DataFrame(output)
        return outDf



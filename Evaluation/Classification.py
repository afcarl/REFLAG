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

    """ returns the embeddings """
    def get_embeddingDF(self, fname):
        df = pd.read_csv(fname, header=None, skiprows=1, delimiter=' ')
        df.sort_values(by=[0], inplace=True)
        # dfs = dfs[:num_nodes]
        df = df.set_index(0)
        return df.as_matrix(columns=df.columns[0:])

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

    def binarizelabels(self, labels, nclasses=None):
        if nclasses == None:
            mlb = preprocessing.MultiLabelBinarizer()
            return mlb.fit_transform(labels)
        mlb = preprocessing.MultiLabelBinarizer(classes=range(nclasses))
        return mlb.fit_transform(labels)

    def getclassifier(self):
        log_reg = linear_model.LogisticRegression(n_jobs=8)
        ors = OneVsRestClassifier(log_reg)
        return ors

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

    def getPredictions(self, clf, X_train, X_test, Y_train):
        if self.multi_label:
            return self.fit_and_predict_multilabel(clf, X_train, X_test, Y_train)
        else:
            clf.fit(X_train, Y_train)
            return clf.predict(X_test)

    def _get_accuracy(self, tlabels, plabels):
        return accuracy_score(tlabels, plabels)

    def _get_f1micro(self, tlabels, plabels):
        return f1_score(tlabels, plabels, average='micro')

    def getF1macro(self,tlabels, plabels):
        return f1_score(tlabels, plabels, average='macro')

    def _add_rows(self, data, output, tr, acc, f1micro, f1macro):
        output['DATASET'].append(data)
        output["TR"].append(tr)
        output["accuracy"].append(np.mean(np.array(acc)))
        output["f1micro"].append(np.mean(np.array(f1micro)))
        output["f1macro"].append(np.mean(np.array(f1macro)))


    def evaluate_tr(self, clf, embedding ,tr):
        num_nodes = self.labels.size
        ss = ShuffleSplit(n_splits=10, train_size=tr, random_state=2)
        reflagAcc = []
        reflagF1macro = []
        reflagF1micro = []
        for train_idx, test_idx in ss.split(self.labels):
            X_train, X_test, Y_train, Y_test = embedding[train_idx], embedding[test_idx], \
                                               self.labels[train_idx], self.labels[test_idx]
            pred = self.getPredictions(clf, X_train, X_test, Y_train)
            reflagAcc.append(self._get_accuracy(Y_test, pred))
            reflagF1micro.append(self._get_f1micro(Y_test, pred))
            reflagF1macro.append(self.getF1macro(Y_test, pred))

        # self.addRows(self.dataset, self.output, tr, reflagAcc, reflagF1micro, reflagF1macro)
        # outDf = pd.DataFrame(self.output)
        return reflagAcc, reflagF1micro, reflagF1macro

    def evaluate(self, model, label = False):
        output = {"DATASET": [], "TR": [], "accuracy": [], "f1micro": [], "f1macro":[]}
        embedding = 0
        if label == False:
            if isinstance(model, str):
                embedding = self.get_embeddingDF(model)
            else:
                embedding = self.get_embeddingDF(model)

        clf = self.getclassifier()
        TR = [0.1, 0.3, 0.5]
        for tr in TR:
            print "TR ... ", tr
            if label == True:
                model = "./embeddings/" + self.dataset + "_reflag_label_" + str(int(tr * 100)) + ".emb"
                if isinstance(model, str):
                    embedding = self.get_embeddingDF(model)

            reflagAcc, reflagF1micro, reflagF1macro = self.evaluate_tr(clf, embedding, tr)
            self._add_rows(self.dataset, output, tr, reflagAcc, reflagF1micro, reflagF1macro)
        print "SVC Training Finished"

        print "results"
        outDf = pd.DataFrame(output)
        print outDf

# c_eval = Classification('M10', multilabel=False)
# c_eval.evaluate('embeddings/M10_reflag.emb')
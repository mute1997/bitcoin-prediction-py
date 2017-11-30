import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import os

import preprocessing
from read_csv import read_csv
from model import save_model, resume_model
import utils

filename = './data/preprocessed_data_2014-10-31_to_2017-10-20.csv'
model_name = 'clf.pkl'

def train_data(filename):
    train_X = []
    train_y = []

    return_index = utils.get_return_index(filename)

    # 30日分ずらした配列を作成する
    days = 30
    for i in range(0, int(len(return_index))-days):
        feature = return_index.ix[i:i+days-1]
        if len(feature) != days:
            break

        train_X.append(feature.values)
        # 上がったら1, 下がったら0
        if feature.values[-1] < return_index[i+days]:
            train_y.append(1)
        else:
            train_y.append(0)

    return np.array(train_X), np.array(train_y)

if __name__ == '__main__':
    # 前処理
    if not os.path.exists('./data/train_data.csv') and not os.path.exists('./data/test_data.csv'):
        data = preprocessing.round_data()
        preprocessing.write_data(data)

    # 学習
    train_X, train_y = train_data('./data/train_data.csv')

    # decision tree
    decision_tree_clf = DecisionTreeClassifier()
    decision_tree_clf.fit(train_X, train_y)
    # save_model(decision_tree_clf, 'decision_tree.pkl')

    # SGDClassifier
    sgdclassifier_clf = SGDClassifier()
    sgdclassifier_clf.fit(train_X, train_y)
    # save_model(sgdclassifier_clf, 'sgdclassifier.pkl')

    # SVM
    svm_clf = SVC()
    svm_clf.fit(train_X, train_y)
    # save_model(svm_clf, 'svm.pkl')

    # 評価
    test_X, test_y = train_data('./data/test_data.csv')
    print('SGDClassifier', decision_tree_clf.score(test_X, test_y))
    print('DecisionTreeClassifier', sgdclassifier_clf.score(test_X, test_y))
    print('SVM', svm_clf.score(test_X, test_y))

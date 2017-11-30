import numpy as np
from sklearn.tree import DecisionTreeClassifier
import os

import eval_model
import preprocessing
from read_csv import read_csv
from model import save_model, resume_model
from eval_model import eval_model
import utils

filename = './data/preprocessed_data_2014-10-31_to_2017-10-20.csv'
model_name = 'clf.pkl'

def train_data():
    train_X = []
    train_y = []

    return_index = utils.get_return_index('./data/train_data.csv')

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
    if not os.path.exists('clf.pkl'):
        train_X, train_y = train_data()
        clf = DecisionTreeClassifier()
        clf.fit(train_X, train_y)
        save_model(clf, model_name)

    # 評価
    if os.path.exists('clf.pkl'):
        clf = resume_model(model_name)
        print(eval_model(clf, './data/test_data.csv'))

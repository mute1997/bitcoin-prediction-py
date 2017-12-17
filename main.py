import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import grid_search
from sklearn.svm import SVC
from sklearn import preprocessing as sk_pp
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
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

    # return_index = utils.get_return_index(filename)
    closes = utils.get_closes(filename)
    return_index = (1 + closes.pct_change()).cumprod()
    return_index[0] = 1

    preproced_closes = sk_pp.scale(closes)

    # days分のデータからpredict_days後にrise_price以上上がっているかを学習する
    days = 6 # days - 1日分のデータ
    predict_days = 6 # predict_days + 1日後の予測

    rise_price = 1000
    for i in range(0, int(len(return_index))-days-predict_days):
        train_X.append(preproced_closes[i:i+days-1])
        if closes[i+days-1] + rise_price < closes[i+days+predict_days]:
            # rise price円以上上がったら1
            train_y.append(1)
        else:
            # 同じか上がってないなら0
            train_y.append(0)

    return np.array(train_X), np.array(train_y)


if __name__ == '__main__':
    # 前処理
    if not os.path.exists('./data/train_data.csv') and not os.path.exists('./data/test_data.csv'):
        data = preprocessing.round_data()
        preprocessing.write_data(data)

    # 学習
    # parameters = {'hidden_layer_sizes': [(100,), (100, 10), (100, 100, 10), (1000, 100, 100, 10)]}
    train_X, train_y = train_data('./data/train_data.csv')
    test_X, test_y = train_data('./data/test_data.csv')

    epoch_count = 0
    clf = MLPClassifier(max_iter=1)
    while(True):
        # clf = grid_search.GridSearchCV(MLPClassifier(), parameters)
        clf.fit(train_X, train_y)
        r = clf.score(test_X, test_y)
        epoch_count += 1
        if epoch_count % 10 == 0:
            print('loop ' + str(epoch_count) + ' times')
        if r >= 0.6:
            save_model(clf, model_name)
            break

    clf = resume_model(model_name)
    print('correct_data', test_y)
    print('predict', clf.predict(test_X))
    # print(classification_report(test_y, clf.predict(test_X)))
    print('MLPClassifier', clf.score(test_X, test_y))
    # print('eval', clf.predict([sk_pp.scale([1285054, 1302136, 1324606, 1562165, 2222165, 1889398])]))

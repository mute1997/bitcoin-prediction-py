import json

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from model import save_model, resume_model

filename = 'bitflyer.json'
model_name = 'clf.pkl'

def get_return_index(prices):
    return_index = []
    return_index = (1 + pandas.Series(prices).pct_change()).cumprod()
    return_index[0] = 1
    return return_index

# 取得したデータのAPIドキュメント
# https://cryptowatch.jp/docs/api#ohlc
def get_train_data():
    train_X = []
    train_y = []

    # APIのURLとデータ構造
    # https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=900
    # [ CloseTime, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume ]
    order_book_data = json.load(open(filename, 'r'))

    # 終値のみを取り出す
    prices = []
    for value in order_book_data['result']['900']:
        prices.append(value[1])

    # pricesをreturn indexにする
    return_index = get_return_index(prices)

    # データの生成
    input_amount = 10
    for index in range(0, len(return_index) - input_amount, input_amount):
        x = return_index[index:index+input_amount].tolist()
        y = 1 if x[-1] < return_index[index+input_amount+1] else 0 # 上がってたら1, 下がってる or 同じだったら0
        train_X.append(x)
        train_y.append(y)

    return np.array(train_X), np.array(train_y)

def get_data_and_split():
    X, y = get_train_data()
    ratio = 0.8

    train_X = X[0:int(len(X) * ratio)]
    train_y = y[0:int(len(y) * ratio)]
    test_X = X[int(len(X) * ratio):]
    test_y = y[int(len(y) * ratio):]

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = get_data_and_split()

    clf = MLPClassifier()
    clf.fit(train_X, train_y)

    print('MLPClassifier: %0.2f' % clf.score(test_X, test_y))
    # print('eval', clf.predict([preprocessing.scale([1285054, 1302136, 1324606, 1562165, 2222165, 1889398])]))

    # ----------------------------------------------------
    # 汎化性能を求める
    # これを参考にスコア算出 -> (https://qiita.com/Lewuathe/items/09d07d3ff366e0dd6b24)
    # ----------------------------------------------------
    train_features, train_labels = get_train_data()
    scores = cross_val_score(clf, train_features, train_labels, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # ----------------------------------------------------

import json

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import save_model, resume_model

filename = 'bitflyer.json'
model_name = 'clf.pkl'

def scale_standard(ary):
    sc = StandardScaler()
    sc.fit(ary)
    return sc.transform(ary)

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
    for value in order_book_data['result']['60']:
        prices.append(value[1])
    print('data size:', len(prices))

    # データの生成
    input_amount = 10
    for index in range(0, len(prices) - input_amount, input_amount):
        x = prices[index:index+input_amount]
        y = 1 if x[-1] < prices[index+input_amount+1] else 0 # 上がってたら1, 下がってる or 同じだったら0
        train_X.append(x)
        train_y.append(y)

    return np.array(train_X), np.array(train_y)


if __name__ == '__main__':
    # ----------------------------------------------------
    # 汎化性能を求める
    # これを参考にスコア算出 -> (https://qiita.com/Lewuathe/items/09d07d3ff366e0dd6b24)
    # ----------------------------------------------------
    train_features, train_labels = get_train_data()

    # train_Xとtest_Xの標準化
    train_X = scale_standard(train_features)

    clf = MLPClassifier()
    clf.fit(train_X, train_labels)

    scores = cross_val_score(clf, train_X, train_labels, cv=5)
    print("Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

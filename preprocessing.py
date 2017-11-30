# Timestamp,Open,High,Low,Close,Volume_(BTC),Volume_(Currency),Weighted_Price
# TODO: 移動平均線など他の数値もデータに含める
# 1日ごとに丸める -> テクニカル分析の数値を計算 -> csv吐き出し

import datetime
import csv
from read_csv import read_csv

ratio = 0.7 # 学習用データの割合

def get_prices(ary):
    return {
            'timestamp': datetime.datetime.fromtimestamp(int(ary[0])).strftime('%Y%m%d'),
            'open': int(ary[1]),
            'high': int(ary[2]),
            'low': int(ary[3]),
            'close': int(ary[4])
            }

def round_data():
    data = []
    csv = read_csv('./data/coincheckJPY_1-min_data_2014-10-31_to_2017-10-20.csv')
    i = 0
    while i != len(csv):
        row = get_prices(csv[i])
        date = row['timestamp']
        opens = []
        highs = []
        lows = []
        closes = []

        while date == row['timestamp'] and i != len(csv):
            opens.append(row['open'])
            highs.append(row['high'])
            lows.append(row['low'])
            closes.append(row['close'])

            row = get_prices(csv[i])
            i += 1
        data.append([opens[0], max(highs), min(lows), closes[-1]])
        if i == len(csv):
            break
        row = get_prices(csv[i])
        i += 1
    return data

def write_data(data):
    # 全てのデータの書き出し
    with open('./data/all_data.csv', 'w') as f:
        f.write('open,high,low,close\n')
        for i in data:
            f.write(','.join(list(map(str, i)))+'\n')

    # 学習用データの書き出し
    with open('./data/train_data.csv', 'w') as f:
        f.write('open,high,low,close\n')
        for i in range(0, int(len(data) * ratio)):
            f.write(','.join(list(map(str, data[i])))+'\n')

    # 評価用データの書き出し
    with open('./data/test_data.csv', 'w') as f:
        f.write('open,high,low,close\n')
        for i in range(int(len(data) * ratio)+1, len(data)):
            f.write(','.join(list(map(str, data[i])))+'\n')

# Timestamp,Open,High,Low,Close,Volume_(BTC),Volume_(Currency),Weighted_Price
# TODO: 移動平均線など他の数値もデータに含める
# 1日ごとに丸める -> テクニカル分析の数値を計算 -> csv吐き出し

import datetime

def get_prices(ary):
    try:
        open_ = int(ary[1])
        close = int(ary[4])
        high = int(ary[2])
        low = int(ary[3])
        timestamp = datetime.datetime.fromtimestamp(int(ary[0])).strftime('%Y%m%d')
        return open_, close, high, low, timestamp, 0
    except:
        return -1, -1, -1, -1, -1, -1

def round_data():
    data = []
    data.append(['open', 'high', 'low', 'close'])
    with open('./data/coincheckJPY_1-min_data_2014-10-31_to_2017-10-20.csv', 'r') as f:
        f.readline()
        open_, close, high, low, timestamp, _ = get_prices(f.readline().split(','))
        while _ != -1:
            date = timestamp
            high_ = -9999999
            low_ = 99999999
            open_, _, high, low, timestamp, _ = get_prices(f.readline().split(','))
            while date == timestamp:
                if high_ < high:
                    high_ = high
                if low_ > low:
                    low_ = low
                _, _, high, low, timestamp, _ = get_prices(f.readline().split(','))
                if _ == -1:
                    break

            if _ != -1:
                _, close, _, _, _, _ = get_prices(f.readline().split(','))
                data.append([open_, high_, low_, close])
        return data

def write_data(data):
    with open('./data/preprocessed_data_2014-10-31_to_2017-10-20.csv', 'w') as f:
        for i in data:
            f.write(','.join(list(map(str, i)))+'\n')

if __name__ == '__main__':
    data = round_data()
    write_data(data)

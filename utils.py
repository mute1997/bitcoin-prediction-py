from read_csv import read_csv
import pandas as pd

def get_prices(ary):
    return {'open': int(ary[0]),
            'high': int(ary[1]),
            'low': int(ary[2]),
            'close': int(ary[3])}

def get_closes(filename):
    data = read_csv(filename)

    closes = []
    row = get_prices(data[0])
    for i in data:
        closes.append(get_prices(i)['close'])

    return pd.Series(closes)

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

filename = './data/preprocessed_data_2014-10-31_to_2017-10-20.csv'
ratio = 0.7 # 学習用データの割合

def get_lines():
    with open(filename, 'r') as f:
        return len(f.readlines())

def get_data(ary):
    try:
        open_ = ary[0]
        high = ary[1]
        low = ary[2]
        close = ary[3]
        return open_, high, low, close, 0
    except:
        return -1, -1, -1, -1, -1

def train_data():
    train_X = []
    train_y = []
    with open(filename, 'r') as f:
        f.readline()
        _, _, _, close, _ = get_data(list(map(float, f.readline().replace('\n','').split(','))))
        last_time_price = close

        # 上がってたら1, 下がってたら0, 変わらないなら2
        open_, high, low, close, _ = get_data(list(map(int, f.readline().replace('\n','').split(','))))
        for i in range(0, int(get_lines() * ratio)):
            if _ == -1:
                return np.array(train_X), np.array(train_y)

            train_X.append([open_, high, low, close])
            if last_time_price < close:
                train_y.append(1)
            elif last_time_price == close:
                train_y.append(2)
            else:
                train_y.append(0)

            last_time_price = close
            open_, high, low, close, _ = get_data(list(map(int, f.readline().replace('\n','').split(','))))
    return np.array(train_X), np.array(train_y)

def save_model(clf, filename):
    joblib.dump(clf, filename)

def resume_model(filename):
    return joblib.load(filename)

if __name__ == '__main__':
    model = 'clf.pkl'

    train_X, train_y = train_data()
    clf = DecisionTreeClassifier()
    clf.fit(train_X, train_y)
    save_model(clf, 'clf.pkl')

    data = np.array([[626668,634030,570000,585681]])
    clf = resume_model(model)

    # TODO: modelの評価をする
    print(data)
    print(clf.predict(data))

    print([[84634, 85389, 84542, 85075]])
    print(clf.predict(np.array([[84634, 85389, 84542, 85075]])))

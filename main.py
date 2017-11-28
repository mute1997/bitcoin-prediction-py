import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

filename = './data/coincheckJPY_1-min_data_2014-10-31_to_2017-10-20.csv'
ratio = 0.7 # 学習用データの割合

def get_lines():
    with open(filename, 'r') as f:
        return len(f.readlines())

def train_data():
    train_X = []
    train_y = []
    with open(filename, 'r') as f:
        f.readline()
        last_time_price = list(map(float, f.readline().replace('\n','').split(',')))[4]
        ary = list(map(float, f.readline().replace('\n','').split(',')))

        # 上がってたら1, 下がってたら0, 変わらないなら2
        for i in range(0, int(get_lines() * ratio)):
            train_X.append(ary)
            if last_time_price < ary[4]:
                train_y.append(1)
            elif last_time_price == ary[4]:
                train_y.append(2)
            else:
                train_y.append(0)

            last_time_price = ary[4]
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

    # data = np.array([[1508457540,637559,637997,636547,637169,27.9909433,17844185.365,637498.53565]])
    # clf = resume_model(model)
    # print(clf.predict(np.array(data)))

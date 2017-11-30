from read_csv import read_csv
import utils

def eval_model(clf, filename):
    return_index = utils.get_return_index('./data/test_data.csv')
    count = []

    days = 30
    for i in range(0, int(len(return_index))-days):
        feature = return_index.ix[i:i+days-1]
        if len(feature) != days:
            break

        # 0であったら正解
        # 1は失敗
        result = clf.predict([feature.values])
        if feature.values[-1] < return_index[i+days] and result == 1:
            count.append(0)
        elif feature.values[-1] > return_index[i+days] and result == 0:
            count.append(0)
        else:
            count.append(1)

    return count.count(0) / len(count)


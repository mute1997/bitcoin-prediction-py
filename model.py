from sklearn.externals import joblib

def save_model(clf, filename):
    joblib.dump(clf, filename)

def resume_model(filename):
    return joblib.load(filename)


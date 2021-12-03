from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from transformers import set_seed
import pandas as pd
import numpy as np
import random
from sklearn.dummy import DummyClassifier

set_seed(42)


def extract_argument_quality_scores(europolis, dataset, output):
    effectiveness, reasonableness, cogency, all = [], [], [], []
    for i in range(len(dataset)):
        comment = dataset["cleaned_comment"].values[i]
        match = europolis[europolis.cleaned_comment == comment]

        effectiveness.append(float(match.effectiveness.values[0]))
        reasonableness.append(float(match.reasonableness.values[0]))
        cogency.append(float(match.cogency.values[0]))
        all.append(float(match.overall.values[0]))
    dataset["cogency"] = cogency
    dataset["effectiveness"] = effectiveness
    dataset["reasonableness"] = reasonableness
    dataset["overall"] = all
    dataset.to_csv(output, sep="\t", index=False)


def train_forest(train, test, label):

    train_x = train[["cogency", "effectiveness", "reasonableness", "overall"]]

    test_x = test[["cogency", "effectiveness", "reasonableness", "overall"]]
    train_y = train[label]
    test_y = test[label]
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    probabilities = classifier.predict_proba(test_x)

    importance = classifier.feature_importances_
    feature_dic = {}
    # summarize feature importance
    for i, v in enumerate(importance):
        feature_dic[list(train_x.columns)[i]] = v
    feature_dic = dict(sorted(feature_dic.items(), key=lambda item: item[1]))
    for k, v in feature_dic.items():
        print("feature : %s, importance: %.2f" % (k, v))
    forest_report = classification_report(y_true=test_y, y_pred=y_pred, output_dict=True)
    print(forest_report)
    return forest_report

def random_baseline(train, test, label):
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(train, list(train[label]))
    test_y = list(test[label])
    predictions = dummy_clf.predict(test)
    report = classification_report(y_true=test_y, y_pred=predictions,  output_dict=True)
    print(report)
    return report

def majority_baseline(train, test, label):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train, list(train[label]))
    test_y = list(test[label])
    predictions = dummy_clf.predict(test)
    report = classification_report(y_true=test_y, y_pred=predictions,  output_dict=True)
    print(report)
    return report



if __name__ == '__main__':
    europolis_whole = pd.read_csv(
        "/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi.csv",
        sep="\t")
    print(europolis_whole.columns)

    test_set = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/10_splits_justification/split0/train.csv",
                           sep="\t")
    dir = "/Users/falkne/PycharmProjects/dqi_predictor/5fold"
    fmacros_val = []
    fmacros_test = []
    accuracy_val = []
    accuracy_test = []
    fmacros_val_random = []
    accuracy_val_random = []
    fmacros_test_random = []
    accuracy_test_random = []
    fmacros_val_maj = []
    accuracy_val_maj = []
    fmacros_test_maj = []
    accuracy_test_maj = []
    cl = "label"

    for i in range(0, 5):
        test = pd.read_csv("%s/split%d/test.csv" % (dir, i), sep="\t")
        train = pd.read_csv("%s/split%d/train.csv" % (dir, i), sep="\t")
        val = pd.read_csv("%s/split%d/val.csv" % (dir, i), sep="\t")

        #report = train_run(outfile="tmp", train=train, test=val, label="resp_gr")
        report_val = train_forest(test=val, train=train, label=cl)
        report_test = train_forest(test=test,train=train, label=cl)
        fmacros_val.append(report_val["macro avg"]["f1-score"])
        fmacros_test.append(report_test["macro avg"]["f1-score"])

        accuracy_val.append(report_val["accuracy"])
        accuracy_test.append(report_test["accuracy"])

        report_val_random = random_baseline(test=val, train=train, label=cl)
        report_test_random = random_baseline(test=test, train=train, label=cl)
        fmacros_val_random.append(report_val_random["macro avg"]["f1-score"])
        fmacros_test_random.append(report_test_random["macro avg"]["f1-score"])

        accuracy_val_random.append(report_val_random["accuracy"])
        accuracy_test_random.append(report_test_random["accuracy"])

        report_val_maj = majority_baseline(test=val, train=train, label=cl)
        report_test_maj = majority_baseline(test=test, train=train, label=cl)
        fmacros_val_maj.append(report_val_maj["macro avg"]["f1-score"])
        fmacros_test_maj.append(report_test_maj["macro avg"]["f1-score"])

        accuracy_val_maj.append(report_val_maj["accuracy"])
        accuracy_test_maj.append(report_test_maj["accuracy"])
    print("result for %s random forest: f1 macro val: %.2f / test: %.2f;  accuracy val %.2f / test %.2f" % (cl, np.average(np.array(fmacros_val)), np.average(np.array(fmacros_test)), np.average(np.array(accuracy_val)), np.average(np.array(accuracy_test))))
    print("result for %s random baseline: f1 macro val: %.2f / test: %.2f;  accuracy val %.2f / test %.2f" % (cl, np.average(np.array(fmacros_val_random)), np.average(np.array(fmacros_test_random)), np.average(np.array(accuracy_val_random)), np.average(np.array(accuracy_test_random))))
    print("result for %s majority baseline: f1 macro val: %.2f / test: %.2f;  accuracy val %.2f / test %.2f" % (cl, np.average(np.array(fmacros_val_maj)), np.average(np.array(fmacros_test_maj)), np.average(np.array(accuracy_val_maj)), np.average(np.array(accuracy_test_maj))))

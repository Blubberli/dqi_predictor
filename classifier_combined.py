import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import numpy as np


# gather predictions for all instances

def get_comment2qualityscores():
    path = "/Users/falkne/PycharmProjects/dqi_predictor/10_splits_justification"
    data = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi.csv", sep="\t")
    train = pd.read_csv("%s/split0/train.csv" % (path), sep="\t")
    dev = pd.read_csv("%s/split0/val.csv" % (path), sep="\t")
    test = pd.read_csv("%s/split0/test.csv" % (path), sep="\t")
    merged = pd.concat([train, dev, test])
    unique_ids = ["%s#%s#%s#%s#%s" % (
        merged['UniqueID'].values[i], merged['small_gr'].values[i], merged['nr'].values[i],
        merged['sequence'].values[i],
        merged['name'].values[i]) for i in range(len(merged))]
    merged["ID"] = unique_ids
    cogency, effectiveness, reasonableness, overall = [], [], [], []
    for i in range(len(data)):
        unique_id = "%s#%s#%s#%s#%s" % (
            data['UniqueID'].values[i], data['small_gr'].values[i], data['nr'].values[i],
            data['sequence'].values[i],
            data['name'].values[i])
        cogency.append(merged[merged.ID == unique_id].cogency.values[0])
        effectiveness.append(merged[merged.ID == unique_id].effectiveness.values[0])
        reasonableness.append(merged[merged.ID == unique_id].reasonableness.values[0])
        overall.append(merged[merged.ID == unique_id].overall.values[0])
    merged["cogency"] = cogency
    merged["effectiveness"] = effectiveness
    merged["reasonableness"] = reasonableness
    merged["overall"] = overall
    merged.to_csv("/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi.csv", index=False, sep="\t")


def extract_predictions(test_frames):
    id2predictions = {}
    for i in range(len(test_frames)):
        frame = test_frames[i]
        preds = list(frame["predictions"])
        unique_ids = ["%s#%s#%s#%s#%s" % (
            frame['UniqueID'].values[i], frame['small_gr'].values[i], frame['nr'].values[i],
            frame['sequence'].values[i],
            frame['name'].values[i]) for i in range(len(frame))]
        for j in range(len(unique_ids)):
            p = preds[j]
            p = p.replace("[", "").replace("]", "").split(",")
            p = [float(el) for el in p]
            id2predictions[unique_ids[j]] = p

    return id2predictions


def extract_argument_scores(data, small_data):
    cogency, effectiveness, reasonableness, overall = [], [], [], []
    for i in range(len(small_data)):
        unique_id = small_data.ID.values[i]
        cogency.append(data[data.ID == unique_id].cogency.values[0])
        effectiveness.append(data[data.ID == unique_id].effectiveness.values[0])
        reasonableness.append(data[data.ID == unique_id].reasonableness.values[0])
        overall.append(data[data.ID == unique_id].overall.values[0])
    small_data["cogency"] = cogency
    small_data["effectiveness"] = effectiveness
    small_data["reasonableness"] = reasonableness
    small_data["overall"] = overall
    return small_data


def add_id_col(df):
    unique_ids = ["%s#%s#%s#%s#%s" % (
        df['UniqueID'].values[i], df['small_gr'].values[i], df['nr'].values[i],
        df['sequence'].values[i],
        df['name'].values[i]) for i in range(len(df))]
    df["ID"] = unique_ids
    return df


def train_combined(train, test, label, id2predictions, merged_file):
    # read overall europolis file
    data = pd.read_csv(merged_file, sep="\t")
    # add ID col to training
    train = add_id_col(train)
    # add ID col to test
    test = add_id_col(test)
    # add AQ scores to train and test
    train = extract_argument_scores(data, train)
    test = extract_argument_scores(data, test)
    # add roberta predictions to train and test
    train_predictions = [id2predictions[id] for id in list(train.ID)]
    test_predictions = [id2predictions[id] for id in list(test.ID)]
    train_x = train[["cogency", "effectiveness", "reasonableness", "overall"]]
    # make a single col for each class probability for train
    for i in range(len(train_predictions[0])):
        col = "prediction_class%d" % i
        values = [val[i] for val in train_predictions]
        train_x[col] = values
    # for test
    test_x = test[["cogency", "effectiveness", "reasonableness", "overall"]]
    for i in range(len(test_predictions[0])):
        col = "prediction_class%d" % i
        values = [val[i] for val in test_predictions]
        test_x[col] = values
    train_y = train[label]
    test_y = test[label]
    # train RF on the AQ scores and class probabilities
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


if __name__ == '__main__':
    path = "/Users/falkne/PycharmProjects/dqi_predictor/5fold/jlev_predictions"
    preds = []
    for file in os.listdir(path):
        preds.append(pd.read_csv(path + "/" + file, sep="\t"))
    # train = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/5fold/split0/train.csv", sep="\t")
    # test = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/5fold/split0/test.csv", sep="\t")
    m = "/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi.csv"
    # test_predictions = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/test_df_with_predictions.csv", sep="\t")
    id2preds = extract_predictions(preds)
    fmacros_val = []
    fmacros_test = []
    accuracy_val  = []
    accuracy_test = []
    accuracy = []
    cl = "resp_gr"
    for i in range(0, 5):
        print("split %d" % i)
        train = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/5fold/split%d/train.csv" % i, sep="\t")
        val = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/5fold/split%d/val.csv" % i, sep="\t")
        test = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/5fold/split%d/test.csv" % i, sep="\t")
        report = train_combined(train=train, test=test, label="label", id2predictions=id2preds,
                                merged_file=m)
        report_val = train_combined(train=train, test=val, label=cl, id2predictions=id2preds,
                                merged_file=m)
        report_test = train_combined(train=train, test=test, label=cl, id2predictions=id2preds,
                                merged_file=m)
        fmacros_val.append(report_val["macro avg"]["f1-score"])
        fmacros_test.append(report_test["macro avg"]["f1-score"])

        accuracy_val.append(report_val["accuracy"])
        accuracy_test.append(report_test["accuracy"])

    print("result for %s random forest: f1 macro val: %.2f / test: %.2f;  accuracy val %.2f / test %.2f" % (
    cl, np.average(np.array(fmacros_val)), np.average(np.array(fmacros_test)), np.average(np.array(accuracy_val)),
    np.average(np.array(accuracy_test))))


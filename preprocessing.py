import pandas as pd
import random
import re
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from collections import Counter


def remove_links(comment):
    pattern = re.compile(r"((https?):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)", re.MULTILINE | re.UNICODE)
    comment = re.sub(pattern, "", comment).strip()
    return comment


def remove_empty_lines(comment):
    pattern = re.compile(r"^\s*$")
    comment = re.sub(pattern, "", comment)
    comment = re.sub(" +", " ", comment)
    return comment


def fix_encoding(comment):
    comment = comment.replace("â€", "\"")
    comment = comment.replace("\"™", "\'")
    comment = comment.replace("œ", "")
    comment = comment.replace("\\", "")
    return comment


def remove_tabs(comment):
    return ' '.join(comment.split(sep=None))


def clean_comment(comment):
    comment = fix_encoding(comment)
    comment = comment.replace("\n\n", " ")
    comment = remove_links(comment)
    comment = remove_empty_lines(comment)
    comment = remove_tabs(comment)
    comment = strip_timestamp(comment)
    return comment


def strip_timestamp(comment):
    pattern = re.compile("\[?\d+\s?\:\s?\d+\:?\d+\]")
    comment = re.sub(pattern, "", comment).strip()
    return comment


def make_splits(original_path, save_path, label_col):
    """Create 10 splits such that all data points are covered in test at least once. Use 20% as test data and 15 % as validation data"""
    kf = KFold(n_splits=10)
    data = pd.read_csv(original_path, sep="\t")
    data = data[data[label_col].notna()]
    data = data.sample(frac=1).reset_index(drop=True)
    print("the overall dataset size is %d" % len(data))
    test_size = int(0.2 * len(data))
    val_size = int(0.15 * len(data))
    train_size = len(data) - test_size - val_size
    print("estimated test size : %d; estimated validation size: %d, estimated training size: %d" % (
        test_size, val_size, train_size))
    y = data[label_col]
    x = data.drop([label_col], axis=1)
    split_counter = 0
    for train_index, test_index in kf.split(x):
        additional_test = test_size - len(test_index)
        additional_test_indices = random.sample(list(train_index), k=additional_test)

        final_test_indices = list(test_index) + list(additional_test_indices)
        final_test_indices.sort()
        train_indices = [el for el in train_index if el not in final_test_indices]

        print(
            "the number of overlap indices between training and test is %d" % len(set(final_test_indices).intersection(
                set(train_indices))))

        # extract instances and label
        X_train, X_test = x.loc[train_indices], x.loc[final_test_indices]

        y_train, y_test = y[train_indices], y[final_test_indices]
        # merge label to instances
        train = X_train
        train["label"] = y_train
        test = X_test
        test["label"] = y_test

        # relative validation size
        relative_val_size = val_size / len(train)
        print("the percentage of validation from the left training is %.2f" % relative_val_size)
        train, val = train_test_split(train, test_size=relative_val_size)

        print("length test %d, length val %d lenght train %d" % (len(test), len(val), len(train)))
        print("class distribution in train")
        print(Counter(train["label"]))

        print("class distribution in val")
        print(Counter(val["label"]))

        print("class distribution in test")
        print(Counter(test["label"]))

        train.to_csv("%s/split%d/train.csv" % (save_path, split_counter), sep="\t", index=False)
        test.to_csv("%s/split%d/test.csv" % (save_path, split_counter), sep="\t", index=False)
        val.to_csv("%s/split%d/val.csv" % (save_path, split_counter), sep="\t", index=False)
        split_counter += 1

def create_whole_dataset():
    datasets = []
    for i in range(0, 10):
        train = pd.read_csv("10_splits_justification_backup/split%d/train.csv" % i, sep="\t")
        val = pd.read_csv("10_splits_justification_backup/split%d/val.csv" % i, sep="\t")
        test = pd.read_csv("10_splits_justification_backup/split%d/test.csv" % i, sep="\t")
        datasets.append(train)
        datasets.append(val)
        datasets.append(test)
    merged = pd.concat(datasets)
    df = merged.drop_duplicates(subset=['cleaned_comment'])
    df.to_csv("europolis_dqi.csv", index=False, sep="\t")

def create_kfold(data):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    counter = 0
    labels = ["label", "int1", "jcon", "resp_gr"]
    for train, test in kfold.split(data, data["int1"]):
        print('train: %d, test: %d' % (len(train), len(test)))
        print('train: %.2f, test: %.2f' % (len(train)/len(data), len(test)/len(data)))
        train_set, test_set = data.loc[train], data.loc[test]
        train_set, val_set = train_test_split(train_set, test_size=0.25)
        """
        for l in labels:
            if len(set(data[l].values)) != len(set(val_set[l].values)):
                target = set(data[l].values).difference(set(val_set[l].values))
                print(target)
                for t in target:
                    additional_val_example = train_set[train_set[l] == t].sample(1)
                    train_set = pd.concat([train_set, additional_val_example]).drop_duplicates(keep=False)
                    val_set = pd.concat([val_set, additional_val_example])
        """
        train_set.to_csv("/Users/falkne/PycharmProjects/dqi_predictor/stratifiedINT/split%d/train.csv" % counter, sep="\t", index=False)
        val_set.to_csv("/Users/falkne/PycharmProjects/dqi_predictor/stratifiedINT/split%d/val.csv" % counter, sep="\t", index=False)
        test_set.to_csv("/Users/falkne/PycharmProjects/dqi_predictor/stratifiedINT/split%d/test.csv" % counter, sep="\t", index=False)
        for label in labels:
            freq_test = Counter(test_set[label].values)
            freq_val = Counter(val_set[label].values)
            freq_train = Counter(train_set[label].values)
            with open("/Users/falkne/PycharmProjects/dqi_predictor/stratifiedINT/split%d/train_%s.txt" % (counter, label), "w") as f:
                f.write(str(freq_train))
            with open("/Users/falkne/PycharmProjects/dqi_predictor/stratifiedINT/split%d/val_%s.txt" % (counter, label), "w") as f:
                f.write(str(freq_val))
            with open("/Users/falkne/PycharmProjects/dqi_predictor/stratifiedINT/split%d/test_%s.txt" % (counter, label), "w") as f:
                f.write(str(freq_test))
        counter+=1


        print("train: %d, val: %d, test:%d" % (len(train_set), len(val_set), len(test_set)))
        print('train: %.2f, val:%.2f, test: %.2f' % (len(train_set)/len(data), len(val_set)/len(data), len(test_set)/len(data)))






if __name__ == '__main__':
    #make_splits("/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi.csv", "10_splits_justification", "jlev")
    data = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi.csv", sep="\t")
    create_kfold(data)
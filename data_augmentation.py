import pandas as pd
from collections import Counter
import numpy as np


# import nltk; nltk.download('wordnet')

def compute_number_of_needed_instances(label):
    for i in range(0, 10):
        training_data = pd.read_csv("%s/split%i/train.csv" % ("10_splits_justification", i), sep="\t")
        labels = training_data[label].values
        freq_dic = Counter(labels)
        freq_dic = dict(sorted(freq_dic.items(), key=lambda item: item[1]))
        overall = sum(freq_dic.values())
        s = ""
        for cat, freq in freq_dic.items():
            if cat == 3.0:
                freq = freq * 2
                overall += freq
            s += ("category: %.1f, frequency: %.2f" % (cat, (freq / overall) * 100)) + "\n"
        print(s + "\n")


def prepare_augmented_data(category, target_label):
    augment_dic = {}
    unique_ids = set()
    data = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi.csv", sep="\t")
    for i in range(len(data)):
        unique_id = "%s#%s#%s#%s#%s" % (
            data['UniqueID'].values[i], data['small_gr'].values[i], data['nr'].values[i], data['sequence'].values[i],
            data['name'].values[i])
        comment = data["cleaned_comment"].values[i]
        label = data[category].values[i]
        if label == target_label and unique_id not in unique_ids:
            if len(comment.split(" ")) > 4:
                augment_dic[comment] = unique_id
                unique_ids.add(unique_id)
    d = pd.DataFrame()
    d["sent"] = list(augment_dic.keys())
    d["ID"] = list(augment_dic.values())
    d.to_csv(
        "/Users/falkne/PycharmProjects/dqi_predictor/augmented_data/int/commet2ID_category%s" % target_label, sep="\t",
        index=False)
    to_augment = pd.DataFrame()
    to_augment["label"] = list(augment_dic.values())
    to_augment["sent"] = list(augment_dic.keys())
    print(to_augment)
    to_augment.to_csv(
        "/Users/falkne/PycharmProjects/dqi_predictor/augmented_data/int/augment_int_%s.csv" % target_label, sep="\t",
        index=False, header=False)


def create_augmented_training_data(category, augmented_data, label_col, train_file):
    augmented_data_file = pd.read_csv(augmented_data, sep="\t")
    print(len(augmented_data_file))

    for j in range(0, 5):
        print(j)
        input_file = "%s/split%i/%s" % ("5fold", j, train_file)
        print(input_file)
        training_data = pd.read_csv(input_file, sep="\t")
        print("length before augmentation: %d" % len(training_data))
        target_sentences = []
        for i in range(len(training_data)):
            unique_id = "%s#%s#%s#%s#%s" % (
                training_data['UniqueID'].values[i], training_data['small_gr'].values[i], training_data['nr'].values[i],
                training_data['sequence'].values[i], training_data['name'].values[i])

            target_sents = augmented_data_file[augmented_data_file.iloc[:, 0] == unique_id]
            for i in range(len(target_sents)):
                target_sentences.append(target_sents.iloc[i, 1])
        # print(len(target_sentences))
        labels = [category] * len(target_sentences)
        additional_training = training_data.sample(len(labels))
        additional_training[label_col] = labels
        additional_training["cleaned_comment"] = target_sentences
        additional_training["UniqueID"] = ["AUGMENTED"] * len(labels)
        train = pd.concat([training_data, additional_training])
        print("length after augmentation: %d" % len(train))
        output_file = "%s/split%i/train_augmented_int.csv" % ("5fold", j)
        print(output_file)
        train.to_csv("%s/split%i/train_augmented_int.csv" % ("5fold", j), sep="\t")


def test_augmented():
    for j in range(0, 5):
        print(j)
        training_data = pd.read_csv("%s/split%i/%s" % ("5fold", j, "train_augmented_int.csv"), sep="\t")
        labels = training_data["int1"].values
        print(Counter(labels))


if __name__ == '__main__':
    # 1.0: 3
    # 2.0: 4
    # 3.0: 4
    #compute_number_of_needed_instances("int1")
    #prepare_augmented_data("int1", 1.0)
    create_augmented_training_data(1.0,
                                   "augmented_data/int/augmented_int_1.0.txt",
                                   "int1", "train.csv")
    test_augmented()

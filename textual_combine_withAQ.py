import pandas as pd
from collections import Counter
import seaborn as sns


def score_range():
    data = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi.csv", sep="\t")
    print(data.columns)
    print(data.cogency.describe)
    bin_labels_2 = ['very low', 'low', 'medium-low', 'medium-high', 'high', 'very high']
    data['tercile_cogency'] = pd.qcut(data['cogency'],
                                      q=[0, 1 / 6, 1 / 3, .5, 2 / 3, 5 / 6, 1],
                                      labels=bin_labels_2)
    data['tercile_effectiveness'] = pd.qcut(data['effectiveness'],
                                            q=[0, 1 / 6, 1 / 3, .5, 2 / 3, 5 / 6, 1],
                                            labels=bin_labels_2)
    data['tercile_reasonableness'] = pd.qcut(data['reasonableness'],
                                             q=[0, 1 / 6, 1 / 3, .5, 2 / 3, 5 / 6, 1],
                                             labels=bin_labels_2)
    data['tercile_overall'] = pd.qcut(data['overall'],
                                      q=[0, 1 / 6, 1 / 3, .5, 2 / 3, 5 / 6, 1],
                                      labels=bin_labels_2)

    data.to_csv("/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi_with_terciles.csv", sep="\t", index=False)

def make_aq_col(df, data):
    aq_col = []
    counter = 0
    for j in range(len(df)):
        text = df.cleaned_comment.values[j]
        id = "%s#%s#%s#%s#%s" % (
            df['UniqueID'].values[j], df['small_gr'].values[j], df['nr'].values[j],
            df['sequence'].values[j],
            df['name'].values[j])
        cogency = data[data.ID == id].tercile_cogency.values[0]
        effectiveness = data[data.ID == id].tercile_effectiveness.values[0]
        reasonableness = data[data.ID == id].tercile_reasonableness.values[0]
        overall = data[data.ID == id].tercile_overall.values[0]
        if cogency == effectiveness == reasonableness == overall:
            equal = True
        else:
            equal = False
        if not equal:
            counter += 1
        s = "cogency is %s [SEP] effectiveness is %s [SEP] reasonableness is %s [SEP] overall quality is %s [SEP] %s" % (
            cogency, effectiveness, reasonableness, overall, text)
        aq_col.append(s)
    df["AQ"] = aq_col
    print("different dimensions : %d" % counter)
    return df

def construct_text_col_from_terciles():
    data = pd.read_csv("/Users/falkne/PycharmProjects/dqi_predictor/europolis_dqi_with_terciles.csv", sep="\t")
    print(data.columns)
    for i in range(0, 5):
        train = pd.read_csv("5fold/split%d/train.csv" % i, sep="\t")
        val = pd.read_csv("5fold/split%d/val.csv" % i, sep="\t")
        test = pd.read_csv("5fold/split%d/test.csv" % i, sep="\t")
        train = make_aq_col(df=train, data=data)
        val = make_aq_col(df=val, data=data)
        test = make_aq_col(df=test, data=data)
        train.to_csv("5fold/split%d/train.csv" % i, sep="\t", index=False)
        val.to_csv("5fold/split%d/val.csv" % i, sep="\t", index=False)
        test.to_csv("5fold/split%d/test.csv" % i, sep="\t", index=False)





    print(data.columns)


if __name__ == '__main__':
    #score_range()
    construct_text_col_from_terciles()

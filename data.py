from collections import Counter
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pandas as pd


class ClassificationDataset(torch.utils.data.Dataset):
    """
    This dataset contains the encoded sentences, the labels and the size of the dataset
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_classification_dataset(df, tokenizer, data_args, labels, label_encoder):
    # create encoded sequences using the max sequence length and the given tokenizer. sentences are stored in the column
    # called 'cleaned_comment'
    encodings = tokenizer(df['cleaned_comment'].tolist(),
                          truncation=True, padding=True,
                          max_length=data_args.max_seq_length)
    # labels are transformed with a label encoder such that label can be converted into unique ID back and forth
    labels = label_encoder.transform(labels)
    return ClassificationDataset(encodings=encodings, labels=labels)


def read_data(data_path, classifcation_label):
    """
    Read in a dataset
    :param data_path: the path to the whole europolis dataset
    :param classifcation_label: the label of the dimension that should be classified
    :return: data frame
    """
    print(data_path)
    df = pd.read_csv(data_path, sep="\t")
    df = df[df['cleaned_comment'].notna()]
    df = df[df[classifcation_label].notna()]
    print(f'full dataset after drop {len(df)}')
    return df


def create_label_encoder(all_labels):
    """
    This method creates a label encoder that encodes each label to a unique number
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    return label_encoder


def get_class_weights(training_labels):
    """
    Given a list of labels compute a weight for each label depending on its frequency
    :param training_labels: a list of labels for the training data
    :return: a dictionary that contains a float weight for each label
    """
    freqs = Counter(training_labels)
    n_samples = len(training_labels)
    n_classes = len(set(training_labels))
    weight_dic = {}
    for label, freq in freqs.items():
        weight = n_samples / (n_classes * freq)
        weight_dic[label] = weight
    return weight_dic


def preprocessing(data, tokenizer, text_col, max_len):
    """
    Given a dataset, extract the text column. For each text encode each sentence with a given tokenizer and pad each sentence to max len.
    """
    sentences = data[text_col].values
    encoded_sentences = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len
        )

        encoded_sentences.append(encoded_sent)
    encoded_sentences = pad_sequences(encoded_sentences, maxlen=max_len, dtype="long",
                                      value=0, truncating="post", padding="post")
    return encoded_sentences


def attention_masks(encoded_sentences):
    """Enocode the attention masks"""
    # attention masks, 0 for padding, 1 for actual token
    attention_masks = []
    for sent in encoded_sentences:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


def get_data_loader(data_set, labels, label_encoder, batchsize, tokenizer, max_len, text_col, is_test):
    """
    For a given dataset (e.g. train) and the corresponding labels: Encode the text of that dataset (and pad to max len).
    Encode the labels with the label encoder. create a dataloader that returns batches
    """
    encoded_sentences = preprocessing(data=data_set, tokenizer=tokenizer, max_len=max_len, text_col=text_col)
    inputs = torch.tensor(encoded_sentences)
    labels = torch.tensor(label_encoder.transform(labels))
    masks = torch.tensor(attention_masks(encoded_sentences))

    data = TensorDataset(inputs, masks, labels)
    if is_test:
        sequential_sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sequential_sampler, batch_size=batchsize)
    else:
        dataloader = DataLoader(data, shuffle=True, batch_size=batchsize)

    return dataloader


def get_class_weights_vector(train_labels, num_classes):
    """For a given list of labels return a tensor of class weights (each class is associated with a weight that is based on its frequency)."""
    class_weights = get_class_weights(train_labels)
    c_weights = []
    for i in range(0, num_classes):
        c_weights.append(class_weights[i])
    return torch.tensor(c_weights)

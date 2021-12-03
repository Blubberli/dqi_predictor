import time
import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import BertTokenizer, RobertaTokenizer, XLMRobertaTokenizer
from transformers import BertForSequenceClassification, AdamW, RobertaForSequenceClassification, BertConfig, \
    BertPreTrainedModel, BertModel
from transformers import get_linear_schedule_with_warmup
import re
from sklearn.metrics import classification_report, f1_score

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU in use:', torch.cuda.get_device_name(1))
else:
    print('using the CPU')
    device = torch.device("cpu")

MAX_LEN = 256  # max sequences length
batch_size = 32
columns_mapping = {
    "resp_gr": "LABEL",
    "cleaned_comment": "sentence"
}
labels_encoding = {
    0.0: 0,
    1.0: 1,
    2.0: 2,
    3.0: 3,
    4.0: 4
}
model_path = "./tmp/model.pt"


def get_class_weights(training_labels):
    freqs = Counter(training_labels)
    n_samples = len(training_labels)
    n_classes = len(set(training_labels))
    weight_dic = {}
    for label, freq in freqs.items():
        weight = n_samples / (n_classes * freq)
        weight_dic[label] = weight
    return weight_dic


def preprocessing(df):
    sentences = df.sentence.values
    labels = np.array([labels_encoding[l] for l in list(df.LABEL)])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

    encoded_sentences = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN
        )

        encoded_sentences.append(encoded_sent)
    encoded_sentences = pad_sequences(encoded_sentences, maxlen=MAX_LEN, dtype="long",
                                      value=0, truncating="post", padding="post")
    return encoded_sentences, labels


def attention_masks(encoded_sentences):
    # attention masks, 0 for padding, 1 for actual token
    attention_masks = []
    for sent in encoded_sentences:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


# load the datasets
train = pd.read_csv("10_splits_justification/split0/train.csv", sep="\t")
val = pd.read_csv("10_splits_justification/split0/val.csv", sep="\t")
test = pd.read_csv("10_splits_justification/split0/test.csv", sep="\t")

train.rename(columns=columns_mapping, inplace=True)
val.rename(columns=columns_mapping, inplace=True)
# drop moderator columns
train = train.dropna()
val = val.dropna()
train_encoded_sentences, train_labels = preprocessing(train)
train_attention_masks = attention_masks(train_encoded_sentences)

print("training labels:")
for k, v in Counter(train_labels).items():
    print(k, v)

class_weights = get_class_weights(train_labels)
c_weights = []
for i in range(0, 4):
    c_weights.append(class_weights[i])
c_weights = torch.tensor(c_weights)
val_encoded_sentences, val_labels = preprocessing(val)
val_attention_masks = attention_masks(val_encoded_sentences)

print("validation labels")
for k, v in Counter(val_labels).items():
    print(k, v)


test.rename(columns=columns_mapping, inplace=True)

# test_encoded_sentences_en, test_labels_en = preprocessing(test_en)
# test_attention_masks_en = attention_masks(test_encoded_sentences_en)
# test_encoded_sentences_de, test_labels_de = preprocessing(test_de)
# test_attention_masks_de = attention_masks(test_encoded_sentences_de)
# test_encoded_sentences_fr, test_labels_fr = preprocessing(test_fr)
# test_attention_masks_fr = attention_masks(test_encoded_sentences_fr)

train_inputs = torch.tensor(train_encoded_sentences)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_attention_masks)

validation_inputs = torch.tensor(val_encoded_sentences)
validation_labels = torch.tensor(val_labels)
validation_masks = torch.tensor(val_attention_masks)

# data loader for training
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

# data loader for validation
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

import random

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False,
)

model.cuda()

optimizer = AdamW(model.parameters(),
                  lr=1e-5,
                  eps=1e-8,
                  weight_decay=0.0
                  )

epochs = 20
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # 10% * datasetSize/batchSize
                                            num_training_steps=total_steps)


def compute_accuracy(preds, labels):
    p = np.argmax(preds, axis=1).flatten()
    l = labels.flatten()
    return np.sum(p == l) / len(l)


def run_train(epochs):
    best_model = None
    losses = []
    print(c_weights)
    critereon = torch.nn.CrossEntropyLoss(weight=c_weights.to(device))
    # critereon = torch.nn.BCELoss(weight=c_weights.to(device))
    best_f1 = 0.0
    for e in range(epochs):
        print('======== Epoch {:} / {:} ========'.format(e + 1, epochs))
        start_train_time = time.time()
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):

            if step % 10 == 0:
                elapsed = time.time() - start_train_time
                print(f'{step}/{len(train_dataloader)} --> Time elapsed {elapsed}')

            # input_data, input_masks, input_labels = batch
            input_data = batch[0].to(device)
            input_masks = batch[1].to(device)
            input_labels = batch[2].to(device)

            model.zero_grad()

            # forward propagation
            out = model(input_data,
                        token_type_ids=None,
                        attention_mask=input_masks,
                        labels=input_labels)

            # loss = out[0]
            # total_loss = total_loss + loss.item()
            logits = out[1]

            # backward propagation
            loss = critereon(logits, input_labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 1)

            optimizer.step()

        epoch_loss = total_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f"Training took {time.time() - start_train_time}")

        # Validation
        start_validation_time = time.time()
        model.eval()
        eval_loss, eval_acc, eval_f1 = 0, 0, 0

        for step, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            eval_data, eval_masks, eval_labels = batch
            with torch.no_grad():
                out = model(eval_data,
                            token_type_ids=None,
                            attention_mask=eval_masks)
            logits = out[0]

            #  Uncomment for GPU execution
            logits = logits.detach().cpu().numpy()
            eval_labels = eval_labels.to('cpu').numpy()
            batch_acc = compute_accuracy(logits, eval_labels)
            p = np.argmax(logits, axis=1).flatten()
            l = eval_labels.flatten()
            f1macro = f1_score(y_true=l, y_pred=p, average='macro')

            # Uncomment for CPU execution
            # batch_acc = compute_accuracy(logits.numpy(), eval_labels.numpy())

            eval_acc += batch_acc
            eval_f1 += f1macro

        if eval_f1 > best_f1:
            output_dir = './model_save'

            # if not os.path.exists(output_dir):
            #    os.makedirs(output_dir)
            torch.save(model.state_dict(), model_path)
            best_f1 = eval_f1
            best_model = model

        print(
            f"Accuracy: {eval_acc / (step + 1)} F1 macro {eval_f1 / (step + 1)}, Time elapsed: {time.time() - start_validation_time}")
    return losses, best_model


losses, best_model = run_train(epochs)


# model = model.load_state_dict(torch.load(model_path))


# tokenizer.save_pretrained(output_dir)

def run_test(df_test, model):
    test_encoded_sentences, test_labels = preprocessing(df_test)
    test_attention_masks = attention_masks(test_encoded_sentences)

    test_inputs = torch.tensor(test_encoded_sentences)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_attention_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model.eval()
    eval_loss, eval_acc = 0, 0
    predictions = []
    gold_labels = []
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        eval_data, eval_masks, eval_labels = batch
        with torch.no_grad():
            out = model(eval_data,
                        token_type_ids=None,
                        attention_mask=eval_masks)
        logits = out[0]
        logits = logits.detach().cpu().numpy()
        eval_labels = eval_labels.to('cpu').numpy()
        batch_acc = compute_accuracy(logits, eval_labels)
        p = np.argmax(logits, axis=1).flatten()
        l = eval_labels.flatten()
        predictions.extend(p)
        gold_labels.extend(l)
        eval_acc += batch_acc
    print(f"Accuracy: {eval_acc / (step + 1)}")
    print("classification report:\n")
    print(classification_report(y_true=gold_labels, y_pred=predictions))

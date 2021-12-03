from scipy.stats import pearsonr
import pandas as pd
import torch
import argparse

# Preliminaries

from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig

# Training

import torch.optim as optim
import numpy as np
# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        config = BertConfig.from_pretrained(options_name, num_labels=1)

        self.encoder = BertForSequenceClassification.from_pretrained(options_name, config=config)
        # set this to get regression output
        self.encoder.config.num_labels = 1

    def forward(self, text, label):
        loss, logits = self.encoder(text, labels=label)[:2]

        return loss, logits


def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def r2_score(outputs, labels):
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def pearson_correlation(outputs, labels):
    preds = np.squeeze(outputs, axis=1).tolist()
    corr, pval = pearsonr(preds, labels.tolist())
    return corr


def train(model, optimizer, train_loader, valid_loader, num_epochs, destination_folder,
          best_valid_loss=float("Inf")):
    # initialize running values
    eval_every = len(train_loader) // 2
    running_loss = 0.0
    valid_mean_square_loss = 0.0
    valid_running_r2 = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    best_valid_corr = 0.0
    pearson_val = 0.0
    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (text, labels), _ in train_loader:
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for (text, labels), _ in valid_loader:
                        labels = labels.to(device)
                        text = text.type(torch.LongTensor)
                        text = text.to(device)
                        output = model(text, labels)
                        loss, out = output
                        valid_mean_square_loss += loss.item()
                        valid_running_r2 += r2_score(out, labels)
                        pearson_val += pearson_correlation(out, labels)

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_mean_square_loss / len(valid_loader)
                average_valid_r2 = valid_running_r2 / len(valid_loader)
                average_valid_correlation = pearson_val / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_mean_square_loss = 0.0
                valid_running_r2 = 0.0
                model.train()

                # print progress
                print(
                    'Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid R2 score: {:.4f}, Valid Pearson Correlation: {:.3f}'
                    .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                            average_train_loss, average_valid_loss, average_valid_r2, average_valid_correlation))

                # checkpoint
                if best_valid_corr < average_valid_correlation:
                    # best_valid_loss = average_valid_loss
                    best_valid_corr = average_valid_correlation
                    save_checkpoint(destination_folder + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(destination_folder + '/' + 'metrics.pt', train_loss_list, valid_loss_list,
                                 global_steps_list)

    save_metrics(destination_folder + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


def evaluate(model, test_loader, result_folder):
    y_pred = []
    y_true = []
    y_scores = []
    predictions_path = result_folder + "/predictions.csv"
    report_path = result_folder + "/classification_report.csv"
    model.eval()
    with torch.no_grad():
        for (text, labels), _ in test_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())
            y_scores.extend(torch.sigmoid(output).tolist())
    with open(predictions_path, "w") as f:
        f.write("gold label\tpredicted label\tprobability\n")
        for i in range(len(y_pred)):
            f.write(str(y_true[i]) + "\t" + str(y_pred[i]) + "\t" + str(y_scores[i]) + "\n")
    f.close()
    report = classification_report(y_true, y_pred, labels=[1, 0], digits=2, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(report_path, sep="\t")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder", type=str)
    parser.add_argument("max_seqlen", type=int)
    parser.add_argument("result_folder", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("quality_dim")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Model parameter
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False,
                       batch_first=True,
                       fix_length=args.max_seqlen, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('cleaned_comment', text_field), (args.quality_dim, label_field)]
    # save relevant columns in tsv
    traincsv = pd.read_csv("%s/train.csv" % args.source_folder, sep="\t")
    valcsv = pd.read_csv("%s/val.csv" % args.source_folder, sep="\t")
    testcsv = pd.read_csv("%s/test.csv" % args.source_folder, sep="\t")
    traincsv = traincsv[["cleaned_comment", args.quality_dim]]
    valcsv = valcsv[["cleaned_comment", args.quality_dim]]
    testcsv = testcsv[["cleaned_comment", args.quality_dim]]
    traincsv.to_csv("%s/train.tsv" % args.source_folder, sep="\t", index=False)
    valcsv.to_csv("%s/val.tsv" % args.source_folder, sep="\t", index=False)
    testcsv.to_csv("%s/test.tsv" % args.source_folder, sep="\t", index=False)

    # TabularDataset

    train_data, valid, test = TabularDataset.splits(path=args.source_folder, train='train.tsv',
                                                    validation='val.tsv',
                                                    test='test.tsv', format='TSV', fields=fields,
                                                    skip_header=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Iterators
    train_iter = BucketIterator(train_data, batch_size=8, sort_key=lambda x: len(x.cleaned_comment),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=8, sort_key=lambda x: len(x.cleaned_comment),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=8, device=device, train=False, shuffle=False, sort=False)

    model = BERT().to(device)
    optimizer = AdamW(model.parameters(),
                      lr=5e-5,
                      eps=1e-8)
    training_labels = list(pd.read_csv("%s/train.tsv" % args.source_folder, sep="\t")[args.quality_dim])

    train(model=model, optimizer=optimizer, train_loader=train_iter,
          valid_loader=valid_iter, destination_folder=args.result_folder, num_epochs=args.epochs)
    best_model = BERT().to(device)

    load_checkpoint(args.result_folder + '/model.pt', best_model)

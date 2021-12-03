import numpy as np
import time
import torch
import wandb
from pathlib import Path
from loggin import get_name_with_hyperparams, get_kfold_aggregated_name, check_output_dirs_do_not_exist, reset_wandb_env
from args import parse_arguments
from data import read_data, create_label_encoder, get_data_loader, get_class_weights_vector, \
    create_classification_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, Trainer, EvalPrediction
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU in use:')
else:
    print('using the CPU')
    device = torch.device("cpu")


def get_one_datasplit(data, classification_label, split_rand_state):
    """This methods takes one dataset and returns a training and test set and the corresponding training and test labels. It uses 20% of the data for testing."""
    x, x_test, y, y_test = train_test_split(data, data[classification_label], test_size=0.2, train_size=0.8,
                                            random_state=split_rand_state)
    return x, x_test, y, y_test


def get_train_dev_test_dataloaders(train_dev_data, test_data, label_encoder, data_args, training_args, tokenizer):
    # given trainin(+dev) data and test data, label encoder and arguments, create data loaders for train, val, test
    x_train, x_dev, y_train, y_dev = train_test_split(train_dev_data, train_dev_data[data_args.quality_dim],
                                                      test_size=0.25, train_size=0.75,
                                                      random_state=data_args.split_rand_state)
    y_test = test_data[data_args.quality_dim].values
    train_loader = get_data_loader(data_set=x_train, labels=y_train, label_encoder=label_encoder,
                                   batchsize=training_args.per_device_train_batch_size,
                                   max_len=data_args.max_seq_length, text_col="cleaned_comment", tokenizer=tokenizer,
                                   is_test=False)
    val_loader = get_data_loader(data_set=x_dev, labels=y_dev, label_encoder=label_encoder,
                                 batchsize=training_args.per_device_eval_batch_size,
                                 max_len=data_args.max_seq_length, text_col="cleaned_comment", tokenizer=tokenizer,
                                 is_test=True)
    test_loader = get_data_loader(data_set=test_data, labels=y_test, label_encoder=label_encoder,
                                  batchsize=training_args.per_device_eval_batch_size,
                                  max_len=data_args.max_seq_length, text_col="cleaned_comment", tokenizer=tokenizer,
                                  is_test=True)
    return train_loader, val_loader, test_loader


def run_train_with_trainer(train_dev_data, test_data, label_encoder, data_args, model_args, training_args, fold_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )
    x_train, x_dev, y_train, y_dev = train_test_split(train_dev_data, train_dev_data[data_args.quality_dim],
                                                      test_size=0.25, train_size=0.75,
                                                      random_state=data_args.split_rand_state)
    y_test = test_data[data_args.quality_dim].values

    train_dataset = create_classification_dataset(df=x_train, tokenizer=tokenizer, label_encoder=label_encoder,
                                                  labels=y_train, data_args=data_args)
    dev_dataset = create_classification_dataset(df=x_dev, tokenizer=tokenizer, label_encoder=label_encoder,
                                                labels=y_dev, data_args=data_args)
    test_dataset = create_classification_dataset(df=test_data, tokenizer=tokenizer, label_encoder=label_encoder,
                                                 labels=y_test, data_args=data_args)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                               num_labels=model_args.labels_num)
    name, _, _ = get_name_with_hyperparams(data_args=data_args, model_args=model_args, training_args=training_args,
                                           fold_id=fold_id)
    print(f'{name=}')
    print(f'{training_args.run_name=}')
    print(f'{training_args.output_dir=}')
    reset_wandb_env()
    wandb_run = wandb.init(project=training_args.project_name,
                           group=name,
                           name=training_args.run_name,
                           reinit=True,
    #                       # to fix "Error communicating with wandb process"
    #                       # see https://docs.wandb.ai/guides/track/launch#init-start-error
                           settings=wandb.Settings(start_method="fork"))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )

    train_results = trainer.evaluate(train_dataset)
    print("train results")
    print(train_results)
    print("dev results")
    dev_results = trainer.evaluate(dev_dataset)
    print(dev_results)

    trainer.train()

    print("train results after training")
    train_results = trainer.evaluate(train_dataset)
    print("train results:\n", "\n".join([f"{k}\t{train_results[k]:.2%}" for k in list(train_results.keys())]))
    print("dev results after training")
    dev_results = trainer.evaluate(dev_dataset)
    print("dev results:\n", "\n".join([f"{k}\t{dev_results[k]:.2%}" for k in list(dev_results.keys())]))
    wandb_run.finish()
    return dev_results


def run_train(train_dev_data, test_data, label_encoder, data_args, model_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )
    train_loader, val_loader, test_loader = get_train_dev_test_dataloaders(train_dev_data=train_dev_data,
                                                                           test_data=test_data,
                                                                           label_encoder=label_encoder,
                                                                           data_args=data_args,
                                                                           training_args=training_args,
                                                                           tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                               num_labels=model_args.labels_num)
    model.to(device)
    optimizer = AdamW(model.parameters(),
                      lr=training_args.learning_rate,
                      eps=1e-8,
                      weight_decay=0.0
                      )
    critereon = torch.nn.CrossEntropyLoss(
        weight=get_class_weights_vector(train_dev_data[data_args.quality_dim].values,
                                        num_classes=model_args.labels_num).to(device))
    best_f1 = 0.0
    losses = []
    epochs = int(training_args.num_train_epochs)
    print("epochs %d" % epochs)
    for e in range(epochs):
        print('======== Epoch {:} / {:} ========'.format(e + 1, epochs))
        start_train_time = time.time()
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):

            if step % 10 == 0:
                elapsed = time.time() - start_train_time
                print(f'{step}/{len(train_loader)} --> Time elapsed {elapsed}')

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

        epoch_loss = total_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Training took {time.time() - start_train_time}")

        # Validation
        start_validation_time = time.time()
        model.eval()
        eval_loss, eval_acc, eval_f1 = 0, 0, 0

        for step, batch in enumerate(val_loader):
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
            p = np.argmax(logits, axis=1).flatten()
            l = eval_labels.flatten()
            f1macro = f1_score(y_true=l, y_pred=p, average='macro')
            batch_acc = accuracy_score(y_true=l, y_pred=p)

            # Uncomment for CPU execution
            # batch_acc = compute_accuracy(logits.numpy(), eval_labels.numpy())

            eval_acc += batch_acc
            eval_f1 += f1macro

        if eval_f1 > best_f1:
            torch.save(model.state_dict(), data_args.save_path)
            best_f1 = eval_f1
            best_model = model

        print(
            f"Accuracy: {eval_acc / (step + 1)} F1 macro {eval_f1 / (step + 1)}, Time elapsed: {time.time() - start_validation_time}")
    return losses, best_model


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1).flatten()
    #macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds,
    #                                                                             average='macro')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    print("this method is called")
    print("fmacro is %.2f" % macro_f1)
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }
    return results


if __name__ == '__main__':
    model_args, data_args, training_args = parse_arguments()
    print("model is trained for %s " % data_args.quality_dim)
    # read in the data
    df = read_data(data_args.data_dir, data_args.quality_dim)
    # init label encoder
    label_encoder = create_label_encoder(df[data_args.quality_dim].values)

    if not training_args.folds_num:
        x, x_test, y, y_test = get_one_datasplit(df, data_args.quality_dim, data_args.split_rand_state)

    else:
        # create a number of folds for cross-validation
        skf = StratifiedKFold(n_splits=training_args.folds_num, shuffle=True, random_state=data_args.split_rand_state)
        # create a column to save the models predictions
        df['predictions'] = np.nan
        # create dictionaries to store dev and test set scores
        all_dev_metrics = {}
        all_test_metrics = {}
        # iterate through each fold of the data and train and evaluate
        for fold, (train_dev_idx, test_idx) in enumerate(skf.split(df, df[data_args.quality_dim]), start=1):
            print(f'{fold=}')
            train_dev_data, test_data = df.iloc[train_dev_idx], df.iloc[test_idx]
            # losses, best_model = run_train(train_dev_data=train_dev_data, test_data=test_data,
            #                               data_args=data_args, model_args=model_args, training_args=training_args,
            #                               label_encoder=label_encoder, fold_id=fold)
            dev_results = run_train_with_trainer(train_dev_data=train_dev_data, test_data=test_data,
                                                 data_args=data_args, model_args=model_args,
                                                 training_args=training_args,
                                                 label_encoder=label_encoder, fold_id=fold)
            # save results for this fold
            all_dev_metrics[fold] = dev_results

            total_kfold_run_name, total_kfold_output_dir = get_kfold_aggregated_name(data_args=data_args,
                                                                                     training_args=training_args,
                                                                                     model_args=model_args)
            wandb_run = wandb.init(project=training_args.project_name,
                                   name=total_kfold_run_name,
                                   reinit=True,
                                   # to fix "Error communicating with wandb process"
                                   # see https://docs.wandb.ai/guides/track/launch#init-start-error
                                   settings=wandb.Settings(start_method="fork"))
            wandb_run.log()
            metrics = list(dev_results.keys())
            for fold, m in all_dev_metrics.items():
                print(f'{fold=}')
                print("dev results:\n", "\n".join([f"{k}\t{all_dev_metrics[fold][k]:.2%}"
                                                   for k in metrics]))

            total_kfold_output_dir = Path(total_kfold_output_dir)
            total_kfold_output_dir.mkdir()
            with open(total_kfold_output_dir / 'metrics.json', 'w') as f:
                for k, v in dev_results.items():
                    f.write("%s - %.2f\n" % (k, v))
            wandb_run.finish()

import numpy as np
import time
import torch
import wandb
from pathlib import Path
from loggin import get_name_with_hyperparams, get_kfold_aggregated_name, check_output_dirs_do_not_exist, reset_wandb_env
from args import parse_arguments
from data import read_data, create_label_encoder, create_classification_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import set_seed
from transformers.integrations import WandbCallback
from collections import Counter
import torch.nn.functional as F

# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.trainer import Trainer

# set seed to 42 for reproducibility
set_seed(42)


# check for GPUs or CPU
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


def log_run_info(train_data, dev_data, test_data, data_args, model_args, training_args, loggin_file):
    loggin_file.write("#########traininig args:######")
    loggin_file.write(str(training_args))
    loggin_file.write("#########model args:######")
    loggin_file.write(str(model_args))
    loggin_file.write("#########data args:######")
    loggin_file.write(str(data_args))
    loggin_file.write("class distribution in training")
    loggin_file.write(str(Counter(train_data.labels)))
    loggin_file.write("class distribution in dev")
    loggin_file.write(str(Counter(dev_data.labels)))
    loggin_file.write("class distribution in text")
    loggin_file.write(str(Counter(test_data.labels)))


def run_train_with_trainer(train_data, dev_data, test_data, data_args, model_args, training_args,
                           fold_id, test_csv):
    # initialize classification model
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                               num_labels=model_args.labels_num)
    general_name, run_name, _ = get_name_with_hyperparams(data_args=data_args, model_args=model_args,
                                                          training_args=training_args,
                                                          fold_id=fold_id)
    loggin_file = open(training_args.output_dir + "/" + run_name + ".txt", "w")
    log_run_info(train_data, dev_data, test_data, data_args, model_args, training_args, loggin_file)
    print(f'{run_name=}')
    print(f'{training_args.run_name=}')
    print(f'{training_args.output_dir=}')

    # start wanDB
    reset_wandb_env()
    wandb_run = wandb.init(project=training_args.project_name,
                           group=general_name,
                           name=run_name,
                           reinit=True,
                           #                       # to fix "Error communicating with wandb process"
                           #                       # see https://docs.wandb.ai/guides/track/launch#init-start-error
                           settings=wandb.Settings(start_method="fork"))
    my_callback = WandbCallback()
    general_dir = training_args.output_dir
    split_dir = Path(training_args.output_dir + "/" + run_name)
    if not split_dir.exists():
        split_dir.mkdir()
    training_args.output_dir = split_dir
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        compute_metrics=compute_metrics,
        callbacks=[my_callback]
    )

    train_results = trainer.evaluate(train_dataset)
    loggin_file.write("results on training set without training")
    loggin_file.write(str(train_results))
    loggin_file.write("results on dev set without training")
    loggin_file.write(str(trainer.evaluate(dev_dataset)))

    # train with trainer
    trainer.train()

    loggin_file.write("results on training after training")
    train_results = trainer.evaluate(train_dataset)
    loggin_file.write(str(train_results))

    loggin_file.write("results on dev after training")
    # evaluate on dev set
    dev_results = trainer.evaluate(dev_dataset)
    loggin_file.write(str(dev_results))

    test_result = trainer.evaluate(test_data)
    dev_report = dev_results["eval_report"]
    test_report = test_result["eval_report"]

    wandb_run.finish()
    save_results(output_dir=str(split_dir), report=dev_report,
                 filename=run_name + "_dev_results.txt")
    save_results(output_dir=str(split_dir), report=test_report,
                 filename=run_name + "_test_results.txt")
    prediction_output = trainer.predict(test_dataset)
    #            probs = F.softmax(out, dim=-1)

    test_csv['predictions'] = F.softmax(torch.tensor(prediction_output.predictions), dim=-1).tolist()
    test_csv.to_csv(f'{str(split_dir)}/test_df_with_predictions.csv', index=False, sep="\t")
    training_args.output_dir = general_dir
    loggin_file.close()

    return dev_results, test_result


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1).flatten()
    precision, recall, macro_f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='macro')
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    report = classification_report(y_true=labels, y_pred=preds)
    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        "report": report
    }
    return results


def save_results(output_dir, report, filename):
    with open(output_dir + "/%s" % filename, "w") as f:
        f.write(str(report))
    print("results saved in %s" % filename)


if __name__ == '__main__':
    # read in arguments
    # model args: all classification details
    # data args: path to dataset etc.
    # training args: learning rate, optimizer etc.
    model_args, data_args, training_args = parse_arguments()
    print("model is trained for %s " % data_args.quality_dim)
    # read in the data
    # df = read_data(data_args.data_dir, data_args.quality_dim)
    # init label encoder
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )
    # either we do
    if not training_args.folds_num:
        all_dev_metrics = {}
        all_test_metrics = {}
        for i in range(0, 5):
            train = read_data(data_args.data_dir + "/split%i/train.csv" % i, data_args.quality_dim)
            dev = read_data(data_args.data_dir + "/split%i/val.csv" % i, data_args.quality_dim)
            test = read_data(data_args.data_dir + "/split%i/test.csv" % i, data_args.quality_dim)
            label_encoder = create_label_encoder(train[data_args.quality_dim].values)
            y_train = train[data_args.quality_dim]
            y_dev = dev[data_args.quality_dim]
            y_test = test[data_args.quality_dim]
            train_dataset = create_classification_dataset(df=train, tokenizer=tokenizer, label_encoder=label_encoder,
                                                          labels=y_train, data_args=data_args)
            dev_dataset = create_classification_dataset(df=dev, tokenizer=tokenizer, label_encoder=label_encoder,
                                                        labels=y_dev, data_args=data_args)
            test_dataset = create_classification_dataset(df=test, tokenizer=tokenizer, label_encoder=label_encoder,
                                                         labels=y_test, data_args=data_args)
            dev_results, test_results = run_train_with_trainer(train_data=train_dataset,
                                                               test_data=test_dataset,
                                                               dev_data=dev_dataset,
                                                               data_args=data_args,
                                                               model_args=model_args,
                                                               training_args=training_args,
                                                               fold_id=str(i), test_csv = test)
            all_dev_metrics[str(i)] = dev_results
            all_test_metrics[str(i)] = test_results

        total_kfold_run_name, total_kfold_output_dir = get_kfold_aggregated_name(data_args=data_args,
                                                                                 training_args=training_args,
                                                                                 model_args=model_args)
        wandb_run = wandb.init(project=training_args.project_name,
                               name=total_kfold_run_name,
                               reinit=True,
                               # to fix "Error communicating with wandb process"
                               # see https://docs.wandb.ai/guides/track/launch#init-start-error
                               settings=wandb.Settings(start_method="fork"))
        f1_average = []
        for split, metrics in all_dev_metrics.items():
            f1_average.append(metrics["eval_macro_f1"])
        wandb_run.log({"f1": np.average(f1_average)})

        total_kfold_output_dir = Path(total_kfold_output_dir)
        if not total_kfold_output_dir.exists():
            total_kfold_output_dir.mkdir()
        # save_results(output_dir=total_kfold_output_dir, reports=all_dev_metrics, filename="dev_results.txt")
        # save_results(output_dir=total_kfold_output_dir, reports=all_test_metrics, filename="test_results.txt")
        wandb_run.finish()

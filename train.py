import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.extmath import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput

from moderator_intervention_full.data import create_classification_dataset, ClassificationDataset
from moderator_intervention_full.args import parse_arguments, ModerationType
from util import read_dataset_complete, extract_improve_quality_df


def main():
    model_args, data_args, training_args, debug = parse_arguments()

    #check_output_dirs_do_not_exist(data_args, training_args)

    print(data_args.moderation_type)
    if data_args.moderation_type == ModerationType.ANY_MODERATION:
        df = read_dataset_complete(data_args)
        classification_label = 'MODERATED'
    elif data_args.moderation_type == ModerationType.QUALITY_MODERATION:
        df = extract_improve_quality_df(data_args)
        classification_label = 'MODFORQUALITY'
    else:
        raise ValueError('unsupported type of moderation')

    print(f'full dataset {len(df)}')
    df = df[df['COMMENT'].notna()]
    df = df[df[classification_label].notna()]
    # moderator comments are not moderated, so must be excluded
    df = df[~df['MODERATOR']]
    print(f'full dataset after drop {len(df)}')
    print(df[['type', 'MODERATED']].value_counts())

    if debug:
        df = df.sample(50, random_state=42)

    eval_utils.optimize_threshold = training_args.optimize_threshold

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )

    if not training_args.folds_num:
        # split to train=60% dev=20% test=20%
        x, x_test, y, y_test = train_test_split(df, df[classification_label], test_size=0.2, train_size=0.8,
                                                random_state=data_args.split_rand_state)
        train_one_split(x, x_test, tokenizer, data_args, model_args, training_args, classification_label)

    else:
        skf = StratifiedKFold(n_splits=training_args.folds_num, shuffle=True, random_state=data_args.split_rand_state)
        df['predictions'] = np.nan
        all_dev_metrics = {}
        all_test_metrics = {}
        for fold, (train_dev_idx, test_idx) in enumerate(skf.split(df, df[classification_label]), start=1):
            print(f'{fold=}')
            dev_metrics, test_prediction_output = train_one_split(df.iloc[train_dev_idx], df.iloc[test_idx], tokenizer,
                                                                  data_args, model_args, training_args,
                                                                  classification_label,
                                                                  fold_id=fold)
            df['predictions'].iloc[test_idx] = pd.Series(test_prediction_output.predictions.tolist())
            all_test_metrics[fold] = test_prediction_output.metrics
            all_dev_metrics[fold] = dev_metrics

        # log accumulated results
        total_kfold_run_name, total_kfold_output_dir = get_kfold_output_parameters_based_on_naming_convention(
            data_args, training_args)
        wandb_run = wandb.init(project=training_args.project_name,
                               name=total_kfold_run_name,
                               reinit=True,
                               # to fix "Error communicating with wandb process"
                               # see https://docs.wandb.ai/guides/track/launch#init-start-error
                               settings=wandb.Settings(start_method="fork"))
        for fold, m in all_test_metrics.items():
            print(f'{fold=}')
            print("dev results:\n", "\n".join([f"{k}\t{all_dev_metrics[fold]['eval_' + k]:.2%}"
                                               for k in eval_utils.metrics]))
            print("test results:\n", "\n".join([f"{k}\t{m['test_' + k]:.2%}"
                                                for k in eval_utils.metrics]))
        # wandb_run.log()
        total_kfold_output_dir = Path(total_kfold_output_dir)
        total_kfold_output_dir.mkdir()
        with open(total_kfold_output_dir / 'metrics.json', 'w') as f:
            json.dump({'dev': all_dev_metrics, 'test': all_test_metrics}, f)
        df.to_csv(total_kfold_output_dir / 'df_with_predictions.csv')
        wandb_run.save(str(total_kfold_output_dir / 'df_with_predictions.csv'))
        wandb_run.finish()


def train_one_split(x, x_test, tokenizer, data_args, model_args, training_args, label, fold_id=None) -> tuple[
    dict[str, float], PredictionOutput]:
    x_train, x_dev, y_train, y_dev = train_test_split(x, x[label],
                                                      test_size=0.25, train_size=0.75,
                                                      random_state=data_args.split_rand_state)

    train_dataset = create_classification_dataset(x_train, tokenizer, data_args, label)
    dev_dataset = create_classification_dataset(x_dev, tokenizer, data_args, label)
    test_dataset = create_classification_dataset(x_test, tokenizer, data_args, label)

    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                               num_labels=model_args.labels_num)

    # run_name is set in __post_init__ so must be overwrite even if it's the same as output_dir
    group, training_args.run_name, training_args.output_dir = get_fold_output_parameters_based_on_naming_convention(
        data_args, training_args, fold_id)
    print(f'{group=}')
    print(f'{training_args.run_name=}')
    print(f'{training_args.output_dir=}')
    reset_wandb_env()
    wandb_run = wandb.init(project=training_args.project_name,
                           group=group if fold_id else None,
                           name=training_args.run_name,
                           reinit=True,
                           # to fix "Error communicating with wandb process"
                           # see https://docs.wandb.ai/guides/track/launch#init-start-error
                           settings=wandb.Settings(start_method="fork"))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )

    # evaluate before training
    results = evaluate(trainer, train_dataset, DatasetType.TRAIN)
    print("train results:\n", "\n".join([f"{k}\t{results['eval_' + k]:.2%}" for k in eval_utils.metrics]))
    results = evaluate(trainer, dev_dataset, DatasetType.DEV)
    print("dev results:\n", "\n".join([f"{k}\t{results['eval_' + k]:.2%}" for k in eval_utils.metrics]))

    trainer.train()

    results = evaluate(trainer, train_dataset, DatasetType.TRAIN)
    print("train results:\n", "\n".join([f"{k}\t{results['eval_' + k]:.2%}" for k in eval_utils.metrics]))
    results = evaluate(trainer, dev_dataset, DatasetType.DEV)
    print("dev results:\n", "\n".join([f"{k}\t{results['eval_' + k]:.2%}" for k in eval_utils.metrics]))
    eval_utils.dataset_type = DatasetType.TEST
    prediction_output = trainer.predict(test_dataset)
    print("test results:\n", "\n".join([f"{k}\t{prediction_output.metrics['test_' + k]:.2%}" for k in eval_utils.metrics]))
    x_test['predictions'] = prediction_output.predictions.tolist()
    x_test.to_csv(f'{training_args.output_dir}/test_df_with_predictions.csv')
    wandb_run.finish()
    return results, prediction_output


def check_output_dirs_do_not_exist(data_args, training_args):
    def assert_dir(path):
        assert not Path(path).exists(), f'output dir {path} already exists!'

    if training_args.folds_num:
        for fold_id in range(training_args.folds_num):
            assert_dir(get_fold_output_parameters_based_on_naming_convention(data_args, training_args, fold_id)[-1])
        assert_dir(get_kfold_output_parameters_based_on_naming_convention(data_args, training_args)[-1])
    else:
        assert_dir(get_fold_output_parameters_based_on_naming_convention(data_args, training_args)[-1])


def get_fold_output_parameters_based_on_naming_convention(data_args, training_args, fold_id=None):
    group = ((training_args.output_dir_prefix +
              f"-seqlen{data_args.max_seq_length}"
              f"-use-parent{data_args.comment_parents_num}"
              f"-lr{training_args.learning_rate}") if training_args.output_dir_prefix else training_args.output_dir)
    run_name = group + (f"-fold{fold_id}" if fold_id else f"-rand{data_args.split_rand_state}")
    return group, run_name, training_args.project_name + '/' + run_name


def get_kfold_output_parameters_based_on_naming_convention(data_args, training_args):
    run_name = get_fold_output_parameters_based_on_naming_convention(data_args, training_args)[0] + '-kfold'
    return run_name, training_args.project_name + '/' + run_name


class DatasetType(Enum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'


@dataclass
class EvalUtils:
    optimize_threshold: bool
    dataset_type: DatasetType
    threshold: float
    metrics: list


# a hack to pass parameters from/to compute_metrics
eval_utils = EvalUtils(False, None, 0, None)


def evaluate(trainer: Trainer, dataset: ClassificationDataset, dataset_type: DatasetType):
    eval_utils.dataset_type = dataset_type
    return trainer.evaluate(dataset)


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    if eval_utils.optimize_threshold:
        prob = softmax(pred.predictions)[:, 1]
        # calculate optimal threshold on train and use for dev/test
        # given the small dataset size would be more stable than calculating on dev
        if eval_utils.dataset_type == DatasetType.TRAIN:
            eval_utils.threshold = threshold_search(labels, prob)['threshold']
        preds = prob > eval_utils.threshold
    else:
        preds = pred.predictions.argmax(-1)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    results = {
        'accuracy': acc,
        'balanced_accuracy':  balanced_accuracy_score(labels, preds),
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'pos_f1': pos_f1,
        'pos_precision': pos_precision,
        'pos_recall': pos_recall
    }
    if eval_utils.optimize_threshold:
        results['threshold'] = eval_utils.threshold
    eval_utils.metrics = list(results.keys())
    return results


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(10, 90)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


# from https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-cross-validation/train-cross-validation.py
# this messes up sweeps configuration if not done with parallelization it seems
# also not all args are logged as config
# but keeping this as a simple hack for now
# otherwise wandb.init just renames the previous run and continues logging there
def reset_wandb_env():
    print('resetting wandb env')
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            print(k, v)
            del os.environ[k]
    # force no watching. seems to have been reset somehow. makes saving slower
    os.environ['WANDB_WATCH'] = 'false'


if __name__ == '__main__':
    main()

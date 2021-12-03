from pathlib import Path
import os

def check_output_dirs_do_not_exist(data_args, training_args, model_args):
    def assert_dir(path):
        # assert whether a given path exists
        assert not Path(path).exists(), f'output dir {path} already exists!'

    if training_args.folds_num:
        for fold_id in range(training_args.folds_num):
            assert_dir(get_name_with_hyperparams(data_args, training_args, model_args, fold_id)[-1])
        assert_dir(get_kfold_aggregated_name(data_args, training_args, model_args)[-1])


def get_name_with_hyperparams(data_args, training_args, model_args, fold_id):
    group = ((training_args.output_dir_prefix +
              f"-seqlen{data_args.max_seq_length}"
              f"-batchsize{training_args.per_device_train_batch_size}"
              f"-model{model_args.model_name_or_path}"
              f"f-quality{data_args.quality_dim}"
              f"-lr{training_args.learning_rate}"))
    run_name = group + (f"-fold{fold_id}")
    return group, run_name, training_args.project_name + '/' + run_name


def get_kfold_aggregated_name(data_args, training_args, model_args):
    run_name = get_name_with_hyperparams(data_args, training_args, model_args, "kfold")[0] + '-kfold'
    return run_name, training_args.project_name + '/' + run_name

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

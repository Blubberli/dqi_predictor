# based on transformers/examples/pytorch/text-classification/run_glue.py

import os
import sys
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments
from transformers.file_utils import ExplicitEnum


class QualityDimension(ExplicitEnum):
    JUSTIFICATION = "jlev"
    COMMON_GOOD = "jcon"
    RESPECT_GROUP = "resp_gr"
    INTERACTIVITY = "int1"
    JUST = "label"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    quality_dim: QualityDimension = field(
        metadata={
            "help": "The quality dimension that should be predicted."
        }
    )
    data_dir: Optional[str] = field(
        default=str('europolis.csv')
    )
    save_path: Optional[str] = field(
        default=str('model.pt')
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    split_rand_state: Optional[int] = field(
        default=42,
        metadata={
            "help": "The random state to split the data according to"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    labels_num: Optional[int] = field(
        default=2, metadata={"help": "number of labels in the model output"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )


@dataclass
class KfoldTrainingArguments(TrainingArguments):
    class_weights: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use class weights for imbalanced data."
        },
    )
    folds_num: Optional[int] = field(
        default=None,
        metadata={"help": "The number of folds."},
    )
    output_dir_prefix: Optional[str] = field(
        default=None,
        metadata={"help": "Output path prefix"}
    )
    project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Project name in wandb"}
    )



def parse_arguments():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, KfoldTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    for x in (model_args, data_args, training_args):
        pprint(x)
    return model_args, data_args, training_args

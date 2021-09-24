import json
import subprocess
from dataclasses import dataclass
from typing import List

from torch.utils.data import random_split
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainerCallback,
)

from chemberta.utils.data_collators import multitask_data_collator
from chemberta.utils.raw_text_dataset import (
    LazyRegressionDataset,
    RawTextDataset,
    RegressionTextDataset,
)
from chemberta.utils.roberta_regression import RobertaForRegression


def create_trainer(
    model_type,
    config,
    training_args,
    dataset_args,
    callbacks: List,
    pretrained_model=None,
):
    tokenizer = RobertaTokenizerFast.from_pretrained(
        dataset_args.tokenizer_path,
    )

    if model_type == "mlm":
        dataset_class = RawTextDataset
        dataset = dataset_class(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_max_length,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=dataset_args.mlm_probability
        )
        model = RobertaForMaskedLM

    elif model_type == "regression":
        dataset_class = RegressionTextDataset
        dataset = dataset_class(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_max_length,
        )

        with open(dataset_args.normalization_path) as f:
            normalization_values = json.load(f)

        config.num_labels = dataset.num_labels
        config.norm_mean = normalization_values["mean"]
        config.norm_std = normalization_values["std"]
        model = RobertaForRegression

        data_collator = multitask_data_collator

    elif model_type == "classification":
        dataset_class = RegressionTextDataset
        dataset = dataset_class(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_max_length,
        )

        config.num_labels = dataset.num_labels
        model = RobertaForSequenceClassification

        data_collator = multitask_data_collator

    else:
        raise ValueError(model_type)

    if pretrained_model:
        model = model.from_pretrained(
            pretrained_model, config=config, use_auth_token=True
        )
    else:
        model = model(config=config)

    train_dataset, eval_dataset = get_dataset_splits(
        dataset, dataset_class, dataset_args, tokenizer
    )

    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )


def get_hyperopt_trainer(
    model_type, config, training_args, dataset_args, callbacks: List
):

    tokenizer = RobertaTokenizerFast.from_pretrained(
        dataset_args.tokenizer_path, max_len=dataset_args.max_tokenizer_len
    )

    if model_type == "mlm":
        dataset = RawTextDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_max_length,
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=dataset_args.mlm_probability
        )

        def model_init_fn():
            model = RobertaForMaskedLM(config=config)
            return model

        model_init_callable = model_init_fn

    elif model_type == "regression":
        dataset = RegressionTextDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_max_length,
        )

        with open(dataset_args.normalization_path) as f:
            normalization_values = json.load(f)

        config.num_labels = dataset.num_labels
        config.norm_mean = normalization_values["mean"]
        config.norm_std = normalization_values["std"]

        def model_init_fn():
            model = RobertaForRegression(config=config)
            return model

        model_init_callable = model_init_fn

        data_collator = multitask_data_collator

    elif model_type == "regression_lazy":
        dataset = LazyRegressionDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_max_length,
        )

        with open(dataset_args.normalization_path) as f:
            normalization_values = json.load(f)

        config.num_labels = dataset.num_labels
        config.norm_mean = normalization_values["mean"]
        config.norm_std = normalization_values["std"]
        model = RobertaForRegression(config=config)

        def model_init_fn():
            model = RobertaForRegression(config=config)
            return model

        model_init_callable = model_init_fn

        data_collator = multitask_data_collator

    else:
        raise ValueError(model_type)

    train_dataset, eval_dataset = get_train_test_split(dataset, dataset_args.frac_train)

    return Trainer(
        model_init=model_init_callable,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )


@dataclass
class DatasetArguments:
    dataset_path: str
    normalization_path: str
    frac_train: float
    eval_path: str
    tokenizer_path: str
    tokenizer_max_length: int
    mlm_probability: float


def get_dataset_splits(dataset, dataset_class, dataset_args, tokenizer):
    if dataset_args.eval_path:
        train_dataset = dataset
        eval_dataset = dataset_class(
            tokenizer=tokenizer,
            file_path=dataset_args.eval_path,
            block_size=dataset_args.tokenizer_max_length,
        )

    else:
        print("No eval set provided, splitting data into train and eval")
        train_dataset, eval_dataset = create_train_test_split(
            dataset, dataset_args.frac_train
        )

    return train_dataset, eval_dataset


def create_train_test_split(dataset, frac_train):
    train_size = max(int(frac_train * len(dataset)), 1)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    return train_dataset, eval_dataset


class AwsS3Callback(TrainerCallback):
    def __init__(self, local_directory, s3_directory):
        self.local_directory = local_directory
        self.s3_directory = s3_directory

    def on_evaluate(self, args, state, control, **kwargs):
        # sync local and remote directories
        subprocess.check_call(
            [
                "aws",
                "s3",
                "sync",
                self.local_directory,
                self.s3_directory,
                "--acl",
                "bucket-owner-full-control",
                "--delete",
            ]
        )
        return

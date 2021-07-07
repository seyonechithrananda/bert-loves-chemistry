import json
import os
from dataclasses import dataclass
from typing import List

from chemberta.utils.data_collators import multitask_data_collator
from chemberta.utils.raw_text_dataset import (LazyRegressionDataset,
                                              RawTextDataset,
                                              RegressionTextDataset)
from chemberta.utils.roberta_regression import \
    RobertaForRegression  # RobertaForSequenceClassification,
from nlp.features import string_to_arrow
from torch.utils.data import random_split
from transformers import (DataCollatorForLanguageModeling, RobertaConfig,
                          RobertaForMaskedLM, RobertaForSequenceClassification,
                          RobertaTokenizerFast, Trainer, TrainingArguments)
from transformers.data.data_collator import default_data_collator


def create_trainer(
    model_type,
    config,
    training_args,
    dataset_args,
    callbacks: List,
    pretrained_model=None,
):
    print(dataset_args.tokenizer_path)
    print(dataset_args.max_tokenizer_len)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        dataset_args.tokenizer_path, max_len=dataset_args.max_tokenizer_len
    )

    if model_type == "mlm":

        dataset_class = RawTextDataset
        dataset = dataset_class(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_block_size,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=dataset_args.mlm_probability
        )
        model = RobertaForMaskedLM

    elif model_type == "regression":
        dataset = RegressionTextDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_block_size,
        )

        with open(dataset_args.normalization_path) as f:
            normalization_values = json.load(f)

        config.num_labels = dataset.num_labels
        config.norm_mean = normalization_values["mean"]
        config.norm_std = normalization_values["std"]
        model = RobertaForRegression

        data_collator = multitask_data_collator

    elif model_type == "regression_lazy":
        dataset = LazyRegressionDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_block_size,
        )

        with open(dataset_args.normalization_path) as f:
            normalization_values = json.load(f)

        config.num_labels = dataset.num_labels
        config.norm_mean = normalization_values["mean"]
        config.norm_std = normalization_values["std"]
        model = RobertaForRegression

        data_collator = multitask_data_collator

    elif model_type == "classification":
        dataset = RegressionTextDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_block_size,
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

    if dataset_args.eval_path:
        train_dataset = dataset
        eval_dataset = dataset_class(
            tokenizer=tokenizer,
            file_path=dataset_args.eval_path,
            block_size=dataset_args.tokenizer_block_size,
        )

    else:
        print("No eval set provided, splitting data into train and eval")
        train_dataset, eval_dataset = get_train_test_split(
            dataset, dataset_args.frac_train
        )

    return Trainer(
        model=model,
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
    max_tokenizer_len: int
    tokenizer_block_size: int
    mlm_probability: float


def get_train_test_split(dataset, frac_train):
    train_size = max(int(frac_train * len(dataset)), 1)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    return train_dataset, eval_dataset

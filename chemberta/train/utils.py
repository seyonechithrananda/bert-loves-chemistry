from dataclasses import dataclass

from chemberta.utils.data_collators import multitask_data_collator
from chemberta.utils.raw_text_dataset import RawTextDataset, RegressionDataset, LazyRegressionDataset
from chemberta.utils.roberta_regression import RobertaForRegression
from nlp.features import string_to_arrow
from torch.utils.data import random_split
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import EarlyStoppingCallback
import json


def create_trainer(model_type, config, training_args, dataset_args):
    print(dataset_args.tokenizer_path)
    print(dataset_args.max_tokenizer_len)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        dataset_args.tokenizer_path, max_len=dataset_args.max_tokenizer_len
    )

    if model_type == "mlm":
        dataset = RawTextDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_block_size,
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=dataset_args.mlm_probability
        )
        model = RobertaForMaskedLM(config=config)

    elif model_type == "regression":
        dataset = RegressionDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_block_size,
        )

        with open(dataset_args.normalization_path) as f:
            normalization_values = json.load(f)

        config.num_labels = dataset.num_labels
        config.norm_mean = normalization_values["mean"]
        config.norm_std = normalization_values["std"]
        model = RobertaForRegression(config=config)

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
        model = RobertaForRegression(config=config)

        data_collator = multitask_data_collator

    else:
        raise ValueError(model_type)

    train_dataset, eval_dataset = get_train_test_split(dataset, dataset_args.frac_train)

    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )


@dataclass
class DatasetArguments:
    dataset_path: str
    normalization_path: str
    frac_train: float
    tokenizer_path: str
    max_tokenizer_len: int
    tokenizer_block_size: int
    mlm_probability: float


def get_train_test_split(dataset, frac_train):
    train_size = max(int(frac_train * len(dataset)), 1)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    return train_dataset, eval_dataset

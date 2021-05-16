from dataclasses import dataclass

from chemberta.utils.data_collators import multitask_data_collator
from chemberta.utils.raw_text_dataset import RawTextDataset, RegressionDataset
from chemberta.utils.roberta_regression import RobertaForRegression
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


def create_trainer(model_type, config, training_args, dataset_args):
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

    if model_type == "regression":
        dataset = RegressionDataset(
            tokenizer=tokenizer,
            file_path=dataset_args.dataset_path,
            block_size=dataset_args.tokenizer_block_size,
        )
        config.num_labels = dataset.num_labels
        config.norm_mean = dataset.norm_mean
        config.norm_std = dataset.norm_std
        model = RobertaForRegression(config=config)

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
    tokenizer_path: str
    tokenizer_len: int
    dataset_path: str
    normalization_path: str
    tokenizer_block_size: int
    mlm_probability: float
    frac_train: float


def get_train_test_split(dataset, frac_train):
    train_size = max(int(frac_train * len(dataset)), 1)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    return train_dataset, eval_dataset

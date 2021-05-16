""" Script for training a Roberta Model (mlm or regression)

Usage:
    python train_roberta.py --dataset_path=<DATASET_PATH> --model_name=<MODEL_NAME>
"""

import os
from dataclasses import dataclass

import pandas as pd
import torch
import transformers
from absl import app, flags
from chemberta.utils.data_collators import multitask_data_collator
from chemberta.utils.raw_text_dataset import (
    RawTextDataset,
    RegressionDataset,
    RegressionDatasetLazy,
)
from chemberta.utils.roberta_regression import RobertaForRegression
from nlp.load import DATASETS_PATH
from tokenizers import ByteLevelBPETokenizer
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

FLAGS = flags.FLAGS

# RobertaConfig params
flags.DEFINE_integer(name="vocab_size", default=512, help="")
flags.DEFINE_integer(name="max_position_embeddings", default=515, help="")
flags.DEFINE_integer(name="num_attention_heads", default=6, help="")
flags.DEFINE_integer(name="num_hidden_layers", default=6, help="")
flags.DEFINE_integer(name="type_vocab_size", default=1, help="")

# Tokenizer params
flags.DEFINE_string(
    name="tokenizer_path",
    default="seyonec/SMILES_tokenized_PubChem_shard00_160k",
    help="",
)
flags.DEFINE_integer(name="max_tokenizer_len", default=512, help="")
flags.DEFINE_integer(name="tokenizer_block_size", default=512, help="")

# Dataset params
flags.DEFINE_string(name="dataset_path", default=None, help="")
flags.DEFINE_string(name="model_name", default="PubChem_10M_SMILES_Tokenizer", help="")

# MLM params
flags.DEFINE_float(
    name="mlm_probability", default=0.15, lower_bound=0.0, upper_bound=1.0, help=""
)

# Regression params
flags.DEFINE_float(name="normalization_path", default=None, help="")

# Train params
flags.DEFINE_float(name="frac_train", default=0.95, help="")
flags.DEFINE_integer(name="eval_steps", default=10, help="")
flags.DEFINE_integer(name="logging_steps", default=10, help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_integer(name="num_train_epochs", default=100, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=64, help="")
flags.DEFINE_integer(name="save_steps", default=100, help="")
flags.DEFINE_integer(name="save_total_limit", default=2, help="")

flags.mark_flag_as_required("dataset_path")


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
    train_size = max(int(FLAGS.frac_train * len(dataset)), 1)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    return train_dataset, eval_dataset


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


def main(argv):
    model_config = RobertaConfig(
        vocab_size=FLAGS.vocab_size,
        max_position_embeddings=FLAGS.max_position_embeddings,
        num_attention_heads=FLAGS.num_attention_heads,
        num_hidden_layers=FLAGS.num_hidden_layers,
        type_vocab_size=FLAGS.type_vocab_size,
        is_gpu=torch.cuda.is_available(),
    )

    dataset_args = DatasetArguments(
        FLAGS.dataset_path,
        FLAGS.normalization_path,
        FLAGS.frac_train,
        FLAGS.tokenizer_path,
        FLAGS.tokenizer_len,
        FLAGS.tokenizer_block_size,
        FLAGS.mlm_probability,
    )

    training_args = TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=FLAGS.eval_steps,
        logging_steps=FLAGS.logging_steps,
        load_best_model_at_end=True,
        output_dir=FLAGS.model_name,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        save_steps=FLAGS.save_steps,
        save_total_limit=FLAGS.save_total_limit,
        fp16=torch.cuda.is_available(),  # fp16 only works on CUDA devices
    )

    trainer = create_trainer(
        FLAGS.model_type, model_config, training_args, dataset_args
    )
    trainer.train()
    trainer.save_model(FLAGS.model_name)


if __name__ == "__main__":
    app.run(main)

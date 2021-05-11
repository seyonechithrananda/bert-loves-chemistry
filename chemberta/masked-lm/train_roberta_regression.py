""" Script for training a Roberta Regression Model (single or multi-task)

Usage:
    python train_roberta_regression.py --dataset_path=<DATASET_PATH> --output_dir=<OUTPUT_DIR> --model_name=<MODEL_NAME> --tokenizer_type=<smiles/bpe>

"""

import torch
import pandas as pd

import os
from absl import app
from absl import flags

import transformers

import torch
from torch.utils.data import random_split
from transformers.trainer_callback import EarlyStoppingCallback
import wandb
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM

from chemberta.utils.roberta_regression import RobertaForRegression
from chemberta.utils.data_collators import multitask_data_collator
from chemberta.utils.raw_text_dataset import RawTextDataset, RegressionDataset, RegressionDatasetLazy

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from tokenizers import ByteLevelBPETokenizer

FLAGS = flags.FLAGS

# RobertaConfig params
flags.DEFINE_integer(name="vocab_size", default=512, help="")
flags.DEFINE_integer(name="max_position_embeddings", default=515, help="")
flags.DEFINE_integer(name="num_attention_heads", default=6, help="")
flags.DEFINE_integer(name="num_hidden_layers", default=6, help="")
flags.DEFINE_integer(name="type_vocab_size", default=1, help="")

# Tokenizer params
flags.DEFINE_enum(name="tokenizer_type", default="smiles", enum_values=["smiles", "bpe", "SMILES", "BPE"], help="")
flags.DEFINE_string(name="tokenizer_path", default="seyonec/SMILES_tokenized_PubChem_shard00_160k", help="")
flags.DEFINE_integer(name="BPE_min_frequency", default=2, help="")
flags.DEFINE_string(name="output_tokenizer_dir", default="tokenizer_dir", help="")
flags.DEFINE_integer(name="max_tokenizer_len", default=512, help="")
flags.DEFINE_integer(name="tokenizer_block_size", default=512, help="")


# Dataset params
flags.DEFINE_string(name="dataset_path", default=None, help="")
flags.DEFINE_string(name="output_dir", default="PubChem_10M_SMILES_Tokenizer", help="")
flags.DEFINE_string(name="model_name", default="PubChem_10M_SMILES_Tokenizer", help="")

# MLM params
flags.DEFINE_float(name="mlm_probability", default=0.15, lower_bound=0.0, upper_bound=1.0, help="")

# Train params
flags.DEFINE_float(name="frac_train", default=0.95, help="")
flags.DEFINE_integer(name="eval_steps", default=10, help="")
flags.DEFINE_integer(name="logging_steps", default=10, help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_integer(name="num_train_epochs", default=100, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=64, help="")
flags.DEFINE_integer(name="save_steps", default=100, help="")
flags.DEFINE_integer(name="save_total_limit", default=2, help="")

flags.mark_flag_as_required('dataset_path')


def main(argv):
    #wandb.login()
    is_gpu = torch.cuda.is_available()

    if FLAGS.tokenizer_path:
        tokenizer_path = FLAGS.tokenizer_path
    elif FLAGS.tokenizer_type.upper() == "BPE":
        tokenizer_path = FLAGS.output_tokenizer_dir
        if not os.path.isdir(tokenizer_path):
            os.makedirs(tokenizer_path)

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=FLAGS.dataset_path, vocab_size=FLAGS.vocab_size, min_frequency=FLAGS.BPE_min_frequency, special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"])
        tokenizer.save_model(tokenizer_path)
    else:
        raise TypeError("Please provide a tokenizer path if using the SMILES tokenizer")

    print("making tokenizer")
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=FLAGS.max_tokenizer_len)
    print("making dataset")
    dataset = RegressionDataset(tokenizer=tokenizer, file_path=FLAGS.dataset_path, block_size=FLAGS.tokenizer_block_size)

    train_size = max(int(FLAGS.frac_train * len(dataset)), 1)
    eval_size = len(dataset) - train_size
    print(f"Train size: {train_size}")
    print(f"Eval size: {eval_size}")

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    config = RobertaConfig(
        vocab_size=FLAGS.vocab_size,
        max_position_embeddings=FLAGS.max_position_embeddings,
        num_attention_heads=FLAGS.num_attention_heads,
        num_hidden_layers=FLAGS.num_hidden_layers,
        type_vocab_size=FLAGS.type_vocab_size,
        num_labels=dataset.num_labels,
        norm_mean=dataset.norm_mean,
        norm_std=dataset.norm_std,
        is_gpu = is_gpu
    )

    model = RobertaForRegression(config=config)
    model = model.cuda()

    training_args = TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=FLAGS.eval_steps,
        logging_steps=FLAGS.logging_steps,
        load_best_model_at_end = True,
        output_dir=FLAGS.output_dir,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        save_steps=FLAGS.save_steps,
        save_total_limit=FLAGS.save_total_limit,
        fp16 = is_gpu # fp16 only works on CUDA devices
    )

    print("training")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=multitask_data_collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model(FLAGS.model_name)

if __name__ == '__main__':
    app.run(main)

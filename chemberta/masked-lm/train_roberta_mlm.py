""" Script for training a Roberta Masked-Language Model

Usage [SMILES tokenizer]:
    python train_roberta_mlm.py --dataset_path=<DATASET_PATH> --output_dir=<OUTPUT_DIR> --run_name=<RUN_NAME> --tokenizer_type=smiles --tokenizer_path="seyonec/SMILES_tokenized_PubChem_shard00_160k"

Usage [BPE tokenizer]:
    python train_roberta_mlm.py --dataset_path=<DATASET_PATH> --output_dir=<OUTPUT_DIR> --run_name=<RUN_NAME> --tokenizer_type=bpe
"""
import os
from absl import app
from absl import flags

import transformers
from transformers.trainer_callback import EarlyStoppingCallback

import torch
from torch.utils.data import random_split

import wandb
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM

from chemberta.utils.raw_text_dataset import RawTextDataset

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from tokenizers import ByteLevelBPETokenizer

FLAGS = flags.FLAGS

# RobertaConfig params
flags.DEFINE_integer(name="vocab_size", default=600, help="")
flags.DEFINE_integer(name="max_position_embeddings", default=515, help="") # This needs to be longer than max_tokenizer_len. max_len is currently 514 in seyonec/SMILES_tokenized_PubChem_shard00_160k
flags.DEFINE_integer(name="num_attention_heads", default=1, help="")
flags.DEFINE_integer(name="num_hidden_layers", default=1, help="")
flags.DEFINE_integer(name="type_vocab_size", default=1, help="")
flags.DEFINE_bool(name="fp16", default=True, help="Mixed precision.")

# Tokenizer params
flags.DEFINE_enum(name="tokenizer_type", default="smiles", enum_values=["smiles", "bpe", "SMILES", "BPE"], help="")
flags.DEFINE_string(name="tokenizer_path", default="", help="")
flags.DEFINE_integer(name="BPE_min_frequency", default=2, help="")
flags.DEFINE_string(name="output_tokenizer_dir", default="tokenizer_dir", help="")
flags.DEFINE_integer(name="max_tokenizer_len", default=512, help="")
flags.DEFINE_integer(name="tokenizer_block_size", default=512, help="")


# Dataset params
flags.DEFINE_string(name="dataset_path", default=None, help="")
flags.DEFINE_string(name="output_dir", default="default_dir", help="")
flags.DEFINE_string(name="run_name", default="default_run", help="")

# MLM params
flags.DEFINE_float(name="mlm_probability", default=0.15, lower_bound=0.0, upper_bound=1.0, help="")

# Train params
flags.DEFINE_float(name="frac_train", default=0.95, help="")
flags.DEFINE_integer(name="eval_steps", default=1000, help="")
flags.DEFINE_integer(name="logging_steps", default=100, help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_integer(name="num_train_epochs", default=1, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=64, help="")
flags.DEFINE_integer(name="save_steps", default=10000, help="")
flags.DEFINE_integer(name="save_total_limit", default=2, help="")

flags.mark_flag_as_required("dataset_path")


def main(argv):
    torch.manual_seed(0)
    
    wandb.login()

    is_gpu = torch.cuda.is_available()

    config = RobertaConfig(
        vocab_size=FLAGS.vocab_size,
        max_position_embeddings=FLAGS.max_position_embeddings,
        num_attention_heads=FLAGS.num_attention_heads,
        num_hidden_layers=FLAGS.num_hidden_layers,
        type_vocab_size=FLAGS.type_vocab_size,
    )

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
        print("Please provide a tokenizer path if using the SMILES tokenizer")

    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path, max_len=FLAGS.max_tokenizer_len)

    model = RobertaForMaskedLM(config=config)
    print(f"Model size: {model.num_parameters()} parameters.")

    dataset = RawTextDataset(tokenizer=tokenizer, file_path=FLAGS.dataset_path, block_size=FLAGS.tokenizer_block_size)

    train_size = max(int(FLAGS.frac_train * len(dataset)), 1)
    eval_size = len(dataset) - train_size
    print(f"Train size: {train_size}")
    print(f"Eval size: {eval_size}")

    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=FLAGS.mlm_probability
    )

    training_args = TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=FLAGS.eval_steps,
        load_best_model_at_end=True,
        logging_steps=FLAGS.logging_steps,
        output_dir=os.path.join(FLAGS.output_dir, FLAGS.run_name),
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        save_steps=FLAGS.save_steps,
        save_total_limit=FLAGS.save_total_limit,
        fp16 = is_gpu and FLAGS.fp16, # fp16 only works on CUDA devices
        report_to="wandb",
        run_name=FLAGS.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    trainer.save_model(os.path.join(FLAGS.output_dir, FLAGS.run_name, "final"))

if __name__ == '__main__':
    app.run(main)

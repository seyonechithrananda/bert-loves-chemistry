""" Script for training a Roberta Model (mlm or regression)

Usage [mlm]:
    python train_roberta.py
        --model_type=mlm
        --dataset_path=<DATASET_PATH>
        --mlm_probability=<MLM_MASKING_PROBABILITY>
        --output_dir=<OUTPUT_DIR>
        --run_name=<RUN_NAME>

Usage [regression]:
    python train_roberta.py
        --model_type=regression
        --dataset_path=<DATASET_PATH>
        --normalization_path=<PATH_TO_CACHED_NORMS>
        --output_dir=<OUTPUT_DIR>
        --run_name=<RUN_NAME>
>
"""

import os

import torch
from absl import app, flags
from transformers import RobertaConfig, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback

from chemberta.train.flags import roberta_model_configuration
from chemberta.train.utils import DatasetArguments, create_trainer

FLAGS = flags.FLAGS

# Model params
flags.DEFINE_enum(
    name="model_type",
    default="mlm",
    enum_values=["mlm", "regression", "regression_lazy"],
    help="",
)

# # RobertaConfig params
roberta_model_configuration()

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
flags.DEFINE_string(name="output_dir", default="default_dir", help="")
flags.DEFINE_string(name="run_name", default="default_run", help="")

# MLM params
flags.DEFINE_float(
    name="mlm_probability", default=0.15, lower_bound=0.0, upper_bound=1.0, help=""
)

# Regression params
flags.DEFINE_string(name="normalization_path", default=None, help="")

# Train params
flags.DEFINE_float(name="frac_train", default=0.95, help="")
flags.DEFINE_integer(name="eval_steps", default=10, help="")
flags.DEFINE_integer(name="early_stopping_patience", default=3, help="")
flags.DEFINE_integer(name="logging_steps", default=10, help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_integer(name="num_train_epochs", default=100, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=64, help="")
flags.DEFINE_integer(name="save_steps", default=100, help="")
flags.DEFINE_integer(name="save_total_limit", default=2, help="")

flags.mark_flag_as_required("dataset_path")
flags.mark_flag_as_required("model_type")


def main(argv):
    torch.manual_seed(0)
    run_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)

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
        FLAGS.max_tokenizer_len,
        FLAGS.tokenizer_block_size,
        FLAGS.mlm_probability,
    )

    training_args = TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=FLAGS.eval_steps,
        logging_steps=FLAGS.logging_steps,
        load_best_model_at_end=True,
        output_dir=run_dir,
        run_name=FLAGS.run_name,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        save_steps=FLAGS.save_steps,
        save_total_limit=FLAGS.save_total_limit,
        fp16=torch.cuda.is_available(),  # fp16 only works on CUDA devices
    )

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=FLAGS.early_stopping_patience)
    ]

    trainer = create_trainer(
        FLAGS.model_type, model_config, training_args, dataset_args, callbacks
    )
    trainer.train()
    trainer.save_model(os.path.join(run_dir, "final"))


if __name__ == "__main__":
    app.run(main)

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
import yaml
from absl import app, flags
from transformers import RobertaConfig, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback

from chemberta.train.flags import (
    dataset_flags,
    roberta_model_configuration_flags,
    tokenizer_flags,
    train_flags,
)
from chemberta.train.utils import DatasetArguments, create_trainer

# Model params
flags.DEFINE_enum(
    name="model_type",
    default="mlm",
    enum_values=["mlm", "regression", "regression_lazy"],
    help="",
)

dataset_flags()
roberta_model_configuration_flags()
tokenizer_flags()
train_flags()

# MLM params
flags.DEFINE_float(
    name="mlm_probability", default=0.15, lower_bound=0.0, upper_bound=1.0, help=""
)

# Regression params
flags.DEFINE_string(name="normalization_path", default=None, help="")

flags.mark_flag_as_required("dataset_path")
flags.mark_flag_as_required("model_type")

FLAGS = flags.FLAGS


flags_dict = {
    k: {vv.name: vv.value for vv in v} for k, v in FLAGS.flags_by_module_dict().items()
}


def main(argv):
    torch.manual_seed(0)
    run_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)

    model_config = RobertaConfig(
        vocab_size=FLAGS.vocab_size,
        max_position_embeddings=FLAGS.max_position_embeddings,
        num_attention_heads=FLAGS.num_attention_heads,
        num_hidden_layers=FLAGS.num_hidden_layers,
        hidden_size=FLAGS.hidden_size_per_attention_head * FLAGS.num_attention_heads,
        intermediate_size=4 * FLAGS.intermediate_size,
        type_vocab_size=FLAGS.type_vocab_size,
        hidden_dropout_prob=FLAGS.hidden_dropout_prob,
        attention_probs_dropout_prob=FLAGS.attention_probs_dropout_prob,
        is_gpu=torch.cuda.is_available(),
    )

    dataset_args = DatasetArguments(
        FLAGS.dataset_path,
        FLAGS.normalization_path,
        FLAGS.frac_train,
        FLAGS.eval_path,
        FLAGS.tokenizer_path,
        FLAGS.tokenizer_max_length,
        FLAGS.mlm_probability,
    )

    training_args = TrainingArguments(
        evaluation_strategy="steps",
        learning_rate=FLAGS.learning_rate,
        eval_steps=FLAGS.eval_steps,
        logging_steps=FLAGS.logging_steps,
        load_best_model_at_end=True,
        output_dir=run_dir,
        run_name=FLAGS.run_name,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        per_device_eval_batch_size=FLAGS.per_device_train_batch_size,
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

    flags_file_path = os.path.join(run_dir, "params.yml")
    with open(flags_file_path, "w") as f:
        yaml.dump(flags_dict, f)
    print(f"Saved command-line flags to {flags_file_path}")

    trainer.train()
    trainer.save_model(os.path.join(run_dir, "final"))


if __name__ == "__main__":
    app.run(main)

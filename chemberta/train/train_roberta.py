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

import glob
import os
import subprocess

import s3fs
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
from chemberta.train.utils import AwsS3Callback, DatasetArguments, create_trainer

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

# Regression params
flags.DEFINE_string(name="normalization_path", default=None, help="")

flags.mark_flag_as_required("dataset_path")
flags.mark_flag_as_required("model_type")

FLAGS = flags.FLAGS


def main(argv):
    torch.manual_seed(0)
    run_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name)
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

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
        save_steps=FLAGS.save_steps,
        load_best_model_at_end=FLAGS.load_best_model_at_end,
        output_dir=run_dir,
        run_name=FLAGS.run_name,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        per_device_eval_batch_size=FLAGS.per_device_train_batch_size,
        save_total_limit=FLAGS.save_total_limit,
        fp16=torch.cuda.is_available(),  # fp16 only works on CUDA devices
    )

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=FLAGS.early_stopping_patience),
    ]

    if FLAGS.cloud_directory is not None:
        # check if remote directory exists, pull down
        fs = s3fs.S3FileSystem()
        full_cloud_dir = os.path.join(FLAGS.cloud_directory, FLAGS.run_name)
        if fs.exists(full_cloud_dir):
            print(f"Found existing directory at {full_cloud_dir}. Downloading...")
            subprocess.check_call(
                [
                    "aws",
                    "s3",
                    "sync",
                    full_cloud_dir,
                    run_dir,
                ]
            )
        callbacks.append(
            AwsS3Callback(local_directory=run_dir, s3_directory=full_cloud_dir)
        )

    trainer = create_trainer(
        FLAGS.model_type, model_config, training_args, dataset_args, callbacks
    )

    flags_dict = {
        k: {vv.name: vv.value for vv in v}
        for k, v in FLAGS.flags_by_module_dict().items()
        if k in ["dataset", "model", "training"]
    }

    flags_file_path = os.path.join(run_dir, "params.yml")
    with open(flags_file_path, "w") as f:
        yaml.dump(flags_dict, f)
    print(f"Saved command-line flags to {flags_file_path}")

    # if there is a checkpoint available, use it
    checkpoints = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    if checkpoints:
        iters = [int(x.split("-")[-1]) for x in checkpoints if "checkpoint" in x]
        iters.sort()
        latest_checkpoint = os.path.join(run_dir, f"checkpoint-{iters[-1]}")
        print(f"Loading model from latest checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()
    trainer.save_model(os.path.join(run_dir, "final"))

    # do a final sync
    if FLAGS.cloud_directory is not None:
        subprocess.check_call(
            [
                "aws",
                "s3",
                "sync",
                run_dir,
                full_cloud_dir,
                "--acl",
                "bucket-owner-full-control",
                "--delete",
            ]
        )


if __name__ == "__main__":
    app.run(main)

"""Script for finetuning and evaluating pre-trained ChemBERTa models on MoleculeNet tasks.

[classification]
python finetune.py --datasets=bbbp --pretrained_model_name_or_path=DeepChem/ChemBERTa-SM-015

[regression]
python finetune.py --datasets=delaney --pretrained_model_name_or_path=DeepChem/ChemBERTa-SM-015

[csv]
python finetune.py --datasets=$HOME/finetune_datasets/logd/ \
                --dataset_types=regression \
                --pretrained_model_name_or_path=DeepChem/ChemBERTa-SM-015 \
                --is_molnet=False

[multiple]
python finetune.py \
--datasets=bace_classification,bace_regression,bbbp,clearance,clintox,delaney,lipo,tox21 \
--pretrained_model_name_or_path=DeepChem/ChemBERTa-SM-015 \
--n_trials=20 \
--output_dir=finetuning_experiments \
--run_name=sm_015

[from scratch (no pretraining)]
python finetune.py --datasets=bbbp

"""
import json
import os
import shutil
import tempfile
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from absl import app, flags
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    roc_auc_score,
)
from transformers import RobertaConfig, RobertaTokenizerFast, Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback

from chemberta.finetune.utils import (
    get_finetune_datasets,
    get_latest_checkpoint,
    prune_state_dict,
)
from chemberta.utils.cloud import check_cloud, sync_with_s3
from chemberta.utils.molnet_dataloader import get_dataset_info
from chemberta.utils.roberta_regression import (
    RobertaForRegression,
    RobertaForSequenceClassification,
)

FLAGS = flags.FLAGS

# Settings
flags.DEFINE_string(name="output_dir", default="default_dir", help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_integer(name="seed", default=0, help="Global random seed.")

# Model params
flags.DEFINE_list(
    name="pretrained_paths",
    default=None,
    help="list of pretrained models. can be local or cloud",
)
flags.DEFINE_list(
    name="model_names", default=None, help="list of names to assign to each model run"
)
flags.DEFINE_boolean(
    name="is_molnet",
    default=True,
    help="If true, assumes all dataset are MolNet datasets.",
)

# Train params
flags.DEFINE_integer(name="logging_steps", default=10, help="")
flags.DEFINE_integer(name="early_stopping_patience", default=5, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=64, help="")
flags.DEFINE_integer(name="per_device_eval_batch_size", default=64, help="")
flags.DEFINE_integer(
    name="n_trials",
    default=5,
    help="Number of different hyperparameter combinations to try. Each combination will result in a different finetuned model.",
)
flags.DEFINE_integer(
    name="n_seeds",
    default=5,
    help="Number of unique random seeds to try. This only applies to the final best model selected after hyperparameter tuning.",
)
flags.DEFINE_integer(
    name="save_total_limit",
    default=3,
    help="Total number of checkpoints to save per model configuration.",
)

# Dataset params
flags.DEFINE_list(
    name="datasets",
    default=None,
    help="Comma-separated list of MoleculeNet dataset names.",
)
flags.DEFINE_string(
    name="split", default="scaffold", help="DeepChem data loader split_type."
)
flags.DEFINE_list(
    name="dataset_types",
    default=None,
    help="List of dataset types (ex: classification,regression). Include 1 per dataset, not necessary for MoleculeNet datasets.",
)

# Tokenizer params
flags.DEFINE_string(
    name="tokenizer_path",
    default="seyonec/SMILES_tokenized_PubChem_shard00_160k",
    help="",
)
flags.DEFINE_integer(name="max_tokenizer_len", default=512, help="")

flags.mark_flag_as_required("datasets")
flags.mark_flag_as_required("pretrained_paths")


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def main(argv):
    is_molnet = FLAGS.is_molnet

    for i, pretrained_model_dir in enumerate(FLAGS.pretrained_paths):
        if FLAGS.model_names is not None:
            model_name = FLAGS.model_names[i]
        else:
            model_name = f"model_{i}"
        # Check that CSV dataset has the proper flags
        if not is_molnet:
            print("Assuming each dataset is a folder containing CSVs...")
            assert (
                len(FLAGS.dataset_types) > 0
            ), "Please specify dataset types for csv datasets"
            for dataset_folder in FLAGS.datasets:
                assert os.path.exists(os.path.join(dataset_folder, "train.csv"))
                assert os.path.exists(os.path.join(dataset_folder, "valid.csv"))
                assert os.path.exists(os.path.join(dataset_folder, "test.csv"))

        for i in range(len(FLAGS.datasets)):
            dataset_name_or_path = FLAGS.datasets[i]
            dataset_name = get_dataset_name(dataset_name_or_path)
            dataset_type = (
                get_dataset_info(dataset_name)["dataset_type"]
                if is_molnet
                else FLAGS.dataset_types[i]
            )

            run_dir = os.path.join(FLAGS.output_dir, model_name, dataset_name)

            if os.path.exists(run_dir) and not FLAGS.overwrite_output_dir:
                print(f"Run dir already exists for dataset: {dataset_name}")
            else:
                print(f"Finetuning on {dataset_name}")
                finetune_model_on_single_dataset(
                    pretrained_model_dir,
                    dataset_name_or_path,
                    dataset_type,
                    run_dir,
                    is_molnet,
                )


def check_cloud(path: str):
    """Naive check to if the path is a cloud path"""
    if path.startswith("s3:"):
        return True
    return False


def sync_with_s3(source_dir: str, target_dir: str):
    """Sync source_dir directory with target_dir"""
    subprocess.check_call(
        [
            "aws",
            "s3",
            "sync",
            source_dir,
            target_dir,
            "--acl",
            "bucket-owner-full-control",
            "--delete",
        ]
    )
    return


def get_latest_checkpoint(saved_model_dir):
    """Get the folder for the latest checkpoint"""
    iters = [
        int(x.split("-")[-1]) for x in os.listdir(saved_model_dir) if "checkpoint" in x
    ]
    iters.sort()
    latest_checkpoint_dir = os.path.join(saved_model_dir, f"checkpoint-{iters[-1]}")
    return latest_checkpoint_dir


def prune_state_dict(model_dir):
    """Remove problematic keys from state dictionary"""
    if not (model_dir and os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))):
        return None

    state_dict_path = os.path.join(model_dir, "pytorch_model.bin")
    assert os.path.exists(
        state_dict_path
    ), f"No `pytorch_model.bin` file found in {model_dir}"
    loaded_state_dict = torch.load(state_dict_path)
    state_keys = loaded_state_dict.keys()
    keys_to_remove = [
        k for k in state_keys if k.startswith("regression") or k.startswith("norm")
    ]

    new_state_dict = OrderedDict({**loaded_state_dict})
    for k in keys_to_remove:
        del new_state_dict[k]
    return new_state_dict


def finetune_model_on_single_dataset(
    pretrained_model_dir: str,
    dataset_name: str,
    dataset_type: str,
    run_dir: str,
    is_molnet: bool,
    use_final: bool = False,
):
    """Pretrains a single model on a single dataset

    Args:
        pretrained_model_dir: local or cloud dir
        dataset_name: name of dataset
        dataset_type: regression or classification
        run_dir: directory for saving results
        is_molnet: whether or not it's a MolNet dataset
        use_final: whether or not to use `final` directory. otherwise, uses the latest checkpoint
    """
    torch.manual_seed(FLAGS.seed)
    os.environ["WANDB_DISABLED"] = "true"

    tokenizer = RobertaTokenizerFast.from_pretrained(
        FLAGS.tokenizer_path, max_len=FLAGS.max_tokenizer_len, use_auth_token=True
    )

    finetune_datasets = get_finetune_datasets(dataset_name, tokenizer, is_molnet)

    if check_cloud(pretrained_model_dir):
        local_dir = os.path.join(
            tempfile.gettempdir(), os.sep.join(pretrained_model_dir.split(os.sep)[2:])
        )
        print(f"Syncing {pretrained_model_dir} to {local_dir}")
        sync_with_s3(pretrained_model_dir, local_dir)

    else:
        local_dir = pretrained_model_dir

    if use_final:
        checkpoint_dir = os.path.join(local_dir, "final")

    else:
        checkpoint_dir = get_latest_checkpoint(local_dir)
        other_checkpoint_dirs = [
            os.path.join(local_dir, x)
            for x in os.listdir(local_dir)
            if "checkpoint" in x
        ]
        other_checkpoint_dirs.remove(checkpoint_dir)
        for dir in other_checkpoint_dirs:
            shutil.rmtree(dir, ignore_errors=True)

    assert os.path.isdir(
        checkpoint_dir
    ), f"Could not find checkpoint dir {checkpoint_dir}"

    config = RobertaConfig.from_pretrained(
        checkpoint_dir,
    )

    if dataset_type == "classification":
        config.num_labels = finetune_datasets.num_labels

    elif dataset_type == "regression":
        config.num_labels = 1
        config.norm_mean = finetune_datasets.norm_mean
        config.norm_std = finetune_datasets.norm_std

    state_dict = prune_state_dict(checkpoint_dir)

    def warmup_model_init():
        if dataset_type == "classification":
            model_class = RobertaForSequenceClassification
        elif dataset_type == "regression":
            model_class = RobertaForRegression

        model = model_class.from_pretrained(
            checkpoint_dir,
            config=config,
            state_dict=state_dict,
            use_auth_token=True,
        )
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False

        return model

    # train for 2 epochs to get the final layer warmed-up
    warmup_dir = os.path.join(run_dir, "warmup/")
    warmup_model_dir = os.path.join(warmup_dir, "warmed_up")
    warmup_training_args = TrainingArguments(
        evaluation_strategy="epoch",
        num_train_epochs=2,
        output_dir=warmup_dir,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        per_device_eval_batch_size=FLAGS.per_device_eval_batch_size,
        logging_steps=FLAGS.logging_steps,
        load_best_model_at_end=True,
        report_to=None,
    )
    warmup_trainer = Trainer(
        model_init=warmup_model_init,
        args=warmup_training_args,
        train_dataset=finetune_datasets.train_dataset,
        eval_dataset=finetune_datasets.valid_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=FLAGS.early_stopping_patience)
        ],
    )
    warmup_trainer.train()
    warmup_trainer.save_model(warmup_model_dir)

    def hp_model_init():
        if dataset_type == "classification":
            model_class = RobertaForSequenceClassification
        elif dataset_type == "regression":
            model_class = RobertaForRegression

        # make sure to leave out the `state_dict` argument
        # since we actually want to use the saved final layer weights
        model = model_class.from_pretrained(
            warmup_model_dir,
            config=config,
            use_auth_token=True,
        )
        # make sure everything is trainable
        for name, param in model.base_model.named_parameters():
            param.requires_grad = True

        return model

    hp_training_args = TrainingArguments(
        evaluation_strategy="epoch",
        num_train_epochs=100,
        output_dir=run_dir,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        per_device_eval_batch_size=FLAGS.per_device_eval_batch_size,
        logging_steps=FLAGS.logging_steps,
        load_best_model_at_end=True,
        report_to=None,
        save_total_limit=FLAGS.save_total_limit,
    )

    hp_trainer = Trainer(
        model_init=hp_model_init,
        args=hp_training_args,
        train_dataset=finetune_datasets.train_dataset,
        eval_dataset=finetune_datasets.valid_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=FLAGS.early_stopping_patience)
        ],
    )

    def custom_hp_space_optuna(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True),
            # "num_train_epochs": trial.suggest_int(
            #     "num_train_epochs", 1, FLAGS.num_train_epochs_max
            # ),
            "seed": trial.suggest_int("seed", 1, 40),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [FLAGS.per_device_train_batch_size]
            ),
        }

    best_trial = hp_trainer.hyperparameter_search(
        backend="optuna",
        direction="minimize",
        hp_space=custom_hp_space_optuna,
        n_trials=FLAGS.n_trials,
    )

    # Set parameters to the best ones from the hp search
    for n, v in best_trial.hyperparameters.items():
        setattr(hp_trainer.args, n, v)

    dir_valid = os.path.join(run_dir, "results", "valid")
    dir_test = os.path.join(run_dir, "results", "test")
    os.makedirs(dir_valid, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)

    metrics_valid = {}
    metrics_test = {}

    # Run with several seeds so we can see std
    for random_seed in range(FLAGS.n_seeds):
        setattr(hp_trainer.args, "seed", random_seed)
        setattr(hp_trainer.args, "run_name", f"run_{random_seed}")
        hp_trainer.train()
        metrics_valid[f"seed_{random_seed}"] = eval_model(
            hp_trainer,
            finetune_datasets.valid_dataset_unlabeled,
            dataset_name,
            dataset_type,
            dir_valid,
            random_seed,
        )
        metrics_test[f"seed_{random_seed}"] = eval_model(
            hp_trainer,
            finetune_datasets.test_dataset,
            dataset_name,
            dataset_type,
            dir_test,
            random_seed,
        )

    with open(os.path.join(dir_valid, "metrics.json"), "w") as f:
        json.dump(metrics_valid, f)
    with open(os.path.join(dir_test, "metrics.json"), "w") as f:
        json.dump(metrics_test, f)

    # Delete checkpoints/runs from hyperparameter search since they use a lot of disk
    for d in glob(os.path.join(run_dir, "run-*")):
        shutil.rmtree(d, ignore_errors=True)
    for d in glob(os.path.join(run_dir, "checkpoint-*")):
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(warmup_dir, ignore_errors=True)

    hp_trainer.save_state()
    hp_trainer.save_model(os.path.join(run_dir, "final"))


def eval_model(trainer, dataset, dataset_name, dataset_type, output_dir, random_seed):
    labels = dataset.labels
    predictions = trainer.predict(dataset)
    fig = plt.figure(dpi=144)

    if dataset_type == "classification":
        if len(np.unique(labels)) <= 2:
            y_pred = softmax(predictions.predictions, axis=1)[:, 1]
            metrics = {
                "roc_auc_score": roc_auc_score(y_true=labels, y_score=y_pred),
                "average_precision_score": average_precision_score(
                    y_true=labels, y_score=y_pred
                ),
            }
            sns.histplot(x=y_pred, hue=labels)
        else:
            y_pred = np.argmax(predictions.predictions, axis=-1)
            metrics = {"mcc": matthews_corrcoef(labels, y_pred)}

    elif dataset_type == "regression":
        y_pred = predictions.predictions.flatten()
        metrics = {
            "pearsonr": pearsonr(y_pred, labels),
            "rmse": mean_squared_error(y_true=labels, y_pred=y_pred, squared=False),
        }
        sns.regplot(x=y_pred, y=labels)
        plt.xlabel("ChemBERTa predictions")
        plt.ylabel("Ground truth")
    else:
        raise ValueError(dataset_type)

    plt.title(f"{dataset_name} {dataset_type} results")
    plt.savefig(os.path.join(output_dir, f"results_seed_{random_seed}.png"))

    return metrics


def get_dataset_name(dataset_name_or_path):
    return os.path.splitext(os.path.basename(dataset_name_or_path))[0]


if __name__ == "__main__":
    app.run(main)

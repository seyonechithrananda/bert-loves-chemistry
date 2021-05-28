"""Script for finetuning and evaluating pre-trained ChemBERTa models on MoleculeNet tasks.

[classification]
python finetune.py --datasets=bbbp --model_dir=/home/ubuntu/chemberta_models/mlm/sm_015/

[regression]
python finetune.py --datasets=delaney --model_dir=/home/ubuntu/chemberta_models/mlm/sm_015/

[multiple]
python finetune.py \
--datasets=bace_classification,bace_regression,bbbp,clearance,clintox,delaney,lipo \
--model_dir=/home/ubuntu/chemberta_models/mlm/sm_015/ \
--n_trials=20 \
--output_dir=finetuning_experiments \
--run_name=run1 \

"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import shutil
import torch
import wandb

from absl import app, flags
from glob import glob
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    RobertaConfig,
    Trainer,
    TrainingArguments,
)

from chemberta.utils.molnet_dataloader import get_dataset_info, load_molnet_dataset
from chemberta.utils.roberta_regression import RobertaForRegression

from transformers.trainer_callback import EarlyStoppingCallback


FLAGS = flags.FLAGS

# Settings
flags.DEFINE_string(name="output_dir", default="default_dir", help="")
flags.DEFINE_boolean(name="overwrite_output_dir", default=True, help="")
flags.DEFINE_string(name="run_name", default="default_run", help="")
flags.DEFINE_integer(name="seed", default=0, help="")

# Model params
flags.DEFINE_string(name="model_dir", default=None, help="")
flags.DEFINE_boolean(name="freeze_base_model", default=False, help="")

# Train params
flags.DEFINE_integer(name="logging_steps", default=10, help="")
flags.DEFINE_integer(name="early_stopping_patience", default=3, help="")
flags.DEFINE_integer(name="num_train_epochs", default=10, help="")
flags.DEFINE_integer(name="per_device_train_batch_size", default=64, help="")
flags.DEFINE_integer(name="per_device_eval_batch_size", default=64, help="")
flags.DEFINE_integer(name="n_trials", default=5, help="")
flags.DEFINE_integer(name="n_seeds", default=5, help="")

# Dataset params
flags.DEFINE_list(name="datasets", default=None, help="")
flags.DEFINE_string(name="split", default="scaffold", help="")

# Tokenizer params
flags.DEFINE_string(
    name="tokenizer_path",
    default="seyonec/SMILES_tokenized_PubChem_shard00_160k",
    help="",
)
flags.DEFINE_integer(name="max_tokenizer_len", default=512, help="")

flags.mark_flag_as_required("datasets")
flags.mark_flag_as_required("model_dir")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def main(argv):
    for dataset_name in FLAGS.datasets:
        print(f"Finetuning on {dataset_name}")
        finetune_single_dataset(dataset_name)
    
    
def finetune_single_dataset(dataset_name):
    torch.manual_seed(FLAGS.seed)
    run_dir = os.path.join(FLAGS.output_dir, FLAGS.run_name, dataset_name)

    tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset(
        dataset_name, split=FLAGS.split, df_format="chemprop"
    )
    assert len(tasks) == 1

    tokenizer = RobertaTokenizerFast.from_pretrained(
        FLAGS.tokenizer_path, max_len=FLAGS.max_tokenizer_len
    )

    train_encodings = tokenizer(
        train_df["smiles"].tolist(), truncation=True, padding=True
    )
    valid_encodings = tokenizer(
        valid_df["smiles"].tolist(), truncation=True, padding=True
    )
    test_encodings = tokenizer(
        test_df["smiles"].tolist(), truncation=True, padding=True
    )

    train_labels = train_df.iloc[:, 1].values
    valid_labels = valid_df.iloc[:, 1].values
    test_labels = test_df.iloc[:, 1].values

    train_dataset = MolNetDataset(train_encodings, train_labels)
    valid_dataset = MolNetDataset(valid_encodings, valid_labels)
    test_dataset = MolNetDataset(test_encodings)

    config = RobertaConfig.from_pretrained(FLAGS.model_dir)

    dataset_type = get_dataset_info(dataset_name)["dataset_type"]
    if dataset_type == "classification":
        model_class = RobertaForSequenceClassification
    elif dataset_type == "regression":
        model_class = RobertaForRegression
        config.num_labels = 1
        config.norm_mean = [np.mean(np.array(train_labels), axis=0)]
        config.norm_std = [np.std(np.array(train_labels), axis=0)]

    def model_init():
        if dataset_type == "classification":
            model_class = RobertaForSequenceClassification
        elif dataset_type == "regression":
            model_class = RobertaForRegression
        model = model_class.from_pretrained(FLAGS.model_dir, config=config)

        if FLAGS.freeze_base_model:
            for name, param in model.base_model.named_parameters():
                param.requires_grad = False

        return model

    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        output_dir=run_dir,
        overwrite_output_dir=FLAGS.overwrite_output_dir,
        num_train_epochs=FLAGS.num_train_epochs,
        per_device_train_batch_size=FLAGS.per_device_train_batch_size,
        per_device_eval_batch_size=FLAGS.per_device_eval_batch_size,
        logging_steps=FLAGS.logging_steps,
        load_best_model_at_end=True,
        report_to=None,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=FLAGS.early_stopping_patience)
        ],
    )

    best_trial = trainer.hyperparameter_search(
        backend="optuna",
        direction="minimize",
        n_trials=FLAGS.n_trials,
    )

    # Remake valid_dataset without labels to force unnormalization in regression model
    valid_dataset = MolNetDataset(valid_encodings)

    # Set parameters to the best ones from the hp search
    for n, v in best_trial.hyperparameters.items():
        setattr(trainer.args, n, v)
        
    dir_valid = os.path.join(run_dir, "results", "valid")
    dir_test = os.path.join(run_dir, "results", "test")
    os.makedirs(dir_valid, exist_ok=True)
    os.makedirs(dir_test, exist_ok=True)
        
    metrics_valid = {}
    metrics_test = {}

    # Run with several seeds so we can see std
    for random_seed in range(FLAGS.n_seeds):
        setattr(trainer.args, "seed", random_seed)
        trainer.train()
        metrics_valid[f"seed_{random_seed}"] = eval_model(
            trainer,
            valid_dataset,
            valid_labels,
            dataset_name,
            dataset_type,
            dir_valid,
            random_seed,
        )
        metrics_test[f"seed_{random_seed}"] = eval_model(
            trainer,
            test_dataset,
            test_labels,
            dataset_name,
            dataset_type,
            dir_test,
            random_seed,
        )

    with open(os.path.join(dir_valid, "metrics.json"), "w") as f:
        json.dump(metrics_valid, f)
    with open(os.path.join(dir_test, "metrics.json"), "w") as f:
        json.dump(metrics_test, f)

    # Delete checkpoints from hyperparameter search since they use a lot of disk
    for d in glob(os.path.join(run_dir, "run-*")):
        shutil.rmtree(d, ignore_errors=True)


def eval_model(trainer, dataset, labels, dataset_name, dataset_type, output_dir, random_seed):
    predictions = trainer.predict(dataset)
    fig = plt.figure(dpi=144)

    if dataset_type == "classification":
        y_pred = softmax(predictions.predictions, axis=1)[:, 1]
        metrics = {
            "roc_auc_score": roc_auc_score(y_true=labels, y_score=y_pred),
            "average_precision_score": average_precision_score(
                y_true=labels, y_score=y_pred
            ),
        }
        sns.histplot(x=y_pred, hue=labels)
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


class MolNetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


if __name__ == "__main__":
    app.run(main)

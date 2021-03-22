""" Script for training a Roberta Masked-Language Model

Usage (supports both BPE and SMILES-tokenizers, just plug in the relative model path from Huggingface's API):
    python molnet_finetune.py --molnet_dataset=<DATASET_NAME> --model_type=<MODEL_TYPE> --model_name=<MODEL_NAME> --num_train_epochs=<NUM_TRAIN_EPOCHS> --output_dir=<OUTPUT_DIR>

"""

import os
from absl import app
from absl import flags

# main packages - transformere (NLP), deepchem (molnet), torch
import transformers
import deepchem
import torch

# scaffold splitting
from rdkit import Chem

# nvidia training tool
from apex import amp

# molnet dataloader uses pandas dataframe to feed into simple-transformers model
import numpy as np
import pandas as pd
from typing import List

# import molnet loaders from deepchem
from deepchem.molnet import load_bbbp, load_clearance, load_clintox, load_delaney, load_hiv, load_qm7, load_tox21

# import MolNet dataloder from bert-loves-chemistry fork
from chemberta.utils.molnet_dataloader import load_molnet_dataset, write_molnet_dataset_for_chemprop

# fine-tuning model package
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
import wandb

# evaluate model with sklearn metrics
import sklearn

FLAGS = flags.FLAGS

# Dataset params
flags.DEFINE_string(name="molnet_dataset", default="clintox", help="")

# Model loading + train params
flags.DEFINE_string(name="model_type", default="roberta", help="")
flags.DEFINE_string(name="model_name", default="seyonec/SMILES_tokenized_PubChem_shard00_160k", help="Model name to retrieve from Huggingface model hub.")
flags.DEFINE_integer(name="num_train_epochs", default=10, help="")
flags.DEFINE_string(name="output_dir", default="MolNet_Benchmark", help="")


def main(argv):
    wandb.login()

    tasks, (train_df, valid_df, test_df), transformers = load_molnet_dataset(FLAGS.molnet_dataset, tasks_wanted=None)

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model = ClassificationModel(FLAGS.model_type, FLAGS.model_name, 
                                args={'evaluate_each_epoch': True, 'evaluate_during_training_verbose': True, 
                                'no_save': True, 'num_train_epochs': FLAGS.num_train_epochs, 'auto_weights': True}) 
                                # You can set class weights by using the optional weight argument

    # check if our train and evaluation dataframes are setup properly. There should only be two columns for the SMILES string and its corresponding label.
    print("Train Dataset: {}".format(train_df.shape))
    print("Eval Dataset: {}".format(valid_df.shape))
    print("TEST Dataset: {}".format(test_df.shape))

    model.train_model(train_df, eval_df=valid_df, output_dir=FLAGS.output_dir, 
                    args={'wandb_project': 'project-name'})


    # accuracy
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

    # ROC-PRC
    result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.average_precision_score)

    # Lets input a molecule with a toxicity value of 1
    predictions, raw_outputs = model.predict(['C1=C(C(=O)NC(=O)N1)F'])
    print(predictions)
    print(raw_outputs)

    
if __name__ == '__main__':
    app.run(main)
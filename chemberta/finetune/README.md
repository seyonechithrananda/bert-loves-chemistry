# Finetuning

Pass in one or more MoleculeNet dataset names separated by a comma.

```
python finetune.py --datasets=bbbp,delaney --model_dir=DeepChem/ChemBERTa-SM-015
```

## MoleculeNet datasets
The full list of supported MoleculeNet datasets is in `chemberta/utils/molnet_dataloader.py`.

## Pretrained models

Pretrained ChemBERTa models are available at [https://huggingface.co/DeepChem](https://huggingface.co/DeepChem). 


Currently, the following models are available:

- **DeepChem/ChemBERTa-SM-015**: 15.6M parameter model (2 layers, 2 attention heads) pretrained on PubChem 77M with 15\% MLM masking rate.
- **DeepChem/ChemBERTa-MD-015**: 44.0M parameter model (6 layers, 6 attention heads) pretrained on PubChem 77M with 15\% MLM masking rate.
- **DeepChem/ChemBERTa-LG-015**: 86.5M parameter model (12 layers, 12 attention heads) pretrained on PubChem 77M with 15\% MLM masking rate.

The DeepChem HuggingFace Model Hub is WIP and more models will be added as they become available.

## Hyperparameter tuning

The fineuning script uses HuggingFace's built-in [hyperparameter search](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.hyperparameter_search) (Optuna backend). First, finetuning is repeated for `n_trials` using different hyperparameter combinations. Then, the best model from the search (according to validation set metrics) is selected and finetuned `n_seeds` times with different random seeds.

- `n_trials`: Number of different hyperparameter combinations to try. Each combination will result in a different finetuned model.
- `n_seeds`: Number of unique random seeds to try. This only applies to the final best model selected after hyperparameter tuning.

## Aggregate metrics

After a finetuning experiment is completed on one or more datasets, the `aggregate_metrics.py` script can be used to collate the metrics from the different datasets.
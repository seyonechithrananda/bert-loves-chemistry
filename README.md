# ChemBERTa
ChemBERTa: A collection of BERT-like models applied to chemical SMILES data for drug design, chemical modelling, and property prediction.

Right now the notebooks are all for the RoBERTa model (a variant of BERT) trained on the task of masked-language modelling (MLM). Training was done over 5 epochs until loss converged to around 0.28. The model weights for ChemBERTA pre-trained on ZINC 100k are available using [HuggingFace](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1), and the ZINC 250k version is available [here](https://huggingface.co/seyonec/ChemBERTa-zinc250k-v1). We expect to release larger models pre-trained on even larger subsets of ZINC, CHEMBL, and Enamine in the near future. 

I hope this is of use to developers, students and researchers exploring the use of transformers and the attention mechanism for chemistry!

You can load the tokenizer + model for MLM prediction tasks using the following code:

```
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

model = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
```
The abstract for this method is detailed [here](https://t.co/dkA5rMvYrE?amp=1). We expect to release a full paper on Arxiv in end-August.

Todo:
- [ ]  Official DeepChem implementation of ChemBERTa using model API
- [ ]  Open-source attention visualization suite used in paper.
- [ ]  Release larger pre-trained models, and support for a wider array of property prediction tasks (BBBP, etc).

- [x]  Finish writing notebook to train model
- [x]  Finish notebook to preload and run predictions on a single molecule —> test if HuggingFace works
- [x]  Train RoBERTa model until convergence
- [x]  Upload weights onto HuggingFace
- [x]  Create tutorial using evaluation + fine-tuning notebook.
- [x]  Create documentation + writing, visualizations for notebook.
- [x]  Setup PR into DeepChem

# ChemBERTa
ChemBERTa: A collection of BERT-like models applied to chemical SMILES data for drug design, chemical modelling, and property prediction. To be presented at [Baylearn](https://baylearn2020.splashthat.com/) and the [Royal Society of Chemistry's Chemical Science Symposium](https://www.rsc.org/events/detail/42791/chemical-science-symposium-2020-how-can-machine-learning-and-autonomy-accelerate-chemistry).

[Tutorial](https://github.com/deepchem/deepchem/blob/master/examples/tutorials/Transfer_Learning_With_ChemBERTa_Transformers.ipynb) <br />
[ArXiv ChemBERTa-2 Paper](https://arxiv.org/abs/2209.01712) <br />
[Arxiv ChemBERTa Paper](https://arxiv.org/abs/2010.09885) <br />
[Poster](https://chemsci20.ipostersessions.com/Default.aspx?s=99-39-E6-B6-B0-0E-E1-D8-FB-66-1A-44-DC-A3-43-BA) <br />
[Abstract](https://t.co/dkA5rMvYrE?amp=1) <br />
[BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:FzDMp7nctLUJ:scholar.google.com/&output=citation&scisdr=CgWWnePlEM-dmmZXtDE:AAGBfm0AAAAAX5RSrDGmJTVdPMFfzRSs5UY9lD4iRvvd&scisig=AAGBfm0AAAAAX5RSrGbFzGg583aNAYQw1Lap1K79xkEm&scisf=4&ct=citation&cd=-1&hl=en)

License: MIT License

Right now the notebooks are all for the RoBERTa model (a variant of BERT) trained on the task of masked-language modelling (MLM). Training was done over 10 epochs until loss converged to around 0.26 on the ZINC 250k dataset. The model weights for ChemBERTA pre-trained on various datasets (ZINC 100k, ZINC 250k, PubChem 100k, PubChem 250k, PubChem 1M, PubChem 10M) are available using [HuggingFace](https://huggingface.co/seyonec). We expect to continue to release larger models pre-trained on even larger subsets of ZINC, CHEMBL, and PubChem in the near future. 

This library is currently primarily a set of notebooks with our pre-training and fine-tuning setup, and will be updated soon with model implementation + attention visualization code, likely after the Arxiv publication. Stay tuned! 

I hope this is of use to developers, students and researchers exploring the use of transformers and the attention mechanism for chemistry!

# Citing Our Work
Please cite ChemBERTa-2's [ArXiv](https://arxiv.org/abs/2209.01712) paper if you have used these models, notebooks, or examples in any way. The BibTex is available below: 
```
@article{ahmad2022chemberta,
  title={Chemberta-2: Towards chemical foundation models},
  author={Ahmad, Walid and Simon, Elana and Chithrananda, Seyone and Grand, Gabriel and Ramsundar, Bharath},
  journal={arXiv preprint arXiv:2209.01712},
  year={2022}
}
```

# Example
You can load the tokenizer + model for MLM prediction tasks using the following code:

```
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

#any model weights from the link above will work here
model = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
```

# Todo:
- [x]  Official DeepChem implementation of ChemBERTa using model API (In progress)
- [X]  Open-source attention visualization suite used in paper (After formal publication - Beginning of September).
- [x]  Release larger pre-trained models, and support for a wider array of property prediction tasks (BBBP, etc). - See [HuggingFace](https://huggingface.co/seyonec)
- [x]  Finish writing notebook to train model
- [x]  Finish notebook to preload and run predictions on a single molecule —> test if HuggingFace works
- [x]  Train RoBERTa model until convergence
- [x]  Upload weights onto HuggingFace
- [x]  Create tutorial using evaluation + fine-tuning notebook.
- [x]  Create documentation + writing, visualizations for notebook.
- [x]  Setup PR into DeepChem

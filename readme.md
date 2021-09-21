# UD2OIE system

This repository contains the code for the papers :
Universal Dependencies for Multilingual Open Information Extraction\
Open Information Extraction: Supervised and Syntactic Approach for French


# Training Datasets.
## dataset folder contains the training dataset 

## Requirements
### requirements.txt : contains the versions of the main needed packages
### pip install -r requirements.txt to install the required packages

## Saved model folder : 

### UD2OIE_Pred_saved : folder containing the trained predicate module of our modele UD2OIE

### UD2OIE_Arg_saved : folder containing the trained argument module of our modele UD2OIE

## Run and Evaluate saved model

### restore.py : python file to restore our saved model and to evaluate against the benchmarks

## Train from Scratch :

### train.py : python file to train our modele from scratch

## Run and Evaluate trained model :

### predict.py : python file to evaluate the last five checkpoint of our model on the developement set (expert.tsv),
### 				    it also evaluates them against against the evaluation benchmarks.
### 				    we build on the developement set results to pick our final model. 


## Evaluate the other baselines :

### baselines.py : python file to evaluate some of the baselines mentioned in the paper against the benchmarks,
### 				    After training a new model, the output file of our model must be updated in this file.

## CITE
If you use this code in your research, please cite:

```
@InProceedings{atmani_et_al:OASIcs.LDK.2021.24,
  author =	{Atmani, Massinissa and Lafourcade, Mathieu},
  title =	{{Universal Dependencies for Multilingual Open Information Extraction}},
  booktitle =	{3rd Conference on Language, Data and Knowledge (LDK 2021)},
  pages =	{24:1--24:15},
  series =	{Open Access Series in Informatics (OASIcs)},
  ISBN =	{978-3-95977-199-3},
  ISSN =	{2190-6807},
  year =	{2021},
  volume =	{93},
  editor =	{Gromann, Dagmar and S\'{e}rasset, Gilles and Declerck, Thierry and McCrae, John P. and Gracia, Jorge and Bosque-Gil, Julia and Bobillo, Fernando and Heinisch, Barbara},
  publisher =	{Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
  address =	{Dagstuhl, Germany},
  URL =		{https://drops.dagstuhl.de/opus/volltexte/2021/14560},
  URN =		{urn:nbn:de:0030-drops-145600},
  doi =		{10.4230/OASIcs.LDK.2021.24},
  annote =	{Keywords: Natural Language Processing, Information Extraction, Machine Learning}
}

@inproceedings{atmani:hal-03265879,
  TITLE = {{Open Information Extraction: Approche Supervis{\'e}e et Syntaxique pour le Fran{\c c}ais}},
  AUTHOR = {Atmani, Massinissa and Lafourcade, Mathieu},
  URL = {https://hal.archives-ouvertes.fr/hal-03265879},
  BOOKTITLE = {{Traitement Automatique des Langues Naturelles}},
  ADDRESS = {Lille, France},
  EDITOR = {Denis, Pascal and Grabar, Natalia and Fraisse, Amel and Cardon, R{\'e}mi and Jacquemin, Bernard and Kergosien, Eric and Balvet, Antonio},
  PUBLISHER = {{ATALA}},
  PAGES = {50-63},
  YEAR = {2021},
  KEYWORDS = {Syntaxe. ; Extraction d'information ; Apprentissage machine},
  PDF = {https://hal.archives-ouvertes.fr/hal-03265879/file/1.pdf},
  HAL_ID = {hal-03265879},
  HAL_VERSION = {v1},
}
```

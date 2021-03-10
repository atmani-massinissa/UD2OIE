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
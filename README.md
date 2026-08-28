# Multivariate models for stroke cause prediction
This repository contains the code for training and evaluating multivariate models for stroke cause prediction, based on clincial data and features extracted from brain MRI. Details can be found in the manuscript.
````
add manuscript
`````

## Installation
First clone this repository. We recommend creating a dedicated conda environment and installing the packages there

````console 
git clone https://github.com/jul-des/multivariate-stroke-cause-models.git
cd multivariate-stroke-cause-models
conda create -n stroke-models python=3.10 -y
conda activate stroke-models
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
````

## Usage
The image processing pipeline used to extract imaging features for the models can be found under: https://github.com/jul-des/stroke-lesion-processing.

The final logistic regression models containing clinical and imaging data can be found in the directory models. The code for the nested cross validation can be found under model_training. Finally, the cod eused to train the final logistic regression models and plot results can be found under model_evaluation. The jupyter notebook "get_model_predicitons" can be used to generate model predictions for new data using the provided models.

## Citation

When using this code please cite:

`````
add manuscript
`````

## Questions
Questions can be directed at julian.deseoe@uzh.ch





# MLFBERT

MLFBERT: Advancing News Recommendation 
with Multi-Layer Fusion over BERT

My COMPSCI 791 project at the University of Auckland.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Acknowledgements](#acknowledgements)

## Requirements

The development environment has been exported to environment.yml

## Installation

Follow these steps to install MLFBERT:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/MLFBERT.git
   ```

2. Navigate to the project directory:
   ```bash
   cd MLFBERT
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   or
   conda env create -f environment.yml
   ```

## Usage

Download the MIND dataset from [here](https://msnews.github.io/). Extract the dataset and place it in the `MIND_DATASET` folder.

Download the GloVe embeddings from [here](https://nlp.stanford.edu/projects/glove/). Extract the embeddings and place it in the `MIND_DATASET` folder.

The structure of the `MIND_DATASET` folder should be as follows:
```
MIND_DATASET
    train
        behaviors.tsv
        news.tsv
        entity_embedding.vec
        relation_embedding.vec
    valid
        behaviors.tsv
        news.tsv
        entity_embedding.vec
        relation_embedding.vec
    test
        behaviors.tsv
        news.tsv
        entity_embedding.vec
        relation_embedding.vec
    
    glove.840B.300d.txt
```
Only behaviors.tsv and news.tsv are required for training and evaluation.

glove.840B.300d.txt is used for Word2vec model.

for the preporcessing, run the following command:
```bash
cd src
python 000.preprocess.py
```

To train the model from scratch and evaluate it on the validation set, run the following command:
```bash
cd src
python 001.train-models.py params/bert.yaml
```

To load from a checkpoint and evaluate it on the validation set, run the following command:
```bash
cd src
python src\003.pred-models.py
```

To make submission, run the following command:
```bash
cd src
python src\005.make-submission.py
```

## Model

4 models are implemented in this project:
- NRMS
- PLM-NR
- Fusion-NR
- MLFBERT

To switch between models, change the `base_model` parameter in the `src\params\bert.yaml` and `src\params\word2vec.yaml` files.

## Acknowledgements

* [aqweteddy/NRMS-Pytorch](https://github.com/aqweteddy/NRMS-Pytorch)

* [akirasosa/nrms-bert](https://github.com/akirasosa/nrms-bert)

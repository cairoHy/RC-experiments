# Reading Comprehension Experiments

## About This Repo

This is the tensorflow version implementation/reproduce of some reading comprehension models in some reading comprehension datasets including the following:

Models:

- Attention Sum Reader model as presented in "Text Comprehension with the Attention Sum Reader Network" (ACL2016) available at [http://arxiv.org/abs/1603.01547](http://arxiv.org/abs/1603.01547). 
- Attention over Attention Reader model as presented in "Attention-over-Attention Neural Networks for Reading Comprehension" (arXiv2016.7) available at https://arxiv.org/abs/1607.04423.

Datasets:

- CBT.http://lanl.arxiv.org/pdf/1506.03340.pdf

## Quick Start

#### 1.Getting data

- Download the dataset used in this repo.

#### 2.Environment Preparation

- Python-64bit >= v3.5.
- Install require libraries using the following command.

```shell
pip install -r requirements.txt
```

- Install nltk punkt for tokenizer.

```shell
python -m nltk.downloader punkt
```

#### 3.Train the model

You can now train the model by entering the following command.

```shell
python main.py --train 1 --test 0 --data_dir [data_path] --train_file [**.txt] --valid_file [**.txt]
```

#### 4. Result of the model

All the trained results and corresponding config params are saved in sub directories of weight_path(by default the `weight` folder) named `args.json` and `result.json`.
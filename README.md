# Reading Comprehension Experiments

## About This Repo

This is the tensorflow version implementation/reproduce of some reading comprehension models including the following:

- Attention Sum Reader model as presented in "Text Comprehension with the Attention Sum Reader Network" (ACL2016) available at [http://arxiv.org/abs/1603.01547](http://arxiv.org/abs/1603.01547). 

## Quick Start

#### 1.Getting data

- Download the dataset used in this repo.

#### 2.Environment Preparation

- Python-64bit $\geq $ v3.5.
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
python main.py --data_dir data_path --train_file **.txt --valid_file **.txt
```




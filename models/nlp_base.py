import argparse
import json
import logging
import os
from pprint import pprint

import numpy as np
import tensorflow as tf

logger = logging.info


class NLPBase(object):
    """
    Base class for training models.
    """

    def __init__(self):
        self.sess = tf.Session()
        self.args = self.get_args()
        logging.basicConfig(filename=self.args.log_file,
                            level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%y-%m-%d %H:%M')

        # set random seed
        np.random.seed(self.args.random_seed)
        tf.set_random_seed(self.args.random_seed)

        self.save_args()

    def save_args(self):
        # save all params
        self.save_obj_to_json(vars(self.args), "args.json")
        pprint(vars(self.args), indent=4)

    def save_obj_to_json(self, obj, filename):
        if not os.path.exists(self.args.weight_path):
            os.mkdir(self.args.weight_path)
        file = os.path.join(self.args.weight_path, filename)
        with open(file, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, sort_keys=True, indent=4)

    def add_args(self, parser):
        pass

    def get_args(self):
        # TODO:Get args according to args.json
        parser = argparse.ArgumentParser()
        # basis argument
        parser.add_argument("--debug", default=False, help="is debug mode on or off")

        parser.add_argument("--train", default=True, help="train or not")

        parser.add_argument("--test", default=False, help="test or not")

        # TODO:Implement ensemble test
        parser.add_argument("--ensemble", default=False, help="ensemble test or not")

        parser.add_argument("--random_seed", default=2088, help="random seed")

        parser.add_argument("--log_file", default=None, help="which file to save the log,if None,use screen")

        parser.add_argument("--weight_path", default="weights", help="path to save all trained models")

        # data specific argument
        parser.add_argument("--data_root", default="D:/source/data/RC-Cloze-CBT/CBTest/CBTest/data/",
                            help="root path of the dataset")

        parser.add_argument("--tmp_dir", default="tmp", help="dataset specific tmp folder")

        parser.add_argument("--train_file", default="cbtest_NE_train.txt", help="train file")

        parser.add_argument("--valid_file", default="cbtest_NE_valid_2000ex.txt", help="validation file")

        parser.add_argument("--test_file", default="cbtest_NE_test_2500ex.txt", help="test file")

        parser.add_argument("--embedding_file", default="D:/source/data/embedding/glove.6B/glove.6B.200d.txt",
                            type=str, help="pre-trained embedding file")

        parser.add_argument("--max_vocab_num", default=100000, help="the max number of words in vocabulary")

        # hyper-parameters
        parser.add_argument("--embedding_dim", default=200, help="dimension of word embeddings")

        parser.add_argument("--batch_size", default=32, help="batch_size")

        parser.add_argument("--keep_prob", default=0.5, help="dropout,percentage to keep during training")

        parser.add_argument("--grad_clipping", default=10, help="the threshold value of gradient clip")

        # train specific
        parser.add_argument("--optimizer", default="ADAM", choices=["SGD", "ADAM"],
                            help="optimize algorithms, SGD or Adam")

        parser.add_argument("--print_every_n", default=10, help="print performance every n steps")

        parser.add_argument("--evaluate_every_n", default=400,
                            help="evaluate performance on validation set and possibly saving the best model")

        parser.add_argument("--num_epoches", default=10, help="max epoch iterations")

        parser.add_argument("--patience", default=5, help="early stopping patience")
        # -----------------------------------------------------------------------------------------------------------

        self.add_args(parser)

        args = parser.parse_args()

        # set debug params
        args.max_count = 22 * 32 * 10 + 22 * 22 if args.debug else None
        args.evaluate_every_n = 5 if args.debug else args.evaluate_every_n
        args.num_epoches = 2 if args.debug else args.num_epoches

        return args

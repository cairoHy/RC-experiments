import os
from functools import reduce

import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.platform.gfile import FastGFile

from dataset.cloze_dataset import ClozeDataset
from models.nlp_base import logger


class CBT(ClozeDataset):
    def __init__(self, args):
        self.A_len = 10
        super().__init__(args)

    def cbt_data_to_token_ids(self, data_file, target_file, vocab_file, max_count=None):
        """
        22 lines for one sample.
        first 20 lines：documents with line number in the front.
        21st line：line-number question\tAnswer\t\tCandidate1|...|Candidate10.
        22nd line：blank.
        """
        if gfile.Exists(target_file):
            return
        logger("Tokenizing data in {}".format(data_file))
        word_dict = self.load_vocab(vocab_file)
        counter = 0

        with gfile.FastGFile(data_file) as f:
            with gfile.FastGFile(target_file, mode="wb") as tokens_file:
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        logger("Tokenizing line %d" % counter)
                    if max_count and counter > max_count:
                        break
                    if counter % 22 == 21:
                        q, a, _, A = line.split("\t")
                        token_ids_q = self.sentence_to_token_ids(q, word_dict)[1:]
                        token_ids_A = [word_dict.get(a.lower(), self.UNK_ID) for a in A.rstrip("\n").split("|")]
                        tokens_file.write(" ".join([str(tok) for tok in token_ids_q]) + "\t"
                                          + str(word_dict.get(a.lower(), self.UNK_ID)) + "\t"
                                          + "|".join([str(tok) for tok in token_ids_A]) + "\n")
                    else:
                        token_ids = self.sentence_to_token_ids(line, word_dict)
                        token_ids = token_ids[1:] if token_ids else token_ids
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

    def prepare_cbt_data(self, data_dir, train_file, valid_file, test_file, max_vocab_num, output_dir=""):
        """
        build vocabulary and translate CBT data to id format.
        """
        if not gfile.Exists(os.path.join(data_dir, output_dir)):
            os.mkdir(os.path.join(data_dir, output_dir))
        os_train_file = os.path.join(data_dir, train_file)
        os_valid_file = os.path.join(data_dir, valid_file)
        os_test_file = os.path.join(data_dir, test_file)
        idx_train_file = os.path.join(data_dir, output_dir, train_file + ".%d.idx" % max_vocab_num)
        idx_valid_file = os.path.join(data_dir, output_dir, valid_file + ".%d.idx" % max_vocab_num)
        idx_test_file = os.path.join(data_dir, output_dir, test_file + ".%d.idx" % max_vocab_num)
        vocab_file = os.path.join(data_dir, output_dir, "vocab.%d" % max_vocab_num)

        if not gfile.Exists(vocab_file):
            word_counter = self.gen_vocab(os_train_file, max_count=self.args.max_count)
            word_counter = self.gen_vocab(os_valid_file, old_counter=word_counter, max_count=self.args.max_count)
            word_counter = self.gen_vocab(os_test_file, old_counter=word_counter, max_count=self.args.max_count)
            self.save_vocab(word_counter, vocab_file, max_vocab_num)

        # translate train/valid/test files to id format
        self.cbt_data_to_token_ids(os_train_file, idx_train_file, vocab_file, max_count=self.args.max_count)
        self.cbt_data_to_token_ids(os_valid_file, idx_valid_file, vocab_file, max_count=self.args.max_count)
        self.cbt_data_to_token_ids(os_test_file, idx_test_file, vocab_file, max_count=self.args.max_count)

        return vocab_file, idx_train_file, idx_valid_file, idx_test_file

    def read_cbt_data(self, file, max_count=None):
        """
        read CBT data in id format.
        :return: (documents,questions,answers,candidates) each elements is a numpy array.
        """
        documents, questions, answers, candidates = [], [], [], []
        with FastGFile(file, mode="r") as f:
            counter = 0
            d, q, a, A = [], [], [], []
            for line in f:
                counter += 1
                if max_count and counter > max_count:
                    break
                if counter % 100000 == 0:
                    logger("Reading line %d in %s" % (counter, file))
                if counter % 22 == 21:
                    tmp = line.strip().split("\t")
                    q = tmp[0].split(" ") + [self.EOS_ID]
                    a = [1 if tmp[1] == i else 0 for i in d]
                    A = [a for a in tmp[2].split("|")]
                    A.remove(tmp[1])
                    A.insert(0, tmp[1])  # put right answer in the first of candidate
                elif counter % 22 == 0:
                    documents.append(d)
                    questions.append(q)
                    answers.append(a)
                    candidates.append(A)
                    d, q, a, A = [], [], [], []
                else:
                    d.extend(line.strip().split(" ") + [self.EOS_ID])  # add EOS ID in the end of each sentence

        d_lens = [len(i) for i in documents]
        q_lens = [len(i) for i in questions]
        avg_d_len = reduce(lambda x, y: x + y, d_lens) / len(documents)
        logger("Document average length: %d." % avg_d_len)
        logger("Document midden length: %d." % len(sorted(documents, key=len)[len(documents) // 2]))
        avg_q_len = reduce(lambda x, y: x + y, q_lens) / len(questions)
        logger("Question average length: %d." % avg_q_len)
        logger("Question midden length: %d." % len(sorted(questions, key=len)[len(questions) // 2]))

        return documents, questions, answers, candidates

    # noinspection PyAttributeOutsideInit
    def get_data_stream(self):
        # prepare data
        vocab_file, idx_train_file, idx_valid_file, idx_test_file = self.prepare_cbt_data(
            self.args.data_root, self.args.train_file, self.args.valid_file,
            self.args.test_file, self.args.max_vocab_num,
            output_dir=self.args.tmp_dir)

        # read data
        self.train_data = self.read_cbt_data(idx_train_file, max_count=self.args.max_count)
        self.valid_data = self.read_cbt_data(idx_valid_file, max_count=self.args.max_count)

        def get_max_length(d_bt):
            lens = [len(i) for i in d_bt]
            return max(lens)

        # data statistics
        self.d_len = get_max_length(self.train_data[0])
        self.q_len = get_max_length(self.train_data[1])
        self.train_sample_num = len(self.train_data[0])
        self.valid_sample_num = len(self.valid_data[0])
        self.train_idx = np.random.permutation(self.train_sample_num // self.args.batch_size)
        self.test_sample_num = 0

        if self.args.test:
            self.test_data = self.read_cbt_data(idx_test_file, max_count=self.args.max_count)
            self.test_sample_num = len(self.test_data[0])

        return vocab_file, self.d_len, self.q_len, self.train_sample_num, self.valid_sample_num, self.test_sample_num

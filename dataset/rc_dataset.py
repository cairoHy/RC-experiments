import abc
import codecs
import re
from collections import Counter

import nltk
import numpy as np
from tensorflow.python.platform import gfile

from utils.log import logger


def default_tokenizer(sentence):
    _DIGIT_RE = re.compile(r"\d+")
    sentence = _DIGIT_RE.sub("0", sentence)
    sentence = " ".join(sentence.split("|"))
    return nltk.word_tokenize(sentence.lower())


# noinspection PyAttributeOutsideInit
class RCDataset(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        # padding,start of sentence,end of sentence,unk,end of question
        self._PAD = "_PAD"
        self._BOS = "_BOS"
        self._EOS = "_EOS"
        self._UNK = "_UNK"
        self._EOQ = "_EOQ"
        self._START_VOCAB = [self._PAD, self._BOS, self._EOS, self._UNK, self._EOQ]
        self.PAD_ID = 0
        self.BOS_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3
        self.EOQ_ID = 4

        self._BLANK = "XXXXX"

        # special character of char embedding: pad and unk
        self._CHAR_PAD = "γ"
        self._CHAR_UNK = "δ"
        self.CHAR_PAD_ID = 0
        self.CHAR_UNK_ID = 1
        self._CHAR_START_VOCAB = [self._CHAR_PAD,self._CHAR_UNK]

    @property
    def train_idx(self):
        return self._train_idx

    @train_idx.setter
    def train_idx(self, value):
        self._train_idx = value

    @property
    def train_sample_num(self):
        return self._train_sample_num

    @train_sample_num.setter
    def train_sample_num(self, value):
        self._train_sample_num = value

    @property
    def valid_sample_num(self):
        return self._valid_sample_num

    @valid_sample_num.setter
    def valid_sample_num(self, value):
        self._valid_sample_num = value

    @property
    def test_sample_num(self):
        return self._test_sample_num

    @test_sample_num.setter
    def test_sample_num(self, value):
        self._test_sample_num = value

    def shuffle(self):
        logger("Shuffle the dataset.")
        np.random.shuffle(self.train_idx)

    def get_next_batch(self, mode, idx):
        """
        return next batch of data samples
        """
        batch_size = self.args.batch_size
        if mode == "train":
            dataset = self.train_data
            sample_num = self.train_sample_num
        elif mode == "valid":
            dataset = self.valid_data
            sample_num = self.valid_sample_num
        else:
            dataset = self.test_data
            sample_num = self.test_sample_num
        if mode == "train":
            start = self.train_idx[idx] * batch_size
            stop = (self.train_idx[idx] + 1) * batch_size
        else:
            start = idx * batch_size
            stop = (idx + 1) * batch_size if start < sample_num and (idx + 1) * batch_size < sample_num else -1
        samples = batch_size if stop != -1 else len(dataset[0]) - start
        _slice = np.index_exp[start:stop]
        data = {
            "questions_bt:0": dataset[0][_slice],
            "documents_bt:0": dataset[1][_slice],
            "candidates_bi:0": dataset[2][_slice],
            "y_true_bi:0": dataset[3][_slice]
        }
        return data, samples

    @staticmethod
    def gen_embeddings(word_dict, embed_dim, in_file=None, init=np.zeros):
        """
        Init embedding matrix with (or without) pre-trained word embeddings.
        """
        num_words = max(word_dict.values()) + 1
        embedding_matrix = init(-0.05, 0.05, (num_words, embed_dim))
        logger('Embeddings: %d x %d' % (num_words, embed_dim))

        if not in_file:
            return embedding_matrix

        def get_dim(file):
            first = gfile.FastGFile(file, mode='r').readline()
            return len(first.split()) - 1

        assert get_dim(in_file) == embed_dim
        logger('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in codecs.open(in_file, encoding="utf-8"):
            sp = line.split()
            if sp[0] in word_dict:
                pre_trained += 1
                embedding_matrix[word_dict[sp[0]]] = np.asarray([float(x) for x in sp[1:]], dtype=np.float32)
        logger("Pre-trained: {}, {:.3f}%".format(pre_trained, pre_trained * 100.0 / num_words))
        return embedding_matrix

    def sentence_to_token_ids(self, sentence, word_dict, tokenizer=default_tokenizer):
        """
        Turn sentence to token ids.
            sentence:       ["I", "have", "a", "dog"]
            word_list:      {"I": 1, "have": 2, "a": 4, "dog": 7"}
            return:         [1, 2, 4, 7]
        """
        return [word_dict.get(token, self.UNK_ID) for token in tokenizer(sentence)]

    def get_embedding_matrix(self, vocab_file, is_char_embedding=False):
        """
        :param is_char_embedding: is the function called for generate char embedding
        :param vocab_file: file containing saved vocabulary.
        :return: a dict with each key as a word, each value as its corresponding embedding vector.
        """
        word_dict = self.load_vocab(vocab_file)
        embedding_file = None if is_char_embedding else self.args.embedding_file
        embedding_dim = self.args.char_embedding_dim if is_char_embedding else self.args.embedding_dim
        embedding_matrix = self.gen_embeddings(word_dict,
                                               embedding_dim,
                                               embedding_file,
                                               init=np.random.uniform)
        return embedding_matrix

    def sort_by_length(self, data):
        # TODO: sort data array according to sequence length in order to speed up training
        pass

    @staticmethod
    def gen_char_vocab(data_file, tokenizer=default_tokenizer, old_counter=None):
        """
         generate character level vocabulary according to train corpus.
        """
        logger("Creating character dict from data {}.".format(data_file))
        char_counter = old_counter if old_counter else Counter()
        with gfile.FastGFile(data_file) as f:
            for line in f:
                tokens = tokenizer(line.rstrip("\n"))
                char_counter.update([char for word in tokens for char in word])

        # summary statistics
        total_chars = sum(char_counter.values())
        distinct_chars = len(list(char_counter))

        logger("STATISTICS" + "-" * 20)
        logger("Total characters: " + str(total_chars))
        logger("Total distinct characters: " + str(distinct_chars))
        return char_counter

    @staticmethod
    def gen_vocab(data_file, tokenizer=default_tokenizer, old_counter=None, max_count=None):
        """
        generate vocabulary according to train corpus.
        """
        logger("Creating word dict from data {}.".format(data_file))
        word_counter = old_counter if old_counter else Counter()
        counter = 0
        with gfile.FastGFile(data_file) as f:
            for line in f:
                counter += 1
                if max_count and counter > max_count:
                    break
                tokens = tokenizer(line.rstrip('\n'))
                word_counter.update(tokens)
                if counter % 100000 == 0:
                    logger("Process line %d Done." % counter)

        # summary statistics
        total_words = sum(word_counter.values())
        distinct_words = len(list(word_counter))

        logger("STATISTICS" + "-" * 20)
        logger("Total words: " + str(total_words))
        logger("Total distinct words: " + str(distinct_words))

        return word_counter

    def save_char_vocab(self, char_counter, char_vocab_file, max_vocab_num=None):
        """
        Save character vocabulary.
        We need two special vo
        """
        with gfile.FastGFile(char_vocab_file, "w") as f:
            for char in self._CHAR_START_VOCAB:
                f.write(char + "\n")
            for char in list(map(lambda x: x[0], char_counter.most_common(max_vocab_num))):
                f.write(char + "\n")

    def save_vocab(self, word_counter, vocab_file, max_vocab_num=None):
        with gfile.FastGFile(vocab_file, "w") as f:
            for word in self._START_VOCAB:
                f.write(word + "\n")
            for word in list(map(lambda x: x[0], word_counter.most_common(max_vocab_num))):
                f.write(word + "\n")

    @staticmethod
    def load_vocab(vocab_file):
        """
        load word(or char) vocabulary file to word/char dict
        """
        if not gfile.Exists(vocab_file):
            raise ValueError("Vocabulary file %s not found.", vocab_file)
        word_dict = {}
        word_id = 0
        for line in codecs.open(vocab_file, encoding="utf-8"):
            word_dict.update({line.strip(): word_id})
            word_id += 1
        return word_dict

    # noinspection PyAttributeOutsideInit
    def preprocess(self):
        self.train_data = self.preprocess_input_sequences(self.train_data)
        self.valid_data = self.preprocess_input_sequences(self.valid_data)
        if self.args.test:
            self.test_data = self.preprocess_input_sequences(self.test_data)

    @abc.abstractmethod
    def preprocess_input_sequences(self, data):
        """
        Preprocess train/valid/test data. Should be specified by sub class.
        """
        pass

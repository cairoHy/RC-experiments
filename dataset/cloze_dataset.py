import codecs
import re
from collections import Counter

import nltk
import numpy as np
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.platform import gfile
from models.nlp_base import logger


def default_tokenizer(sentence):
    _DIGIT_RE = re.compile(r"\d+")
    sentence = _DIGIT_RE.sub("0", sentence)
    sentence = " ".join(sentence.split("|"))
    return nltk.word_tokenize(sentence.lower())


# noinspection PyUnresolvedReferences
class ClozeDataset(object):
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

    def shuffle(self):
        logger("Shuffle the dataset.")
        np.random.shuffle(self.train_idx)

    def get_next_batch(self, mode, idx):
        """
        return next batch of cloze data samples 
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
        start = idx * batch_size
        stop = (idx + 1) * batch_size if start < sample_num and (idx + 1) * batch_size < sample_num else -1
        samples = batch_size if stop != -1 else len(dataset[0]) - start
        _slice = np.index_exp[start:stop]
        data = {
            "questions_bt:0": dataset[0][_slice],
            "context_bt:0": dataset[1][_slice],
            "candidates_bi:0": dataset[2][_slice],
            "y_true_bi:0": dataset[3][_slice]
        }
        return data, samples

    @staticmethod
    def gen_embeddings(word_dict, embed_dim, in_file=None, init=np.zeros):
        """
        为词表建立一个初始化的词向量矩阵，如果某个词不在词向量文件中，会随机初始化一个向量。

        :param word_dict: 词到id的映射。
        :param embed_dim: 词向量的维度。
        :param in_file: 预训练的词向量文件。 
        :param init: 对于预训练文件中找不到的词，如何初始化。
        :return: 词向量矩阵。
        """
        num_words = max(word_dict.values()) + 1
        embedding_matrix = init(-0.1, 0.1, (num_words, embed_dim))
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
        把句子中的单词转化为相应的ID。
        比如：
            句子：["I", "have", "a", "dog"]
            word_list：{"I": 1, "have": 2, "a": 4, "dog": 7"}
            返回：[1, 2, 4, 7]
        """
        return [word_dict.get(token, self.UNK_ID) for token in tokenizer(sentence)]

    def get_embedding_matrix(self, vocab_file):
        # 初始化词向量矩阵，使用(-0.1,0.1)区间内的随机均匀分布
        word_dict = self.load_vocab(vocab_file)
        embedding_matrix = self.gen_embeddings(word_dict,
                                               self.args.embedding_dim,
                                               self.args.embedding_file,
                                               init=np.random.uniform)
        return embedding_matrix

    def sort_by_length(self, data):
        # TODO: sort data array according to sequence length in order to speed up training
        pass

    @staticmethod
    def gen_vocab(data_file, tokenizer=default_tokenizer, old_counter=None, max_count=None):
        """
        分析语料库，根据词频返回词典。
        """
        logger("Creating word_dict from data %s" % data_file)
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

    def save_vocab(self, word_counter, vocab_file, max_vocab_num=None):
        with gfile.FastGFile(vocab_file, "w") as f:
            for word in self._START_VOCAB:
                f.write(word + "\n")
            for word in list(map(lambda x: x[0], word_counter.most_common(max_vocab_num))):
                f.write(word + "\n")

    @staticmethod
    def load_vocab(vocab_file):
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

    def preprocess_input_sequences(self, data):
        """
        预处理输入：
        shuffle
        PAD/TRUNC到固定长度的序列
        y_true是长度为self.A_len的向量，index=0为正确答案，one-hot编码
        """
        documents, questions, answer, candidates = data

        questions_ok = pad_sequences(questions, maxlen=self.q_len, dtype="int32", padding="post", truncating="post")
        documents_ok = pad_sequences(documents, maxlen=self.d_len, dtype="int32", padding="post", truncating="post")
        candidates_ok = pad_sequences(candidates, maxlen=self.A_len, dtype="int32", padding="post", truncating="post")
        y_true = np.zeros_like(candidates_ok)
        y_true[:, 0] = 1
        return questions_ok, documents_ok, candidates_ok, y_true

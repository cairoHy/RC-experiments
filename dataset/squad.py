import json
import os

import numpy as np
from tensorflow.contrib.keras.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.platform import gfile

from dataset.rc_dataset import RCDataset
from utils.log import logger


class SQuAD(RCDataset):
    def __init__(self, args):
        super(SQuAD, self).__init__(args)
        self.w_len = 10

    def next_batch_feed_dict_by_dataset(self, dataset, _slice, samples):
        data = {
            "documents_bt:0": dataset[0][_slice],
            "questions_bt:0": dataset[1][_slice],
            # TODO: substitute with real data
            "documents_btk:0": np.zeros([samples, self.d_len, self.w_len]),
            "questions_btk:0": np.zeros([samples, self.q_len, self.w_len]),
            "answer_start:0": dataset[2][_slice],
            "answer_end:0": dataset[3][_slice]
        }
        return data, samples

    def preprocess_input_sequences(self, data):
        documents, questions, answer_spans = data
        documents_ok = pad_sequences(documents, maxlen=self.d_len, dtype="int32", padding="post", truncating="post")
        questions_ok = pad_sequences(questions, maxlen=self.q_len, dtype="int32", padding="post", truncating="post")
        answer_start = [np.array([int(i == answer_span[0]) for i in range(self.d_len)]) for answer_span in answer_spans]
        answer_end = [np.array([int(i == answer_span[1]) for i in range(self.d_len)]) for answer_span in answer_spans]
        return documents_ok, questions_ok, np.asarray(answer_start), np.asarray(answer_end)

    def prepare_data(self, data_dir, train_file, valid_file, max_vocab_num, output_dir=""):
        """
        build word vocabulary and character vocabulary.
        """
        if not gfile.Exists(os.path.join(data_dir, output_dir)):
            os.mkdir(os.path.join(data_dir, output_dir))
        os_train_file = os.path.join(data_dir, train_file)
        os_valid_file = os.path.join(data_dir, valid_file)
        vocab_file = os.path.join(data_dir, output_dir, "vocab.%d" % max_vocab_num)
        char_vocab_file = os.path.join(data_dir, output_dir, "char_vocab")

        vocab_data_file = os.path.join(data_dir, output_dir, "data.txt")

        def save_data(d_data, q_data):
            """
            save all data to a file and use it build vocabulary.
            """
            with open(vocab_data_file, mode="w", encoding="utf-8") as f:
                f.write("\t".join(d_data) + "\n")
                f.write("\t".join(q_data) + "\n")

        if not gfile.Exists(vocab_data_file):
            d, q, _ = self.read_squad_data(os_train_file)
            v_d, v_q, _ = self.read_squad_data(os_valid_file)
            save_data(d, q)
            save_data(v_d, v_q)
        if not gfile.Exists(vocab_file):
            logger("Start create vocabulary.")
            word_counter = self.gen_vocab(vocab_data_file, max_count=self.args.max_count)
            self.save_vocab(word_counter, vocab_file, max_vocab_num)
        if not gfile.Exists(char_vocab_file):
            logger("Start create character vocabulary.")
            char_counter = self.gen_char_vocab(vocab_data_file)
            self.save_char_vocab(char_counter, char_vocab_file, max_vocab_num=70)

        return os_train_file, os_valid_file, vocab_file, char_vocab_file

    def read_squad_data(self, file):
        """
        read squad data file in string form
        :return tuple of (documents, questions, answer_spans)
        """
        logger("Reading SQuAD data.")

        def extract(sample_data):
            document = sample_data["context"]
            for qas in sample_data["qas"]:
                question = qas["question"]
                for ans in qas["answers"]:
                    answer_len = len(ans["text"])
                    answer_span = [ans["answer_start"], ans["answer_start"] + answer_len]
                    assert (ans["text"] == document[ans["answer_start"]:(ans["answer_start"] + answer_len)])
                    documents.append(document)
                    questions.append(question)
                    answer_spans.append(answer_span)

        documents, questions, answer_spans = [], [], []
        f = json.load(open(file, encoding="utf-8"))
        data_list, version = f["data"], f["version"]
        logger("SQuAD version: {}".format(version))
        [extract(sample) for data in data_list for sample in data["paragraphs"]]
        if self.args.debug:
            documents, questions, answer_spans = documents[:500], questions[:500], answer_spans[:500]

        return documents, questions, answer_spans

    def squad_data_to_idx(self, vocab_file, *args):
        """
        convert string list to index list form.         
        """
        logger("Convert string data to index.")
        word_dict = self.load_vocab(vocab_file)
        res_data = [0, ] * len(args)
        for idx, i in enumerate(args):
            tmp = [self.sentence_to_token_ids(document, word_dict) for document in i]
            res_data[idx] = tmp.copy()
        logger("Convert string2index done.")
        return res_data

    # noinspection PyAttributeOutsideInit
    def get_data_stream(self):
        # prepare data
        os_train_file, os_valid_file, self.vocab_file, self.char_vocab_file = self.prepare_data(self.args.data_root,
                                                                                                self.args.train_file,
                                                                                                self.args.valid_file,
                                                                                                self.args.max_vocab_num,
                                                                                                self.args.tmp_dir)

        # read data
        documents, questions, answer_spans = self.read_squad_data(os_train_file)
        v_documents, v_questions, v_answer_spans = self.read_squad_data(os_valid_file)
        documents, questions, v_documents, v_questions = self.squad_data_to_idx(self.vocab_file, documents, questions,
                                                                                v_documents, v_questions)
        # SQuAD cannot access the test data
        # first 9/10 train data     ->     train data
        # last  1/10 train data     ->     valid data
        # valid data                ->     test data
        train_num = len(documents) * 9 // 10
        self.train_data = (documents[:train_num], questions[:train_num], answer_spans[:train_num])
        self.valid_data = (documents[train_num:], questions[train_num:], answer_spans[train_num:])
        self.test_data = (v_documents, v_questions, v_answer_spans)

        def get_max_length(d_bt):
            lens = [len(i) for i in d_bt]
            return max(lens)

        # data statistics
        self.d_len = get_max_length(self.train_data[0])
        self.q_len = get_max_length(self.train_data[1])
        self.train_sample_num = len(self.train_data[0])
        self.valid_sample_num = len(self.valid_data[0])
        self.test_sample_num = len(self.test_data[0])
        self.train_idx = np.random.permutation(self.train_sample_num // self.args.batch_size)

        return self.d_len, self.q_len, self.train_sample_num, self.valid_sample_num, self.test_sample_num

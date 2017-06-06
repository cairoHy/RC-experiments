import argparse
import logging
import sys
import unittest

# noinspection PyUnresolvedReferences
import dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(filename=None,
                            level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%y-%m-%d %H:%M')
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", default=True, type=str, help="is debug mode on or off")

        parser.add_argument("--data_root", default="../data/SQuAD/",
                            help="root path of the dataset")

        parser.add_argument("--tmp_dir", default="tmp", help="dataset specific tmp folder")

        parser.add_argument("--train_file", default="train-v1.1.json", help="train file")

        parser.add_argument("--valid_file", default="dev-v1.1.json", help="validation file")

        parser.add_argument("--max_count", default=None, type=int,
                            help="read n lines of data file, if None, read all data")

        parser.add_argument("--max_vocab_num", default=100000, type=int, help="the max number of words in vocabulary")

        parser.add_argument("--batch_size", default=32, type=int, help="batch_size")

        parser.add_argument("--train", default=True, type=bool, help="train or not")

        parser.add_argument("--test", default=True, type=bool, help="test or not")

        self.args = parser.parse_known_args()[0]


class TestCBT(TestDataset):
    def runTest(self):
        self.args.data_root = "../data/CBTest/CBTest/data/"
        self.args.train_file = "cbtest_NE_train.txt"
        self.args.valid_file = "cbtest_NE_valid_2000ex.txt"
        self.args.test_file = "cbtest_NE_test_2500ex.txt"
        self.dataset = getattr(sys.modules["dataset"], "CBT")(self.args)
        statistics = self.dataset.get_data_stream()
        for i in statistics[1:]:
            self.assertEqual(type(i), int, "Some data statistic not int.")
            self.assertGreater(i, 0, "Some data number not greater than zero.")


class TestSQuAD(TestDataset):
    def runTest(self):
        self.dataset = getattr(sys.modules["dataset"], "SQuAD")(self.args)
        data_dir, train_file, valid_file = self.args.data_root, self.args.train_file, self.args.valid_file
        max_vocab_num, output_dir = self.args.max_vocab_num, self.args.tmp_dir

        os_train_file, os_valid_file, vocab_file, char_vocab_file = self.dataset.prepare_data(data_dir, train_file,
                                                                                              valid_file, max_vocab_num,
                                                                                              output_dir)

        documents, questions, answer_spans = self.dataset.read_squad_data(os_train_file)
        v_documents, v_questions, v_answer_spans = self.dataset.read_squad_data(os_valid_file)
        data = self.dataset.squad_data_to_idx(vocab_file, documents, questions,
                                              v_documents, v_questions)
        # make sure that each one of (d,q,v_d,v_q) is a list, and each element is a list too.
        for i in data:
            self.assertEqual(type(i), list, "some data in train set or valid set is not a list.")
            self.assertGreater(len(i), 0, "some data in train set or valid set is None.")
            self.assertEqual(type(i[0]), list, "some elements in train set or valid set is not a list.")
            for word in i[0]:
                self.assertEqual(type(word), int, "Not all the word is index form.")
                self.assertGreaterEqual(word, 0, "Invalid index for some word.")

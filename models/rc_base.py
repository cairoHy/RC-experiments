import abc
import os
import sys

import tensorflow as tf

# noinspection PyUnresolvedReferences
import dataset
from models import models_in_datasets
from models.nlp_base import NLPBase
from utils.log import logger, save_obj_to_json, err


# noinspection PyAttributeOutsideInit
class RcBase(NLPBase, metaclass=abc.ABCMeta):
    """
    Base class of reading comprehension experiments.
    Reads different reading comprehension datasets according to specific class.
    creates a model and starts training it.
    Any deep learning model should inherit from this class and implement the create_model method.
    """

    def __init__(self):
        super(RcBase, self).__init__()
        self.model_name = self.__class__.__name__

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        self._loss = value

    @property
    def correct_prediction(self):
        return self._correct_prediction

    @correct_prediction.setter
    def correct_prediction(self, value):
        self._correct_prediction = value

    def get_train_op(self):
        """
        define optimization operation 
        """
        if self.args.optimizer == "SGD":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.args.lr)
        elif self.args.optimizer == "ADAM":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        else:
            raise NotImplementedError("Other Optimizer Not Implemented.-_-||")

        # gradient clip
        grad_vars = optimizer.compute_gradients(self.loss)
        grad_vars = [
            (tf.clip_by_norm(grad, self.args.grad_clipping), var)
            if grad is not None else (grad, var)
            for grad, var in grad_vars]
        self.train_op = optimizer.apply_gradients(grad_vars, self.step)
        return

    @abc.abstractmethod
    def create_model(self):
        """
        should be override by sub-class and create some operations include [loss, correct_prediction]
        as class fields.
        """
        return

    def execute(self):
        """
        main method to train and test
        """
        self.confirm_fitness()

        self.dataset = getattr(sys.modules["dataset"], self.args.dataset)(self.args)

        # Get the statistics of data
        # [document length] and [question length] to build the model
        # train/valid/test sample number to train and validate and test the model
        statistics = self.dataset.get_data_stream()
        self.d_len, self.q_len, self.train_nums, self.valid_nums, self.test_num = statistics
        self.dataset.preprocess()

        # Get the word embedding and character embedding(if necessary)
        self.embedding_matrix = self.dataset.get_embedding_matrix(self.dataset.vocab_file)
        if self.args.use_char_embedding and getattr(self.dataset, "char_vocab_file"):
            self.char_embedding_matrix = self.dataset.get_embedding_matrix(self.dataset.char_vocab_file, True)

        self.create_model()

        self.saver = tf.train.Saver(max_to_keep=20)

        if self.args.train:
            self.train()
        if self.args.test:
            self.test()

        self.sess.close()

    def get_batch_data(self, mode, idx):
        """
        Get batch data and feed it to tensorflow graph
        Modify it in sub-class if needed.
        """
        return self.dataset.get_next_batch(mode, idx)

    def train(self):
        """
        train model
        """
        self.step = tf.Variable(0, name="global_step", trainable=False)
        batch_size = self.args.batch_size
        epochs = self.args.num_epoches
        self.get_train_op()
        self.sess.run(tf.global_variables_initializer())
        self.load_weight()

        # early stopping params, by default val_acc is the metric
        self.patience, self.best_val_acc = self.args.patience, 0.
        # Start training
        corrects_in_epoch, samples_in_epoch, loss_in_epoch = 0, 0, 0
        batch_num = self.train_nums // batch_size
        logger("Train on {} batches, {} samples per batch, {} total.".format(batch_num, batch_size, self.train_nums))

        step = self.sess.run(self.step)
        while step < batch_num * epochs:
            step = self.sess.run(self.step)
            # on Epoch start
            if step % batch_num == 0:
                corrects_in_epoch, samples_in_epoch, loss_in_epoch = 0, 0, 0
                logger("{}Epoch : {}{}".format("-" * 40, step // batch_num + 1, "-" * 40))
                self.dataset.shuffle()

            data, samples = self.get_batch_data("train", step % batch_num)
            loss, _, corrects_in_batch = self.sess.run([self.loss, self.train_op, self.correct_prediction],
                                                       feed_dict=data)
            corrects_in_epoch += corrects_in_batch
            loss_in_epoch += loss * samples
            samples_in_epoch += samples

            # logger
            if step % self.args.print_every_n == 0:
                logger("Samples : {}/{}.\tStep : {}/{}.\tLoss : {:.4f}.\tAccuracy : {:.4f}".format(
                    samples_in_epoch, self.train_nums,
                    step % batch_num, batch_num,
                    loss_in_epoch / samples_in_epoch, corrects_in_epoch / samples_in_epoch))

            # evaluate on the valid set and early stopping
            if step and step % self.args.evaluate_every_n == 0:
                val_acc, val_loss = self.validate()
                self.early_stopping(val_acc, val_loss, step)

    def validate(self):
        batch_size = self.args.batch_size
        v_batch_num = self.valid_nums // batch_size
        # ensure the entire valid set is selected
        v_batch_num = v_batch_num + 1 if (self.valid_nums % batch_size) != 0 else v_batch_num
        logger("Validate on {} batches, {} samples per batch, {} total."
               .format(v_batch_num, batch_size, self.valid_nums))
        val_num, val_corrects, v_loss = 0, 0, 0
        for i in range(v_batch_num):
            data, samples = self.get_batch_data("valid", i)
            if samples != 0:
                loss, v_correct = self.sess.run([self.loss, self.correct_prediction], feed_dict=data)
                val_num += samples
                val_corrects += v_correct
                v_loss += loss * samples
        assert (val_num == self.valid_nums)
        val_acc = val_corrects / val_num
        val_loss = v_loss / val_num
        logger("Evaluate on : {}/{}.\tVal acc : {:.4f}.\tVal Loss : {:.4f}".format(val_num,
                                                                                   self.valid_nums,
                                                                                   val_acc,
                                                                                   val_loss))
        return val_acc, val_loss

    # noinspection PyUnusedLocal
    def early_stopping(self, val_acc, val_loss, step):
        if val_acc > self.best_val_acc:
            self.patience = self.args.patience
            self.best_val_acc = val_acc
            self.save_weight(val_acc, step)
        elif self.patience == 1:
            logger("Oh u, stop training.")
            exit(0)
        else:
            self.patience -= 1
            logger("Remaining/Patience : {}/{} .".format(self.patience, self.args.patience))

    def save_weight(self, val_acc, step):
        path = self.saver.save(self.sess,
                               os.path.join(self.args.weight_path,
                                            "{}-val_acc-{:.4f}.models".format(self.model_name, val_acc)),
                               global_step=step)
        logger("Save models to {}.".format(path))

    def load_weight(self):
        ckpt = tf.train.get_checkpoint_state(self.args.weight_path)
        if ckpt is not None:
            logger("Load models from {}.".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            logger("No previous models.")

    def test(self):
        if not self.args.train:
            self.sess.run(tf.global_variables_initializer())
            self.load_weight()
        batch_size = self.args.batch_size
        batch_num = self.test_num // batch_size
        batch_num = batch_num + 1 if (self.test_num % batch_size) != 0 else batch_num
        correct_num, total_num = 0, 0
        for i in range(batch_num):
            data, samples = self.get_batch_data("test", i)
            if samples != 0:
                correct, = self.sess.run([self.correct_prediction], feed_dict=data)
                correct_num, total_num = correct_num + correct, total_num + samples
        assert (total_num == self.test_num)
        logger("Test on : {}/{}".format(total_num, self.test_num))
        test_acc = correct_num / total_num
        logger("Test accuracy is : {:.5f}".format(test_acc))
        res = {
            "model": self.model_name,
            "test_acc": test_acc
        }
        save_obj_to_json(self.args.weight_path, res, "result.json")

    def confirm_fitness(self):
        # make sure the models_in_datasets var is correct
        try:
            assert (models_in_datasets.get(self.args.dataset, None) is not None)
        except AssertionError:
            err("Models_in_datasets doesn't have the specified dataset key: {}.".format(self.args.dataset))
            exit(1)
        # make sure the model fit the dataset
        try:
            assert (self.model_name in models_in_datasets.get(self.args.dataset, None))
        except AssertionError:
            err("The model -> {} doesn't support the dataset -> {}".format(self.model_name, self.args.dataset))
            exit(1)

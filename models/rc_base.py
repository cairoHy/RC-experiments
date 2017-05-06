import abc
import os

import tensorflow as tf

from dataset.cbt import CBT
from models.nlp_base import NLPBase
from models.nlp_base import logger


# noinspection PyAttributeOutsideInit
class RcBase(NLPBase, metaclass=abc.ABCMeta):
    """
    Base class that reads different reading comprehension datasets, creates a model and starts training it.
    """

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

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

    def add_args(self, parser):
        # data specific
        parser.add_argument("--dataset", default="cbt", choices=["cbt", "cnn", "dailymail"],
                            help='type of the dataset to load')

        parser.add_argument("--d_len_range", default=(0, 2000), help="length scope of document to load")

        parser.add_argument("--q_len_range", default=(0, 200), help="length scope of question to load")

        # hyper-parameters
        parser.add_argument("--lr", default=0.001, help="learning rate")

        parser.add_argument("--hidden_size", default=128, help="RNN hidden size")

        parser.add_argument("--num_layers", default=1, help="RNN layer number")

        parser.add_argument("--use_lstm", default=False, help="RNN kind, if False, use GRU else LSTM")

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
        self.train_op = optimizer.apply_gradients(grad_vars)
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
        self.A_len = 10
        self.dataset = CBT(self.args)

        statistics = self.dataset.get_data_stream()
        vocab_file, self.d_len, self.q_len, self.train_nums, self.valid_nums, self.test_num = statistics
        self.dataset.preprocess()

        self.embedding_matrix = self.dataset.get_embedding_matrix(vocab_file)

        self.create_model()

        self.saver = tf.train.Saver(max_to_keep=20)

        if self.args.train:
            self.train()
        if self.args.test:
            self.test()

    def train(self):
        """
        train model
        """
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

        for step in range(batch_num * epochs):
            # on Epoch end
            if step % batch_num == 0:
                corrects_in_epoch, samples_in_epoch, loss_in_epoch = 0, 0, 0
                logger("{}Epoch : {}{}".format("-" * 40, step // batch_num + 1, "-" * 40))
                self.dataset.shuffle()

            data, samples = self.dataset.get_next_batch("train", step % batch_num)
            loss, _, corrects_in_batch = self.sess.run([self.loss, self.train_op, self.correct_prediction],
                                                       feed_dict=data)
            corrects_in_epoch += corrects_in_batch
            loss_in_epoch += loss * samples
            samples_in_epoch += samples

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
        v_batch_num = self.valid_nums // batch_size + 1
        logger("Validate on {} batches, {} samples per batch, {} total."
               .format(v_batch_num, batch_size, self.valid_nums))
        val_num, val_corrects, v_loss = 0, 0, 0
        for i in range(v_batch_num):
            data, samples = self.dataset.get_next_batch("valid", i)
            loss, v_correct = self.sess.run([self.loss, self.correct_prediction], feed_dict=data)
            val_num += samples
            val_corrects += v_correct
            v_loss += loss * samples
        # assert (val_num == self.valid_nums)
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
        batch_num = self.test_num // batch_size + 1
        correct_num, total_num = 0, 0
        for i in range(batch_num):
            data, samples = self.dataset.get_next_batch("test", i)
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
        self.save_obj_to_json(res, "result.json")

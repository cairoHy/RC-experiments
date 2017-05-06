import os

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell

from models.nlp_base import logger
from models.rc_base import RcBase


class AoAReader(RcBase):
    """
    Attention-over-Attention reader in "Attention-over-Attention Neural Networks for Reading Comprehension"
    (arXiv2016.7) available at https://arxiv.org/abs/1607.04423.
    """

    def __init__(self):
        super().__init__()
        self.model_name = os.path.basename(__file__)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        hidden_size = self.args.hidden_size
        self.max_steps = self.d_len + self.q_len
        logger(" [*] Building Bidirectional GRU layer...")
        self.fw_cell = GRUCell(hidden_size)
        self.bw_cell = GRUCell(hidden_size)
        self.initial_state_fw = self.fw_cell.zero_state(self.args.batch_size, tf.float32)
        self.initial_state_bw = self.bw_cell.zero_state(self.args.batch_size, tf.float32)

        init_embedding = tf.constant(self.embedding_matrix, dtype=tf.float32, name="embedding_init")
        embedding = tf.get_variable(initializer=init_embedding,
                                    name="embedding_matrix",
                                    dtype=tf.float32)
        if self.args.train and self.args.keep_prob < 1:
            embedding = tf.nn.dropout(embedding, self.args.keep_prob)

        inputs = tf.placeholder(tf.int32, [self.args.batch_size, self.max_steps], name="inputs")
        embed_inputs = tf.nn.embedding_lookup(embedding, inputs)

        _seq_len = tf.fill(tf.expand_dims(self.args.batch_size, 0),
                           tf.constant(self.max_steps, dtype=tf.int64))

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            self.fw_cell,
            self.bw_cell,
            embed_inputs,
            sequence_length=_seq_len,
            initial_state_fw=self.initial_state_fw,
            initial_state_bw=self.initial_state_bw,
            dtype=tf.float32)

        # concat output
        outputs = tf.concat(outputs, -1)

        # select document & query
        d = outputs[:, :self.d_len, :]
        q = outputs[:, self.d_len:, :]

        # batch pair-wise matching
        i_att = tf.matmul(d, q, adjoint_b=True)  # shape = (batch_size, self.d, self.q)
        # individual attentions
        alpha = tf.map_fn(lambda x: tf.nn.softmax(tf.transpose(x)), i_att)
        # attention-over-attentions
        beta_t = tf.map_fn(tf.nn.softmax, i_att)
        beta = tf.map_fn(lambda x: tf.reduce_mean(x, 0), beta_t)  # shape = (batch_size, self.q, )
        beta = tf.reshape(beta, [self.args.batch_size, self.q_len, 1])
        # document-level attention
        s = tf.matmul(alpha, beta, adjoint_a=True)  # shape = (batch_size, self.d, 1)

        document = inputs[:, :self.d_len]

        mask = tf.map_fn(predict, vocab, dtype=tf.float32)
        mask = tf.reshape(mask, [self.args.batch_size, self.vocab_size, self.d_len])

        # prediction
        self.y_ = tf.reshape(tf.matmul(mask, s), [self.args.batch_size, self.vocab_size])

        # answer
        self.y = tf.placeholder(tf.float32, [self.args.batch_size, self.vocab_size])

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y)
        tf.summary.scalar("loss", tf.reduce_mean(self.loss))

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
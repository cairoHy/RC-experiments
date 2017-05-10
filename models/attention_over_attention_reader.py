import os

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, LSTMCell

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
        #########################
        # b ... position of the example within the batch
        # t ... position of the word within the document/question
        #       ... d for max length of document
        #       ... q for max length of question
        # f ... features of the embedding vector or the encoded feature vector
        # i ... position of the word in candidates list
        # v ... position of the word in vocabulary
        #########################
        _EPSILON = 10e-8
        num_layers = self.args.num_layers
        hidden_size = self.args.hidden_size
        cell = LSTMCell if self.args.use_lstm else GRUCell

        # model input
        questions_bt = tf.placeholder(dtype=tf.int32, shape=(None, self.q_len), name="questions_bt")
        documents_bt = tf.placeholder(dtype=tf.int32, shape=(None, self.d_len), name="documents_bt")
        candidates_bi = tf.placeholder(dtype=tf.int32, shape=(None, self.A_len), name="candidates_bi")
        y_true_bi = tf.placeholder(shape=(None, self.A_len), dtype=tf.float32, name="y_true_bi")

        init_embedding = tf.constant(self.embedding_matrix, dtype=tf.float32, name="embedding_init")
        embedding = tf.get_variable(initializer=init_embedding,
                                    name="embedding_matrix",
                                    dtype=tf.float32)
        if self.args.train and self.args.keep_prob < 1:
            embedding = tf.nn.dropout(embedding, self.args.keep_prob)

        # shape=(None) the length of inputs
        document_lengths = tf.reduce_sum(tf.sign(tf.abs(documents_bt)), 1)
        question_lengths = tf.reduce_sum(tf.sign(tf.abs(questions_bt)), 1)
        document_mask_bt = tf.sequence_mask(document_lengths, self.d_len, dtype=tf.float32)
        question_mask_bt = tf.sequence_mask(question_lengths, self.q_len, dtype=tf.float32)

        with tf.variable_scope('q_encoder', initializer=tf.orthogonal_initializer()):
            # encode question to fixed length of vector
            # output shape: (None, max_q_length, embedding_dim)
            question_embed_btf = tf.nn.embedding_lookup(embedding, questions_bt)
            logger("q_embed_btf shape {}".format(question_embed_btf.get_shape()))
            q_cell_fw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
            q_cell_bw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=q_cell_bw,
                                                                   cell_fw=q_cell_fw,
                                                                   dtype="float32",
                                                                   sequence_length=question_lengths,
                                                                   inputs=question_embed_btf,
                                                                   swap_memory=True)
            # q_encoder output shape: (None, max_t_length, hidden_size * 2)
            q_encoded_bqf = tf.concat(outputs, axis=-1)
            logger("q_encoded_bqf shape {}".format(q_encoded_bqf.get_shape()))

        with tf.variable_scope('d_encoder', initializer=tf.orthogonal_initializer()):
            # encode each document(context) word to fixed length vector
            # output shape: (None, max_d_length, embedding_dim)
            d_embed_btf = tf.nn.embedding_lookup(embedding, documents_bt)
            logger("d_embed_btf shape {}".format(d_embed_btf.get_shape()))
            d_cell_fw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
            d_cell_bw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=d_cell_bw,
                                                                   cell_fw=d_cell_fw,
                                                                   dtype="float32",
                                                                   sequence_length=document_lengths,
                                                                   inputs=d_embed_btf,
                                                                   swap_memory=True)
            # d_encoder output shape: (None, max_d_length, hidden_size * 2)
            d_encoded_bdf = tf.concat(outputs, axis=-1)
            logger("d_encoded_bdf shape {}".format(d_encoded_bdf.get_shape()))

        # mask of the pair-wise matrix
        M_mask = tf.einsum("bi,bj->bij", document_mask_bt, question_mask_bt)
        # batch pair-wise matching
        M_bdq = tf.matmul(d_encoded_bdf, q_encoded_bqf, adjoint_b=True)

        # individual attentions
        alpha_bdq = self.softmax_with_mask(M_bdq, 1, M_mask, name="alpha")
        beta_bdq = self.softmax_with_mask(M_bdq, 2, M_mask, name="beta")
        beta_bq1 = tf.expand_dims(tf.reduce_sum(beta_bdq, 1) / tf.to_float(tf.expand_dims(document_lengths, -1)), -1)
        logger("beta_bq1 shape:{}".format(beta_bq1.get_shape()))
        # document-level attention
        s_bd = tf.squeeze(tf.einsum("bdq,bqi->bdi", alpha_bdq, beta_bq1), -1)

        vocab_size = self.embedding_matrix.shape[0]
        # attention sum operation and gather within candidate_index
        y_hat_bi = tf.scan(fn=lambda prev, cur: tf.gather(tf.unsorted_segment_sum(cur[0], cur[1], vocab_size), cur[2]),
                           elems=[s_bd, documents_bt, candidates_bi],
                           initializer=tf.Variable([0] * self.A_len, dtype="float32"))

        # manual computation of crossentropy
        output_bi = y_hat_bi / tf.reduce_sum(y_hat_bi, axis=-1, keep_dims=True)
        epsilon = tf.convert_to_tensor(_EPSILON, output_bi.dtype.base_dtype, name="epsilon")
        output_bi = tf.clip_by_value(output_bi, epsilon, 1. - epsilon)

        # loss and correct number
        self.loss = tf.reduce_mean(- tf.reduce_sum(y_true_bi * tf.log(output_bi), axis=-1))
        self.correct_prediction = tf.reduce_sum(
            tf.sign(tf.cast(tf.equal(tf.argmax(output_bi, 1),
                                     tf.argmax(y_true_bi, 1)), "float")))

    @staticmethod
    def softmax_with_mask(logits, axis, mask, epsilon=10e-8, name=None):
        with tf.name_scope(name, 'softmax', [logits, mask]):
            max_axis = tf.reduce_max(logits, axis, keep_dims=True)
            target_exp = tf.exp(logits - max_axis) * mask
            normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
            softmax = target_exp / (normalize + epsilon)
            return softmax

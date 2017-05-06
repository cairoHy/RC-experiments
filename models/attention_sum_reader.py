import os

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, GRUCell

from models.nlp_base import logger
from models.rc_base import RcBase

_EPSILON = 10e-8


class AttentionSumReader(RcBase):
    """
    Attention Sum Reader model as presented in "Text Comprehension with the Attention Sum Reader Network" 
    (ACL2016) available at http://arxiv.org/abs/1603.01547. 
    """

    def __init__(self):
        super().__init__()
        self.model_name = os.path.basename(__file__)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        #########################
        # b ... position of the example within the batch
        # t ... position of the word within the document/question
        # f ... features of the embedding vector or the encoded feature vector
        # i ... position of the word in candidates list
        #########################
        num_layers = self.args.num_layers
        hidden_size = self.args.hidden_size
        cell = LSTMCell if self.args.use_lstm else GRUCell

        # model input
        questions_bt = tf.placeholder(dtype=tf.int32, shape=(None, self.q_len), name="questions_bt")
        context_bt = tf.placeholder(dtype=tf.int32, shape=(None, self.d_len), name="context_bt")
        candidates_bi = tf.placeholder(dtype=tf.int32, shape=(None, self.A_len), name="candidates_bi")
        y_true_bi = tf.placeholder(shape=(None, self.A_len), dtype=tf.float32, name="y_true_bi")

        # shape=(None) the length of inputs
        context_lengths = tf.reduce_sum(tf.sign(tf.abs(context_bt)), 1)
        question_lengths = tf.reduce_sum(tf.sign(tf.abs(questions_bt)), 1)
        context_mask_bt = tf.sequence_mask(context_lengths, self.d_len, dtype=tf.float32)

        init_embedding = tf.constant(self.embedding_matrix, dtype=tf.float32, name="embedding_init")
        embedding = tf.get_variable(initializer=init_embedding,
                                    name="embedding_matrix",
                                    dtype=tf.float32)

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
            # q_encoder output shape: (None, hidden_size * 2)
            q_encoded_bf = tf.concat([last_states[0][-1], last_states[1][-1]], axis=-1)
            logger("q_encoded_bf shape {}".format(q_encoded_bf.get_shape()))

        with tf.variable_scope('d_encoder', initializer=tf.orthogonal_initializer()):
            # encode each document(context) word to fixed length vector
            # output shape: (None, max_d_length, embedding_dim)
            d_embed_btf = tf.nn.embedding_lookup(embedding, context_bt)
            logger("d_embed_btf shape {}".format(d_embed_btf.get_shape()))
            d_cell_fw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
            d_cell_bw = MultiRNNCell(cells=[cell(hidden_size) for _ in range(num_layers)])
            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=d_cell_bw,
                                                                   cell_fw=d_cell_fw,
                                                                   dtype="float32",
                                                                   sequence_length=context_lengths,
                                                                   inputs=d_embed_btf,
                                                                   swap_memory=True)
            # d_encoder output shape: (None, max_d_length, hidden_size * 2)
            d_encoded_btf = tf.concat(outputs, axis=-1)
            logger("d_encoded_btf shape {}".format(d_encoded_btf.get_shape()))

        def att_dot(x):
            """attention dot product function"""
            d_btf, q_bf = x
            res = tf.matmul(tf.expand_dims(q_bf, -1), d_btf, adjoint_a=True, adjoint_b=True)
            return tf.reshape(res, [-1, self.d_len])

        with tf.variable_scope('merge'):
            mem_attention_pre_soft_bt = att_dot([d_encoded_btf, q_encoded_bf])
            mem_attention_pre_soft_masked_bt = tf.multiply(mem_attention_pre_soft_bt,
                                                           context_mask_bt,
                                                           name="attention_mask")
            mem_attention_bt = tf.nn.softmax(logits=mem_attention_pre_soft_masked_bt, name="softmax_attention")

        # attention-sum process
        def sum_prob_of_word(word_ix, sentence_ixs, sentence_attention_probs):
            word_ixs_in_sentence = tf.where(tf.equal(sentence_ixs, word_ix))
            return tf.reduce_sum(tf.gather(sentence_attention_probs, word_ixs_in_sentence))

        # noinspection PyUnusedLocal
        def sum_probs_single_sentence(prev, cur):
            candidate_indices_i, sentence_ixs_t, sentence_attention_probs_t = cur
            result = tf.scan(
                fn=lambda previous, x: sum_prob_of_word(x, sentence_ixs_t, sentence_attention_probs_t),
                elems=[candidate_indices_i],
                initializer=tf.constant(0., dtype="float32"))
            return result

        def sum_probs_batch(candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt):
            result = tf.scan(
                fn=sum_probs_single_sentence,
                elems=[candidate_indices_bi, sentence_ixs_bt, sentence_attention_probs_bt],
                initializer=tf.Variable([0] * self.A_len, dtype="float32"))
            return result

        # output shape: (None, i) i = max_candidate_length = 10
        y_hat = sum_probs_batch(candidates_bi, context_bt, mem_attention_bt)

        # crossentropy
        output = y_hat / tf.reduce_sum(y_hat,
                                       reduction_indices=len(y_hat.get_shape()) - 1,
                                       keep_dims=True)
        # manual computation of crossentropy
        epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype, name="epsilon")
        output = tf.clip_by_value(output, epsilon, 1. - epsilon)
        self.loss = tf.reduce_mean(- tf.reduce_sum(y_true_bi * tf.log(output),
                                                   reduction_indices=len(output.get_shape()) - 1))

        # correct prediction nums
        self.correct_prediction = tf.reduce_sum(tf.sign(tf.cast(tf.equal(tf.argmax(y_hat, 1),
                                                                         tf.argmax(y_true_bi, 1)), "float")))

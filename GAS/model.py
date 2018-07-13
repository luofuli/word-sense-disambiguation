# -*- coding: utf-8 -*-
"""
 @version: python2.7
 @author: luofuli
 @time: 2017/7/1
"""

import tensorflow as tf
import numpy as np

rnn = tf.contrib.rnn


class Model:
    def __init__(self, config, n_senses_from_target_id, init_word_vecs=None):
        self.name = 'GAS'

        batch_size = config.batch_size
        n_step_f = config.n_step_f
        n_step_b = config.n_step_b

        embedding_size = config.embedding_size
        n_units = config.n_lstm_units
        context_size = 2 * n_units

        forget_bias = config.forget_bias
        keep_prob = config.keep_prob
        lambda_l2_reg = config.lambda_l2_reg
        momentum = config.momentum
        max_grad_norm = config.max_grad_norm

        memory_update_type = config.memory_update_type

        if config.concat_target_gloss:
            n_units /= 2

        max_n_sense = config.max_n_sense
        max_gloss_words = config.max_gloss_words

        lr_start = config.lr_start  # 0.2
        lr_decay_factor = 0.96
        lr_min = 0.01

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.inputs_f = tf.placeholder(tf.int32, shape=[batch_size, n_step_f], name='inputs_f')
        self.inputs_b = tf.placeholder(tf.int32, shape=[batch_size, n_step_b], name='inputs_b')
        self.inputs = tf.placeholder(tf.int32, shape=[batch_size, n_step_f + n_step_b + 1], name='inputs')
        self.target_ids = tf.placeholder(tf.int32, shape=[batch_size], name='target_ids')
        self.sense_ids = tf.placeholder(tf.int32, shape=[batch_size], name='sense_ids')
        self.glosses = tf.placeholder(tf.int32, shape=[batch_size, max_n_sense, max_gloss_words], name='glosses')
        self.glosses_lenth = tf.placeholder(tf.int32, shape=[batch_size, max_n_sense], name='glosses_lenth')
        self.hyper_lenth = tf.placeholder(tf.int32, shape=[batch_size, max_n_sense], name='hyper_lenth')
        self.hypo_lenth = tf.placeholder(tf.int32, shape=[batch_size, max_n_sense], name='hypo_lenth')
        self.sense_mask = sense_mask = tf.placeholder(tf.float32, shape=[batch_size, max_n_sense, 2 * n_units])

        self.predictions = tf.Variable(tf.zeros([batch_size], dtype=tf.int32), trainable=False)
        self.correct = tf.Variable(tf.zeros([batch_size], dtype=tf.int32), trainable=False)
        self.global_step = global_step = tf.Variable(0, trainable=False)

        tot_n_senses = sum(n_senses_from_target_id.values())
        tot_n_target_words = len(n_senses_from_target_id)

        global_initializer = tf.random_uniform_initializer(-0.1, 0.1)
        lstm_initializer = tf.orthogonal_initializer()

        with tf.device('/cpu:0'):
            with tf.variable_scope('word_emb'):
                if config.use_pre_trained_embedding:
                    word_embeddings = tf.get_variable('word_embeddings', initializer=init_word_vecs, trainable=False)
                else:
                    word_embeddings = tf.get_variable('word_embeddings',
                                                      shape=[config.vocab_size, embedding_size],
                                                      initializer=global_initializer, trainable=True)

        n_senses_sorted_by_target_id = [n_senses_from_target_id[target_id] for target_id
                                        in range(len(n_senses_from_target_id))]
        n_senses_sorted_by_target_id_tf = tf.constant(n_senses_sorted_by_target_id, tf.int32)
        _W_starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)) * context_size)[:-1]
        _W_lenghts = np.array(n_senses_sorted_by_target_id) * context_size
        W_starts = tf.constant(_W_starts, tf.int32)
        W_lengths = tf.constant(_W_lenghts, tf.int32)
        _b_starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)))[:-1]
        _b_lengths = np.array(n_senses_sorted_by_target_id)
        b_starts = tf.constant(_b_starts, tf.int32)
        b_lengths = tf.constant(_b_lengths, tf.int32)

        with tf.variable_scope('target_params', initializer=global_initializer):
            W_targets = tf.get_variable('W_targets', [tot_n_senses * context_size], dtype=tf.float32)
            b_target = tf.get_variable('b_target', [tot_n_senses], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
            w_ratio = tf.get_variable('r_target', [tot_n_target_words], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.8))

        with tf.variable_scope('memory_params', initializer=global_initializer):
            W_memory = tf.get_variable('W_memory', [2 * context_size, context_size], dtype=tf.float32)
            U_memory = tf.get_variable('U_memory', [context_size, context_size], dtype=tf.float32)

        self.keep_prob = keep_prob = tf.cond(tf.equal(self.is_training, tf.constant(True)),
                                             lambda: tf.constant(config.keep_prob),
                                             lambda: tf.constant(1.0))    # val or test: 1.0 means no dropout

        def lstm_cell(num_units):
            cell = rnn.LSTMCell(num_units, initializer=lstm_initializer, forget_bias=forget_bias)
            if tf.__version__ == '1.2.0' and config.state_dropout:
                cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob, state_keep_prob=keep_prob)
            else:
                cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        with tf.variable_scope('sentence', initializer=lstm_initializer):
            s_cell_fw = lstm_cell(n_units)  # forward
            s_cell_bw = lstm_cell(n_units)  # backward
            inputs_s = tf.nn.embedding_lookup(word_embeddings, self.inputs)  # [batch_size, n_step, dim]
            inputs_s = tf.nn.dropout(inputs_s, keep_prob)
            s_outputs, s_final_state = tf.nn.bidirectional_dynamic_rnn(s_cell_fw,
                                                                       s_cell_bw,
                                                                       inputs_s,
                                                                       dtype=tf.float32,
                                                                       time_major=False)
            s_outputs_fw = s_outputs[0]
            s_outputs_bw = s_outputs[1]  # [batch_size, n_step, n_units]
            s_output = [s_outputs_fw[:, n_step_f - 1, :], s_outputs_bw[:, n_step_f + 1, :]]
            s_output = tf.concat(s_output, 1)  # [batch_size, 2*n_units]

        with tf.variable_scope('gloss', initializer=lstm_initializer):
            g_cell_fw = lstm_cell(n_units)  # forward
            g_cell_bw = lstm_cell(n_units)  # backward
            inputs_g = tf.reshape(self.glosses, [batch_size * max_n_sense, max_gloss_words])
            inputs_g = tf.nn.embedding_lookup(word_embeddings, inputs_g)  # [batch_size*max_n_sense, max_words, dim]
            inputs_g = tf.nn.dropout(inputs_g, self.keep_prob)
            sequence_length = tf.reshape(self.glosses_lenth, [batch_size * max_n_sense])
            g_outputs, g_final_states = tf.nn.bidirectional_dynamic_rnn(g_cell_fw,
                                                                        g_cell_bw,
                                                                        inputs_g,
                                                                        sequence_length=sequence_length,
                                                                        time_major=False,
                                                                        dtype=tf.float32)
            g_outputs_fw = g_outputs[0]
            g_outputs_bw = g_outputs[1]  # [batch_size*max_n_sense, n_step, n_units]
            g_output = [g_outputs_fw[:, -1, :], g_outputs_bw[:, 0, :]]
            g_output = tf.concat(g_output, 1)  # [batch_size*max_n_sense, 2*n_units]
            g_output = tf.reshape(g_output, [batch_size, max_n_sense, 2 * n_units])
            g_output = tf.nn.dropout(g_output, keep_prob)

        def batch_norm(x, is_training=self.is_training):
            if config.batch_norm:
                return tf.layers.batch_normalization(x, momentum=0.8, training=is_training)
            else:
                return x

        memory_p = []

        def memory(gloss, context):
            memory_size = gloss.get_shape().as_list()[-1]
            Cin = Ain = gloss * sense_mask  # [batch_size, max_n_sense, 2*n_units]
            Bin = tf.reshape(context, [batch_size, memory_size, 1])  # [batch_size, 2*n_units, 1]
            Aout = tf.matmul(Ain, Bin)  # [batch_size, max_n_sense, 1]
            Aout_exp = tf.exp(Aout) * sense_mask[:, :, :1]
            p = Aout_exp / tf.reduce_sum(Aout_exp, axis=1, keepdims=True)  # [batch_size, max_n_sense, 1]
            memory_p.append(tf.squeeze(p))
            Mout = tf.squeeze(tf.matmul(Cin, p, transpose_a=True))  # [batch_size, 2*n_units]
            if memory_update_type == 'concat':
                state = tf.concat((Mout, context), 1)  # [batch_size, 4*n_units]
                state = tf.nn.relu(batch_norm(tf.matmul(state, W_memory)))  # [batch_size, 2*n_units]
            else:  # linear
                state = batch_norm(tf.add(Mout, tf.matmul(context, U_memory)))  # [batch_size, 2*n_units]

            return state, tf.squeeze(Aout)

        state = s_output
        for i in range(config.memory_hop + 1):
            state, Aout = memory(batch_norm(g_output), batch_norm(state))

        self.memory_p = tf.stack(memory_p)

        # prediction
        hidden_state = tf.split(s_output, batch_size, 0)
        target_ids = tf.split(self.target_ids, batch_size, 0)
        sense_ids = tf.split(self.sense_ids, batch_size, 0)
        self.w_ratio = w_ratio = tf.clip_by_value(w_ratio, 0, 1)  # the ratio of memory
        # self.w_ratio = w_ratio = tf.sigmoid(w_ratio)

        loss = tf.Variable(0.0, trainable=False)
        n_correct = tf.Variable(0, trainable=False)

        # add memory attention and make predictions for all instances in a batch
        for i in range(batch_size):
            target_id = target_ids[i]
            sense_id = sense_ids[i]
            a_ = hidden_state[i]
            n_sense = tf.squeeze(tf.slice(n_senses_sorted_by_target_id_tf, target_id, [1]))  # target word 义项个数

            one = tf.constant(1, tf.int32, [1])
            W = tf.slice(W_targets,
                         tf.slice(W_starts, target_id, one),
                         tf.slice(W_lengths, target_id, one))
            W = tf.reshape(W, [n_sense, context_size])  # [n_sense, context_size]
            b = tf.slice(b_target,
                         tf.slice(b_starts, target_id, one),
                         tf.slice(b_lengths, target_id, one))

            logits_w = tf.matmul(a_, W, False, True) + b  # [1, n_sense]

            logits_m = Aout[i, :n_sense]  # [n_sense]
            logits_m = tf.reshape(logits_m, [1, n_sense])  # [1, n_sense]
            r = tf.squeeze(tf.slice(w_ratio, target_id, [1]))

            logits = r * logits_w + (1 - r) * logits_m

            # calculate cross-entropy loss
            loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=sense_id))

            predicted_sense = tf.argmax(logits, 1, name='prediction')
            predicted_sense = tf.cast(predicted_sense, tf.int32)
            self.predictions = tf.scatter_update(self.predictions, tf.constant(i, shape=[1]), predicted_sense)
            n_correct += tf.squeeze(tf.cast(tf.equal(sense_id, predicted_sense), tf.int32))
            self.correct = tf.scatter_update(self.correct, tf.constant(i, shape=[1]),
                                             tf.cast(tf.equal(sense_id, predicted_sense), tf.int32))

            if i == batch_size - 1:
                tf.summary.histogram('logits', logits)
                tf.summary.histogram('W_targets', W_targets)
                tf.summary.histogram('b_target', b_target)

        self.loss_op = tf.div(loss, batch_size)
        self.accuracy_op = tf.div(tf.cast(n_correct, tf.float32), batch_size)

        # Summaries
        tf.summary.scalar('loss', self.loss_op)
        tf.summary.scalar('accuracy', self.accuracy_op)
        self.summary_op = tf.summary.merge_all()

        print 'TRAINABLE VARIABLES'
        tvars = tf.trainable_variables()
        for tvar in tvars:
            print tvar.name

        # Weight Penalty
        if lambda_l2_reg:
            print 'USING L2 regularization'
            w_cost = tf.constant(0.0)
            n_w = tf.constant(0.0)
            for tvar in tvars:
                if 'lstm' in tvar.name or 'memory' in tvar.name:
                    print(tvar.name)
                    w_cost += tf.nn.l2_loss(tvar)
                    n_w += tf.to_float(tf.size(tvar))
            self.loss_op += lambda_l2_reg * w_cost / n_w

        # Update Parameters
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_op, tvars), max_grad_norm)
        self.lr = tf.maximum(lr_min, tf.train.exponential_decay(lr_start, global_step, 60, lr_decay_factor))
        optimizer = tf.train.MomentumOptimizer(self.lr, momentum)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

        # Both ok
        # self.lr = tf.Variable(lr_start, trainable=False)
        # optimizer = tf.train.AdamOptimizer(self.lr)
        # self.train_op = optimizer.minimize(self.loss_op, global_step=global_step, var_list=tvars)

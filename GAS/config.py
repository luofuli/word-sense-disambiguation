# -*- coding: utf-8 -*-
"""
 @version: python2.7
 @author: luofuli
 @time: 2017/7/1
"""
import time, random


class MemNNConfig(object):
    """MemNN+ parameters"""

    def __init__(self):
        # parameters for pre-trained word embedding
        self.embedding_size = 300
        self.use_pre_trained_embedding = True

        # parameters for input data
        self.n_step_f = 30  # forward context length
        self.n_step_b = 30  # backward context length

        # parameters for knowledge(glosses)
        self.gloss_expand_type = 2  # 0=None 1=hyper，2=hypo 3=hyper+hypo 4：hierarchical
        self.max_gloss_words = 100
        self.max_n_sense = 10
        self.min_sense_freq = 1

        # parameters for model
        self.n_lstm_units = 512
        self.forget_bias = 0.0
        self.memory_update_type = 'concat'  # 'concat' or 'linear'
        self.memory_hop = 1
        self.concat_target_gloss = False
        self.keep_prob = 0.5  # dropout keep_prob

        # parameters for train
        self.n_epochs = 50
        self.batch_size = 8
        self.lr_start = 0.001
        self.lambda_l2_reg = 0  # L2 regularization
        self.momentum = 0.1
        self.max_grad_norm = 10  # clip gradient

        # Validation info
        self.evaluate_gap = 1000
        self.store_log_gap = 100
        self.validate = True
        self.min_no_improvement = 10
        self.print_batch = True

        self.batch_norm = False

    def random_config(self):
        random.seed(time.time())
        # self.memory_update_type = random.choice(['concat', 'linear'])
        # self.memory_hop = random.choice([1, 2, 3])
        self.n_lstm_units = random.choice([128, 256, 512])
        self.max_n_sense = random.choice([10, 20, 40])
        self.min_sense_freq = random.choice([1, 3, 5, 10])
        # self.gloss_emb_size = random.choice([64, 128, 256])
        # self.lr_start = random.choice([0.01, 0.001])



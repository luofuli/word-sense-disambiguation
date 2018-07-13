# -*- coding: utf-8 -*-
"""
 @version: python2.7
 @author: luofuli
 @time: 2017/7/1
"""

import os


class WSD_path(object):
    def __init__(self):
        self.DATA_DIR = '../data/'

        # Lexical Example task dataset
        self.LS_DATASET = ['senseval2_LS', 'senseval3_LS']
        # ALL-words task dataset
        self.ALL_WORDS_TRAIN_DATASET = ['semcor', 'semcor+omsti']
        self.ALL_WORDS_TEST_DATASET = ['ALL', 'senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']
        self.ALL_WORDS_VAL_DATASET = 'semeval2007'

        # Lexical Example task path
        self.LS_BASE_PATH =_LS_BASE_PATH= '../data/Lexical_Sample_WSD/'
        self.LS_OLD_TRAIN_PATH = _LS_BASE_PATH + '{}/train.xml'
        self.LS_TRAIN_PATH = _LS_BASE_PATH + '{}/train.new.xml'
        self.LS_VAL_OLD_PATH = _LS_BASE_PATH + '{}/test.haskey.xml'
        self.LS_VAL_PATH = _LS_BASE_PATH + '{}/test.haskey.new.xml'
        self.LS_TEST_PATH = _LS_BASE_PATH + '{}/test.xml'
        self.LS_DIC_PATH = _LS_BASE_PATH + '{}/dictionary.new.xml'
        self.LS_OLD_DIC_PATH = _LS_BASE_PATH + '{}/dictionary.xml'
        self.LS_TEST_KEY_OLD_PATH = _LS_BASE_PATH + '{}/test.key'
        self.LS_TEST_KEY_PATH = _LS_BASE_PATH + '{}/test.new.key'
        self.LS_SENSEMAP_PATH = _LS_BASE_PATH + '{}/sensemap.txt'

        # ALL-words task path
        self.ALL_WORDS_BASE_PATH = _ALL_WORDS_BASE_PATH = '../data/All_Words_WSD/'
        # path for all-words train
        self.ALL_WORDS_TRAIN_PATH = _ALL_WORDS_BASE_PATH + 'Training_Corpora/{0}/{0}.data.xml'
        self.ALL_WORDS_TRAIN_KEY_PATH = _ALL_WORDS_BASE_PATH + 'Training_Corpora/{0}/{0}.gold.key.txt'
        self.ALL_WORDS_DIC_PATH = _ALL_WORDS_BASE_PATH + 'Training_Corpora/{0}/{0}.dict.xml'
        # path for all_words test
        self.ALL_WORDS_TEST_PATH = _ALL_WORDS_BASE_PATH + 'Evaluation_Datasets/{0}/{0}.data.xml'
        self.ALL_WORDS_TEST_KEY_PATH = _ALL_WORDS_BASE_PATH + 'Evaluation_Datasets/{0}/{0}.gold.key.txt'
        self.ALL_WORDS_TEST_KEY_WPATH = _ALL_WORDS_BASE_PATH + 'Evaluation_Datasets/{0}/{0}.gold.key.withPos.txt'
        # MFS / FS result
        self.BASE_OTHER_SYSTEM_PATH = _ALL_WORDS_BASE_PATH + 'Output_Systems_ALL/'
        self.MFS_PATH = _ALL_WORDS_BASE_PATH + 'Output_Systems_ALL/MFS_{0}.key'
        self.WNFS_PATH = _ALL_WORDS_BASE_PATH + 'Output_Systems_ALL/WNFirstsense.key'

        self.WORDNET_PATH = '../data/nltk_data/corpora/wordnet'  # version 3.0
        self.GLOVE_VECTOR = '../data/glove.42B.300d.txt'  # todo: change to yours word2vec path

        if not os.path.exists('../tmp'):
            os.makedirs('../tmp')
        self.BACK_OFF_RESULT_PATH = '../tmp/back_off_results-{}.txt'

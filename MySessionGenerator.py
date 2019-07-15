#!/usr/bin/python3
# -*- coding: utf-8 -*-
# =============================================================================
#         FILE: MySessionGenerator.py
#  DESCRIPTION: Creates common setting of sessions for tensorflow
#        USAGE:
#      OPTIONS:
# REQUIREMENTS:
#
#      LICENCE:
#
#         BUGS:
#        NOTES:
#       AUTHOR: Jiří Tumpach (tumpji),
# ORGANIZATION:
#      VERSION: 1.0
#      CREATED: 2019 06.23.
# =============================================================================

import tensorflow as tf

class SessionGenerator():
    def __init__(self):
        config = tf.ConfigProto()
        config.inter_op_parallelism_threads = 2 # 
        config.intra_op_parallelism_threads = 1
        config.gpu_options.allow_growth = True

        #config.gpu_options.per_process_gpu_memory_fraction = (1-0.05)/5

        self.session = tf.Session(config=config, graph=tf.Graph())

    def finalize(self):
        self.session.graph.finalize()

    def free(self):
        self.session.close()



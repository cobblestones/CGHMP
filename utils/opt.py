#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint
from utils import log
import sys
import time

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether it is to evaluate the model')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--skip_rate', type=int, default=5, help='skip rate of samples')
        self.parser.add_argument('--skip_rate_test', type=int, default=5, help='skip rate of samples for test')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--in_features', type=int, default=54, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='size of each model layer')
        self.parser.add_argument('--d_model', type=int, default=256, help='past frame number')
        self.parser.add_argument('--kernel_size', type=int, default=5, help='past frame number')
        self.parser.add_argument('--alpha', type=float, default=0.05, help='the RATIO of the label predict loss')
        self.parser.add_argument('--beta', type=float, default=0.2,help="the RATIO of the  multi-scale loss")
        self.parser.add_argument('--gama', type=float, default=0.5, help="the RATIO of the multi-temporal-smooth loss")
        # self.parser.add_argument('--drop_out', type=float, default=0.5, help='drop out probability')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--input_n', type=int, default=50, help='past frame number')
        self.parser.add_argument('--output_n', type=int, default=10, help='future frame number')
        self.parser.add_argument('--test_output_n', type=int, default=25, help='future frame number')
        self.parser.add_argument('--dct_n', type=int, default=10, help='future frame number')
        self.parser.add_argument('--lr_now', type=float, default=0.0005)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--epoch', type=int, default=200)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--test_batch_size', type=int, default=16)
        self.parser.add_argument('--is_load', dest='is_load', action='store_true',help='whether to load existing model')
        self.parser.add_argument('--cuda_id', type=int, default=0, help='the index of GPU id')



    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        self.current_train_id = time.strftime("%Y_%m_%d_%HH_%MM_%SS", time.localtime())
        # logger_name = current_train_id + '-train_logger.log'


        if not self.opt.is_eval:
            script_name = os.path.basename(sys.argv[0])[:-3]
            log_name = '{}_in{}_out{}_ks{}_dctn{}_{}_alpha{}_beta{}_gama{}_Total'.format(script_name, self.opt.input_n,
                                                                            self.opt.output_n,
                                                                            self.opt.kernel_size,
                                                                            self.opt.dct_n,
                                                                            self.current_train_id,
                                                                            self.opt.alpha,
                                                                            self.opt.beta,
                                                                            self.opt.gama
                                                                            )
            self.opt.exp = log_name
            # do some pre-check
            ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                log.save_options(self.opt)
            self.opt.ckpt = ckpt
            log.save_options(self.opt)
        self._print()
        # log.save_options(self.opt)
        return self.opt


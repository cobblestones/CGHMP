#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import json
import os
import torch
import pandas as pd
import numpy as np


def save_csv_log(opt, head, value, is_create=False, file_name='training'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = opt.ckpt + '/{}.csv'.format(file_name)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)

def save_csv_log_for_predict(opt, head, value, is_create=False, file_name='training'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    head_path, tail_path= os.path.split(opt.ckpt)
    file_path = head_path + '/{}.csv'.format(tail_path)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)

def save_ckpt(epo,lr_now,err,state, is_best=True, file_name=['ckpt_best.pth.tar', 'ckpt_last.pth.tar'], opt=None):
    # file_path = os.path.join(opt.ckpt, file_name[1])
    file_path = os.path.join(opt.ckpt, 'ckpt_cur_epoch{}_lr{:.4f}_{}.pth'.format(epo,lr_now,err))
    torch.save(state, file_path)
    if is_best:
        file_path = os.path.join(opt.ckpt, 'ckpt_Best_epoch{}_lr{:.4f}_{}.pth'.format(epo,lr_now,err))
        torch.save(state, file_path)

def save_options(opt):
    with open(opt.ckpt + '/option.json', 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=False, indent=4))

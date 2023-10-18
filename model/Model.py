from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
import sys
sys.path.append("..")
from model import GCN
import utils.util as util
import numpy as np


class AttModel(Module):
    def __init__(self, in_features=48, kernel_size=5, d_model=512, num_stage=2, dct_n=10):
        super(AttModel, self).__init__()
        self.kernel_size = kernel_size
        self.d_model = d_model
        # self.seq_in = seq_in
        self.dct_n = dct_n
        # ks = int((kernel_size + 1) / 2)
        assert kernel_size == 10
        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, \
                                              kernel_size=6, bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, \
                                             kernel_size=5, bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, \
                                             kernel_size=6, bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, \
                                             kernel_size=5, bias=False),
                                   nn.ReLU())

        self.gcn = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.predictlabl=nn.Sequential(
                                      nn.Linear(in_features*(dct_n) * 2, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 15)
                                      )

        self.label_to_input_feature = nn.Sequential(
                                     nn.Linear(15, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, in_features * dct_n * 2),
                                     nn.ReLU(),

        )

        self.gcn2 = GCN.GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3,
                       num_stage=num_stage,
                       node_n=in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        """
        :param src: [batch_size,seq_len,feat_dim]
        :param output_n:
        :param input_n:
        :param frame_n:
        :param dct_n:
        :param itera:
        :return:
        """
        dct_n = self.dct_n
        src = src[:, :input_n]  # [bs,in_n,dim]
        src_tmp = src.clone()
        bs = src.shape[0]
        # input value for key
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        # input value for query
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        # coefficient for dct_m and idct_m
        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().cuda()
        idct_m = torch.from_numpy(idct_m).float().cuda()

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        value_idx = np.expand_dims(np.arange(vl), axis=0) + np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, value_idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  # [32,40,66*11]

        # this index is for get lastest 10 slice and repeat the last silce 10 times
        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n #[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] 前面10帧加上把最后1帧复制10次
        outputs = []
        outputs_labels = []

        # get key value form key_slice_data：[:,0:input_n - output_n]→[:,0:50-10]
        key_tmp = self.convK(src_key_tmp / 1000.0)

        for i in range(itera):
            # get query value from query_slice_data: [:, -self.kernel_size:]→[:,0:-10:0]
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            # get score
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            # normalize score
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))

            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape(
                [bs, -1, dct_n])

            # this index is for get lastest 10 slice and repeat the last silce 10 times
            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            dct_out_tmp = self.gcn(dct_in_tmp) #input:256,66,40  output:256,66,40

            dct_out_tmp_to_input_classlabel= dct_out_tmp.transpose(1, 2).reshape(bs,-1)
            dct_out_predict_class=self.predictlabl(dct_out_tmp_to_input_classlabel)
            outputs_labels.append(dct_out_predict_class.unsqueeze(1))

            # add the preditied label to the gcn to generate human motion pose
            label_to_features = self.label_to_input_feature(dct_out_predict_class)
            batch_size, joints_number, frames = dct_out_tmp.size()
            dct_out_tmp=dct_out_tmp+label_to_features.reshape( batch_size, joints_number, frames)

            dct_out_tmp = self.gcn2(dct_out_tmp)

            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                                   dct_out_tmp[:, :, :dct_n].transpose(1, 2))#input:256,66,40  output:256,20,66
            outputs.append(out_gcn.unsqueeze(2))#torch.Size([256, 20, 1, 66])


            if itera > 1:
                # update key-value query
                out_tmp = out_gcn.clone()[:, 0 - output_n:]
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)
                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + \
                          np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)
                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)
                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape(
                    [bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape(
                    [bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape(
                    [bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)
                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)
        outputs = torch.cat(outputs, dim=2)
        outputs_labels = torch.cat(outputs_labels, dim=1)
        return outputs,outputs_labels



if __name__=='__main__':
    net_pred = AttModel(in_features=66, kernel_size=10, d_model=256,num_stage=12, dct_n=20).cuda()
    history_poses=torch.randn([128,60,66]).cuda()
    future_poses,p3d_out_all_class = net_pred(history_poses, input_n=50, output_n=10, itera=3)
    print("future_poses.shape:{},predicted_class.shape:{}".format(future_poses.shape,p3d_out_all_class.shape))
from utils import h36motion3d as datasets
from model import Model
from utils.opt import Options
from utils import util
from utils import log
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
from  tqdm import tqdm
from utils.multi_scale import downs_from_22


def main(opt):

    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> create models')
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = Model.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.cuda()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters())/1000000.0))

    if opt.is_load or opt.is_eval:
        model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

    print('>>> loading datasets')

    if not opt.is_eval:
        dataset = datasets.Datasets(opt, split=0)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, \
                                 num_workers=0, pin_memory=True)
        valid_dataset = datasets.Datasets(opt, split=1)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, \
                                  shuffle=True, num_workers=0, pin_memory=True)

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, \
                             num_workers=0,
                             pin_memory=True)

    # evaluation
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')


    # training
    if not opt.is_eval:
        err_best = 1000
        for epo in range(start_epoch, opt.epoch + 1):
            is_best = False
            # if epo % opt.lr_decay == 0:
            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
            print('>>> training epoch: {:d}'.format(epo))
            ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, \
                                  epo=epo, opt=opt)
            print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))
            ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
            print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))
            ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
            print('testing error: {:.3f}'.format(ret_test['#80']))
            ret_log = np.array([epo, lr_now])
            head = np.array(['epoch', 'lr'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            for k in ret_valid.keys():
                ret_log = np.append(ret_log, [ret_valid[k]])
                head = np.append(head, ['valid_' + k])
            for k in ret_test.keys():
                ret_log = np.append(ret_log, [ret_test[k]])
                head = np.append(head, ['test_' + k])
            log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
            valid_value=ret_valid['m_p3d_h36']
            if valid_value < err_best:
                err_best = valid_value
                is_best = True
            average_error = np.mean((ret_test['#80'], ret_test['#160'], ret_test['#320'], ret_test['#400'],ret_test['#560'], ret_test['#1000']))
            err_value = 'AverageError{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}'.format(
                average_error, ret_test['#80'], ret_test['#160'], ret_test['#320'], ret_test['#400'], ret_test['#560'], ret_test['#1000'])
            log.save_ckpt(epo, lr_now, err_value,
                          {'epoch': epo, 'lr': lr_now,
                           'err': ret_valid['m_p3d_h36'],
                           'state_dict': net_pred.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          is_best=is_best, opt=opt)




def L2MultiTemporalSmoothLoss_ByVelocityAndacceleration(out, gt):
    '''
    # out: prediction results (batch_size, seq_len, feature dimesion: 22*3)
    # gt：ground truth results (batch_size, seq_len, feature dimesion: 22*3)
    adopt the velocity of prediction and the corresponding ground truth to smooth sequence
    adopt the acceleration of prediction and the corresponding ground truth to smooth sequence
    '''
    batch_size, seq_len,_,_  = gt.shape

    gt_smooth_velocity = gt[:, 1:seq_len, :] - gt[:, 0:seq_len - 1, :]
    out_smooth_velocity = out[:, 1:seq_len, :] - out[:, 0:seq_len - 1, :]
    loss_velocity = torch.mean(torch.norm(gt_smooth_velocity - out_smooth_velocity, 2, dim=-1))

    batch_size_velocity, seq_len_velocity,_,_  = gt_smooth_velocity.shape

    gt_smooth_acceleration = gt_smooth_velocity[:, 1:seq_len_velocity, :] - gt_smooth_velocity[:, 0:seq_len_velocity - 1, :]
    out_smooth_acceleration = out_smooth_velocity[:, 1:seq_len_velocity, :] - out_smooth_velocity[:, 0:seq_len_velocity - 1, :]  # 这儿的loss最开始居然写错了
    loss_accleration= torch.mean(torch.norm(gt_smooth_acceleration - out_smooth_acceleration, 2, dim=-1))

    loss_all=loss_velocity+loss_accleration

    return loss_all



def L2MultiScaleLoss_train(out, gt,down_key_transform):
    '''
    # out: prediction results (batch_size, seq_len, feature dimesion: 22*3)
    # gt：ground truth results (batch_size, seq_len, feature dimesion: 22*3)
    # down_key_transform: downsample stragy: 22→12→7→4
    '''
    batch_size, seq_len, _, _ = gt.shape

    # gt_all_scales = {'p32': gt_32, 'p22': gt_22}
    gt_all_scales = {'p22': (gt.view(batch_size, seq_len, -1)).permute(0,2,1)}
    # 将预测出来的特征全部进行降采样 然后多个维度做loss
    gt_all_scales_down = downs_from_22(gt_all_scales, down_key=down_key_transform)

    out_all_scales = {'p22': (out.view(batch_size, seq_len, -1)).permute(0,2,1)}
    out_all_scales_down = downs_from_22(out_all_scales, down_key=down_key_transform)

    gt_p12=(gt_all_scales_down['p12']).permute(0,2,1)
    out_p12=(out_all_scales_down['p12']).permute(0,2,1)
    loss_p12=torch.mean(torch.norm(gt_p12.view(batch_size,seq_len,-1,3) - out_p12.view(batch_size,seq_len,-1,3), 2, dim=-1))

    gt_p7=(gt_all_scales_down['p7']).permute(0,2,1)
    out_p7=(out_all_scales_down['p7']).permute(0,2,1)
    loss_p7=torch.mean(torch.norm(gt_p7.view(batch_size,seq_len,-1,3) - out_p7.view(batch_size,seq_len,-1,3), 2, dim=-1))

    gt_p4 = (gt_all_scales_down['p4']).permute(0,2,1)
    out_p4 = (out_all_scales_down['p4']).permute(0,2,1)
    loss_p4 = torch.mean( torch.norm(gt_p4.view(batch_size, seq_len, -1, 3) - out_p4.view(batch_size, seq_len, -1, 3), 2, dim=-1))

    loss = loss_p12+loss_p7+loss_p4
    return loss



def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):

    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.test_output_n)) + 1
        m_p3d_h36 = np.zeros([opt.test_output_n])

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1,joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    n = 0
    in_n = opt.input_n
    out_n = opt.output_n

    # itera = 1
    if is_train <= 1:
        itera = 1
        out_n = opt.output_n
    else:
        itera = 3
        out_n = opt.test_output_n

    down_key = [('p22', 'p12', [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11], [12], [13], [14, 15, 16], [17], [18], [19, 20, 21]]),
                ('p12', 'p7', [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11]]),
                ('p7', 'p4', [[0, 2], [1, 2], [3, 4], [5, 6]])]

    st = time.time()

    for i, (label, p3d_h36) in tqdm(enumerate(data_loader)):
        label=label.cuda()
        batch_size, seq_n, _ = p3d_h36.shape
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()
        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        # p3d_src = p3d_h36.clone()[:, :, dim_used]
        p3d_src = p3d_h36.clone()[:, :, dim_used]

        p3d_out_all,p3d_out_all_class = net_pred(p3d_src, input_n=in_n, output_n=10, itera=itera)


        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        if is_train == 0:
            p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0, ]
        else:
            p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:,:out_n]


        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
        p3d_out = p3d_out.reshape([-1, out_n, 32, 3])
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])

        # 2d joint loss:
        grad_norm = 0
        if is_train == 0:
            p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))

            criterion = nn.CrossEntropyLoss()
            loss_label_guided=criterion(p3d_out_all_class[:,0],label.long())
            loss_multi_temporal_smooth_loos=L2MultiTemporalSmoothLoss_ByVelocityAndacceleration(p3d_out_all[:, :, 0],p3d_sup)
            loss_multi_scale_loss=L2MultiScaleLoss_train(p3d_out_all[:, :, 0],p3d_sup,down_key)
            loss_all=loss_p3d+opt.alpha*loss_label_guided+opt.beta*loss_multi_scale_loss+opt.gama*loss_multi_temporal_smooth_loos

            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            # update log values
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:  # if is validation or train simply output the overall mean error
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt, time.time() - st, grad_norm))

    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n
    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j]*40)] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)



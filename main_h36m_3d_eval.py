from utils import h36motion3d as datasets
from model import Model
from utils.opt import Options
from utils import log
from torch.utils.data import DataLoader
import torch
import numpy as np
import time


def main(opt):
    print('>>> create models')
    in_features = 66
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    net_pred = Model.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.cuda()
    model_path_len=opt.ckpt
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    ckpt = torch.load(model_path_len)
    net_pred.load_state_dict(ckpt['state_dict'])

    print('>>> loading datasets')

    head = np.array(['act'])
    for k in range(1, opt.output_n + 1):
        head = np.append(head, [f'#{k}'])

    acts = ["walking", "eating", "smoking", "discussion", "directions",
            "greeting", "phoning", "posing", "purchases", "sitting",
            "sittingdown", "takingphoto", "waiting", "walkingdog",
            "walkingtogether"]
    errs = np.zeros([len(acts) + 1, opt.output_n])
    for i, act in enumerate(acts):
        test_dataset = datasets.Datasets(opt, split=2, actions=[act])
        print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
        test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True)

        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        average_error = np.mean(( ret_test['#80'], ret_test['#160'], ret_test['#320'], ret_test['#400'], ret_test['#560'],
                                ret_test['#1000']))
        print('>>> AverageError{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}'.format(
            average_error, ret_test['#80'], ret_test['#160'], ret_test['#320'], ret_test['#400'], ret_test['#560'],
            ret_test['#1000']))
        ret_log = np.array([])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
        errs[i] = ret_log
    errs[-1] = np.mean(errs[:-1], axis=0)
    acts = np.expand_dims(np.array(acts + ["average"]), axis=1)
    value = np.concatenate([acts, errs.astype(np.str_)], axis=1)
    TotalAverageError = np.mean((errs[-1][1], errs[-1][3], errs[-1][7], errs[-1][9], errs[-1][13], errs[-1][24]))
    print('>>>>>>>TotalAverageError{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}_err{:.4f}'.format(
        TotalAverageError, errs[-1][1], errs[-1][3], errs[-1][7], errs[-1][9], errs[-1][13],
        errs[-1][24]))
    log.save_csv_log_for_predict(opt, head, value, is_create=True, file_name='test_H36M')


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
    net_pred.eval()
    titles = np.array(range(opt.output_n)) + 1
    m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    seq_in = opt.kernel_size

    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    itera = 3

    for i, (label,p3d_h36) in enumerate(data_loader):

        batch_size, seq_n, _ = p3d_h36.shape
        # when only one sample in this batch
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().cuda()
        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_src = p3d_h36.clone()[:, :, dim_used]
        p3d_out_all_pred,p3d_out_all_class = net_pred(p3d_src, input_n=in_n, output_n=10, itera=itera)

        p3d_out_all = p3d_out_all_pred[:, seq_in:].transpose(1, 2).reshape([batch_size, 10 * itera, -1])[:, :out_n]

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
        p3d_out = p3d_out.reshape([-1, out_n, 32, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

    ret = {}
    m_p3d_h36 = m_p3d_h36 / n
    for j in range(out_n):
        ret["#{:d}".format(titles[j]*40)] = m_p3d_h36[j]
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)

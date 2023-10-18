from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils import data_utils
from matplotlib import pyplot as plt
import torch


class Datasets(Dataset):
    def __init__(self, opt, actions=None, split=0):
        self.path_to_data = "./datasets/h3.6m/"
        self.split = split
        self.in_n = opt.input_n     # 50
        if self.split <= 1:
            self.out_n = opt.output_n #10
        else:
            self.out_n = opt.test_output_n #25
        self.sample_rate = 2
        self.p3d = {}
        self.data_idx = []
        self.cuda_id=opt.cuda_id
        seq_len = self.in_n + self.out_n
        subs = np.array([[1, 6, 7, 8, 9], [11], [5]])

        # acts = data_utils.define_actions(actions)
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions

        acts_sum=["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]

        joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                      "RightForeArm",
                      "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]
        subs = subs[split]
        key = 0
        clas_number=np.zeros([len(subs)+1,len(acts)])
        for sub_idx,subj in enumerate(subs):

            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1:
                    num=0
                    for subact in [1, 2]:  # subactions
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence = data_utils.readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float().cuda(self.cuda_id)
                        # remove global rotation and translation
                        the_sequence[:, 0:6] = 0
                        p3d = data_utils.expmap2xyz_torch(the_sequence)
                        #get label
                        # labels = np.zeros(len(acts),np.float32)
                        # labels[acts.index(action)] = 1
                        label_index = acts.index(action)
                        # print(label_index)
                        self.p3d[key] =(label_index, p3d.view(num_frames, -1).cpu().data.numpy())
                        # self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()
                        valid_frames = np.arange(0, num_frames - seq_len + 1, opt.skip_rate)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        # print(len(tmp_data_idx_2))
                        num=num+len(tmp_data_idx_2)
                        key += 1
                    clas_number[sub_idx,action_idx]=num
                else:
                    num=0
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)
                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float().cuda(self.cuda_id)
                    the_seq1[:, 0:6] = 0
                    p3d1 = data_utils.expmap2xyz_torch(the_seq1)
                    # labels = np.zeros(len(acts),np.float32)
                    # labels[acts.index(action)] = 1
                    label_index=acts_sum.index(action)
                    # print(label_index)
                    self.p3d[key] = (label_index,p3d1.view(num_frames1, -1).cpu().data.numpy())
                    # self.p3d[key] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))

                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)
                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float().cuda(self.cuda_id)
                    the_seq2[:, 0:6] = 0
                    p3d2 = data_utils.expmap2xyz_torch(the_seq2)
                    # labels = np.zeros(len(acts),np.float32)
                    # labels[acts.index(action)] = 1
                    label_index = acts_sum.index(action)
                    self.p3d[key + 1] =(label_index,p3d2.view(num_frames2, -1).cpu().data.numpy())
                    # self.p3d[key + 1] = p3d2.view(num_frames2, -1).cpu().data.numpy()

                    # test on 256
                    # fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                    #                                                input_n=self.in_n)
                    # valid_frames_1 = fs_sel1[:, 0]

                    fs_sel1 = np.arange(0, num_frames1 - 100)
                    valid_frames_1=fs_sel1

                    tmp_data_idx_1 = [key] * len(valid_frames_1)
                    tmp_data_idx_2 = list(valid_frames_1)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    num = num + len(tmp_data_idx_2)
                    fs_sel2 = np.arange(0, num_frames2 - 100)
                    valid_frames_2 = fs_sel2
                    # valid_frames_2 = fs_sel2[:, 0] # test on 256
                    tmp_data_idx_1 = [key + 1] * len(valid_frames_2)
                    tmp_data_idx_2 = list(valid_frames_2)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    num = num + len(tmp_data_idx_2)
                    key += 2
                    clas_number[sub_idx, action_idx] = num

        clas_number[-1] = np.sum(clas_number[:-1], axis=0)
        # ignore constant joints and joints at same position with other joints
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)



    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        # return self.p3d[key][fs]
        return self.p3d[key][0],self.p3d[key][1][fs]

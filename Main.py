# coding: utf-8
"""
Author: Inki
Email: inki.yinji@qq.com
Create: 2021 0224
Last modify: 2021 0412
"""

import torch
import torch.nn as nn
import torch.nn.functional as func_nn
import torch.optim as opt
import numpy as np
import torch.utils.data as data_utils

from Prototype import MIL
from Function import get_k_cross_validation_index
from Model import ANET, ENET


class MINN(MIL):
    """"""

    def __init__(self,
                 file_name: str,
                 epoch: int = 10,
                 lr: float = 0.001,
                 max_dim: int = 1000,
                 net_type: str = "a",
                 bag_space: None = None):
        super(MINN, self).__init__(file_name, bag_space=bag_space)
        self.epoch = epoch
        self.lr = lr
        self.max_dim = max_dim
        self.net_type = net_type
        self.net = None
        self.opt = None
        self.loss = nn.CrossEntropyLoss()

    def __get_net(self, mapping_mat):
        if self.net_type == "a":
            self.net = ANET(self.num_att, self.num_class, mapping_mat)
        elif self.net_type == "e":
            self.net = ENET(self.num_att, self.num_class, mapping_mat)

    def __get_optimizer(self):
        """"""
        self.opt = opt.Adam(self.net.parameters(), self.lr)

    def __mapping_data(self, model, mapping_mat):
        """"""
        ret_vec = []
        # weight, b = np.transpose(model[-2].numpy()), model[-1].numpy()
        vec_weight = np.zeros(mapping_mat * self.num_class)
        for i in range(self.num_bag):

            temp_vec, weight = self.__mapping_bag(i, model)
            for j in range(self.num_class):
                # vec_weight[j * mapping_mat: (j + 1) * mapping_mat] = temp_vec * weight[j] + b[j]
                vec_weight[j * mapping_mat: (j + 1) * mapping_mat] = temp_vec * weight[j]
            ret_vec.append(temp_vec.tolist())

        return np.array(ret_vec)

    def __mapping_bag(self, idx, model):
        bag = self.bag_space[idx, 0][:, :-1]
        bag = torch.from_numpy(bag)

        if self.net_type == "a":
            bag = torch.mm(bag.float(), model[0]) + model[1]
            bag = func_nn.relu(bag)

            bag_v = torch.mm(bag, model[2]) + model[3]
            bag_v = func_nn.tanh(bag_v)

            bag_u = torch.mm(bag, model[4]) + model[5]
            bag_u = func_nn.sigmoid(bag_u)

            bag_w = torch.mm(bag_v * bag_u, model[6]) + model[7]
            bag_w = func_nn.softmax(bag_w)

            bag_m = bag + bag_w
            bag_m = torch.mm(bag_m, model[8]) + model[9]
            bag_m = func_nn.sigmoid(bag_m)
            bag_m = torch.mm(bag_m, model[10]) + model[11]
            bag_m = func_nn.sigmoid(bag_m)
        else:
            bag_m = torch.mm(bag.float(), model[0]) + model[1]
            bag_m = func_nn.sigmoid(bag_m)
            bag_m = torch.mm(bag_m, model[2]) + model[3]
            bag_m = func_nn.sigmoid(bag_m)

        weight = torch.mm(bag_m, model[-2]) + model[-1]

        return np.average(bag_m.numpy(), 0), np.average(weight.numpy(), 0)

    def get_mapping(self):
        """"""
        tr_idxes, te_idxes = get_k_cross_validation_index(self.num_bag)
        for i, (tr_idx, te_idx) in enumerate(zip(tr_idxes, te_idxes)):
            # print("%d-th loop of 10-cv" % (i + 1))
            tr_ins, tr_ins_lab, _ = self.get_sub_ins_space(tr_idx)
            tr_ins, tr_ins_lab = torch.from_numpy(tr_ins), torch.from_numpy(tr_ins_lab)
            tr_data = data_utils.TensorDataset(tr_ins, tr_ins_lab)
            tr_loader = data_utils.DataLoader(tr_data, batch_size=10, shuffle=True)
            del tr_ins, tr_ins_lab, tr_data
            # if self.num_att > self.num_ins:
            #     num_mapping = self.num_ins
            # else:
            num_mapping = len(tr_idx)
            self.__get_net(num_mapping)
            self.__get_optimizer()
            batch_count = 0
            for epoch in range(self.epoch):
                tr_loss, tr_acc, num_test_ins = 0, 0, 0
                for batch, (data, label) in enumerate(tr_loader):
                    if label.min() == -1:
                        label[label == -1] = 0
                    pre_lab = self.net(data.float())
                    loss = self.loss(pre_lab, label.long())
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    tr_loss += loss.cpu().item()
                    # pre_lab, tr_label = pre_lab.argmax(dim=1).numpy(), label.numpy()
                    tr_acc += (pre_lab.argmax(dim=1) == label).sum().cpu().item()
                    batch_count += 1
                    num_test_ins += len(label)
                # print("Epoch %d, loss %.4f, tr acc %.4f" % (epoch + 1, tr_loss / batch_count, tr_acc / num_test_ins))
            model = [weight.detach().numpy() for weight in list(self.net.parameters())]
            model = [np.transpose(weight) for weight in model]
            model = [torch.from_numpy(weight) for weight in model]
            temp_model = []
            for weight in model:
                if len(weight.shape) == 1:
                    temp_model.append(torch.reshape(weight, (1, weight.shape[0])).float())
                else:
                    temp_model.append(weight.float())
            mapping_mat = self.__mapping_data(model, num_mapping)

            yield mapping_mat[tr_idx], self.bag_lab[tr_idx], mapping_mat[te_idx], self.bag_lab[te_idx], None


def test_10cv():
    """
    """
    po_label = 0
    file_name = "D:/Data/OneDrive/文档/Code/MIL1/Data/Web/web4+.mat"  # "mnist" + str(po_label) + ".none"
    net_type = "a"
    epoch = 5 * 1

    """======================================================="""
    epoch = 1 if epoch == 0 else epoch
    loops = 5
    # tr_f1_k, tr_acc_k, tr_roc_k = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    # tr_f1_s, tr_acc_s, tr_roc_s = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    # tr_f1_j, tr_acc_j, tr_roc_j = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    te_f1_k, te_acc_k, te_roc_k = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    te_f1_s, te_acc_s, te_roc_s = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    te_f1_j, te_acc_j, te_roc_j = np.zeros(loops), np.zeros(loops), np.zeros(loops)
    print("=================================================")
    print("File name: %s; Net type: %s; Epoch: %d" % (file_name.split(".")[-2].split("/")[-1], net_type, epoch))
    # from MnistLoad import MnistLoader
    # bag_space = MnistLoader(seed=1, po_label=po_label, data_type="mnist", data_path=file_name).bag_space
    dsk = MINN(file_name, epoch=epoch, net_type=net_type)
    from Classifier import Classifier
    for i in range(loops):

        classifier = Classifier(["knn", "svm", "j48"], ["f1_score", "acc", "roc"])
        data_iter = dsk.get_mapping()
        te_per = classifier.test(data_iter)
        # tr_f1_k[i], tr_acc_k[i], tr_roc_k[i] = tr_per["knn"][0], tr_per["knn"][1], tr_per["knn"][2]
        # tr_f1_s[i], tr_acc_s[i], tr_roc_s[i] = tr_per["svm"][0], tr_per["svm"][1], tr_per["svm"][2]
        # tr_f1_j[i], tr_acc_j[i], tr_roc_j[i] = tr_per["j48"][0], tr_per["j48"][1], tr_per["j48"][2]
        te_f1_k[i], te_acc_k[i], te_roc_k[i] = te_per["knn"][0], te_per["knn"][1], te_per["knn"][2]
        te_f1_s[i], te_acc_s[i], te_roc_s[i] = te_per["svm"][0], te_per["svm"][1], te_per["svm"][2]
        te_f1_j[i], te_acc_j[i], te_roc_j[i] = te_per["j48"][0], te_per["j48"][1], te_per["j48"][2]
        print("%.4lf, %.4lf, %.4lf; %.4lf, %.4lf, %.4lf; %.4lf, %.4lf, %.4lf; \n"
              % (te_f1_k[i], te_acc_k[i], te_roc_k[i],
                 te_f1_s[i], te_acc_s[i], te_roc_s[i],
                 te_f1_j[i], te_acc_j[i], te_roc_j[i]
                 ), end=" ")

    # print("%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf "
    #       "%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf "
    #       "%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf " % (np.sum(tr_f1_k) / loops, np.std(tr_f1_k),
    #                                                 np.sum(tr_acc_k) / loops, np.std(tr_acc_k),
    #                                                 np.sum(tr_roc_k) / loops, np.std(tr_roc_k),
    #                                                 np.sum(tr_f1_s) / loops, np.std(tr_f1_s),
    #                                                 np.sum(tr_acc_s) / loops, np.std(tr_acc_s),
    #                                                 np.sum(tr_roc_s) / loops, np.std(tr_roc_s),
    #                                                 np.sum(tr_f1_j) / loops, np.std(tr_f1_j),
    #                                                 np.sum(tr_acc_j) / loops, np.std(tr_acc_j),
    #                                                 np.sum(tr_roc_j) / loops, np.std(tr_roc_j)), end="")
    print("knn-f1 std   knn-acc std   knn-roc std   svm-f1 std    svm-acc std   svm-roc std   "
          "j48-f1 std    j48-acc std   j48-roc std")
    print("%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf "
          "%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf "
          "%.4lf %.4lf %.4lf %.4lf %.4lf %.4lf " % (np.sum(te_f1_k) / loops, np.std(te_f1_k),
                                                    np.sum(te_acc_k) / loops, np.std(te_acc_k),
                                                    np.sum(te_roc_k) / loops, np.std(te_roc_k),
                                                    np.sum(te_f1_s) / loops, np.std(te_f1_s),
                                                    np.sum(te_acc_s) / loops, np.std(te_acc_s),
                                                    np.sum(te_roc_s) / loops, np.std(te_roc_s),
                                                    np.sum(te_f1_j) / loops, np.std(te_f1_j),
                                                    np.sum(te_acc_j) / loops, np.std(te_acc_j),
                                                    np.sum(te_roc_j) / loops, np.std(te_roc_j)), end="")


if __name__ == '__main__':
    import time
    s_t = time.time()
    test_10cv()
    print("%.4f" % (time.time() - s_t))

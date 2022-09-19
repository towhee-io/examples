# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/7/31 20:06
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from Utils.script import calc_acc, calc_rmse, calc_rec

class Trainer:
    def __init__(self, model, dataset, data, optimizer, root, topK):
        self.model = model
        self.dataset = dataset
        self.data = data
        self.optimizer = optimizer
        self.root = root
        self.topK = topK

    def negative_sample(self, train=True):
        # sample enough number of negative edges in dataset
        if train:
            neg_edge_index = negative_sampling(edge_index=self.data.edge_index,
                                               num_nodes=(self.data.num_users, self.data.num_items),
                                               num_neg_samples=self.data.train_gt.size(0), method='sparse')
            # print("neg_edge_index:", neg_edge_index)
            # print("num_neg_edge_index:", neg_edge_index.size(1))
            neg_train_idx = neg_edge_index[0, :] * self.data.num_items + neg_edge_index[1, :]
            aug_edge_index = torch.cat([self.data.edge_index, neg_edge_index], dim=-1)
            aug_gt = torch.cat([self.data.train_gt, torch.zeros_like(self.data.train_gt)], dim=0)
            aug_edge_type = torch.cat([self.data.edge_type, torch.zeros_like(self.data.train_gt)], dim=0)
            aug_idx = torch.cat([self.data.train_idx, neg_train_idx])
        else:
            neg_edge_index = negative_sampling(edge_index=self.data.edge_index,
                                               num_nodes=(self.data.num_users, self.data.num_items),
                                               num_neg_samples=self.data.test_gt.size(0), method='sparse')
            # print("neg_edge_index:", neg_edge_index)
            # print("num_neg_edge_index:", neg_edge_index.size(1))
            neg_test_idx = neg_edge_index[0, :] * self.data.num_items + neg_edge_index[1, :]
            aug_edge_index = torch.cat([self.data.edge_index, neg_edge_index], dim=-1)
            aug_gt = torch.cat([self.data.test_gt, torch.zeros_like(self.data.test_gt)], dim=0)
            aug_edge_type = torch.cat([self.data.edge_type, torch.zeros_like(self.data.test_gt)], dim=0)
            aug_idx = torch.cat([self.data.test_idx, neg_test_idx])
        return aug_edge_index, aug_gt, aug_edge_type, aug_idx

    def training(self, epochs):
        self.epochs = epochs
        for epoch in range(self.epochs):
            # loss, train_rmse = self.train_one(epoch)
            # test_rmse = self.test()
            loss, train_acc = self.train_one(epoch)
            test_acc = self.test()
            # self.summary(epoch, loss, train_rmse=train_rmse, test_rmse=test_rmse)
            metrics = {'loss': loss,
                       'train_acc': train_acc,
                       'test_acc': test_acc}

            self.summary(epoch, metrics)

        self.recommend()

    def train_one(self, epoch):
        self.model.train()
        aug_edge_index, aug_gt, aug_edge_type, aug_idx = self.negative_sample(train=True)
        out = self.model(self.data.x, aug_edge_index,
                         aug_edge_type)
        loss = F.cross_entropy(out[aug_idx], aug_gt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # rmse = calc_rmse(out[self.aug_train_idx], self.aug_train_gt)
        # return loss.item(), rmse.item()
        acc = calc_acc(out[aug_idx], aug_gt)
        return loss.item(), acc.item()

    def test(self):
        self.model.eval()
        aug_edge_index, aug_gt, aug_edge_type, aug_idx = self.negative_sample(train=False)
        out = self.model(self.data.x, aug_edge_index,
                         aug_edge_type)
        # rmse = calc_rmse(out[self.data.test_idx], self.data.test_gt)
        # return rmse.item()
        accuracy = calc_acc(out[aug_idx], aug_gt)
        return accuracy.item()

    def recommend(self):
        self.model.eval()
        aug_edge_index, aug_gt, aug_edge_type, aug_idx = self.negative_sample(train=True)
        out = self.model(self.data.x, aug_edge_index,
                         aug_edge_type)
        precision, recall, rec_lst = calc_rec(out, self.data.num_user_items, self.data.num_items, self.data.train_idx,
                                     self.data.test_idx, self.root, self.topK)

        print(f"overall precision:{precision:.6f}, recall:{recall}")

    def summary(self, epoch, metrics):
        # if test_rmse is None:
        if metrics['test_acc'] is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch, self.epochs, metrics['loss']))
        else:
            # print('[ Epoch: {:>4}/{} | Loss: {:.6f} | TRAIN RMSE: {:.6f} | Test RMSE: {:.6f} ]'.format(
            #    epoch, self.epochs, loss, train_rmse, test_rmse))
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | Train ACC: {:.6f} | Test ACC: {:.6f}'.format(epoch, self.epochs,
                    metrics['loss'], metrics['train_acc'], metrics['test_acc']))
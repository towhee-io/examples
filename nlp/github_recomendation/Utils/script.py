# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/6/23 12:37
import os
import torch
import math
from torch import nn
import torch.nn.functional as F
import pandas as pd


def merge_data(filepath, index, fold_num):
    filename_list = os.listdir(filepath)
    print("filename_list:", filename_list)
    csv_list = [pd.read_csv(os.path.join(filepath, filename)) for filename in filename_list if '.csv' in filename]
    df = pd.concat(csv_list)

    df.to_csv(filepath + f'../dataAll/data{index * fold_num + 1}-{(index + 1) * fold_num}.csv', index=False)


def drop_repeat_row(filepath, key):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates(subset=key, keep="first")

    df.to_csv(filepath, index=False)

def split_users(filepath, n_split, fold_num):
    df = pd.read_csv(filepath)

    for i in range(0, n_split):
        filename = f'../github_users/filtered_user{i}-{i + 1}W.csv'
        if i != n_split - 1:
            temp_df = df[(i * fold_num):(i + 1) * fold_num]
        else:
            temp_df = df[i * fold_num:]
        temp_df.to_csv(filename, index=False)


class Config(dict):
    def __init__(self, config):
        super(Config, self).__init__()
        self._conf = config

    def __getattr__(self, name):
        if self._conf.get(name) is not None:
            return self._conf[name]
        return None

def random_init(tensor, in_dim, out_dim):
    thresh = math.sqrt(6.0 / (in_dim + out_dim))
    if tensor is not None:
        try:
            tensor.data.uniform_(-thresh, thresh)
        except:
            nn.init.uniform_(tensor, a=-thresh, b=thresh)


def init_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            truncated_normal(m.bias)
        except:
            pass

def truncated_normal(tensor, mean=0, std=1):
    tensor.data.fill_(std * 2)
    with torch.no_grad():
        while (True):
            if tensor.max() >= std * 2:
                tensor[tensor >= std * 2] = tensor[tensor >= std * 2].normal_(mean, std)
                tensor.abs_()
            else:
                break


def calc_rmse(pred, gt):
    pred = F.softmax(pred, dim=1)
    expected_pred = torch.zeros(gt.shape)
    pred = pred.cpu().data
    gt = gt.cpu().data
    print(f"pred:{pred}, pred.shape:{pred.shape}, gt:{gt}, gt.shape:{gt.shape}")
    for relation in range(pred.shape[1]):
        expected_pred += pred[:, relation] * (relation + 1)

    rmse = (gt.to(torch.float) + 1) - expected_pred
    rmse = torch.pow(rmse, 2)
    rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)
    return rmse

def calc_acc(pred, gt):
    # print(f"pred:{pred}, pred.shape:{pred.shape}, gt:{gt}, gt.shape:{gt.shape}")
    pred = F.softmax(pred, dim=1)
    mean_acc = (gt == pred.argmax(dim=-1)).float().mean()
    return mean_acc


def calc_rec(out, user_item, num_item, train_idx, test_idx, root, topK):
    out = F.softmax(out, dim=1)
    out = out.cpu().data.numpy()
    precision = 0
    recall = 0
    num_user = user_item.shape[0]
    rec_lst = []
    df_user = pd.read_csv(os.path.join(root, 'users.csv'))
    df_project = pd.read_csv(os.path.join(root, 'projects.csv'))
    user_dict = df_user['name'].to_dict()
    project_dict = df_project['name'].to_dict()

    for i in range(num_user):
        hit = 0
        items = [str(j) for j in range(num_item)]
        probs = out[i * num_item:(i + 1) * num_item, -1].tolist()
        item_prob = dict(zip(items, probs))
        sorted_items = sorted(item_prob.items(), key=lambda x: x[1], reverse=True)
        # print("sorted_items:", sorted_items)
        rec_items = []
        for item, _ in sorted_items:
            item_idx = i * num_item + int(item)
            if item_idx not in train_idx:
                rec_items.append(project_dict[int(item)])
                if item_idx in test_idx:
                    hit += 1
            if len(rec_items) >= topK:
                break
        user_precision = 1.0 * hit / topK
        user_recall = 1.0 * hit / user_item[i]
        precision += user_precision
        recall += user_recall
        rec_lst.append(rec_items)
        if i % 100 == 0:
            print(f"recommended projects for user {user_dict[i]}: {rec_items}")
        # print(f"precision:{user_precision:.6f}, recall:{user_recall:.6f}")

    precision = precision / num_user
    recall = recall / num_user

    return precision, recall, rec_lst


# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/7/17 22:42
import os.path

import numpy as np
import pandas as pd
import torch
import xxhash
from torch_geometric.data import InMemoryDataset, Data


class GithubDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GithubDataset, self).__init__(root, transform, pre_transform)
        # processed_path[0]是处理后的数据，由process method定义
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def num_nodes(self):
        return self.data.x.shape[0]

    @property
    def raw_file_names(self):
        return ['data.csv', 'train.csv', 'test.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self, pivot=0.7):
        # split the training and test dataset
        data_csv = self.raw_paths[0]
        train_csv = self.raw_paths[1]
        test_csv = self.raw_paths[2]

        if not os.path.isfile(train_csv) and not os.path.isfile(test_csv):
            raw_df = pd.read_csv(data_csv)
            train_idx = []
            test_idx = []
            for index, row in raw_df.iterrows():
                user = row['user']
                item = row['project']
                unique_id = str(user) + str(item)
                encoder_id = xxhash.xxh32(unique_id).intdigest()
                # split the data by pivot
                if encoder_id % 100 < int(pivot * 100):
                    train_idx.append(index)
                else:
                    test_idx.append(index)
                if index % 100000 == 0:
                    print(f"loading data {index}")

            print('split training dataset and test dataset success')
            print(f'length of training dataset: {len(train_idx)}')
            print(f'length of test dataset: {len(test_idx)}')

            train_df = raw_df.iloc[train_idx]
            test_df = raw_df.iloc[test_idx]

            train_df.to_csv(train_csv, index=False)
            test_df.to_csv(test_csv, index=False)

        train_df, train_nums = self.create_df(train_csv)
        test_df, test_nums = self.create_df(test_csv)

        train_idx, train_gt = self.create_gt_idx(train_df, train_nums)
        test_idx, test_gt = self.create_gt_idx(test_df, train_nums)
        print("train_idx:", train_idx)
        print("test_idx:", test_idx)

        train_df['project'] = train_df['project'] + train_nums['user']
        x = torch.arange(train_nums['node'], dtype=torch.long)

        #  Prepare edges
        edge_user = torch.tensor(train_df['user'].values)
        print("edge_user:", edge_user, edge_user.shape)
        edge_item = torch.tensor(train_df['project'].values)
        print("edge_item:", edge_item, edge_item.shape)
        edge_index = torch.stack((torch.cat((edge_user, edge_item), 0),
                                  torch.cat((edge_item, edge_user), 0)), 0)
        print("edge_index:", edge_index, edge_index.shape)
        edge_index = edge_index.to(torch.long)

        edge_type = torch.tensor(train_df['has_star'])
        edge_type = torch.cat((edge_type, edge_type), 0)

        # Prepare data
        data = Data(x=x, edge_index=edge_index)
        data.edge_type = edge_type
        data.train_idx = train_idx
        data.test_idx = test_idx
        data.train_gt = train_gt
        data.test_gt = test_gt
        data.num_users = torch.tensor([train_nums['user']])
        data.num_items = torch.tensor([train_nums['project']])
        data.num_user_items = torch.tensor(test_nums['user_item'])

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def create_df(self, csv_path):
        df = pd.read_csv(csv_path)
        num_user = df.max()['user'] + 1
        num_project = df.max()['project'] + 1
        num_node = num_user + num_project + 2
        num_edge = len(df)
        user_item_dict = df['user'].value_counts(sort=False).to_dict()
        num_user_item = np.zeros(num_user)
        for usr, prj in user_item_dict.items():
            num_user_item[int(usr)] = prj

        nums = {'user': num_user,
                'project': num_project,
                'node': num_node,
                'edge': num_edge,
                'user_item': num_user_item}

        return df, nums

    def create_gt_idx(self, df, nums):
        df['idx'] = df['user'] * nums['project'] + df['project']
        idx = torch.tensor(df['idx'])
        # gt: ground truth
        gt = torch.tensor(df['has_star'])
        return idx, gt

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data[0]

    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = GithubDataset(root='../data/tiny/')
data = dataset[0]
data = data.to(device)
print("data:", data)













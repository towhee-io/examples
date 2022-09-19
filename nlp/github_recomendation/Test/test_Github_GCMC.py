# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/8/19 12:40
import torch
import yaml

from Utils import RSDataset
from Model import GCMC
from Train.trainer import Trainer
from Utils.script import random_init, init_xavier, Config

def main(cfg):
    cfg = Config(cfg)

    # device and dataset setting
    device = (torch.device(f'cuda:{cfg.gpu_id}')
        if torch.cuda.is_available() and cfg.gpu_id >= 0
        else torch.device('cpu'))

    # dataset = MVDataset.MCDataset(cfg.root, cfg.dataset_name)
    dataset = RSDataset.GithubDataset(cfg.root)
    data = dataset[0].to(device)

    # add some params to config
    cfg.num_nodes = dataset.num_nodes
    cfg.num_relations = dataset.num_relations
    cfg.num_users = int(data.num_users)

    # set and init model
    model = GCMC.GAE(cfg, random_init).to(device)
    model.apply(init_xavier)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # train
    trainer = Trainer(model, dataset, data, optimizer, cfg.root, cfg.topK)
    trainer.training(cfg.epochs)


if __name__ == '__main__':
    with open('../Resource/config.yml') as f:
        cfg = yaml.safe_load(f)
    print("config:", cfg)
    main(cfg)
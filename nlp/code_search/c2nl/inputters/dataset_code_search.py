# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from c2nl.inputters.vector_code_search import vectorize


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class CommentDataset(Dataset):
    '''
    common dataset
    '''
    def __init__(self, examples, model, max_src_len):
        self.model = model
        self.examples = examples
        self.ex_num_epoch = len(examples)
        self.count_ex = 0
        self.max_src_len = max_src_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # ex = data_augumentation(self.examples[index], self.max_src_len)
        return vectorize(self.examples[index], self.model)

    def lengths(self):
        return [(len(ex['code'].tokens), len(ex['summary'].tokens))
                for ex in self.examples]


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):
    '''
    sampler which sorted samples by length
    '''
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)

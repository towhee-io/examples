# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/misc.py
# -*- coding: utf-8 -*-

import string
import torch
import subprocess
from nltk.stem import PorterStemmer
from c2nl.inputters import constants

ps = PorterStemmer()


def normalize_string(s, dostem=False):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def stem(text):
        if not dostem:
            return text
        return ' '.join([ps.stem(w) for w in text.split()])

    return stem(white_space_fix(remove_punc(lower(s))))


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        'Not all arguments have the same value: ' + str(args)


def validate(sequence):
    seq_wo_punc = sequence.translate(str.maketrans('', '', string.punctuation))
    return len(seq_wo_punc.strip()) > 0


def tens2sen(t, word_dict=None, src_vocabs=None):
    sentences = []
    # loop over the batch elements
    for idx, s in enumerate(t):
        sentence = []
        for wt in s:
            word = wt if isinstance(wt, int) \
                else wt.item()
            if word in [constants.BOS]:
                continue
            if word in [constants.EOS]:
                break
            if word_dict and word < len(word_dict):
                sentence += [word_dict[word]]
            elif src_vocabs:
                word = word - len(word_dict)
                sentence += [src_vocabs[idx][word]]
            else:
                sentence += [str(word)]

        if len(sentence) == 0:
            # NOTE: just a trick not to score empty sentence
            # this has no consequence
            sentence = [str(constants.PAD)]

        sentence = ' '.join(sentence)
        # if not validate(sentence):
        #     sentence = str(constants.PAD)
        sentences += [sentence]
    return sentences


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    :param lengths: 1d tensor [batch_size]
    :param max_len: int
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)  # (0 for pad positions)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
           (hasattr(opt, 'gpu') and opt.gpu > -1)


def generate_relative_positions_matrix(length,
                                       max_relative_positions,
                                       use_neg_dist,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)

    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)

    # Shift values to be >= 0
    if use_neg_dist:
        final_mat = distance_mat_clipped + max_relative_positions
    else:
        final_mat = torch.abs(distance_mat_clipped)

    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])

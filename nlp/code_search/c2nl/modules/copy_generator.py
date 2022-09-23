# src: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/copy_generator.py
""" Generator module """
import torch.nn as nn
import torch

from c2nl.inputters import constants
from c2nl.utils.misc import aeq


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.
    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.
    The copy generator is an extended version of the standard
    generator that computes three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary
    """

    def __init__(self, input_size, tgt_dict, generator, eps=1e-20):
        super(CopyGenerator, self).__init__()
        self.linear = generator
        self.linear_copy = nn.Linear(input_size, 1)
        self.tgt_dict = tgt_dict
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.eps = eps

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.
        Args:
           hidden (`FloatTensor`): hidden outputs `[batch, tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch, tlen, slen]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[batch, src_len, extra_words]`
        """
        # CHECKS
        batch, tlen, _ = hidden.size()  # pylint: disable=unused-variable
        batch_, tlen_, slen = attn.size()  # pylint: disable=unused-variable
        batch, slen_, cvocab = src_map.size()  # pylint: disable=unused-variable
        aeq(tlen, tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, :, self.tgt_dict[constants.PAD_WORD]] = -self.eps
        prob = self.softmax(logits)

        # Probability of copying p(z=1) batch.
        p_copy = self.sigmoid(self.linear_copy(hidden))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy.expand_as(prob))
        mul_attn = torch.mul(attn, p_copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn, src_map)  # `[batch, tlen, extra_words]`
        return torch.cat([out_prob, copy_prob], 2)


class CopyGeneratorCriterion(object):
    """ Copy generator criterion """

    def __init__(self, vocab_size, force_copy, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size

    def __call__(self, scores, align, target):
        # CHECKS
        batch, tlen, _ = scores.size()  # pylint: disable=unused-variable
        _, tlen_ = target.size()
        aeq(tlen, tlen_)
        _, tlen_ = align.size()
        aeq(tlen, tlen_)

        align = align.view(-1)
        target = target.view(-1)
        scores = scores.view(-1, scores.size(2))

        # Compute unks in align and target for readability
        align_unk = align.eq(constants.UNK).float()
        align_not_unk = align.ne(constants.UNK).float()
        target_unk = target.eq(constants.UNK).float()
        target_not_unk = target.ne(constants.UNK).float()

        # Copy probability of tokens in source
        out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
        # Set scores for unk to 0 and add eps
        out = out.mul(align_not_unk) + self.eps
        # Get scores for tokens in target
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)

        # Regular prob (no unks and unks that can't be copied)
        if not self.force_copy:
            # Add score for non-unks in target
            out = out + tmp.mul(target_not_unk)
            # Add score for when word is unk in both align and tgt
            out = out + tmp.mul(align_unk).mul(target_unk)
        else:
            # Forced copy. Add only probability for not-copied tokens
            out = out + tmp.mul(align_unk)

        loss = -out.log()
        return loss

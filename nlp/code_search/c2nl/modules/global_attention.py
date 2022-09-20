# copied from https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/global_attention.py

"""" Global attention modules (Luong / Bahdanau) """
import torch
import torch.nn as nn

from c2nl.utils.misc import aeq, sequence_mask


# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention

class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.
    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.
    .. mermaid::
       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G
    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].
    However they
    differ on how they compute the attention score.
    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`
    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]
    """

    def __init__(self, dim, coverage=False, attn_type="dot"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
            "Please select a valid attention type.")

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self._coverage = coverage

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`
        """
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None,
                coverage=None, softmax_weights=True):
        """
        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
        Returns:
          (`FloatTensor`, `FloatTensor`):
          * Computed vector `[batch x tgt_len x dim]`
          * Attention distribtutions for each query
             `[batch x tgt_len x src_len]`
        """

        # one step input
        assert source.dim() == 3
        one_step = True if source.size(1) == 1 else False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.data.masked_fill_(~mask, -float("inf"))

        # We adopt coverage attn described in Paulus et al., 2018
        # REF: https://arxiv.org/abs/1705.04304
        if self._coverage:
            maxes = torch.max(align, 2, keepdim=True)[0]
            exp_score = torch.exp(align - maxes)

            if one_step:
                if coverage is None:
                    # t = 1 in Eq(3) from Paulus et al., 2018
                    unnormalized_score = exp_score
                else:
                    # t = otherwise in Eq(3) from Paulus et al., 2018
                    assert coverage.dim() == 3  # B x 1 x slen
                    unnormalized_score = exp_score.div(coverage + 1e-20)
            else:
                multiplier = torch.tril(torch.ones(target_l - 1, target_l - 1))
                multiplier = multiplier.unsqueeze(0).expand(batch, *multiplier.size())
                multiplier = multiplier.cuda() if align.is_cuda else multiplier

                penalty = torch.bmm(multiplier, exp_score[:, :-1, :])  # B x tlen-1 x slen
                no_penalty = torch.ones_like(penalty[:, -1, :])  # B x slen
                penalty = torch.cat([no_penalty.unsqueeze(1), penalty], dim=1)  # B x tlen x slen
                assert exp_score.size() == penalty.size()
                unnormalized_score = exp_score.div(penalty + 1e-20)

            # Eq.(4) from Paulus et al., 2018
            align_vectors = unnormalized_score.div(unnormalized_score.sum(2, keepdim=True))

        # Softmax to normalize attention weights
        else:
            align_vectors = self.softmax(align)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch * target_l, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        # Check output sizes
        batch_, target_l_, dim_ = attn_h.size()
        aeq(target_l, target_l_)
        aeq(batch, batch_)
        aeq(dim, dim_)
        batch_, target_l_, source_l_ = align_vectors.size()
        aeq(target_l, target_l_)
        aeq(batch, batch_)
        aeq(source_l, source_l_)

        covrage_vector = None
        if self._coverage and one_step:
            covrage_vector = exp_score  # B x 1 x slen

        if softmax_weights:
            return attn_h, align_vectors, covrage_vector
        else:
            return attn_h, align, covrage_vector

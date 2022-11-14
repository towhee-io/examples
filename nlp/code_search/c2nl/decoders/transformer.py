"""
Implementation of "Attention is All You Need"
"""

import torch
from torch import nn
from c2nl.decoders.decoder import DecoderBase
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.utils.misc import sequence_mask
from c2nl.modules.util_class import LayerNorm


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_k,
                 d_v,
                 d_ff,
                 dropout,
                 max_relative_positions=0,
                 coverage_attn=False):
        super(TransformerDecoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(
            heads, d_model, d_k, d_v, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.layer_norm = LayerNorm(d_model)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, d_k, d_v, dropout=dropout,
            coverage=coverage_attn)
        self.layer_norm_2 = LayerNorm(d_model)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self,
                inputs,
                memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=None,
                step=None,
                coverage=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``
        Returns:
            (FloatTensor, FloatTensor):
            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``
        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        query, _, _ = self.attention(inputs,
                                     inputs,
                                     inputs,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     attn_type="self")
        query_norm = self.layer_norm(self.drop(query) + inputs)

        mid, attn, coverage = self.context_attn(memory_bank,
                                                memory_bank,
                                                query_norm,
                                                mask=src_pad_mask,
                                                layer_cache=layer_cache,
                                                attn_type="context",
                                                step=step,
                                                coverage=coverage)
        mid_norm = self.layer_norm_2(self.drop(mid) + query_norm)

        output = self.feed_forward(mid_norm)
        return output, attn, coverage


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O
    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.2,
                 max_relative_positions=0,
                 coverage_attn=False):
        super(TransformerDecoder, self).__init__()

        self.num_layers = num_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        self._coverage = coverage_attn
        self.layer = nn.ModuleList(
            [TransformerDecoderLayer(d_model,
                                     heads,
                                     d_k,
                                     d_v,
                                     d_ff,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     coverage_attn=coverage_attn)
             for i in range(num_layers)])

    def init_state(self, src_len, max_len):
        """Initialize decoder state."""
        state = dict()
        state["src_len"] = src_len  # [B]
        state["src_max_len"] = max_len  # an integer
        state["cache"] = None
        return state

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self,
                tgt_pad_mask,
                emb,
                memory_bank,
                state,
                step=None,
                layer_wise_coverage=None):
        if step == 0:
            self._init_cache(state)

        assert emb.dim() == 3  # batch x len x embedding_dim
        output = emb

        src_pad_mask = ~sequence_mask(state["src_len"],
                                      max_len=state["src_max_len"]).unsqueeze(1)
        tgt_pad_mask = tgt_pad_mask.unsqueeze(1)  # [B, 1, T_tgt]

        new_layer_wise_coverage = []
        representations = []
        std_attentions = []
        for i, layer in enumerate(self.layer):
            layer_cache = state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            mem_bank = memory_bank[i] if isinstance(memory_bank, list) else memory_bank
            output, attn, coverage = layer(
                output,
                mem_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                coverage=None if layer_wise_coverage is None
                else layer_wise_coverage[i]
            )
            representations.append(output)
            std_attentions.append(attn)
            new_layer_wise_coverage.append(coverage)

        attns = dict()
        attns["std"] = std_attentions[-1]
        attns["coverage"] = None
        if self._coverage:
            attns["coverage"] = new_layer_wise_coverage

        return representations, attns

    def _init_cache(self, state):
        state["cache"] = {}
        for i, _ in enumerate(self.layer):
            layer_cache = dict()
            layer_cache["memory_keys"] = None
            layer_cache["memory_values"] = None
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            state["cache"]["layer_{}".format(i)] = layer_cache

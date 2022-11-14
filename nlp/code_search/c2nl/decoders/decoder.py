import torch
import torch.nn as nn

from c2nl.utils.misc import aeq
from c2nl.decoders.state import RNNDecoderState
from c2nl.modules.global_attention import GlobalAttention


class DecoderBase(nn.Module):
    """Abstract class for decoders.
    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor.
        Subclasses should override this method.
        """

        raise NotImplementedError


# many part of the codes are copied from OpenNMT-Py sources
class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.

    .. mermaid::
       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [LSTM, GRU]
       bidirectional (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`nqa.modules.GlobalAttention`
       dropout (float) : dropout value for :obj:`nn.Dropout`
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 bidirectional_encoder,
                 num_layers,
                 hidden_size,
                 attn_type=None,
                 coverage_attn=False,
                 copy_attn=False,
                 reuse_copy_attn=False,
                 dropout=0.0):

        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        kwargs = {'input_size': input_size,
                  'hidden_size': hidden_size,
                  'num_layers': num_layers,
                  'dropout': dropout,
                  'batch_first': True}
        self.rnn = getattr(nn, rnn_type)(**kwargs)

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = None
        if attn_type:
            self.attn = GlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type
            )
        else:
            assert not self._coverage
            if copy_attn and reuse_copy_attn:
                raise RuntimeError('Attn is turned off, so reuse_copy_attn flag must be false')

        # Set up a separated copy attention layer, if needed.
        self._copy = copy_attn
        self._reuse_copy_attn = reuse_copy_attn
        self.copy_attn = None
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = GlobalAttention(
                hidden_size, attn_type=attn_type
            )

    def count_parameters(self):
        params = list(self.rnn.parameters())
        if self.attn is not None:
            params = params + list(self.attn.parameters())
        if self.copy_attn is not None:
            params = params + list(self.copy_attn.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[batch x tgt_len x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[batch x src_len x hidden]`.
            state (:obj:`onmt.models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[batch x tgt_len x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[batch x tgt_len x src_len]`.
        """
        # Check
        assert isinstance(state, RNNDecoderState)
        # tgt.size() returns tgt length and batch
        tgt_batch, _, _ = tgt.size()
        if self.attn is not None:
            memory_batch, _, _ = memory_bank.size()
            aeq(tgt_batch, memory_batch)
        # END

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths)

        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"]
        # Update the state with the result.
        state.update_state(decoder_final, coverage)

        return decoder_outputs, state, attns

    def init_decoder_state(self, encoder_final):
        """ Init decoder state with last state of the encoder """

        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))

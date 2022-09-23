import torch
from torch import nn


class CharEmbedding(nn.Module):
    """Embeds words based on character embeddings using CNN."""

    def __init__(self, vocab_size, emsize, filter_size, nfilters):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emsize)
        self.convolution = nn.ModuleList([nn.Conv1d(emsize, int(num_filter), int(k))
                                          for (k, num_filter) in zip(filter_size, nfilters)])

    def forward(self, inputs):
        """
        Embed words from character embeddings using CNN.
        Parameters
        --------------------
            inputs      -- 3d tensor (N,sentence_len,word_len)
        Returns
        --------------------
            loss        -- total loss over the input mini-batch (N,sentence_len,char_embed_size)
        """
        # step1: embed the characters
        char_emb = self.embedding(inputs.view(-1, inputs.size(2)))  # (N*sentence_len,word_len,char_emb_size)

        # step2: apply convolution to form word embeddings
        char_emb = char_emb.transpose(1, 2)  # (N*sentence_len,char_emb_size,word_len)
        output = []
        for conv in self.convolution:
            cnn_out = conv(char_emb).transpose(1, 2)  # (N*sentence_len,word_len-filter_size,num_filters)
            cnn_out = torch.max(cnn_out, 1)[0]  # (N*sentence_len,num_filters)
            output.append(cnn_out.view(*inputs.size()[:2], -1))  # appended (N,sentence_len,num_filters)

        output = torch.cat(output, 2)
        return output

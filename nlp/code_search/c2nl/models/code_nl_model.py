import torch
from torch import nn
import torch.nn.functional as f

from prettytable import PrettyTable
from c2nl.modules.char_embedding import CharEmbedding
from c2nl.modules.embeddings import Embeddings
from c2nl.modules.highway import Highway
from c2nl.encoders.transformer import TransformerEncoder
from c2nl.inputters import constants


class Embedder(nn.Module):
    '''
    embedder layer
    '''
    def __init__(self, args):
        super().__init__()

        self.enc_input_size = 0
        self.dec_input_size = 0

        # at least one of word or char embedding options should be True
        assert args.use_src_word or args.use_src_char
        assert args.use_tgt_word or args.use_tgt_char

        self.use_src_word = args.use_src_word
        self.use_tgt_word = args.use_tgt_word
        if self.use_src_word:
            self.src_word_embeddings = Embeddings(args.emsize,
                                                  args.src_vocab_size,
                                                  constants.PAD)
            self.enc_input_size += args.emsize
        if self.use_tgt_word:
            self.tgt_word_embeddings = Embeddings(args.emsize,
                                                  args.tgt_vocab_size,
                                                  constants.PAD)
            self.dec_input_size += args.emsize

        self.use_src_char = args.use_src_char
        self.use_tgt_char = args.use_tgt_char
        if self.use_src_char:
            assert len(args.filter_size) == len(args.nfilters)
            self.src_char_embeddings = CharEmbedding(args.n_characters,
                                                     args.char_emsize,
                                                     args.filter_size,
                                                     args.nfilters)
            self.enc_input_size += sum(list(map(int, args.nfilters)))
            self.src_highway_net = Highway(self.enc_input_size, num_layers=2)

        if self.use_tgt_char:
            assert len(args.filter_size) == len(args.nfilters)
            self.tgt_char_embeddings = CharEmbedding(args.n_characters,
                                                     args.char_emsize,
                                                     args.filter_size,
                                                     args.nfilters)
            self.dec_input_size += sum(list(map(int, args.nfilters)))
            self.tgt_highway_net = Highway(self.dec_input_size, num_layers=2)

        self.use_type = args.use_code_type
        if self.use_type:
            self.type_embeddings = nn.Embedding(len(constants.TOKEN_TYPE_MAP),
                                                self.enc_input_size)

        self.src_pos_emb = args.src_pos_emb
        self.tgt_pos_emb = args.tgt_pos_emb
        self.no_relative_pos = all(v == 0 for v in args.max_relative_pos)

        if self.src_pos_emb and self.no_relative_pos:
            self.src_pos_embeddings = nn.Embedding(args.max_src_len,
                                                   self.enc_input_size)

        if self.tgt_pos_emb:
            self.tgt_pos_embeddings = nn.Embedding(args.max_tgt_len,
                                                   self.dec_input_size)

        self.dropout = nn.Dropout(args.dropout_emb)

    def forward(self,
                sequence,
                sequence_char,
                sequence_type=None,
                mode='encoder',
                step=None):

        if mode == 'encoder':
            word_rep = None
            if self.use_src_word:
                word_rep = self.src_word_embeddings(sequence.unsqueeze(2))  # B x P x d
            if self.use_src_char:
                char_rep = self.src_char_embeddings(sequence_char)  # B x P x f
                if word_rep is None:
                    word_rep = char_rep
                else:
                    word_rep = torch.cat((word_rep, char_rep), 2)  # B x P x d+f
                word_rep = self.src_highway_net(word_rep)  # B x P x d+f

            if self.use_type:
                type_rep = self.type_embeddings(sequence_type)
                word_rep = word_rep + type_rep

            if self.src_pos_emb and self.no_relative_pos:
                pos_enc = torch.arange(start=0,
                                       end=word_rep.size(1)).type(torch.LongTensor)
                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.src_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep

        elif mode == 'decoder':
            word_rep = None
            if self.use_tgt_word:
                word_rep = self.tgt_word_embeddings(sequence.unsqueeze(2))  # B x P x d
            if self.use_tgt_char:
                char_rep = self.tgt_char_embeddings(sequence_char)  # B x P x f
                if word_rep is None:
                    word_rep = char_rep
                else:
                    word_rep = torch.cat((word_rep, char_rep), 2)  # B x P x d+f
                word_rep = self.tgt_highway_net(word_rep)  # B x P x d+f
            if self.tgt_pos_emb:
                if step is None:
                    pos_enc = torch.arange(start=0,
                                           end=word_rep.size(1)).type(torch.LongTensor)
                else:
                    pos_enc = torch.LongTensor([step])  # used in inference time

                pos_enc = pos_enc.expand(*word_rep.size()[:-1])
                if word_rep.is_cuda:
                    pos_enc = pos_enc.cuda()
                pos_rep = self.tgt_pos_embeddings(pos_enc)
                word_rep = word_rep + pos_rep

        else:
            raise ValueError('Unknown embedder mode!')

        word_rep = self.dropout(word_rep)
        return word_rep


class Encoder(nn.Module):
    '''
    encoder built by TransformerEncoder
    '''
    def __init__(self,
                 args,
                 input_size):
        super(Encoder, self).__init__()

        self.transformer = TransformerEncoder(num_layers=args.nlayers,
                                              d_model=input_size,
                                              heads=args.num_head,
                                              d_k=args.d_k,
                                              d_v=args.d_v,
                                              d_ff=args.d_ff,
                                              dropout=args.trans_drop,
                                              max_relative_positions=args.max_relative_pos,
                                              use_neg_dist=args.use_neg_dist)
        self.use_all_enc_layers = args.use_all_enc_layers
        if self.use_all_enc_layers:
            self.layer_weights = nn.Linear(input_size, 1, bias=False)

    def count_parameters(self):
        return self.transformer.count_parameters()

    def forward(self,
                inputt,
                input_len):
        layer_outputs, _ = self.transformer(inputt, input_len)  # B x seq_len x h
        if self.use_all_enc_layers:
            output = torch.stack(layer_outputs, dim=2)  # B x seq_len x nlayers x h
            layer_scores = self.layer_weights(output).squeeze(3)
            layer_scores = f.softmax(layer_scores, dim=-1)
            memory_bank = torch.matmul(output.transpose(2, 3),
                                       layer_scores.unsqueeze(3)).squeeze(3)
        else:
            memory_bank = layer_outputs[-1]
        return memory_bank, layer_outputs


class Aggregate(nn.Module):
    """scale output from encoder"""
    def __init__(self, args):
        super().__init__()
        self.m1 = nn.Parameter(torch.randn(args.max_src_len, 1))

    def forward(self, x):
        """
        Args: x, shape: [batch size, args.max_src_len, args.emsize]
        Rerturn: output, shape: [batch size, args.emsize]
        """
        x_ = x.permute([0, 2, 1])
        x = torch.matmul(x_, self.m1)
        output = torch.squeeze(x, dim=2)
        return output


class Compress(nn.Module):
    '''
    compress output from code encoder and output from comment encoder
    '''
    def __init__(self, args, dropout=0.3):
        super(Compress, self).__init__()
        self.fn1 = nn.Linear(args.max_src_len, 1)
        self.fn2 = nn.Linear(args.emsize, args.emsize)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        Args:
            x: from code_encoder, shape: [batch size, args.max_src_len, args.emsize]
            y: shape: [batch size, args.emsize]
        Return:
            which shape is: x: [batch size, args.emsize], y: [batch size, args.emsize]
        """
        x_ = x.permute([0, 2, 1])
        outout_x = self.dropout_1(self.relu(self.fn1(x_)))
        output_x = torch.squeeze(outout_x, dim=2)

        output_y = self.dropout_2(self.fn2(y))
        return output_x, output_y


class Transformer(nn.Module):
    """Module that writes an answer for the question given a passage."""

    def __init__(self, args):
        """"Constructor of the class."""
        super().__init__()

        self.name = 'Transformer'
        if len(args.max_relative_pos) != args.nlayers:
            assert len(args.max_relative_pos) == 1
            args.max_relative_pos = args.max_relative_pos * args.nlayers

        self.embedder = Embedder(args)
        self.encoder = Encoder(args, self.embedder.enc_input_size)
        self.aggregate = Aggregate(args)
        # self.compress = Compress(args)
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))  # pylint: disable=not-callable
        self.layer_wise_attn = args.layer_wise_attn

        self.criterion = nn.CrossEntropyLoss()

    def _run_forward_ml(self,
                        code_word_rep,
                        code_char_rep,
                        code_type_rep,
                        code_len,
                        summ_word_rep,
                        summ_char_rep,
                        summ_len,
                        tgt_seq,
                        return_embed=False,
                        valid=False,
                        **kwargs):  # pylint: disable=unused-argument
        code_batch = None
        nl_batch = None
        if code_word_rep is not None:
            batch_size = code_word_rep.shape[0]
            # embed and encode the source sequence
            code_rep = self.embedder(code_word_rep,
                                    code_char_rep,
                                    code_type_rep,
                                    mode='encoder')
            memory_bank, layer_wise_outputs = self.encoder(code_rep, code_len)  # pylint: disable=unused-variable
            code_batch = self.aggregate(memory_bank)
            # code_batch, nl_batch = self.compress(memory_bank, tgt_seq)
            code_batch = nn.functional.normalize(code_batch, dim=1)

        if summ_word_rep is not None:
            summ_rep = self.embedder(summ_word_rep,
                                    summ_char_rep,
                                    summ_len,
                                    mode='encoder')
            memory_bank, layer_wise_outputs = self.encoder(summ_rep, summ_len)  # B x seq_len x h
            nl_batch = self.aggregate(memory_bank)
            # code_batch, nl_batch = self.compress(memory_bank, tgt_seq)
            nl_batch = nn.functional.normalize(nl_batch, dim=1)

        if return_embed:
            if code_batch is None:
                return nl_batch
            elif nl_batch is None:
                return code_batch
            else:
                return code_batch, nl_batch

        # scaled pairwise cosine similarities
        logits = torch.mm(code_batch, nl_batch.t()) * torch.exp(self.temperature)
        labels = torch.arange(0, batch_size)
        labels = labels.cuda() if logits.is_cuda else labels
        loss_1 = self.criterion(logits, labels)
        loss_2 = self.criterion(logits.t(), labels)
        loss = (loss_1 + loss_2) / 2
        acc1 = (labels == torch.argmax(logits, dim=1)).sum().float() / batch_size
        acc2 = (labels == torch.argmax(logits, dim=0)).sum().float() / batch_size
        if valid:
            return loss, acc1.float(), acc2.float(), code_batch, nl_batch
        else:
            return loss, acc1.float(), acc2.float()

    def forward(self,
                code_word_rep,
                code_char_rep,
                code_type_rep,
                code_len,
                summ_word_rep,
                summ_char_rep,
                summ_len,
                tgt_seq,
                return_embed=False,
                **kwargs):
        """
        Input:
            - code_word_rep: ``(batch_size, max_doc_len)``
            - code_char_rep: ``(batch_size, max_doc_len, max_word_len)``
            - code_len: ``(batch_size)``
            - summ_word_rep: ``(batch_size, max_que_len)``
            - summ_char_rep: ``(batch_size, max_que_len, max_word_len)``
            - summ_len: ``(batch_size)``
            - tgt_seq: ``(batch_size, max_len)``
        Output:
            - ``(batch_size, P_LEN)``, ``(batch_size, P_LEN)``
        """
        return self._run_forward_ml(code_word_rep,
                                    code_char_rep,
                                    code_type_rep,
                                    code_len,
                                    summ_word_rep,
                                    summ_char_rep,
                                    summ_len,
                                    tgt_seq,
                                    return_embed,
                                    **kwargs)


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_encoder_parameters(self):
        return self.encoder.count_parameters()

    def layer_wise_parameters(self):
        table = PrettyTable()
        table.field_names = ['Layer Name', 'Output Shape', 'Param #']
        table.align['Layer Name'] = 'l'
        table.align['Output Shape'] = 'r'
        table.align['Param #'] = 'r'
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table

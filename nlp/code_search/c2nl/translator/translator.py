#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import torch
import torch.nn.functional as f

from c2nl.translator.beam import Beam
from c2nl.inputters import constants


class Translator(object):
    """
    Uses a model to translate a batch of sentences.
    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
    """

    def __init__(self,
                 model,
                 use_gpu,
                 beam_size,
                 n_best=1,
                 max_length=100,
                 global_scorer=None,
                 copy_attn=False,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 ignore_when_blocking=[],
                 replace_unk=False):

        self.use_gpu = use_gpu
        self.model = model
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.min_length = min_length
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = set(ignore_when_blocking)
        self.replace_unk = replace_unk

    def translate_batch(self, batch_inputs):
        # Eval mode
        self.model.network.eval()

        code_word_rep = batch_inputs['code_word_rep']
        code_char_rep = batch_inputs['code_char_rep']
        code_type_rep = batch_inputs['code_type_rep']
        code_mask_rep = batch_inputs['code_mask_rep']
        code_len = batch_inputs['code_len']
        source_map = batch_inputs['src_map']
        alignment = batch_inputs['alignment']
        blank = batch_inputs['blank']
        fill = batch_inputs['fill']

        beam_size = self.beam_size
        batch_size = code_len.size(0)

        # Define a list of tokens to exclude from ngram-blocking
        # exclusion_list = ["<t>", "</t>", "."]
        exclusion_tokens = set([self.model.tgt_dict[t]
                                for t in self.ignore_when_blocking])

        beam = [Beam(beam_size,
                     n_best=self.n_best,
                     cuda=self.use_gpu,
                     global_scorer=self.global_scorer,
                     pad=self.model.tgt_dict[constants.PAD_WORD],
                     eos=self.model.tgt_dict[constants.EOS_WORD],
                     bos=self.model.tgt_dict[constants.BOS_WORD],
                     min_length=self.min_length,
                     stepwise_penalty=self.stepwise_penalty,
                     block_ngram_repeat=self.block_ngram_repeat,
                     exclusion_tokens=exclusion_tokens)
                for __ in range(batch_size)]

        # Help functions for working with beams and batches
        def var(a):
            return torch.tensor(a)

        def rvar(a):
            return var(a.repeat(beam_size, 1, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        model = self.model.network.module \
            if hasattr(self.model.network, 'module') else self.model.network
        model_name = model.name
        embedder = model.embedder
        encoder = model.encoder
        decoder = model.decoder
        generator = model.generator
        copy_generator = model.copy_generator if self.copy_attn else None

        # (1) Run the encoder on the src.
        code_rep = embedder(code_word_rep,
                            code_char_rep,
                            code_type_rep,
                            mode='encoder')
        # memory_bank: B x P x h; enc_states: l*num_directions x B x h
        if model_name == 'Transformer':
            memory_bank, layer_wise_outputs = encoder(code_rep, code_len)
            src_lens = code_len.repeat(beam_size)
            dec_states = decoder.init_decoder(src_lens, memory_bank.shape[1])
        else:
            enc_states, memory_bank = encoder(code_rep, code_len)
            dec_states = decoder.init_decoder(enc_states)
            # (2) Repeat src objects `beam_size` times.
            if isinstance(dec_states, tuple):  # for split decoder
                dec_states[0].repeat_beam_size_times(beam_size)
                dec_states[1].repeat_beam_size_times(beam_size)
            else:
                dec_states.repeat_beam_size_times(beam_size)

        src_lengths = code_len
        if src_lengths is None:
            src_lengths = torch.Tensor(batch_size).type_as(memory_bank) \
                .long() \
                .fill_(memory_bank.size(1))

        # (2) Repeat src objects `beam_size` times.
        if model_name == 'Transformer' and model.layer_wise_attn:
            memory_bank = [rvar(lwo.data) for lwo in layer_wise_outputs]
        else:
            memory_bank = rvar(memory_bank.data)

        memory_lengths = src_lengths.repeat(beam_size)
        if code_mask_rep is not None:
            code_mask_rep = code_mask_rep.repeat(beam_size, 1)
        src_map = rvar(source_map) if self.copy_attn else None
        attn = {"coverage": None}

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length + 1):
            if all((b.done for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = torch.stack([b.get_current_state() for b in beam])
            # Making it beam_size x batch_size and then apply view
            inp = var(inp.t().contiguous().view(-1, 1))

            # Turn any copied words to UNKs
            if self.copy_attn:
                inp = inp.masked_fill(inp.gt(len(self.model.tgt_dict) - 1),
                                      constants.UNK)

            inp_chars = None
            if embedder.use_tgt_char:
                words = [self.model.tgt_dict[w] for w in inp[:, 0].tolist()]
                inp_chars = [self.model.tgt_dict.word_to_char_ids(w).tolist() for w in words]
                inp_chars = torch.Tensor(inp_chars).to(inp).unsqueeze(1)

            if model_name == 'Transformer':
                tgt = embedder(inp, inp_chars, mode='decoder', step=i)
                # Run one step.
                # applicable for Transformer
                tgt_pad_mask = inp.data.eq(constants.PAD)
                layer_wise_dec_out, attn = decoder.decode(tgt_pad_mask,
                                                          tgt,
                                                          memory_bank,
                                                          dec_states,
                                                          step=i,
                                                          layer_wise_coverage=attn['coverage'])

                dec_out = layer_wise_dec_out[-1]
                # attn["std"] is a list (of size num_heads),
                # so we pick the attention from first head
                attn["std"] = attn["std"][0]
                if self.copy_attn:
                    _, copy_score, _ = model.copy_attn(dec_out,
                                                       memory_bank,
                                                       memory_lengths=memory_lengths,
                                                       softmax_weights=False)

                    # mask copy_attn weights here if needed
                    if code_mask_rep is not None:
                        mask = code_mask_rep.byte().unsqueeze(1)  # Make it broadcastable.
                        copy_score.data.masked_fill_(mask, -float('inf'))
                    attn["copy"] = f.softmax(copy_score, dim=-1)

            else:
                # rnn-based model
                tgt = embedder(inp, inp_chars, mode='decoder')
                dec_out, attn = decoder.decode(tgt,
                                               dec_states,
                                               memory_bank,
                                               memory_lengths)
                if "std" in attn:
                    attn["std"] = f.softmax(attn["std"], dim=-1)
                if self.copy_attn:
                    copy_score = attn["copy"]
                    if code_mask_rep is not None:
                        mask = code_mask_rep.byte().unsqueeze(1)  # Make it broadcastable.
                        copy_score.data.masked_fill_(mask, -float('inf'))
                    attn["copy"] = f.softmax(copy_score, dim=-1)

            # (b) Compute a vector of batch x beam word scores.
            if self.copy_attn:
                out = copy_generator.forward(dec_out, attn["copy"], src_map)
                out = out.squeeze(1)
                # beam x batch_size x tgt_vocab
                out = unbottle(out.data)
                for b in range(out.size(0)):
                    for bx in range(out.size(1)):
                        if blank[bx]:
                            blank_b = torch.Tensor(blank[bx]).to(code_word_rep)
                            fill_b = torch.Tensor(fill[bx]).to(code_word_rep)
                            out[b, bx].index_add_(0, fill_b,
                                                  out[b, bx].index_select(0, blank_b))
                            out[b, bx].index_fill_(0, blank_b, 1e-10)
                beam_attn = unbottle(attn["copy"].squeeze(1))
            else:
                out = generator.forward(dec_out.squeeze(1))
                # beam x batch_size x tgt_vocab
                out = unbottle(f.softmax(out, dim=1))
                # beam x batch_size x tgt_vocab
                beam_attn = unbottle(attn["std"].squeeze(1))

            out = out.log()

            # (c) Advance each beam.
            for j, b in enumerate(beam):
                if not b.done:
                    b.advance(out[:, j],
                              beam_attn.data[:, j, :memory_lengths[j]])
                if model_name != 'Transformer':
                    if isinstance(dec_states, tuple):  # for split decoder
                        dec_states[0].beam_update(j, b.get_current_origin(), beam_size)
                        dec_states[1].beam_update(j, b.get_current_origin(), beam_size)
                    else:
                        dec_states.beam_update(j, b.get_current_origin(), beam_size)

        # (4) Extract sentences from beam.
        ret = self._from_beam(beam)
        return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

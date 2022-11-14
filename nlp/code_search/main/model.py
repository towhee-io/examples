import copy
import math
import logging
from tqdm import tqdm

import torch
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
from c2nl.config import override_model_args
from c2nl.models.transformer import Transformer
from c2nl.utils.copy_utils import collapse_copy_scores, replace_unknown, \
    make_src_map, align
from c2nl.utils.misc import tens2sen, count_file_lines

logger = logging.getLogger(__name__)


class Code2NaturalLanguage(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, src_dict, tgt_dict, state_dict=None):
        # Book-keeping.
        self.args = args
        self.src_dict = src_dict
        self.args.src_vocab_size = len(src_dict)
        self.tgt_dict = tgt_dict
        self.args.tgt_vocab_size = len(tgt_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        if args.model_type == 'transformer':
            self.network = Transformer(self.args, tgt_dict)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)

        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.args.fix_embeddings:
            self.network.embedder.src_word_embeddings.fix_word_lut()
            self.network.embedder.tgt_word_embeddings.fix_word_lut()

        if self.args.optimizer == 'sgd':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(parameters,
                                       self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)

        elif self.args.optimizer == 'adam':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(parameters,
                                        self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)

        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex

            source_map = make_src_map(ex['src_map'])
            source_map = source_map.cuda(non_blocking=True) if self.use_cuda \
                else source_map

            alignment = align(ex['alignment'])
            alignment = alignment.cuda(non_blocking=True) if self.use_cuda \
                else alignment

            blank, fill = collapse_copy_scores(self.tgt_dict, ex['src_vocab'])

        code_word_rep = ex['code_word_rep']
        code_char_rep = ex['code_char_rep']
        code_type_rep = ex['code_type_rep']
        code_mask_rep = ex['code_mask_rep']
        code_len = ex['code_len']
        summ_word_rep = ex['summ_word_rep']
        summ_char_rep = ex['summ_char_rep']
        summ_len = ex['summ_len']
        tgt_seq = ex['tgt_seq']

        if any(l is None for l in ex['language']):
            ex_weights = None
        else:
            ex_weights = [self.args.dataset_weights[lang] for lang in ex['language']]
            ex_weights = torch.FloatTensor(ex_weights)

        if self.use_cuda:
            code_len = code_len.cuda(non_blocking=True)
            summ_len = summ_len.cuda(non_blocking=True)
            tgt_seq = tgt_seq.cuda(non_blocking=True)
            if code_word_rep is not None:
                code_word_rep = code_word_rep.cuda(non_blocking=True)
            if code_char_rep is not None:
                code_char_rep = code_char_rep.cuda(non_blocking=True)
            if code_type_rep is not None:
                code_type_rep = code_type_rep.cuda(non_blocking=True)
            if code_mask_rep is not None:
                code_mask_rep = code_mask_rep.cuda(non_blocking=True)
            if summ_word_rep is not None:
                summ_word_rep = summ_word_rep.cuda(non_blocking=True)
            if summ_char_rep is not None:
                summ_char_rep = summ_char_rep.cuda(non_blocking=True)
            if ex_weights is not None:
                ex_weights = ex_weights.cuda(non_blocking=True)

        # Run forward
        net_loss = self.network(code_word_rep=code_word_rep,
                                code_char_rep=code_char_rep,
                                code_type_rep=code_type_rep,
                                code_len=code_len,
                                summ_word_rep=summ_word_rep,
                                summ_char_rep=summ_char_rep,
                                summ_len=summ_len,
                                tgt_seq=tgt_seq,
                                src_map=source_map,
                                alignment=alignment,
                                src_dict=self.src_dict,
                                tgt_dict=self.tgt_dict,
                                max_len=self.args.max_tgt_len,
                                blank=blank,
                                fill=fill,
                                source_vocab=ex['src_vocab'],
                                code_mask_rep=code_mask_rep,
                                example_weights=ex_weights)

        loss = net_loss['ml_loss'].mean() if self.parallel \
            else net_loss['ml_loss']
        loss_per_token = net_loss['loss_per_token'].mean() if self.parallel \
            else net_loss['loss_per_token']
        ml_loss = loss.item()
        loss_per_token = loss_per_token.item()
        loss_per_token = 10 if loss_per_token > 10 else loss_per_token
        perplexity = math.exp(loss_per_token)

        loss.backward()

        clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.updates += 1
        return {
            'ml_loss': ml_loss,
            'perplexity': perplexity
        }

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, replace_unk=False):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch examples
            replace_unk: replace `unk` tokens while generating predictions
            src_raw: raw source (passage); required to replace `unk` term
        Output:
            predictions: #batch predicted sequences
        """
        # Eval mode
        self.network.eval()

        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:
            assert 'src_map' in ex and 'alignment' in ex

            source_map = make_src_map(ex['src_map'])
            source_map = source_map.cuda(non_blocking=True) if self.use_cuda \
                else source_map

            blank, fill = collapse_copy_scores(self.tgt_dict, ex['src_vocab'])

        code_word_rep = ex['code_word_rep']
        code_char_rep = ex['code_char_rep']
        code_type_rep = ex['code_type_rep']
        code_mask_rep = ex['code_mask_rep']
        code_len = ex['code_len']
        if self.use_cuda:
            code_len = code_len.cuda(non_blocking=True)
            if code_word_rep is not None:
                code_word_rep = code_word_rep.cuda(non_blocking=True)
            if code_char_rep is not None:
                code_char_rep = code_char_rep.cuda(non_blocking=True)
            if code_type_rep is not None:
                code_type_rep = code_type_rep.cuda(non_blocking=True)
            if code_mask_rep is not None:
                code_mask_rep = code_mask_rep.cuda(non_blocking=True)

        decoder_out = self.network(code_word_rep=code_word_rep,
                                   code_char_rep=code_char_rep,
                                   code_type_rep=code_type_rep,
                                   code_len=code_len,
                                   summ_word_rep=None,
                                   summ_char_rep=None,
                                   summ_len=None,
                                   tgt_seq=None,
                                   src_map=source_map,
                                   alignment=alignment,
                                   max_len=self.args.max_tgt_len,
                                   src_dict=self.src_dict,
                                   tgt_dict=self.tgt_dict,
                                   blank=blank, fill=fill,
                                   source_vocab=ex['src_vocab'],
                                   code_mask_rep=code_mask_rep)

        predictions = tens2sen(decoder_out['predictions'],
                               self.tgt_dict,
                               ex['src_vocab'])
        if replace_unk:
            for i in range(len(predictions)):
                enc_dec_attn = decoder_out['attentions'][i]
                if self.args.model_type == 'transformer':
                    assert enc_dec_attn.dim() == 3
                    enc_dec_attn = enc_dec_attn.mean(1)
                predictions[i] = replace_unknown(predictions[i],
                                                 enc_dec_attn,
                                                 src_raw=ex['code_tokens'][i])
                if self.args.uncase:
                    predictions[i] = predictions[i].lower()

        targets = [summ for summ in ex['summ_text']]
        return predictions, targets, decoder_out['copy_info']

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'src_dict': self.src_dict,
            'tgt_dict': self.tgt_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'src_dict': self.src_dict,
            'tgt_dict': self.tgt_dict,
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        tgt_dict = saved_params['tgt_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return Code2NaturalLanguage(args, src_dict, tgt_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        tgt_dict = saved_params['tgt_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = Code2NaturalLanguage(args, src_dict, tgt_dict, state_dict)
        model.updates = updates
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
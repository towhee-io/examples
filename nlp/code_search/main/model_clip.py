import copy
import logging
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from c2nl.config import override_model_args
from code_search.c2nl.models.code_nl_model import Transformer


logger = logging.getLogger(__name__)


class Code2NaturalLanguage(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, src_dict, state_dict=None):
        # Book-keeping.
        self.args = args
        self.src_dict = src_dict
        self.args.src_vocab_size = len(src_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        if args.model_type == 'transformer':
            self.network = Transformer(self.args)
        else:
            raise RuntimeError(f'Unsupported model: {args.model_type}')

        # Load saved state
        if state_dict:
            # Load buffer separately
            model_dict = self.network.state_dict()
            pretrained_dict = {k:v for k, v in state_dict.items() if k in model_dict}
            if model_dict.keys() != state_dict.keys():
                print('\n'*3)
                print('WARNNING!!! There are keys not match!')
                print('miss keys: ', [key for key in model_dict if key not in state_dict])
                print('\n'*3)
                del pretrained_dict['embedder.tgt_pos_embeddings.weight']
                del pretrained_dict['embedder.src_word_embeddings.make_embedding.emb_luts.0.weight']
                # print([i for i in pretrained_dict.keys()])
            model_dict.update(pretrained_dict)
            self.network.load_state_dict(model_dict)

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
            raise RuntimeError(f'Unsupported optimizer: {self.args.optimizer}')

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
        loss, acc1, acc2 = self.network(code_word_rep=code_word_rep,
                                        code_char_rep=code_char_rep,
                                        code_type_rep=code_type_rep,
                                        code_len=code_len,
                                        summ_word_rep=summ_word_rep,
                                        summ_char_rep=summ_char_rep,
                                        summ_len=summ_len,
                                        tgt_seq=tgt_seq)

        loss = loss.mean() if self.parallel else loss
        acc1 = acc1.mean() if self.parallel else acc1
        acc2 = acc2.mean() if self.parallel else acc2
        loss.backward()

        clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.updates += 1
        return {
            'ml_loss': loss.item(),
            'acc1': acc1.item(),
            'acc2': acc2.item()
        }

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex):
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

        code_word_rep = ex['code_word_rep']
        code_char_rep = ex['code_char_rep']
        code_type_rep = ex['code_type_rep']
        code_mask_rep = ex['code_mask_rep']
        code_len = ex['code_len']
        summ_word_rep = ex['summ_word_rep']
        summ_char_rep = ex['summ_char_rep']
        summ_len = ex['summ_len']
        tgt_seq = ex['tgt_seq']
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

        loss, acc1, acc2, code_batch, nl_batch = self.network(
                                        code_word_rep=code_word_rep,
                                        code_char_rep=code_char_rep,
                                        code_type_rep=code_type_rep,
                                        code_len=code_len,
                                        summ_word_rep=summ_word_rep,
                                        summ_char_rep=summ_char_rep,
                                        summ_len=summ_len,
                                        tgt_seq=tgt_seq,
                                        valid=True)
        loss = loss.mean() if self.parallel else loss
        acc1 = acc1.mean() if self.parallel else acc1
        acc2 = acc2.mean() if self.parallel else acc2
        return {
            'ml_loss': loss.item(),
            'acc1': acc1.item(),
            'acc2': acc2.item(),
            'code_batch': code_batch.cpu().numpy(),
            'nl_batch': nl_batch.cpu().numpy()
        }


    def get_code_embed_batch(self, ex):

        self.network.eval()

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
        with torch.no_grad():
            code_batch = self.network(code_word_rep=code_word_rep,
                                      code_char_rep=code_char_rep,
                                      code_type_rep=code_type_rep,
                                      code_len=code_len,
                                      tgt_seq=None,
                                      return_embed=True)
            code_batch = code_batch.cpu().numpy()
        return code_batch


    def get_code_nl_embed_batch(self, ex):

        self.network.eval()

        code_word_rep = ex['code_word_rep']
        code_char_rep = ex['code_char_rep']
        code_type_rep = ex['code_type_rep']
        code_mask_rep = ex['code_mask_rep']
        code_len = ex['code_len']
        summ_word_rep = ex['summ_word_rep']
        summ_char_rep = ex['summ_char_rep']
        summ_len = ex['summ_len']
        tgt_seq = ex['tgt_seq']
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
        with torch.no_grad():
            code_batch, nl_batch = self.network(code_word_rep=code_word_rep,
                                                code_char_rep=code_char_rep,
                                                code_type_rep=code_type_rep,
                                                code_len=code_len,
                                                summ_word_rep=summ_word_rep,
                                                summ_char_rep=summ_char_rep,
                                                summ_len=summ_len,
                                                tgt_seq=tgt_seq,
                                                return_embed=True)
            code_batch = code_batch.cpu().numpy()
            nl_batch = nl_batch.cpu().numpy()
        return code_batch, nl_batch


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
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:  # pylint: disable=broad-except
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'src_dict': self.src_dict,
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:  # pylint: disable=broad-except
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s', filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return Code2NaturalLanguage(args, src_dict, state_dict)

    @staticmethod
    def load_pretrained_model(filename, new_args, src_dict):
        logger.info('Loading model %s', filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        state_dict = saved_params['state_dict']
        return Code2NaturalLanguage(new_args, src_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True):
        logger.info('Loading model %s', filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        src_dict = saved_params['src_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = Code2NaturalLanguage(args, src_dict, state_dict)
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

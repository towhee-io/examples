# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import torch
import logging
import subprocess
import argparse
import numpy as np

import c2nl.config as config
import c2nl.inputters.utils as util
import c2nl.inputters.vector as vector
import c2nl.inputters.dataset as data
from tqdm import tqdm
from main.model import Code2NaturalLanguage
from main.train import compute_eval_score

from collections import OrderedDict
from c2nl.utils.copy_utils import collapse_copy_scores, make_src_map, align
from c2nl.inputters.timer import AverageMeter, Timer
from c2nl.inputters import constants
from c2nl.eval.bleu import Bleu, nltk_corpus_bleu, corpus_bleu
from c2nl.eval.rouge import Rouge
from c2nl.eval.meteor import Meteor
from c2nl.translator.translator import Translator
from c2nl.translator.beam import GNMTGlobalScorer
from c2nl.translator.translation import TranslationBuilder

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_test_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--data_workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--test_batch_size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dataset_name', nargs='+', type=str, required=True,
                       help='Name of the experimental dataset')
    files.add_argument('--model_dir', type=str, default='/tmp/qa_models/',
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model_name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data_dir', type=str, default='/data/',
                       help='Directory of training/validation data')
    files.add_argument('--dev_src', nargs='+', type=str, required=True,
                       help='Preprocessed dev source file')
    files.add_argument('--dev_src_tag', nargs='+', type=str,
                       help='Preprocessed dev source tag file')
    files.add_argument('--dev_tgt', nargs='+', type=str,
                       help='Preprocessed dev target file')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncase', type='bool', default=False,
                            help='Code and summary words will be lower-cased')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_generate', type='bool', default=False,
                         help='Only generate code summaries')

    # Beam Search
    bsearch = parser.add_argument_group('Beam Search arguments')
    bsearch.add_argument('--beam_size', type=int, default=4,
                         help='Set the beam size (=1 means greedy decoding)')
    bsearch.add_argument('--n_best', type=int, default=1,
                         help="""If verbose is set, will output the n_best
                           decoded sentences""")
    bsearch.add_argument('--stepwise_penalty', type='bool', default=False,
                         help="""Apply penalty at every decoding step.
                           Helpful for summary penalty.""")
    bsearch.add_argument('--length_penalty', default='none',
                         choices=['none', 'wu', 'avg'],
                         help="""Length Penalty to use.""")
    bsearch.add_argument('--coverage_penalty', default='none',
                         choices=['none', 'wu', 'summary'],
                         help="""Coverage Penalty to use.""")
    bsearch.add_argument('--block_ngram_repeat', type=int, default=0,
                         help='Block repetition of ngrams during decoding.')
    bsearch.add_argument('--ignore_when_blocking', nargs='+', type=str,
                         default=[],
                         help="""Ignore these strings when blocking repeats.
                           You want to block sentence delimiters.""")
    bsearch.add_argument('--gamma', type=float, default=0.,
                         help="""Google NMT length penalty parameter
                            (higher = longer generation)""")
    bsearch.add_argument('--beta', type=float, default=0.,
                         help="""Coverage penalty parameter""")
    bsearch.add_argument('--replace_unk', action="store_true",
                         help="""Replace the generated UNK tokens with the
                           source token that had highest attention weight. If
                           phrase_table is provided, it will lookup the
                           identified source token and give the corresponding
                           target token. If it is not provided(or the identified
                           source token does not exist in the table) then it
                           will copy the source token""")
    bsearch.add_argument('--verbose', action="store_true",
                         help='Print scores and predictions for each sentence')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.dev_src_files = []
    args.dev_tgt_files = []
    args.dev_src_tag_files = []

    num_dataset = len(args.dataset_name)
    if num_dataset > 1:
        if len(args.dev_src) == 1:
            args.dev_src = args.dev_src * num_dataset
        if len(args.dev_tgt) == 1:
            args.dev_tgt = args.dev_tgt * num_dataset
        if len(args.dev_src_tag) == 1:
            args.dev_src_tag = args.dev_src_tag * num_dataset

    for i in range(num_dataset):
        dataset_name = args.dataset_name[i]
        data_dir = os.path.join(args.data_dir, dataset_name)
        dev_src = os.path.join(data_dir, args.dev_src[i])
        if not os.path.isfile(dev_src):
            raise IOError('No such file: %s' % dev_src)
        if args.only_generate:
            dev_tgt = None
        else:
            dev_tgt = os.path.join(data_dir, args.dev_tgt[i])
            if not os.path.isfile(dev_tgt):
                raise IOError('No such file: %s' % dev_tgt)
        if args.use_code_type:
            dev_src_tag = os.path.join(data_dir, args.dev_src_tag[i])
            if not os.path.isfile(dev_src_tag):
                raise IOError('No such file: %s' % dev_src_tag)
        else:
            dev_src_tag = None

        args.dev_src_files.append(dev_src)
        args.dev_tgt_files.append(dev_tgt)
        args.dev_src_tag_files.append(dev_src_tag)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '_beam.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.pred_file = os.path.join(args.model_dir, args.model_name + '_beam.json')


# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------

def build_translator(model, args):
    scorer = GNMTGlobalScorer(args.gamma,
                              args.beta,
                              args.coverage_penalty,
                              args.length_penalty)

    translator = Translator(model,
                            args.cuda,
                            args.beam_size,
                            n_best=args.n_best,
                            max_length=args.max_tgt_len,
                            copy_attn=model.args.copy_attn,
                            global_scorer=scorer,
                            min_length=0,
                            stepwise_penalty=args.stepwise_penalty,
                            block_ngram_repeat=args.block_ngram_repeat,
                            ignore_when_blocking=args.ignore_when_blocking,
                            replace_unk=args.replace_unk)
    return translator


def prepare_batch(batch, model):
    # To enable copy attn, collect source map and alignment info
    batch_inputs = dict()

    if model.args.copy_attn:
        assert 'src_map' in batch and 'alignment' in batch
        source_map = make_src_map(batch['src_map'])
        source_map = source_map.cuda(non_blocking=True) if args.cuda \
            else source_map
        alignment = None
        blank, fill = collapse_copy_scores(model.tgt_dict, batch['src_vocab'])
    else:
        source_map, alignment = None, None
        blank, fill = None, None

    batch_inputs['src_map'] = source_map
    batch_inputs['alignment'] = alignment
    batch_inputs['blank'] = blank
    batch_inputs['fill'] = fill

    code_word_rep = batch['code_word_rep']
    code_char_rep = batch['code_char_rep']
    code_type_rep = batch['code_type_rep']
    code_mask_rep = batch['code_mask_rep']
    code_len = batch['code_len']
    if args.cuda:
        code_len = batch['code_len'].cuda(non_blocking=True)
        if code_word_rep is not None:
            code_word_rep = code_word_rep.cuda(non_blocking=True)
        if code_char_rep is not None:
            code_char_rep = code_char_rep.cuda(non_blocking=True)
        if code_type_rep is not None:
            code_type_rep = code_type_rep.cuda(non_blocking=True)
        if code_mask_rep is not None:
            code_mask_rep = code_mask_rep.cuda(non_blocking=True)

    batch_inputs['code_word_rep'] = code_word_rep
    batch_inputs['code_char_rep'] = code_char_rep
    batch_inputs['code_type_rep'] = code_type_rep
    batch_inputs['code_mask_rep'] = code_mask_rep
    batch_inputs['code_len'] = code_len
    return batch_inputs


def validate_official(args, data_loader, model):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """

    eval_time = Timer()
    translator = build_translator(model, args)
    builder = TranslationBuilder(model.tgt_dict,
                                 n_best=args.n_best,
                                 replace_unk=args.replace_unk)

    # Run through examples
    examples = 0
    trans_dict, sources = dict(), dict()
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for batch_no, ex in enumerate(pbar):
            batch_size = ex['batch_size']
            ids = list(range(batch_no * batch_size,
                             (batch_no * batch_size) + batch_size))
            batch_inputs = prepare_batch(ex, model)

            ret = translator.translate_batch(batch_inputs)
            targets = [[summ] for summ in ex['summ_text']]
            translations = builder.from_batch(ret,
                                              ex['code_tokens'],
                                              targets,
                                              ex['src_vocab'])

            src_sequences = [code for code in ex['code_text']]
            for eid, trans, src in zip(ids, translations, src_sequences):
                trans_dict[eid] = trans
                sources[eid] = src

            examples += batch_size

    hypotheses, references = dict(), dict()
    for eid, trans in trans_dict.items():
        hypotheses[eid] = [' '.join(pred) for pred in trans.pred_sents]
        hypotheses[eid] = [constants.PAD_WORD if len(hyp.split()) == 0
                           else hyp for hyp in hypotheses[eid]]
        references[eid] = trans.targets

    if args.only_generate:
        with open(args.pred_file, 'w') as fw:
            json.dump(hypotheses, fw, indent=4)

    else:
        bleu, rouge_l, meteor, precision, recall, f1, ind_bleu, ind_rouge = \
            eval_accuracies(hypotheses, references)
        logger.info('beam evaluation official: '
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                    (bleu, rouge_l, meteor) +
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
                    'examples = %d | ' %
                    (precision, recall, f1, examples) +
                    'test time = %.2f (s)' % eval_time.time())

        with open(args.pred_file, 'w') as fw:
            for eid, translation in trans_dict.items():
                out_dict = OrderedDict()
                out_dict['id'] = eid
                out_dict['code'] = sources[eid]
                # printing all beam search predictions
                out_dict['predictions'] = [' '.join(pred) for pred in translation.pred_sents]
                out_dict['references'] = references[eid]
                out_dict['bleu'] = ind_bleu[eid]
                out_dict['rouge_l'] = ind_rouge[eid]
                fw.write(json.dumps(out_dict) + '\n')


def eval_accuracies(hypotheses, references):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert sorted(references.keys()) == sorted(hypotheses.keys())

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, _ = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(references, hypotheses)

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    for key in references.keys():
        _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0], references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)

    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100, ind_bleu, ind_rouge


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')
    dev_exs = []
    for dev_src, dev_src_tag, dev_tgt, dataset_name in \
            zip(args.dev_src_files, args.dev_src_tag_files,
                args.dev_tgt_files, args.dataset_name):
        dev_files = dict()
        dev_files['src'] = dev_src
        dev_files['src_tag'] = dev_src_tag
        dev_files['tgt'] = dev_tgt
        exs = util.load_data(args,
                             dev_files,
                             max_examples=args.max_examples,
                             dataset_name=dataset_name,
                             test_split=True)
        dev_exs.extend(exs)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    if not os.path.isfile(args.model_file):
        raise IOError('No such file: %s' % args.model_file)
    model = Code2NaturalLanguage.load(args.model_file)

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    dev_dataset = data.CommentDataset(dev_exs, model)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
        drop_last=args.parallel
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST
    validate_official(args, dev_loader, model)


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Code to Natural Language Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_test_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = torch.cuda.is_available()
    args.parallel = torch.cuda.device_count() > 1

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)

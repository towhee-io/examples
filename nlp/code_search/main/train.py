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
from c2nl.inputters import constants

from collections import OrderedDict, Counter
from tqdm import tqdm
from c2nl.inputters.timer import AverageMeter, Timer
import c2nl.inputters.vector as vector
import c2nl.inputters.dataset as data

from main.model import Code2NaturalLanguage
from c2nl.eval.bleu import corpus_bleu
from c2nl.eval.rouge import Rouge
from c2nl.eval.meteor import Meteor

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def add_train_args(parser):
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
    runtime.add_argument('--num_epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=32,
                         help='Batch size for training')
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
    files.add_argument('--train_src', nargs='+', type=str,
                       help='Preprocessed train source file')
    files.add_argument('--train_src_tag', nargs='+', type=str,
                       help='Preprocessed train source tag file')
    files.add_argument('--train_tgt', nargs='+', type=str,
                       help='Preprocessed train target file')
    files.add_argument('--dev_src', nargs='+', type=str, required=True,
                       help='Preprocessed dev source file')
    files.add_argument('--dev_src_tag', nargs='+', type=str,
                       help='Preprocessed dev source tag file')
    files.add_argument('--dev_tgt', nargs='+', type=str, required=True,
                       help='Preprocessed dev target file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--max_examples', type=int, default=-1,
                            help='Maximum number of examples for training')
    preprocess.add_argument('--uncase', type='bool', default=False,
                            help='Code and summary words will be lower-cased')
    preprocess.add_argument('--src_vocab_size', type=int, default=None,
                            help='Maximum allowed length for src dictionary')
    preprocess.add_argument('--tgt_vocab_size', type=int, default=None,
                            help='Maximum allowed length for tgt dictionary')
    preprocess.add_argument('--max_characters_per_token', type=int, default=30,
                            help='Maximum number of characters allowed per token')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid_metric', type=str, default='bleu',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--sort_by_len', type='bool', default=True,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--print_copy_info', type='bool', default=False,
                     help='Print copy information')
    log.add_argument('--print_one_target', type='bool', default=False,
                     help='Print only one target sequence')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    if not args.only_test:
        args.train_src_files = []
        args.train_tgt_files = []
        args.train_src_tag_files = []

        num_dataset = len(args.dataset_name)
        if num_dataset > 1:
            if len(args.train_src) == 1:
                args.train_src = args.train_src * num_dataset
            if len(args.train_tgt) == 1:
                args.train_tgt = args.train_tgt * num_dataset
            if len(args.train_src_tag) == 1:
                args.train_src_tag = args.train_src_tag * num_dataset

        for i in range(num_dataset):
            dataset_name = args.dataset_name[i]
            data_dir = os.path.join(args.data_dir, dataset_name)
            train_src = os.path.join(data_dir, args.train_src[i])
            train_tgt = os.path.join(data_dir, args.train_tgt[i])
            if not os.path.isfile(train_src):
                raise IOError('No such file: %s' % train_src)
            if not os.path.isfile(train_tgt):
                raise IOError('No such file: %s' % train_tgt)
            if args.use_code_type:
                train_src_tag = os.path.join(data_dir, args.train_src_tag[i])
                if not os.path.isfile(train_src_tag):
                    raise IOError('No such file: %s' % train_src_tag)
            else:
                train_src_tag = None

            args.train_src_files.append(train_src)
            args.train_tgt_files.append(train_tgt)
            args.train_src_tag_files.append(train_src_tag)

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
        dev_tgt = os.path.join(data_dir, args.dev_tgt[i])
        if not os.path.isfile(dev_src):
            raise IOError('No such file: %s' % dev_src)
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
    suffix = '_test' if args.only_test else ''
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.log_file = os.path.join(args.model_dir, args.model_name + suffix + '.txt')
    args.pred_file = os.path.join(args.model_dir, args.model_name + suffix + '.json')
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')

    if args.use_src_word or args.use_tgt_word:
        # Make sure fix_embeddings and pretrained are consistent
        if args.fix_embeddings and not args.pretrained:
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    else:
        args.fix_embeddings = False

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    src_dict = util.build_word_and_char_dict(args,
                                             examples=train_exs + dev_exs,
                                             fields=['code', 'summary'],
                                             dict_size=args.src_vocab_size,
                                             no_special_token=True)
    tgt_dict = util.build_word_and_char_dict(args,
                                             examples=train_exs + dev_exs,
                                             fields=['summary'],
                                             dict_size=args.tgt_vocab_size,
                                             no_special_token=False)
    logger.info('Num words in source = %d and target = %d' % (len(src_dict), len(tgt_dict)))

    # Initialize model
    model = Code2NaturalLanguage(config.get_model_args(args), src_dict, tgt_dict)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    perplexity = AverageMeter()
    epoch_time = Timer()

    current_epoch = global_stats['epoch']
    pbar = tqdm(data_loader)

    pbar.set_description("%s" % 'Epoch = %d [perplexity = x.xx, ml_loss = x.xx]' %
                         current_epoch)

    # Run one epoch
    for idx, ex in enumerate(pbar):
        bsz = ex['batch_size']
        if args.optimizer in ['sgd', 'adam'] and current_epoch <= args.warmup_epochs:
            cur_lrate = global_stats['warmup_factor'] * (model.updates + 1)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = cur_lrate
        net_loss = model.update(ex)
        ml_loss.update(net_loss['ml_loss'], bsz)
        perplexity.update(net_loss['perplexity'], bsz)
        log_info = 'Epoch = %d [perplexity = %.2f, ml_loss = %.2f]' % \
                   (current_epoch, perplexity.avg, ml_loss.avg)

        pbar.set_description("%s" % log_info)

    logger.info('train: Epoch %d | perplexity = %.2f | ml_loss = %.2f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, perplexity.avg, ml_loss.avg, epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint', current_epoch + 1)


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats, mode='dev'):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for idx, ex in enumerate(pbar):
            batch_size = ex['batch_size']
            ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
            predictions, targets, copy_info = model.predict(ex, replace_unk=True)

            src_sequences = [code for code in ex['code_text']]
            examples += batch_size
            for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                hypotheses[key] = [pred]
                references[key] = tgt if isinstance(tgt, list) else [tgt]
                sources[key] = src

            if copy_info is not None:
                copy_info = copy_info.cpu().numpy().astype(int).tolist()
                for key, cp in zip(ex_ids, copy_info):
                    copy_dict[key] = cp

            pbar.set_description("%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])

    copy_dict = None if len(copy_dict) == 0 else copy_dict
    bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(hypotheses,
                                                                   references,
                                                                   copy_dict,
                                                                   sources=sources,
                                                                   filename=args.pred_file,
                                                                   print_copy_info=args.print_copy_info,
                                                                   mode=mode)
    result = dict()
    result['bleu'] = bleu
    result['rouge_l'] = rouge_l
    result['meteor'] = meteor
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    if mode == 'test':
        logger.info('test valid official: '
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                    (bleu, rouge_l, meteor) +
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
                    'examples = %d | ' %
                    (precision, recall, f1, examples) +
                    'test time = %.2f (s)' % eval_time.time())

    else:
        logger.info('dev valid official: Epoch = %d | ' %
                    (global_stats['epoch']) +
                    'bleu = %.2f | rouge_l = %.2f | '
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | examples = %d | ' %
                    (bleu, rouge_l, precision, recall, f1, examples) +
                    'valid time = %.2f (s)' % eval_time.time())

    return result


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1


def eval_accuracies(hypotheses, references, copy_info, sources=None,
                    filename=None, print_copy_info=False, mode='dev'):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, ind_bleu = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        meteor = 0

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    fw = open(filename, 'w') if filename else None
    for key in references.keys():
        _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0],
                                              references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
        if fw:
            if copy_info is not None and print_copy_info:
                prediction = hypotheses[key][0].split()
                pred_i = [word + ' [' + str(copy_info[key][j]) + ']'
                          for j, word in enumerate(prediction)]
                pred_i = [' '.join(pred_i)]
            else:
                pred_i = hypotheses[key]

            logobj = OrderedDict()
            logobj['id'] = key
            if sources is not None:
                logobj['code'] = sources[key]
            logobj['predictions'] = pred_i
            logobj['references'] = references[key][0] if args.print_one_target \
                else references[key]
            logobj['bleu'] = ind_bleu[key]
            logobj['rouge_l'] = ind_rouge[key]
            fw.write(json.dumps(logobj) + '\n')

    if fw: fw.close()
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')

    train_exs = []
    if not args.only_test:
        args.dataset_weights = dict()
        for train_src, train_src_tag, train_tgt, dataset_name in \
                zip(args.train_src_files, args.train_src_tag_files,
                    args.train_tgt_files, args.dataset_name):
            train_files = dict()
            train_files['src'] = train_src
            train_files['src_tag'] = train_src_tag
            train_files['tgt'] = train_tgt
            exs = util.load_data(args,
                                 train_files,
                                 max_examples=args.max_examples,
                                 dataset_name=dataset_name)
            lang_name = constants.DATA_LANG_MAP[dataset_name]
            args.dataset_weights[constants.LANG_ID_MAP[lang_name]] = len(exs)
            train_exs.extend(exs)

        logger.info('Num train examples = %d' % len(train_exs))
        args.num_train_examples = len(train_exs)
        for lang_id in args.dataset_weights.keys():
            weight = (1.0 * args.dataset_weights[lang_id]) / len(train_exs)
            args.dataset_weights[lang_id] = round(weight, 2)
        logger.info('Dataset weights = %s' % str(args.dataset_weights))

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
    start_epoch = 1
    if args.only_test:
        if args.pretrained:
            model = Code2NaturalLanguage.load(args.pretrained)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError('No such file: %s' % args.model_file)
            model = Code2NaturalLanguage.load(args.model_file)
    else:
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            # Just resume training, no modifications.
            logger.info('Found a checkpoint...')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = Code2NaturalLanguage.load_checkpoint(checkpoint_file, args.cuda)
        else:
            # Training starts fresh. But the model state is either pretrained or
            # newly (randomly) initialized.
            if args.pretrained:
                logger.info('Using pretrained model...')
                model = Code2NaturalLanguage.load(args.pretrained, args)
            else:
                logger.info('Training model from scratch...')
                model = init_from_scratch(args, train_exs, dev_exs)

            # Set up optimizer
            model.init_optimizer()
            # log the parameter details
            logger.info('Trainable #parameters [encoder-decoder] {} [total] {}'.format(
                human_format(model.network.count_encoder_parameters() +
                             model.network.count_decoder_parameters()),
                human_format(model.network.count_parameters())))
            table = model.network.layer_wise_parameters()
            logger.info('Breakdown of the trainable paramters\n%s' % table)

    # Use the GPU?
    if args.cuda:
        model.cuda()

    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    if not args.only_test:
        train_dataset = data.CommentDataset(train_exs, model)
        if args.sort_by_len:
            train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                    args.batch_size,
                                                    shuffle=True)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.cuda,
            drop_last=args.parallel
        )

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

    if args.only_test:
        stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
        validate_official(args, dev_loader, model, stats, mode='test')

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info('-' * 100)
        logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}

        if args.optimizer in ['sgd', 'adam'] and args.warmup_epochs >= start_epoch:
            logger.info("Use warmup lrate for the %d epoch, from 0 up to %s." %
                        (args.warmup_epochs, args.learning_rate))
            num_batches = len(train_loader.dataset) // args.batch_size
            warmup_factor = (args.learning_rate + 0.) / (num_batches * args.warmup_epochs)
            stats['warmup_factor'] = warmup_factor

        for epoch in range(start_epoch, args.num_epochs + 1):
            stats['epoch'] = epoch
            if args.optimizer in ['sgd', 'adam'] and epoch > args.warmup_epochs:
                model.optimizer.param_groups[0]['lr'] = \
                    model.optimizer.param_groups[0]['lr'] * args.lr_decay

            train(args, train_loader, model, stats)
            result = validate_official(args, dev_loader, model, stats)

            # Save best valid
            if result[args.valid_metric] > stats['best_valid']:
                logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                            (args.valid_metric, result[args.valid_metric],
                             stats['epoch'], model.updates))
                model.save(args.model_file)
                stats['best_valid'] = result[args.valid_metric]
                stats['no_improvement'] = 0
            else:
                stats['no_improvement'] += 1
                if stats['no_improvement'] >= args.early_stop:
                    break


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Code to Natural Language Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
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
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
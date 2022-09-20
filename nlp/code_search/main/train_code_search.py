# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py
import sys
import os
import json
import torch
import logging
import subprocess
import argparse
import numpy as np
from tqdm import tqdm
sys.path.append('.')
sys.path.append('..')

import c2nl.config as config  # pylint: disable=wrong-import-position
import c2nl.inputters.utils_code_search as util  # pylint: disable=wrong-import-position
from c2nl.inputters import constants  # pylint: disable=wrong-import-position
from c2nl.inputters.timer import AverageMeter, Timer  # pylint: disable=wrong-import-position
import c2nl.inputters.vector_code_search as vector  # pylint: disable=wrong-import-position
import c2nl.inputters.dataset_code_search as data  # pylint: disable=wrong-import-position
from main.model_clip import Code2NaturalLanguage  # pylint: disable=wrong-import-position

import tracemalloc  # pylint: disable=wrong-import-position
tracemalloc.start()

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def human_format(num):
    num = float(f'{num:.3g}')
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    a = f'{num:f}'.rstrip('0').rstrip('.')
    b = ['', 'K', 'M', 'B', 'T'][magnitude]
    return a + b
    # return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
    #                      ['', 'K', 'M', 'B', 'T'][magnitude])


def add_train_args(parser):  # pylint: disable=redefined-outer-name
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
    general.add_argument('--valid_metric', type=str, default='MRR',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display_iter', type=int, default=25,
                         help='Log state after every <display_iter> batches')
    general.add_argument('--sort_by_len', type='bool', default=False,
                         help='Sort batches by length for speed')
    general.add_argument('--only_test', type='bool', default=False,
                         help='Only do testing')
    general.add_argument('--only_eval_calculate_MRR', type='bool', default=False,
                         help='Only evalating and calculating MRR')

    # Log results Learning
    log = parser.add_argument_group('Log arguments')
    log.add_argument('--print_copy_info', type='bool', default=False,
                     help='Print copy information')
    log.add_argument('--print_one_target', type='bool', default=False,
                     help='Print only one target sequence')


def set_defaults(args):  # pylint: disable=redefined-outer-name
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    if not args.only_test and not args.only_eval_calculate_MRR:
        args.train_src_files = []
        args.train_tgt_files = []

        num_dataset = len(args.dataset_name)
        if num_dataset > 1:
            if len(args.train_src) == 1:
                args.train_src = args.train_src * num_dataset
            if len(args.train_tgt) == 1:
                args.train_tgt = args.train_tgt * num_dataset

        for i in range(num_dataset):
            dataset_name = args.dataset_name[i]
            data_dir = os.path.join(args.data_dir, dataset_name)
            train_src = os.path.join(data_dir, args.train_src[i])
            train_tgt = os.path.join(data_dir, args.train_tgt[i])
            if not os.path.isfile(train_src):
                raise IOError(f'No such file: {train_src}')
            if not os.path.isfile(train_tgt):
                raise IOError(f'No such file: {train_tgt}')

            args.train_src_files.append(train_src)
            args.train_tgt_files.append(train_tgt)

    args.dev_src_files = []
    args.dev_tgt_files = []

    # num_dataset = len(args.dataset_name)
    # if num_dataset > 1:
    #     if len(args.dev_src) == 1:
    #         args.dev_src = args.dev_src * num_dataset
    #     if len(args.dev_tgt) == 1:
    #         args.dev_tgt = args.dev_tgt * num_dataset
    if len(args.dev_src) == 1 and len(args.dev_tgt) == 1:
        num_dataset = 1

    for i in range(num_dataset):
        dataset_name = args.dataset_name[i]
        data_dir = os.path.join(args.data_dir, dataset_name)
        dev_src = os.path.join(data_dir, args.dev_src[i])
        dev_tgt = os.path.join(data_dir, args.dev_tgt[i])
        if not os.path.isfile(dev_src):
            raise IOError(f'No such file: {dev_src}')
        if not os.path.isfile(dev_tgt):
            raise IOError(f'No such file: {dev_tgt}')

        args.dev_src_files.append(dev_src)
        args.dev_tgt_files.append(dev_tgt)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid  # pylint: disable=import-outside-toplevel
        import time  # pylint: disable=import-outside-toplevel
        args.model_name = time.strftime('%Y%m%d-') + str(uuid.uuid4())[:8]

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


def init_from_scratch(args, train_exs, dev_exs):  # pylint: disable=redefined-outer-name
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    src_dict = util.build_word_and_char_dict(args,
                                             examples=train_exs + dev_exs,
                                             fields=['code'],
                                             dict_size=args.src_vocab_size,
                                             no_special_token=True)
    logger.info('Num words in source = %d', len(src_dict))

    # Initialize model
    model = Code2NaturalLanguage(config.get_model_args(args), src_dict)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):  # pylint: disable=redefined-outer-name
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    epoch_time = Timer()

    current_epoch = global_stats['epoch']
    pbar = tqdm(data_loader)

    pbar.set_description(f'Epoch = {current_epoch} [ml_loss = x.xx, acc1 = x.xx, acc2 = x.xx]')

    # Run one epoch
    for ex in pbar:
        bsz = ex['batch_size']
        if args.optimizer in ['sgd', 'adam'] and current_epoch <= args.warmup_epochs:
            cur_lrate = global_stats['warmup_factor'] * (model.updates + 1)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = cur_lrate

        net_loss = model.update(ex)
        ml_loss.update(net_loss['ml_loss'], bsz)
        log_info = f'Epoch = {current_epoch} [ml_loss = {net_loss["ml_loss"]:.2f}, ' + \
            f'acc1 = {net_loss["acc1"]:.2f}, acc2 = {net_loss["acc2"]:.2f}]'

        pbar.set_description(log_info)

    logger.info('train: Epoch %d | ml_loss = %.2f | '
                'Time for epoch = %.2f (s)', current_epoch, ml_loss.avg, epoch_time.time())

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint', current_epoch + 1)


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(data_loader, model, global_stats, mode='dev'):
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
    results = []
    code_embeds = []
    nl_embeds = []
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for ex in pbar:
            batch_size = ex['batch_size']
            result = model.predict(ex)
            code_embeds.append(result['code_batch'])
            nl_embeds.append(result['nl_batch'])
            results.append(result)
            pbar.set_description(f'Epoch = {global_stats["epoch"]} [validating ... ]')
            examples += batch_size

    code_embeds = np.concatenate(code_embeds, axis=0)
    nl_embeds = np.concatenate(nl_embeds, axis=0)
    assert code_embeds.shape == nl_embeds.shape
    mrr = calculate_mrr(code_embeds, nl_embeds)

    result = dict()
    result['loss'] = float(sum([i['ml_loss'] for i in results])) / len(results)
    result['acc1'] = float(sum([i['acc1'] for i in results])) / len(results)
    result['acc2'] = float(sum([i['acc2'] for i in results])) / len(results)
    result['MRR'] = mrr

    if mode == 'test':
        logger.info(f'test valid official: '  # pylint: disable=logging-fstring-interpolation
                    f'loss = {result["loss"]:.2f} | acc1 = {result["acc1"]:.2f}'
                    f' | acc2 = {result["acc2"]:.2f} | MRR = {mrr:.2f} | '
                    f'examples = {examples} | '
                    f'test time = {eval_time.time():.2f} (s)')

    else:
        logger.info(f'dev valid official: Epoch = {(global_stats["epoch"])} | '  # pylint: disable=logging-fstring-interpolation
                    f'loss = {result["loss"]:.2f} | acc1 = {result["acc1"]:.2f}'
                    f' | acc2 = {result["acc2"]:.2f} | MRR = {mrr:.2f} | '
                    f'examples = {examples} | '
                    f'test time = {eval_time.time():.2f} (s)')

    return result


def calculate_mrr(code_embeds, nl_embeds):

    scores = np.matmul(nl_embeds, code_embeds.T)
    # np.save('/workspace/NLP/code_search/NeuralCodeSum/scores_from_{}.npy'.format(
    #     args.model_dir.split('/')[-1]), scores)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

    ranks = []
    for count, ids in tqdm(enumerate(sort_ids)):
        rank = 1
        for i in ids[:1000]:
            if rank == 1000:
                ranks.append(0)
                break
            if i != count:
                rank += 1
            else:
                ranks.append(1/rank)
                break

    return float(np.mean(ranks))


def eval_calculate_mrr(data_loader, model):
    code_embeds = []
    nl_embeds = []
    for ex in tqdm(data_loader):
        code_batch, nl_batch = model.get_code_nl_embed_batch(ex)
        code_embeds.append(code_batch)
        nl_embeds.append(nl_batch)
    code_embeds = np.concatenate(code_embeds, axis=0)
    nl_embeds = np.concatenate(nl_embeds, axis=0)
    assert code_embeds.shape == nl_embeds.shape

    return calculate_mrr(code_embeds, nl_embeds)


def main(args):  # pylint: disable=redefined-outer-name
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')

    train_exs = []
    if not args.only_test and not args.only_eval_calculate_MRR:
        args.dataset_weights = {}
        for train_src, train_tgt, dataset_name in \
                zip(args.train_src_files, args.train_tgt_files, args.dataset_name):
            train_files = {}
            train_files['src'] = train_src
            train_files['tgt'] = train_tgt
            exs = util.load_data(args,
                                 train_files,
                                 max_examples=args.max_examples,
                                 dataset_name=dataset_name)
            lang_name = constants.DATA_LANG_MAP[dataset_name]
            args.dataset_weights[constants.LANG_ID_MAP[lang_name]] = len(exs)
            train_exs.extend(exs)

        logger.info('Num train examples = %d', len(train_exs))
        args.num_train_examples = len(train_exs)
        for lang_id in args.dataset_weights.keys():
            weight = (1.0 * args.dataset_weights[lang_id]) / len(train_exs)
            args.dataset_weights[lang_id] = round(weight, 2)
        # logger.info('Dataset weights = %s' % str(args.dataset_weights))

    dev_exs = []
    for dev_src, dev_tgt, dataset_name in \
            zip(args.dev_src_files, args.dev_tgt_files, args.dataset_name):
        dev_files = {}
        dev_files['src'] = dev_src
        dev_files['tgt'] = dev_tgt
        exs = util.load_data(args,
                             dev_files,
                             max_examples=args.max_examples,
                             dataset_name=dataset_name)
        dev_exs.extend(exs)
    logger.info('Num dev examples = %d', len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 1
    if args.only_test or args.only_eval_calculate_MRR:
        if args.pretrained:
            model = Code2NaturalLanguage.load(args.pretrained)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError(f'No such file: {args.model_file}')
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

                # logger.info('-' * 100)
                # logger.info('Rebuild word dictionary')
                # src_dict = util.build_word_and_char_dict(args,
                #                                         examples=train_exs + dev_exs,
                #                                         fields=['code'],
                #                                         dict_size=args.src_vocab_size,
                #                                         no_special_token=True)
                # logger.info('Num words in source = %d' % (len(src_dict)))
                # # Initialize model
                # model = Code2NaturalLanguage.load_pretrained_model(args.pretrained, args, src_dict)
            else:
                logger.info('Training model from scratch...')
                model = init_from_scratch(args, train_exs, dev_exs)

            # Set up optimizer
            model.init_optimizer()
            # log the parameter details
            logger.info('Trainable #parameters [encoder-decoder] %s [total] %s',
                human_format(model.network.count_encoder_parameters()),
                human_format(model.network.count_parameters()))
            table = model.network.layer_wise_parameters()
            logger.info('Breakdown of the trainable paramters\n%s', table)

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

    if not args.only_test and not args.only_eval_calculate_MRR:
        train_dataset = data.CommentDataset(train_exs, model, args.max_src_len)
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

    dev_dataset = data.CommentDataset(dev_exs, model, args.max_src_len)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
        drop_last=False
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s', json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST

    if args.only_test:
        stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
        validate_official(dev_loader, model, stats, mode='test')
    elif args.only_eval_calculate_MRR:
        mrr = eval_calculate_mrr(dev_loader, model)
        print('MRR:', mrr)
    else:
        # TRAIN/VALID LOOP
        logger.info('-' * 100)
        logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}

        if args.optimizer in ['sgd', 'adam'] and args.warmup_epochs >= start_epoch:
            logger.info('Use warmup lrate for the %d epoch, from 0 up to %s.',
                        args.warmup_epochs, args.learning_rate)
            num_batches = len(train_loader.dataset) // args.batch_size
            warmup_factor = (args.learning_rate + 0.) / (num_batches * args.warmup_epochs)
            stats['warmup_factor'] = warmup_factor

        for epoch in range(start_epoch, args.num_epochs + 1):
            stats['epoch'] = epoch
            if args.optimizer in ['sgd', 'adam'] and epoch > args.warmup_epochs:
                model.optimizer.param_groups[0]['lr'] = \
                    model.optimizer.param_groups[0]['lr'] * args.lr_decay

            train(args, train_loader, model, stats)
            result = validate_official(dev_loader, model, stats)

            # Save best valid
            result_ = None
            if args.valid_metric == 'acc1 and acc2':
                result_ = (result['acc1'] + result['acc2']) / 2
            elif args.valid_metric == 'MRR':
                result_ = result['MRR']
            else:
                raise KeyError
            if result_ > stats['best_valid']:
                logger.info('Best valid: %s = %.2f (epoch %d, %d updates)',
                            args.valid_metric, result_,
                            stats['epoch'], model.updates)
                model.save(args.model_file)
                stats['best_valid'] = result_
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
    logger.info('COMMAND: %s', ' '.join(sys.argv))

    # Run!
    main(args)

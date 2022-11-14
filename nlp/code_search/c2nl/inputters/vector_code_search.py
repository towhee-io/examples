import re
import torch


def pretokenize(tokens, select):
    tokens_ = []
    for token in tokens:
        # snake case pretokenize
        token = token.replace('_', ' ')
        # camel case pretokenize
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
        match = [m.group(0) for m in matches]
        if len(match) == 0:
            tokens_.append(token)
        else:
            if select == 0:
                tokens_.append(' '.join(match))
            else:
                tokens_.append(match[0])
    return tokens_


def remove_punctuation(tokens):
    tokens_ = []
    for token in tokens:
        token = token.replace('(', ' ')
        token = token.replace(')', ' ')
        token = token.replace('=', ' ')
        token = token.replace(',', ' ')
        token = token.replace(':', ' ')
        token = token.replace('.', ' ')
        token = token.strip()
        tokens_.append(token)
    return tokens_


def extract_token(tokens):
    text = ' '.join(tokens)
    tokens = re.findall(r'[a-zA-Z0-9_]+', text)
    return tokens


def remove_part_tokens(tokens, select, n=10):
    head = tokens[:5]
    body = tokens[5:]
    length = len(body)
    part_num = length // n
    if select == 0:
        # remove a 1/n part of body
        idx_low = torch.randint(0, n, (1,)).item()
        idx_high = idx_low + 1
        stay = body[:idx_low*part_num] + body[idx_high*part_num-1:]
        return head + stay
    else:
        # remove 1/n tokens of body
        remove_idx = torch.randint(0, length, (part_num,)).tolist()
        stay = [body[i] for i in range(length) if i not in remove_idx]
        return head + stay


# def data_augumentation(ex, count_ex, ex_num_epoch, max_src_len):
#     code = ex['code']
#     selects = torch.randint(0, 2, (3,)).tolist()
#     if selects[0] == 0:
#         tokens = extract_token(code.tokens)
#     else:
#         tokens = remove_punctuation(code.tokens)
#     tokens = pretokenize(tokens, selects[1])
#     if count_ex > ex_num_epoch:  # after one epoch, remove 1/20 --> 1/4
#         # 2 epoch a move
#         n = 20 - count_ex // ex_num_epoch // 2
#         n = 4 if n < 4 else n
#         tokens = remove_part_tokens(tokens, selects[2], n=n)
#     # truncate and pad
#     tokens = tokens[:max_src_len]
#     pad_length = max_src_len - len(tokens)
#     tokens += ['<blank>'] * pad_length

#     code.tokens = tokens
#     return ex


# def data_augumentation(ex, max_src_len):
#     code = ex['code']
#     tokens = remove_punctuation(code.tokens)
#     tokens = pretokenize(tokens, 1)
#     # truncate and pad
#     tokens = tokens[:max_src_len]
#     pad_length = max_src_len - len(tokens)
#     tokens += ['<blank>'] * pad_length
#     code.tokens = tokens
#     return ex


def vectorize(ex, model):
    """Vectorize a single example."""
    src_dict = model.src_dict

    code, summary = ex['code'], ex['summary']
    vectorized_ex = dict()
    vectorized_ex['id'] = code.id
    vectorized_ex['language'] = code.language

    vectorized_ex['code'] = code.text
    vectorized_ex['code_tokens'] = code.tokens
    vectorized_ex['code_char_rep'] = None
    vectorized_ex['code_type_rep'] = None
    vectorized_ex['code_mask_rep'] = None
    vectorized_ex['use_code_mask'] = False

    vectorized_ex['code_word_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict))
    if model.args.use_src_char:
        vectorized_ex['code_char_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict, _type='char'))
    if model.args.use_code_type:
        vectorized_ex['code_type_rep'] = torch.LongTensor(code.type)
    if code.mask:
        vectorized_ex['code_mask_rep'] = torch.LongTensor(code.mask)
        vectorized_ex['use_code_mask'] = True

    vectorized_ex['summ'] = None
    vectorized_ex['summ_tokens'] = None
    vectorized_ex['stype'] = None
    vectorized_ex['summ_word_rep'] = None
    vectorized_ex['summ_char_rep'] = None
    vectorized_ex['target'] = None

    if summary is not None:
        vectorized_ex['summ'] = summary.text
        vectorized_ex['summ_tokens'] = summary.tokens
        vectorized_ex['stype'] = summary.type
        vectorized_ex['summ_word_rep'] = torch.LongTensor(summary.vectorize(word_dict=src_dict))
        if model.args.use_tgt_char:
            vectorized_ex['summ_char_rep'] = torch.LongTensor(summary.vectorize(word_dict=src_dict, _type='char'))
        # target is only used to compute loss during training
        vectorized_ex['target'] = torch.LongTensor(summary.vectorize(src_dict))

    vectorized_ex['src_vocab'] = code.src_vocab
    vectorized_ex['use_src_word'] = model.args.use_src_word
    vectorized_ex['use_tgt_word'] = model.args.use_tgt_word
    vectorized_ex['use_src_char'] = model.args.use_src_char
    vectorized_ex['use_tgt_char'] = model.args.use_tgt_char
    vectorized_ex['use_code_type'] = model.args.use_code_type

    return vectorized_ex


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_src_word = batch[0]['use_src_word']
    use_tgt_word = batch[0]['use_tgt_word']
    use_src_char = batch[0]['use_src_char']
    use_tgt_char = batch[0]['use_tgt_char']
    use_code_type = batch[0]['use_code_type']
    use_code_mask = batch[0]['use_code_mask']
    ids = [ex['id'] for ex in batch]
    language = [ex['language'] for ex in batch]

    # --------- Prepare Code tensors ---------
    code_words = [ex['code_word_rep'] for ex in batch]
    code_chars = [ex['code_char_rep'] for ex in batch]
    code_type = [ex['code_type_rep'] for ex in batch]
    code_mask = [ex['code_mask_rep'] for ex in batch]
    max_code_len = max([d.size(0) for d in code_words])
    if use_src_char:
        max_char_in_code_token = code_chars[0].size(1)

    # Batch Code Representations
    code_len_rep = torch.zeros(batch_size, dtype=torch.long)
    code_word_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_src_word else None
    code_type_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_code_type else None
    code_mask_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
        if use_code_mask else None
    code_char_rep = torch.zeros(batch_size, max_code_len, max_char_in_code_token, dtype=torch.long) \
        if use_src_char else None

    for i in range(batch_size):
        code_len_rep[i] = code_words[i].size(0)
        if use_src_word:
            code_word_rep[i, :code_words[i].size(0)].copy_(code_words[i])
        if use_code_type:
            code_type_rep[i, :code_type[i].size(0)].copy_(code_type[i])
        if use_code_mask:
            code_mask_rep[i, :code_mask[i].size(0)].copy_(code_mask[i])
        if use_src_char:
            code_char_rep[i, :code_chars[i].size(0), :].copy_(code_chars[i])

    # --------- Prepare Summary tensors ---------
    no_summary = batch[0]['summ_word_rep'] is None
    if no_summary:
        summ_len_rep = None
        summ_word_rep = None
        summ_char_rep = None
        tgt_tensor = None
    else:
        summ_words = [ex['summ_word_rep'] for ex in batch]
        summ_chars = [ex['summ_char_rep'] for ex in batch]
        max_sum_len = max([q.size(0) for q in summ_words])
        if use_tgt_char:
            max_char_in_summ_token = summ_chars[0].size(1)

        summ_len_rep = torch.zeros(batch_size, dtype=torch.long)
        summ_word_rep = torch.zeros(batch_size, max_sum_len, dtype=torch.long) \
            if use_tgt_word else None
        summ_char_rep = torch.zeros(batch_size, max_sum_len, max_char_in_summ_token, dtype=torch.long) \
            if use_tgt_char else None

        max_tgt_length = max([ex['target'].size(0) for ex in batch])
        tgt_tensor = torch.zeros(batch_size, max_tgt_length, dtype=torch.long)
        for i in range(batch_size):
            summ_len_rep[i] = summ_words[i].size(0)
            if use_tgt_word:
                summ_word_rep[i, :summ_words[i].size(0)].copy_(summ_words[i])
            if use_tgt_char:
                summ_char_rep[i, :summ_chars[i].size(0), :].copy_(summ_chars[i])
            tgt_len = batch[i]['target'].size(0)
            tgt_tensor[i, :tgt_len].copy_(batch[i]['target'])

    return {
        'ids': ids,
        'language': language,
        'batch_size': batch_size,
        'code_word_rep': code_word_rep,
        'code_char_rep': code_char_rep,
        'code_type_rep': code_type_rep,
        'code_mask_rep': code_mask_rep,
        'code_len': code_len_rep,
        'summ_word_rep': summ_word_rep,
        'summ_char_rep': summ_char_rep,
        'summ_len': summ_len_rep,
        'tgt_seq': tgt_tensor,
        'code_text': [ex['code'] for ex in batch],
        'code_tokens': [ex['code_tokens'] for ex in batch],
        'summ_text': [ex['summ'] for ex in batch],
        'summ_tokens': [ex['summ_tokens'] for ex in batch],
        'stype': [ex['stype'] for ex in batch]
    }

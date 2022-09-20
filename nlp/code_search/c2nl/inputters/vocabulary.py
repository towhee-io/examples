# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py
import unicodedata
import numpy as np
from c2nl.inputters.constants import PAD, PAD_WORD, UNK, UNK_WORD, \
    BOS, BOS_WORD, EOS, EOS_WORD


class Vocabulary(object):
    '''
    vocabulary of token and index
    '''
    def __init__(self, no_special_token=False):
        if no_special_token:
            self.tok2ind = {PAD_WORD: PAD,
                            UNK_WORD: UNK}
            self.ind2tok = {PAD: PAD_WORD,
                            UNK: UNK_WORD}
        else:
            self.tok2ind = {PAD_WORD: PAD,
                            UNK_WORD: UNK,
                            BOS_WORD: BOS,
                            EOS_WORD: EOS}
            self.ind2tok = {PAD: PAD_WORD,
                            UNK: UNK_WORD,
                            BOS: BOS_WORD,
                            EOS: EOS_WORD}

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if isinstance(key, int):
            return key in self.ind2tok
        elif isinstance(key, str):
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.ind2tok.get(key, UNK_WORD)
        elif isinstance(key, str):
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(UNK_WORD))
        else:
            raise RuntimeError('Invalid key type.')

    def __setitem__(self, key, item):
        if isinstance(key, int) and isinstance(item, str):
            self.ind2tok[key] = item
        elif isinstance(key, str) and isinstance(item, int):
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def add_tokens(self, token_list):
        assert isinstance(token_list, list)
        for token in token_list:
            self.add(token)

    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()  # pylint: disable=consider-iterating-dictionary
                  if k not in {PAD_WORD, UNK_WORD}]
        return tokens

    def remove(self, key):
        if key in self.tok2ind:
            ind = self.tok2ind[key]
            del self.tok2ind[key]
            del self.ind2tok[ind]
            return True
        return False


class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.
    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.
    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.
    """

    def __init__(self, words, max_word_length,
                 no_special_token):
        super(UnicodeCharsVocabulary, self).__init__(no_special_token)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bow_char = 256  # <begin word>
        self.eow_char = 257  # <end word>
        self.pad_char = 258  # <padding>

        for w in words:
            self.add(w)
        num_words = len(self.ind2tok)

        self._word_char_ids = np.zeros([num_words, max_word_length],
                                       dtype=np.int32)

        for i, word in self.ind2tok.items():
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length - 2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[k + 1] = self.eow_char  # pylint: disable=undefined-loop-variable

        return code

    def word_to_char_ids(self, word):
        if word in self.tok2ind:
            return self._word_char_ids[self.tok2ind[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, split=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence]

        return chars_ids

from c2nl.inputters.vocabulary import Vocabulary, BOS_WORD, EOS_WORD


class Code(object):
    """
    Code containing annotated text, original text, selection label and
    all the extractive spans that can be an answer for the associated question.
    """

    def __init__(self, id_=None):
        self.id_ = id_
        self._language = None
        self._text = None
        self._tokens = []
        self.type_ = []
        self._mask = []
        self.src_vocab = None  # required for Copy Attention

    @property
    def id(self) -> str:
        return self.id_

    @property
    def language(self) -> str:
        return self._language

    @language.setter
    def language(self, param: str) -> None:
        self._language = param

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def type(self) -> list:
        return self.type_

    @type.setter
    def type(self, param: list) -> None:
        assert isinstance(param, list)
        self.type_ = param

    @property
    def mask(self) -> list:
        return self._mask

    @mask.setter
    def mask(self, param: list) -> None:
        assert isinstance(param, list)
        self._mask = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens = param
        self.form_src_vocab()

    def form_src_vocab(self) -> None:
        self.src_vocab = Vocabulary()
        assert self.src_vocab.remove(BOS_WORD)
        assert self.src_vocab.remove(EOS_WORD)
        self.src_vocab.add_tokens(self.tokens)

    def vectorize(self, word_dict, type_='word') -> list:
        if type_ == 'word':
            return [word_dict[w] for w in self.tokens]
        elif type_ == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False

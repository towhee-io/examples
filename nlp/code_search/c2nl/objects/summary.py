from c2nl.inputters.vocabulary import EOS_WORD, BOS_WORD


class Summary(object):
    """
    Summary containing annotated text, original text, a list of
    candidate documents, answers and well formed answers.
    """

    def __init__(self, id_=None):
        self.id_ = id_
        self._text = None
        self._tokens = []
        self.type_ = None  # summary, comment etc

    @property
    def id(self) -> str:
        return self.id_

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens = param

    def append_token(self, tok=EOS_WORD):
        assert isinstance(tok, str)
        self._tokens.append(tok)

    def prepend_token(self, tok=BOS_WORD):
        assert isinstance(tok, str)
        self._tokens.insert(0, tok)

    @property
    def type(self) -> str:
        return self.type_

    @type.setter
    def type(self, param: str) -> None:
        assert isinstance(param, str)
        self.type_ = param

    def vectorize(self, word_dict, type_='word') -> list:
        if type_ == 'word':
            return [word_dict[w] for w in self.tokens]
        elif type_ == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False

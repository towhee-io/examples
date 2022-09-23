#!/usr/bin/env python3
# Author : Saikat Chakraborty (saikatc@cs.columbia.edu)
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Basic tokenizer that splits text into alpha-numeric tokens and
non-whitespace tokens.
"""

import logging
from .tokenizer import Tokens, Tokenizer
import re

logger = logging.getLogger(__name__)


def tokenize_with_camel_case(token):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
    return [m.group(0) for m in matches]


def tokenize_with_snake_case(token):
    return token.split('_')


class CodeTokenizer(Tokenizer):
    '''
    CodeTokenizer
    '''
    def __init__(self, camel_case=True, snake_case=True, **kwargs):
        """
        Args:
            camel_case: Boolean denoting whether CamelCase split is desired
            snake_case: Boolean denoting whether snake_case split is desired
            annotators: None or empty set (only tokenizes).
        """
        super().__init__()
        self.snake_case = snake_case
        self.camel_case = camel_case
        assert self.snake_case or self.camel_case, \
            'To call CodeIdentifierTokenizer at least one of camel_case or ' \
            'snake_case flag has to be turned on in the initializer'
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s',
                           type(self).__name__, kwargs.get('annotators'))
        self.annotators = set()

    def tokenize(self, text):
        tokens = text.split()
        snake_case_tokenized = []
        if self.snake_case:
            for token in tokens:
                snake_case_tokenized.extend(tokenize_with_snake_case(token))
        else:
            snake_case_tokenized = tokens
        camel_case_tokenized = []
        if self.camel_case:
            for token in snake_case_tokenized:
                camel_case_tokenized.extend(tokenize_with_camel_case(token))
        else:
            camel_case_tokenized = snake_case_tokenized
        data = []
        for token in camel_case_tokenized:
            data.append((token, token, token))

        return Tokens(data, self.annotators)

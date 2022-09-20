PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

TOKEN_TYPE_MAP = {
    # Java
    '<pad>': 0,
    '<unk>': 1,
    'other': 2,
    'var': 3,
    'method': 4,
    # Python
    's': 5,
    'None': 6,
    'value': 7,
    'asname': 8,
    'n': 9,
    'level': 10,
    'is_async': 11,
    'arg': 12,
    'attr': 13,
    'id': 14,
    'name': 15,
    'module': 16
}

AST_TYPE_MAP = {
    '<pad>': 0,
    'N': 1,
    'T': 2
}

DATA_LANG_MAP = {
    'java': 'java',
    'python': 'python',
    'codeSearchNet-py': 'python',
    'code_summarization_public': 'python'
}

LANG_ID_MAP = {
    'java': 0,
    'python': 1,
    'c#': 2
}

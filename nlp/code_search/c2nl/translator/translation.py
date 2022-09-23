# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/translation.py
""" Translation main class """
from __future__ import division, unicode_literals
from __future__ import print_function

from c2nl.inputters import constants


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.
    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`
    Args:
       data (DataSet):
       tgt_vocab : Vocabulary
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
    """

    def __init__(self, tgt_vocab, n_best=1, replace_unk=False):
        self.tgt_vocab = tgt_vocab
        self.n_best = n_best
        self.replace_unk = replace_unk

    def _build_target_tokens(self, src_vocab, src_raw, pred, attn):
        tokens = []
        for tok in pred:
            tok = tok if isinstance(tok, int) \
                else tok.item()
            if tok == constants.BOS:
                continue
            if tok == constants.EOS:
                break

            if tok < len(self.tgt_vocab):
                tokens.append(self.tgt_vocab[tok])
            else:
                tokens.append(src_vocab[tok - len(self.tgt_vocab)])

        if self.replace_unk and (attn is not None):
            for i in range(len(tokens)):
                if tokens[i] == constants.UNK_WORD:
                    _, max_index = attn[i].max(0)
                    tokens[i] = src_raw[max_index.item()]
        return tokens

    def from_batch(self, translation_batch, src_raw, targets, src_vocabs):
        batch_size = len(translation_batch["predictions"])
        preds = translation_batch["predictions"]
        pred_score = translation_batch["scores"]
        attn = translation_batch["attention"]

        translations = []
        for b in range(batch_size):
            src_vocab = src_vocabs[b] if src_vocabs else None
            pred_sents = [self._build_target_tokens(
                src_vocab, src_raw[b],
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            translation = Translation(targets[b], pred_sents,
                                      attn[b], pred_score[b])
            translations.append(translation)

        return translations


class Translation(object):
    """
    Container for a translated sentence.
    Attributes:
        target ([str]): list of targets
        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
    """

    def __init__(self, targets, pred_sents, attn, pred_scores):
        self.targets = targets
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores

    def log(self, sent_number):
        """
        Log translation.
        """
        output = "\nTARGET {}: {}\n".format(sent_number, "\t".join(self.targets))

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = " ".join(best_pred)
        new_line = "\n"
        output += f"PRED {sent_number}: {pred_sent}{new_line}"
        output += f"PRED SCORE: {best_score:.4f}{new_line}"

        if len(self.pred_sents) > 1:
            output += "\nBEST HYP:\n"
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += f"[{score:.4f}] {sent}{new_line}"

        return output

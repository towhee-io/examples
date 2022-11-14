import nltk
from nltk.translate.bleu_score import SmoothingFunction


def nltk_sentence_bleu(hypothesis, reference, order=4):
    cc = SmoothingFunction()
    return nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method4)


def nltk_corpus_bleu(hypotheses, references, order=4):
    refs = []
    hyps = []
    count = 0
    total_score = 0.0

    cc = SmoothingFunction()

    assert (sorted(hypotheses.keys()) == sorted(references.keys()))
    Ids = list(hypotheses.keys())
    ind_score = dict()

    for id in Ids:
        hyp = hypotheses[id][0].split()
        ref = [r.split() for r in references[id]]
        hyps.append(hyp)
        refs.append(ref)

        score = nltk.translate.bleu(ref, hyp, smoothing_function=cc.method4)
        total_score += score
        count += 1
        ind_score[id] = score

    avg_score = total_score / count
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(refs, hyps)
    return corpus_bleu, avg_score, ind_score

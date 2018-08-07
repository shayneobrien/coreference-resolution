import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from boltons.iterutils import windowed
from itertools import groupby, combinations
from collections import defaultdict


def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def prune(spans, T, LAMBDA=0.40):
    """ Prune mention scores to the top lambda percent. Returns list of tuple(scores, indices, g_i) """
    STOP = int(LAMBDA * T) # lambda * document_length

    sorted_spans = sorted(spans, key=lambda s: s.si, reverse=True) # sort by mention score
    nonoverlapping = remove_overlapping(sorted_spans) # remove overlapping spans
    pruned_spans = nonoverlapping[:STOP] # prune to the top λT spans, sort by idx

    spans = sorted(pruned_spans, key=lambda s: (s.i1, s.i2))
    return spans

def remove_overlapping(sorted_spans):
    """ Remove spans that are overlapping by order of decreasing mention score """
    overlap = lambda s1, s2: s1.i1 < s2.i1 <= s1.i2 < s2.i2

    accepted = []
    for s1 in sorted_spans: # for every combo of spans with already accepted spans
        flag = True
        for s2 in accepted:
            if overlap(s1, s2) or overlap(s2, s1): # if i overlaps j or vice versa
                flag = False # let the function know not to accept this span
                break        # break this loop, since we will not accept span i

        if flag: # if span i does not overlap with any previous spans
            accepted.append(s1) # accept it

    return accepted

def pairwise_indexes(spans):
    """ Get indices for indexing into pairwise_scores """
    indexes = [0] + [len(s.yi) for s in spans]
    indexes = [sum(indexes[:idx+1]) for idx, _ in enumerate(indexes)]
    return indexes

def extract_gold_corefs(document):
    """ Parse coreference dictionary of a document to get coref links """
    # Initialize defaultdict for keeping track of corefs
    gold_links = defaultdict(list)

    # Compute number of mentions
    gold_mentions = set([coref['span'] for coref in document.corefs])
    total_mentions = len(gold_mentions)

    # Compute number of coreferences
    for coref_entry in document.corefs:

        # Parse label of coref span, the span itself
        label, span_idx = coref_entry['label'], coref_entry['span']

        # All spans corresponding to the same label
        gold_links[label].append(span_idx) # get all spans corresponding to some label

    # Flatten all possible corefs, sort, get number
    gold_corefs = flatten([[coref
                            for coref in combinations(gold, 2)]
                            for gold in gold_links.values()])
    gold_corefs = sorted(gold_corefs)
    total_corefs = len(gold_corefs)

    return gold_corefs, total_corefs, gold_mentions, total_mentions

def fix_coref_spans(doc):
    """ Add in token spans to corefs dict.
    Done post-hoc due to way text variable is updated """
    token_idxs = range(len(doc.tokens))

    for idx, coref in enumerate(doc.corefs):
        doc.corefs[idx]['word_span'] = tuple(doc.tokens[coref['start']:coref['end']+1])
        doc.corefs[idx]['span'] = tuple([coref['start'], coref['end']])
    return doc

def compute_idx_spans(tokens, L=10):
    """ Compute all possible token spans """
    return flatten([windowed(range(len(tokens)), length) for length in range(1, L)])

def s_to_speaker(span, speakers):
    """ Compute speaker of a span """
    i1, i2 = span[0], span[-1]
    if speakers[i1] == speakers[i2]:
        return speakers[i1]
    return None

def safe_divide(x, y):
    if y != 0:
        return x / y
    return 1

def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]

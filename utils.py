import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from loader import flatten
from boltons.iterutils import windowed
from itertools import groupby, permutations
from collections import defaultdict


def to_var(x):
    """ Convert a tensor to a backprop tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x.requires_grad_()

def prune(spans, LAMBDA=0.40):
    """ Prune mention scores to the top lambda percent """
    STOP = int(LAMBDA * len(spans[0].doc)) # lambda * document_length

    sorted_spans = sorted(spans, key=lambda s: s.si, reverse=True) # sort by mention score
    nonoverlapping = remove_overlapping(sorted_spans) # remove any overlapping spans
    pruned_spans = sorted(nonoverlapping[:STOP], key=lambda s: (s.i1, s.i2)) # prune to the top the top λT

    return pruned_spans

def remove_overlapping(sorted_spans):
    """ Remove spans that are overlapping by order of decreasing mention score """
    overlap = lambda s1, s2: s1.i1 < s2.i1 <= s1.i2 < s2.i2

    accepted = []
    for s1 in sorted_spans: # for every combination of spans with accepted spans
        flag = True
        for s2 in accepted:
            if overlap(s1, s2) or overlap(s2, s1): # i overlaps j or vice versa
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
    gold_links = defaultdict(list)

    for coref_entry in document.corefs:

        label, span_idx = coref_entry['label'], coref_entry['span'] # parse

        gold_links[label].append(span_idx) # get spans corresponding to some label

    gold_corefs = flatten([[gold for gold in permutations(gold, 2)]
                            for gold in gold_links.values()]) # all coref spans

    total_golds = len(gold_links) / 2 # (x, y), (y, x) both valid due to laziness

    return gold_corefs, total_golds

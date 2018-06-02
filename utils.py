import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from loader import flatten
from boltons.iterutils import windowed
from itertools import groupby, combinations
from collections import defaultdict


def to_var(x):
    """ Convert a tensor to a backprop tensor """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def s_to_speaker(span, speakers):
    i1, i2 = span[0], span[-1]
    if speakers[i1] == speakers[i2]:
        return speakers[i1]
    return None

def prune(spans, T, LAMBDA=0.40):
    """ Prune mention scores to the top lambda percent. Returns list of tuple(scores, indices, g_i) """
    STOP = int(LAMBDA * T) # lambda * document_length

    sorted_spans = sorted(spans, key=lambda s: s.si, reverse=True) # sort spans by mention score
    nonoverlapping = remove_overlapping(sorted_spans) # remove any overlapping spans
    pruned_spans = nonoverlapping[:STOP] # prune to just the top the top λT, sort by idx
    
    spans = sorted(pruned_spans, key=lambda s: (s.i1, s.i2))
    return spans

def remove_overlapping(sorted_spans):
    """ Remove spans that are overlapping by order of decreasing mention score """
    overlap = lambda s1, s2: s1.i1 < s2.i1 <= s1.i2 < s2.i2
    
    accepted = []
    for s1 in sorted_spans: # for every combination of spans with already accepted spans
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
    gold_links = defaultdict(list)
    
    for coref_entry in document.corefs:
        
        label, span_idx = coref_entry['label'], coref_entry['span'] # parse label of coref span, the span itself
        
        gold_links[label].append(span_idx) # get all spans corresponding to some label

    gold_corefs = flatten([[gold for gold in combinations(gold, 2)] for gold in gold_links.values()]) # all possible coref spans
    
    total_golds = len(gold_corefs) # the actual number of gold spans (we list (x, y) and (y, x) as both valid due to laziness)
       
    return gold_corefs, total_golds
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
    """ Prune mention scores to the top lambda percent. Returns list of tuple(scores, indices, g_i) """
    STOP = int(LAMBDA * len(spans[0].doc)) # lambda * document_length

    sorted_spans = sorted(spans, key=lambda span:span.si, reverse=True) # sort spans by mention score
    nonoverlapping = remove_overlapping(sorted_spans) # remove any overlapping spans
    pruned_spans = nonoverlapping[:STOP] # prune to just the top the top λT, sort by idx
    
    return pruned_spans

def remove_overlapping(sorted_spans):
    """ Remove spans that are overlapping by order of decreasing mention score """
    nonoverlapping, accepted = [], []
    for span_i in sorted_spans: # for every combination of spans with already accepted spans
        flag = True
        for span_j in accepted:
            if ((span_i.i1 < span_j.i1 <= span_i.i2 < span_j.i2) # if i overlaps j or vice versa
                or 
                (span_j.i1 < span_i.i1 <= span_j.i2 < span_i.i2)): 
                    flag = False # let the function know not to accept this span
                    break        # break this loop, since we will not accept span i

        if flag: # if span i does not overlap with any previous spans
            accepted.append(span_i) # accept it

        nonoverlapping.append(flag) # for span i's list idx, let True if it should be accepted and False otherwise

    nonoverlapping = [sorted_spans[idx] for idx, keep in enumerate(nonoverlapping) if keep] # prune
    return nonoverlapping

def pairwise_indexes(spans):
    """ Get indices for indexing into pairwise_scores """
    indexes = [0] + [len(s.yi) for s in spans]
    indexes = [sum(indexes[:idx+1]) for idx, _ in enumerate(indexes)]
    return indexes

def pair(spans):
    return windowed(spans, 2)

def extract_gold_corefs(document):
    """ Parse coreference dictionary of a document to get coref links """
    gold_links = defaultdict(list)
    
    for coref_entry in document.corefs:
        
        label, span_idx = coref_entry['label'], coref_entry['span'] # parse label of coref span, the span itself
        
        gold_links[label].append(span_idx) # get all spans corresponding to some label

    gold_corefs = flatten([[gold for gold in permutations(gold, 2)] for gold in gold_links.values()]) # all possible coref spans
    
    total_golds = len(gold_links) / 2 # the actual number of gold spans (we list (x, y) and (y, x) as both valid due to laziness)
       
    return gold_corefs, total_golds
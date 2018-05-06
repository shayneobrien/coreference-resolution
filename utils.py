import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from loader import flatten
from boltons import iterutils
from itertools import groupby, permutations, windowed
from collections import defaultdict

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def compute_idx_spans(size, L = 5):
    """ Compute all possible token spans """
    return flatten([iterutils.windowed(range(size), length) for length in range(1, L)])

def make_mention_reprs(lstm_out, embedded, idx_spans, model):
    """ Make span representations for unary mention scoring """
    reprs = []
    
    for span in idx_spans:
        
        start, end = span[0], span[-1] # get start, end id of span
        
        x_start, x_end, x_attn = lstm_out[start], lstm_out[end], model.attention(lstm_out[start:end+1], embedded[start:end+1]) # g_i
        
        reprs.append(torch.cat([x_start, x_end, x_attn], dim = 0)) # No additional features yet
    
    return torch.stack(reprs)

def prune_mentions(mention_scores, idx_spans, LAMBDA):
    """ Prune mention scores to the top lambda percent. Returns list of tuple(scores, indices, g_i) """
    STOP = int(LAMBDA * mention_scores.shape[0]) # lambda * document_length

    _, sorted_idx = torch.sort(mention_scores, descending=True) # sort scores by highest mention
    
    sorted_idx = sorted_idx.data.numpy() # don't need these ids in variable, so convert to numpy

    sorted_spans = [idx_spans[i] for i in sorted_idx] # get sorted spans

    overlaps = remove_overlapping(sorted_spans) # identify spans that overlap with a previously accepted span

    nonoverlapping = [sorted_idx[idx] for idx, val in enumerate(overlaps) if val] # remove overlapping spans

    pruned_idx = nonoverlapping[:STOP] # prune to just the top the top λT 

    return sorted(pruned_idx)

def remove_overlapping(sorted_spans):
    """ Remove spans that are overlapping by order of decreasing mention score """
    nonoverlapping, accepted_spans = [], []
    
    for span_i in sorted_spans:
        
        start_i, end_i, flag = span_i[0], span_i[-1], True # get start, end indices of span i
        
        for span_j in accepted_spans:
            
            start_j, end_j = span_j[0], span_j[-1] # get start, end indices of already accepted span j
            
            if (start_i < start_j <= end_i < end_j) or (start_j < start_i <= end_j < end_i): # i and j overlap
                flag = False # let the function know not to accept this span
                break # break this loop, since we will not accept span i
                
        if flag: # if span i does not overlap with any previous spans
            accepted_spans.append(span_i) # accept it
            
        nonoverlapping.append(flag) # for span i's list idx, let True if it should be accepted and False otherwise

    return nonoverlapping

def get_coref_pairs(mention_reprs, pruned_idx, idx_spans, K):
    """ Compute coreference pairings for pruned mentions. Returns tuple(i idx, j idx, span_i score, span_j score, span_ij representation) """
    
    pairwise_reprs = []
    for idx, i in enumerate(pruned_idx):

        g_i = mention_reprs[i] # span i representation g_i

        for j in pruned_idx[max(0, idx-K):idx]:

            g_j = mention_reprs[j] # span j representation g_j

            span_ij = torch.cat([g_i, g_j, g_i*g_j], dim = 0)   # coref between span i, span j representation g_ij

            pairwise_reprs += [span_ij] # append it to the other coref representations

    pairwise_reprs = torch.stack(pairwise_reprs).squeeze() 

    return pairwise_reprs

def extract_gold_corefs(document):
    """ Parse coreference dictionary of a document to get coref links """
    gold_corefs = defaultdict(list)
    
    for coref_entry in document.corefs:
        
        label, span_idx = coref_entry['label'], coref_entry['span'] # parse label of coref span, the span itself
        
        gold_corefs[label].append(span_idx) # get all spans corresponding to some label

    gold_links = flatten([[gold for gold in permutations(gold, 2)] for gold in gold_corefs.values()]) # all possible coref spans
    
    total_golds = len(gold_links) / 2 # the actual number of gold spans (we list (x, y) and (y, x) as both valid due to laziness)
       
    return gold_links, total_golds
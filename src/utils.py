import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

def unpack_and_unpad(lstm_out, reorder):
    """ Given a padded and packed sequence and its reordering indexes,
    unpack and unpad it. Inverse of pad_and_pack """

    # Restore a packed sequence to its padded version
    unpacked, sizes = pad_packed_sequence(lstm_out, batch_first=True)

    # Restored a packed sequence to its original, unequal sized tensors
    unpadded = [unpacked[idx][:val] for idx, val in enumerate(sizes)]

    # Restore original ordering
    regrouped = [unpadded[idx] for idx in reorder]

    return regrouped

def pad_and_pack(tensors):
    """ Pad and pack a list of sentences to allow for batching of
    variable length sequences in LSTM modules """

    # Pad variables to longest observed length, stack them
    padded, sizes = pad_and_stack(tensors)

    # Pack the variables to mask the padding
    return pack(padded, sizes)

def pad_and_stack(tensors, pad_size=None, value=0):
    """ Pad and stack an uneven tensor of token lookup ids.
    Assumes num_sents in first dimension (batch_first=True)"""

    # Get their original sizes (measured in number of tokens)
    sizes = [s.shape[0] for s in tensors]

    # Pad size will be the max of the sizes
    if not pad_size:
        pad_size = max(sizes)

    # Pad all sentences to the max observed size
    padded = torch.stack([F.pad(input=sent[:pad_size],
                                pad=(0, 0, 0, max(0, pad_size-size)),
                                value=value)
                          for sent, size in zip(tensors, sizes)], dim=0)

    return padded, sizes

def pack(padded, sizes, batch_first=True):
    """Pack padded variables, provide reorder indexes """

    # Get indexes for sorted sizes (largest to smallest)
    size_sort = np.argsort(sizes)[::-1].tolist()

    # Resort the tensor accordingly
    padded = padded[torch.tensor(size_sort, requires_grad=False)]

    # Resort sizes in descending order
    sizes = sorted(sizes, reverse=True)

    # Pack the padded sequences
    packed = pack_padded_sequence(padded, sizes, batch_first)

    # Regroup indexes for restoring tensor to its original order
    reorder = torch.tensor(np.argsort(size_sort), requires_grad=False)

    return packed, reorder

def prune(spans, T, LAMBDA=0.40):
    """ Prune mention scores to the top lambda percent.
    Returns list of tuple(scores, indices, g_i) """
    STOP = int(LAMBDA * T) # lambda * document_length

    sorted_spans = sorted(spans, key=lambda s: s.si, reverse=True) # sort by mention score
    nonoverlapping = remove_overlapping(sorted_spans) # remove overlapping spans
    pruned_spans = nonoverlapping[:STOP] # prune to the top λT spans, sort by idx

    spans = sorted(pruned_spans, key=lambda s: (s.i1, s.i2))
    return spans

def remove_overlapping(sorted_spans):
    """ Remove spans that are overlapping by order of decreasing mention score """
    nonoverlapping, seen = [], set()
    for s in sorted_spans:
        indexes = range(s.i1, s.i2+1)
        taken = [i in seen for i in indexes]
        if len(set(taken)) == 1 or (taken[0] == taken[-1] == False):
            nonoverlapping.append(s)
            seen.update(indexes)

    return nonoverlapping

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
    if speakers[span.i1] == speakers[span.i2]:
        return speakers[span.i1]
    return None

def speaker_label(s1, s2):
    # Same speaker
    if s1.speaker == s2.speaker:
        idx = torch.tensor(1)

    # Different speakers
    elif s1.speaker != s2.speaker:
        idx = torch.tensor(2)

    # No speaker
    else:
        idx = torch.tensor(0)

    return to_cuda(idx)

def safe_divide(x, y):
    if y != 0:
        return x / y
    return 1

def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]

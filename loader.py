import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vectors
from torch.autograd import Variable

import os, io, re, attr, random
from copy import deepcopy as c
from fnmatch import fnmatch
from boltons.iterutils import windowed
from cached_property import cached_property

NORMALIZE_DICT = {"/.": ".", "/?": "?", "-LRB-": "(", "-RRB-": ")", "-LCB-": "{", "-RCB-": "}", "-LSB-": "[", "-RSB-": "]"}
REMOVED_CHAR = ["/", "%", "*"]

class Corpus:
    def __init__(self, documents):
        self.documents = documents
        
    def __getitem__(self, idx):
        return self.documents[idx]   
    
    def __repr__(self):
        return 'Corpus containg %d documents' % len(self.documents)
    
    def get_vocab(self):
        """ Set vocabulary for LazyVectors """
        vocab = set()
        
        for document in self.documents:
            vocab.update(document.tokens)
            
        return vocab

    
class Document:
    def __init__(self, tokens, corefs, speakers, genre):
        self.tokens = tokens
        self.spans = compute_idx_spans(tokens)
        self.corefs = corefs
        self.speakers = speakers
        self.genre = genre
    
    def __getitem__(self, idx):
        return (self.text[idx], self.corefs[idx], self.speakers[idx], self.genre)
    
    def __repr__(self):
        return 'Document containing %d tokens' % len(self.tokens)
    
    def __len__(self):
        return len(self.tokens)
    
    def truncate(self, MAX=50):
        """ Randomly truncate the document to up to MAX sentences """
        sentences = [idx for idx, token in enumerate(self.tokens) if token in ['.', '?', '!']]
        if len(sentences) > MAX:
            i = random.sample(range(MAX, len(sentences)), 1)[0]
            tokens = self.tokens[sentences[i-50]:sentences[i]]
            return self.__class__(tokens, c(self.corefs), c(self.speakers), c(self.genre))
        return self

    
@attr.s(frozen=True, repr=False)
class Span:
    
    # Parent document reference.
    doc = attr.ib()

    # Left / right token indexes.
    i1 = attr.ib()
    i2 = attr.ib()

    # Span embedding tensor.
    g = attr.ib()

    # Unary mention score, as tensor.
    si = attr.ib(default=None)
    
    # List of candidate antecedent spans.
    yi = attr.ib(default=None)

    # Pairwise scores for each yi.
    sij = attr.ib(default=None)
    
    # Corresponding span ids to each yi
    yi_idx = attr.ib(default = None)
    
    def __repr__(self):
        return "'" + " ".join(self.doc.tokens[self.i1:self.i2+1]) + "'"

    @cached_property
    def tokens(self):
        return self.doc.tokens[self.i1:self.i2+1]


class LazyVectors:

    unk_idx = 1

    def __init__(self, name='glove.840B.300d.txt'):
        """ Load only those vectors from GloVE that are in the vocab. 
        
        Requires the glove vectors to be in a folder named .vector_cache 
        (linux: mv glove.840B.300d.text .vector_cache/glove.840B.300d.text)
        """
        self.name = name

    @cached_property
    def loader(self):
        return Vectors(self.name)

    def set_vocab(self, vocab):
        """Set corpus vocab.
        """
        # Intersect with model vocab.
        self.vocab = [v for v in vocab if v in self.loader.stoi]

        # Map string -> intersected index.
        self._stoi = {s: i for i, s in enumerate(self.vocab)}

    def weights(self):
        """Build weights tensor for embedding layer.
        """
        # Select vectors for vocab words.
        weights = torch.stack([
            self.loader.vectors[self.loader.stoi[s]]
            for s in self.vocab
        ])

        # Padding + UNK zeros rows.
        return torch.cat([
            torch.zeros((2, self.loader.dim)),
            weights,
        ])

    def stoi(self, s):
        """Map string -> embedding index.
        """
        idx = self._stoi.get(s)
        return idx + 2 if idx else self.unk_idx


def read_corpus(dirname):
    conll_files = parse_filenames(dirname = dirname, pattern = "*gold_conll")
    return Corpus(flatten([load_file(file) for file in conll_files]))
    
def load_file(filename):
    """ Load a *._conll file
    Input: 
        filename: path to the file
    Output:
        documents: list of Document class for each document in the file containing:
            tokens:                   split list of text
            utts_corefs:            
                coref['label']:     id of the coreference cluster
                coref['start']:     start index (index of first token in the utterance)
                coref['end':        end index (index of last token in the utterance)
                coref['span']:      corresponding span
            utts_speakers:          list of speakers
            genre:                  genre of input
    """
    documents = []
    with io.open(filename, 'rt', encoding='utf-8', errors='strict') as f:
        tokens, text, utts_corefs, utts_speakers, corefs, index = [], [], [], [], [], 0
        genre = filename.split('/')[6]
        for line in f:
            cols = line.split()
            
            # End of utterance within a document: update lists, reset variables for next utterance.
            if len(cols) == 0:
                if text:
                    tokens.extend(text), utts_corefs.extend(corefs), utts_speakers.extend([speaker]*len(text))
                    text, corefs = [], []
                    continue
                    
            # End of document: organize the data, append to output, reset variables for next document.
            elif len(cols) == 2: 
                doc = fix_coref_spans(Document(tokens, utts_corefs, utts_speakers, genre))
                documents.append(doc)
                tokens, text, utts_corefs, utts_speakers, index = [], [], [], [], 0
                
            # Inside an utterance: grab text, speaker, coreference information.
            elif len(cols) > 7:
                text.append(clean_token(cols[3]))
                speaker = cols[9]
                
                # If the last column isn't a '-', there is a coreference link
                if cols[-1] != u'-':
                    coref_expr = cols[-1].split(u'|')
                    for token in coref_expr:
                        
                        # Check if coref column token entry contains (, a number, or ).
                        match = re.match(r"^(\(?)(\d+)(\)?)$", token)
                        label = match.group(2)
                        
                        # If it does, extract the coref label, its start index, and end index.
                        if match.group(1) == u'(':
                            corefs.append({'label': label, 'start': index, 'end': None, 'span': None})
                        if match.group(3) == u')':
                            for i in range(len(corefs)-1, -1, -1):
                                if corefs[i]['label'] == label and corefs[i]['end'] is None:
                                    break
                            corefs[i]['end'] = index
                index += 1
            else:
                # Beginning of document, beginning of file, end of file: nothing to scrape off
                continue
                
    return documents
            
def parse_filenames(dirname, pattern = "*conll"):
    """ Walk a nested directory to get all filename ending in a pattern """
    filenames = []
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if fnmatch(name, pattern):
                yield os.path.join(path, name)
            
def clean_token(token):
    """ Substitute in /?(){}[] for equivalent CoNLL-2012 representations, remove /%* """
    cleaned_token = token
    if cleaned_token in NORMALIZE_DICT:
        cleaned_token = NORMALIZE_DICT[cleaned_token]
    if cleaned_token not in REMOVED_CHAR:
        for char in REMOVED_CHAR:
            cleaned_token = cleaned_token.replace(char, u'')
    if len(cleaned_token) == 0:
        cleaned_token = ","
    return cleaned_token 

def fix_coref_spans(document):
    """ Add in token spans to corefs dict. Done post-hoc due to way text variable is updated """
    token_idxs = range(len(document.tokens))
    for idx, coref in enumerate(document.corefs):
        document.corefs[idx]['word_span'] = tuple(document.tokens[coref['start']:coref['end']+1])
        document.corefs[idx]['span'] = tuple([coref['start'], coref['end']])
    return document  
                
def pair(spans):
    return windowed(spans, 2)

def compute_idx_spans(tokens, L = 10):
    """ Compute all possible token spans """
    return flatten([windowed(range(len(tokens)), length) for length in range(1, L)])

def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]
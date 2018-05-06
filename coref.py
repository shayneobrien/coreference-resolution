
# coding: utf-8

# In[ ]:


# TODO:
# Preprocessing steps
# Char cnn over UNKs
# Regularization
# Fix loss.backward()
# Tidy up gold permutations
# Other feature representations
# Loss goes to zero (?)
# Add in recall


# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import Vectors

import time
import numpy as np
from loader import *
from utils import *

from tqdm import tqdm, tqdm_notebook
from random import sample

def token_to_id(token):
    """ Lookup word ID for a token """
    return VECTORS.stoi(token)

def doc_to_tensor(document):
    """ Convert a sentence to a tensor """
    return to_var(torch.LongTensor([token_to_id(token) for token in document.tokens]))

# Load in corpus, lazily load in word vectors.
train_corpus = read_corpus('../data/train/')
VECTORS = LazyVectors()
VECTORS.set_vocab(train_corpus.get_vocab())


# In[3]:


class DocumentEncoder(nn.Module):
    def __init__(self, hidden_dim):
        """ Document encoder for tokens """
        super().__init__()
        
        weights = VECTORS.weights()
        
        self.embeddings = nn.Embedding(weights.shape[0], weights.shape[1])
        self.embeddings.weight.data.copy_(weights)
        
        self.lstm = nn.LSTM(weights.shape[1], hidden_dim, bidirectional = True, batch_first = True)
        self.emb_dropout, self.out_dropout = nn.Dropout(0.50), nn.Dropout(0.20)
        
    def forward(self, document):
        """ Convert document words to ids, embed them, pass through LSTM. 
        
        Returns hidden states at each time step, embedded tokens
        """
        tensor = doc_to_tensor(document)
        
        tensor = tensor.unsqueeze(0)
        
        embeds = self.embeddings(tensor)
        
        embeds = self.emb_dropout(embeds)
        
        lstm_out, _ = self.lstm(embeds)
        
        lstm_out = self.out_dropout(lstm_out)
    
        return lstm_out.squeeze(), embeds.squeeze()


class MentionScore(nn.Module):
    """ Mention scoring module """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        gi_dim = input_dim * 3
                    
        self.score = nn.Sequential(
            nn.Linear(gi_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.attention = SpanAttention(input_dim)
        
    def forward(self, lstm_out, embeds, document, LAMBDA = 0.40):
        """ Compute mention score for span s_i given its representation g_i
        
        Returns scalar score for whether span s_i is a mention
        """
        mention_reprs = []

        for span in document.spans:

            start, end = span[0], span[-1] # get start, end id of span

            x_start, x_end, x_attn = lstm_out[start], lstm_out[end], self.attention(lstm_out[start:end+1], embeds[start:end+1]) # g_i

            mention_reprs.append(torch.cat([x_start, x_end, x_attn], dim = 0)) # No additional features yet

        mention_reprs = torch.stack(mention_reprs)

        mention_scores = self.score(mention_reprs).squeeze()
        
        _, pruned_idx = prune(mention_scores, document, LAMBDA)
                
        return mention_scores, mention_reprs, pruned_idx


class SpanAttention(nn.Module):
    """ Attention module """
    def __init__(self, input_dim):
        super().__init__()
        
        self.activate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
        )        
        
    def forward(self, lstm_span, embeds_span):
        """ Compute attention for headedness as in Lee et al., 2017 
        
        Returns weighted sum of word vectors in a given span.
        """
        
        alpha = self.activate(lstm_span)
        
        weights = F.softmax(alpha, dim = 1)
        
        weighted_sum = torch.sum(weights * embeds_span, dim = 0)
                
        return weighted_sum 


class PairwiseScore(nn.Module):
    """ Coreference pair scoring module """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        gij_dim = input_dim * 9
        
        self.score = nn.Sequential(
            nn.Linear(gij_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, mention_reprs, pruned_idx, K = 50):
        """ Compute pairwise score for spans s_i and s_j given representation g_ij
        
        Returns scalar score for whether span s_i and s_j are coreferent mentions
        """        
        pairwise_reprs = []
        for idx, i in enumerate(pruned_idx):

            g_i = mention_reprs[i] # span i representation g_i

            for j in pruned_idx[max(0, idx-K):idx]:

                g_j = mention_reprs[j] # span j representation g_j

                span_ij = torch.cat([g_i, g_j, g_i*g_j], dim = 0) # coref between span i, span j representation g_ij

                pairwise_reprs += [span_ij] # append it to the other coref representations

        pairwise_reprs = torch.stack(pairwise_reprs)
    
        return self.score(pairwise_reprs).squeeze()


class CorefScore(nn.Module):
    """ Super class to compute coreference links between spans """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Initialize neural modules
        self.encode_doc = DocumentEncoder(hidden_dim)
        self.score_spans = MentionScore(input_dim, hidden_dim)
        self.score_pairs = PairwiseScore(input_dim, hidden_dim)
        
    def forward(self, document):
        """ Enocde document, predict and prune mentions, get pairwise mention scores, get coreference link score
        
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        lstm_out, embeds = self.encode_doc(document)

        # Get pruned mention scores
        mention_scores, mention_reprs, pruned_idx = self.score_spans(lstm_out, embeds, document)

        # Get pairwise scores
        pairwise_scores = self.score_pairs(mention_reprs, pruned_idx)
        
        return mention_scores, pairwise_scores, pruned_idx


# In[6]:


model = CorefScore(input_dim = 300, hidden_dim = 150)
K = 50
document = train_corpus[1]


# In[7]:


# Get gold corefs
gold_corefs, num_golds = extract_gold_corefs(document)

# Forward pass over the document
mention_scores, pairwise_scores, pruned_idx = model(document)

# Initialize ij for indexing pairwise_scores, losses for log-likelihood loss of each span
ij, losses = 0, []

# Get coreference scores:
for idx, i in tqdm_notebook(enumerate(pruned_idx)):

    antecedents = []

    # Get up to K antecedents for each span i
    for j in pruned_idx[max(0, idx - K):idx]:
        antecedents.append((i, j, ij))
        ij += 1

    # Recoop the tuples for span i, span j
    y_i = [(document.spans[j], document.spans[i]) for i, j, _ in antecedents]

    # Check which of these tuples are in the gold set, if any
    gold_pred_idx = [idx for idx, span in enumerate(y_i) if span in gold_corefs]
    if gold_pred_idx:
        print(idx, gold_pred_idx)

    # If gold_pred_idx is empty, all golds have been pruned or there are none; set gold to dummy
    if not gold_pred_idx:
        gold_pred_idx = [len(y_i)]

    # Compute the coreference mention score for span i and all of its antecedents
    coref_repr = [(mention_scores[i] + mention_scores[j] + pairwise_scores[ij]) 
                for _, j, ij in antecedents] 

    # Stack them to softmax, add in the dummy variable for possibility of no antecedent
    dummy = to_var(torch.tensor(0.))
    score_ij = torch.stack(coref_repr + [dummy])

    # Conditional probability distribution over all possible previous spans
    probs = F.softmax(score_ij, dim = 0)

    # Marginal log-likelihood of correct antecedents implied by gold clustering
    mass = torch.log(sum([probs[idx] for idx in gold_pred_idx]))

    # Save loss for this score_ij
    losses.append(mass)

# Negative marginal log-likelihood loss
loss = torch.mean(torch.stack(losses)) * -1


# In[ ]:


loss.backward()


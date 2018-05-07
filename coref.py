# TODO:
# Preprocessing steps
# Char cnn over UNKs
# Regularization
# Tidy up gold permutations
# Other feature representations
# Add in recall

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.vocab import Vectors

import time, random
import numpy as np

from functools import reduce
from loader import *
from utils import *

from tqdm import tqdm, tqdm_notebook
from random import sample

def token_to_id(token):
    """ Lookup word ID for a token """
    return VECTORS.stoi(token)

def doc_to_tensor(document):
    """ Convert a sentence to a tensor """
    return to_var(torch.tensor([token_to_id(token) for token in document.tokens]))

# # Load in corpus, lazily load in word vectors.
train_corpus = read_corpus('../data/train/')
VECTORS = LazyVectors()
VECTORS.set_vocab(train_corpus.get_vocab())

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
        tensor = doc_to_tensor(document) # Convert document tokens to look up ids
        tensor = tensor.unsqueeze(0) 
        
        embeds = self.embeddings(tensor) # Embed the tokens, dropout
        embeds = self.emb_dropout(embeds)
        
        states, _ = self.lstm(embeds) # Pass an LSTM over the embeds, dropout
        states = self.out_dropout(states)
    
        return states.squeeze(), embeds.squeeze()


class Score(nn.Module):
    """ Generic scoring module """
    def __init__(self, input_dim, hidden_dim = 150):
        super().__init__()
                            
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, x):
        return self.score(x)


class MentionScore(nn.Module):
    """ Mention scoring module """
    def __init__(self, attn_dim, gi_dim):
        super().__init__()
                
        self.attention = Score(attn_dim)
        self.score = Score(gi_dim)
        
    def forward(self, states, embeds, document, LAMBDA = 0.40):
        """ Compute mention score for span s_i given its representation g_i
        
        Returns scalar score for whether span s_i is a mention
        """
        attns = self.attention(states)
        
        spans = []

        for span in document.spans:
                
            i1, i2 = span[0], span[-1]
        
            span_embeds = embeds[i1:i2+1]
            span_states = states[i1:i2+1]
            span_attn = attns[i1:i2+1]

            attn = F.softmax(span_attn, dim = 0)
            attn = sum(attn * span_embeds)

            g_i = torch.cat([span_states[0], span_states[-1], attn])

            spans.append(Span(document, i1, i2, g_i))
            
        mention_reprs = torch.stack([s.g for s in spans])

        mention_scores = self.score(mention_reprs).squeeze()
        
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores)
        ]
                
        return spans


class PairwiseScore(Score):
    """ Coreference pair scoring module """
    # input_dim * 9
        
    def forward(self, spans, K = 50):
        """ Compute pairwise score for spans s_i and s_j given representation g_ij
        
        Returns scalar score for whether span s_i and s_j are coreferent mentions
        """        
        spans = [
            attr.evolve(span, yi=spans[max(0, idx-K):idx])
            for idx, span in enumerate(spans)
        ]

        pairs = torch.stack([
            torch.cat([i.g, j.g, i.g*j.g])
            for i in spans for j in i.yi
        ])

        pairwise_scores = self.score(pairs).squeeze()

        sa_idx = pairwise_indexes(spans)

        spans_ij = []
        for span, (i1, i2) in zip(spans, pair(sa_idx)):

            sij = [
                (span.si + yi.si + pair)
                for yi, pair in zip(span.yi, pairwise_scores[i1:i2])
            ]

            epsilon = to_var(torch.tensor(0.))
            sij = torch.stack(sij + [epsilon])

            spans_ij.append(attr.evolve(span, sij=sij))
            
        spans = [
            attr.evolve(span, yi_idx = [((span.i1, span.i2), (y.i1, y.i2)) for y in span.yi])
            for span in spans_ij
        ]
        
        return spans



class CorefScore(nn.Module):
    """ Super class to compute coreference links between spans """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Compute hidden state sizes for each module
        attn_dim = hidden_dim * 2 # Forward, backward of LSTM
        gi_dim = attn_dim * 3     # Forward, backward of LSTM for: gi_start, gi_end, gi_attn 
        gij_dim = gi_dim * 3      # g_i, g_j, g_i*g_j
        
        # Initialize neural modules
        self.encode_doc = DocumentEncoder(hidden_dim)
        self.score_spans = MentionScore(attn_dim, gi_dim)
        self.score_pairs = PairwiseScore(gij_dim)
        
    def forward(self, document):
        """ Enocde document, predict and prune mentions, get pairwise mention scores, get coreference link score
        
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encode_doc(document)

        # Get mention scores for each span
        spans = self.score_spans(states, embeds, document)
        
        # Prune the spans by decreasing mention score
        spans = prune(spans)
        
        # Get pairwise scores for each span combo
        spans = self.score_pairs(spans)
        
        return spans


class Trainer:
    
    def __init__(self, train_corpus, model, lr = 1e-3):
        self.train_corpus = list(train_corpus)
        self.model = model
        self.optimizer = optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr = lr)
        
        if torch.cuda.is_available():
            self.model.cuda()
            
            
    def train(self, num_epochs, *args, **kwargs):
        for epoch in range(num_epochs):
            self.train_epoch(epoch, *args, **kwargs)
            
    def train_epoch(self, epoch, steps = 100):
        
        self.model.train()
        
        docs = random.sample(self.train_corpus, steps)
        
        epoch_loss = []
        
        for document in tqdm_notebook(docs):
            
            loss = self.train_doc(document)
            
            epoch_loss.append(loss)
            
        print('Epoch: %d | Loss: %f' % (epoch, np.mean(epoch_loss)))
        
        
    def train_doc(self, document, CLIP = 5):
        
        # Extract gold coreference links
        gold_corefs, total_golds = extract_gold_corefs(document)
        
        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        losses = []
        for span in self.model(document):

            # Check which of these tuples are in the gold set, if any
            gold_idx = [
                idx for idx, link in enumerate(span.yi_idx)
                if link in gold_corefs
            ]

            # If gold_pred_idx is empty, all golds have been pruned or there are none; set gold to dummy
            if not gold_idx:
                gold_idx = [len(span.yi_idx)-1]

            # Conditional probability distribution over all possible previous spans
            probs = F.softmax(span.sij, dim = 0)

            # Marginal log-likelihood of correct antecedents implied by gold clustering
            mass = torch.log(sum([probs[i] for i in gold_idx]))

            # Save the loss for this span
            losses.append(mass)

        # Negative marginal log-likelihood for minimizing, backpropagate
        loss = torch.mean(torch.stack(losses)) * -1
        loss.backward()
        
        # Clip parameters
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        
        # Step the optimizer
        self.optimizer.step()
        
        return loss.item()


model = CorefScore(input_dim = 300, hidden_dim = 150)
trainer = Trainer(train_corpus, model)
trainer.train(10)


# coding: utf-8

# In[ ]:


# TODO:
# Comment everything
# Tidy up gold permutations
# Char cnn over UNKs (?)


# In[1]:


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


# In[144]:


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
        
        """
        # Convert document tokens to look up ids
        tensor = doc_to_tensor(document) 
        tensor = tensor.unsqueeze(0) 
        
        # Embed the tokens, regularize
        embeds = self.embeddings(tensor) 
        embeds = self.emb_dropout(embeds)
        
        # Pass an LSTM over the embeds, regularize
        states, _ = self.lstm(embeds)
        states = self.out_dropout(states)
    
        return states.squeeze(), embeds.squeeze()

    
class Feature(nn.Module):
    def __init__(self, bins, feature_dim=20):
        super().__init__()
        
        self.bins = bins
        self.dim = feature_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(bins), feature_dim),
            nn.Dropout(0.20),
        )
        
    def forward(self, *args):
        return self.embeds(self._idx(*args)).squeeze()
        
    def _idx(self, num):
        return to_var(torch.LongTensor(
            [idx for idx, bucket in enumerate(self.bins) if num in bucket]
        ))


class Score(nn.Module):
    """ Generic scoring module """
    def __init__(self, input_dim, hidden_dim=150):
        super().__init__()
                            
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, x):
        return self.score(x)


class MentionScore(nn.Module):
    """ Mention scoring module """
    def __init__(self, attn_dim, gi_dim):
        super().__init__()
                
        self.attention = Score(attn_dim)
        self.width = Feature(bins)
        self.score = Score(gi_dim+self.width.dim)
        
    def forward(self, states, embeds, document, LAMBDA=0.40):
        """ Compute unary mention score for each span
        
        """
        # Compute first part of attention over span states
        attns = self.attention(states) 
        
        spans = []

        for span in document.spans:
                
            i1, i2 = span[0], span[-1]
        
            span_embeds = embeds[i1:i2+1] 
            span_states = states[i1:i2+1]
            span_attn = attns[i1:i2+1]

            attn = F.softmax(span_attn, dim = 0)
            attn = sum(attn * span_embeds)
            
            size = self.width(len(span))

            g_i = torch.cat([span_states[0], span_states[-1], attn, size])

            spans.append(Span(document, i1, i2, g_i))
            
        mention_reprs = torch.stack([s.g for s in spans])

        mention_scores = self.score(mention_reprs).squeeze()
        
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores)
        ]
                
        return spans


class PairwiseScore(nn.Module):
    """ Coreference pair scoring module """
    def __init__(self, input_dim):
        super().__init__()
        
        self.score = Score(input_dim)
        self.distance = Feature(bins)
        
    def forward(self, spans, K=250):
        """ Compute pairwise score for spans and their up to K antecedents

        """        
        spans = [
            attr.evolve(span, yi=spans[max(0, idx-K):idx])
            for idx, span in enumerate(spans)
        ]

        pairs = torch.stack([
            torch.cat([i.g, j.g, i.g*j.g, self.distance(i.i2-j.i1)])
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
    def __init__(self, input_dim, hidden_dim, feature_dim=20):
        super().__init__()
        
        # Compute hidden state sizes for each module
        attn_dim = hidden_dim * 2 # Forward, backward of LSTM
        gi_dim = attn_dim * 3     # Forward, backward of LSTM for: gi_start, gi_end, gi_attn 
        gij_dim = (gi_dim+feature_dim) * 3      # g_i, g_j, g_i*g_j
        
        # Initialize neural modules
        self.encode_doc = DocumentEncoder(hidden_dim)
        self.score_spans = MentionScore(attn_dim, gi_dim)
        self.score_pairs = PairwiseScore(gij_dim)
        
    def forward(self, document):
        """ Enocde document
            Predict unary mention scores, prune them 
            Predict pairwise coreference scores
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


# In[142]:


class Trainer:
    """ Class dedicated to training the model """
    def __init__(self, train_corpus, model, lr = 1e-3):
        self.train_corpus = list(train_corpus)
        self.model = model
        self.optimizer = optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.001)
        
        if torch.cuda.is_available():
            self.model.cuda()
            
    def train(self, num_epochs, *args, **kwargs):
        """ Train a model """
        for epoch in range(num_epochs):
            self.train_epoch(epoch, *args, **kwargs)
            
    def train_epoch(self, epoch, steps = 25):
        """ Run a training epoch over 'steps' documents """
        self.model.train()
        
        docs = random.sample(self.train_corpus, steps)
        
        epoch_loss, epoch_recall = [], []
        
        for doc in tqdm_notebook(docs):
            
            document = doc.truncate() # Randomly truncate document to up to 50 sentences
            
            loss, recall, total_golds = self.train_doc(document)
            
            print(document, 'Loss: %f | Recall: %f | Total Golds: %d' % (loss, recall, total_golds))
            
            epoch_loss.append(loss), epoch_recall.append(recall)
            
            self.scheduler.step()
            
        print('Epoch: %d | Loss: %f | Recall: %f' % (epoch, np.mean(epoch_loss), np.mean(epoch_recall)))
        
    def train_doc(self, document, CLIP = 5):
        """ Compute loss for a forward pass over a document """
        # Extract gold coreference links
        gold_corefs, total_golds = extract_gold_corefs(document)
        
        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        losses, golds_found = [], []
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
        nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        
        # Step the optimizer
        self.optimizer.step()
        
        # Compute recall
        recall = sum(golds_found)
        
        return loss.item(), recall, total_golds
    
    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')
    
    def load_model(self, loadpath,  model = None):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        return model


# In[152]:


model = CorefScore(input_dim = 300, hidden_dim = 150)
trainer = Trainer(train_corpus, model)
trainer.train(100)


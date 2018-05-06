# TODO:
# Preprocessing steps
# Char cnn over UNKs
# Regularization
# Fix loss.backward()
# Tidy up gold permutations
# Other feature representations
# Loss goes to zero (?)

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

class DocumentEncoder(nn.Module):
    def __init__(self, hidden_dim):
        """ Document encoder for tokens """
        super().__init__()
        
        weights = VECTORS.weights()
        
        self.embeddings = nn.Embedding(weights.shape[0], weights.shape[1])
        self.embeddings.weight.data.copy_(weights)
        
        self.lstm = nn.LSTM(weights.shape[1], hidden_dim, bidirectional = True, batch_first = True, dropout = 0.20)
        self.dropout = nn.Dropout(p = 0.50)
        
    def forward(self, document):
        """ Convert document words to ids, embed them, pass through LSTM. 
        
        Returns hidden states at each time step, embedded tokens
        """
        tensor = doc_to_tensor(document)
        
        tensor = tensor.unsqueeze(0)
        
        embedded = self.embeddings(tensor)
        
        embedded = self.dropout(embedded)
        
        lstm_out, _ = self.lstm(embedded)
                
        return lstm_out.squeeze(), embedded.squeeze()


class SpanAttention(nn.Module):
    """ Attention module """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.activate = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, span_hdn, span_embeds):
        """ Compute attention for headedness as in Lee et al., 2017 
        
        Returns weighted sum of word vectors in a given span.
        """
        activated = F.relu(self.activate(span_hdn))
        
        alpha = self.attn(activated)
        
        weights = F.softmax(alpha, dim = 1)
        
        weighted_sum = torch.sum(weights * span_embeds, dim = 0)
        
        return weighted_sum


class MentionScore(nn.Module):
    """ Mention scoring module """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.activate = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        
    def forward(self, span_reprs):
        """ Compute mention score for span s_i given its representation g_i
        
        Returns scalar score for whether span s_i is a mention
        """
        activated = F.relu(self.activate(span_reprs))
        
        return self.score(activated).squeeze()


class PairwiseScore(nn.Module):
    """ Coreference link scoring module """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.activate = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        
    def forward(self, pairwise_reprs):
        """ Compute pairwise score for spans s_i and s_j given representation g_ij
        
        Returns scalar score for whether span s_i and s_j are coreferent mentions
        """
        activated = F.relu(self.activate(pairwise_reprs))
        
        return self.score(activated).squeeze()


class CorefScore(nn.Module):
    """ Super class to compute coreference links between spans """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Compute input dimensions for subsequent operation modules
        attn_dim = hidden_dim * 2
        mention_dim = attn_dim * 2 + input_dim
        pair_dim = mention_dim * 3
        
        # Initialize neural modules
        self.docencoder = DocumentEncoder(hidden_dim)
        self.attention = SpanAttention(attn_dim, hidden_dim)
        self.mention_scorer = MentionScore(mention_dim, hidden_dim)
        self.pairwise_scorer = PairwiseScore(pair_dim, hidden_dim)
        
        # Initialize epsilon / dummy variable
        self.dummy = to_var(torch.FloatTensor([0]))
        
    def forward(self, document, LAMBDA = 0.40, K = 50):
        """ Enocde document, predict and prune mentions, get pairwise mention scores, get coreference link score
        
        Returns a tuple of number of correct gold mentions found, log_prob_sum loss
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        idx_spans = compute_idx_spans(len(document))
        lstm_out, embedded = model.docencoder(document)

        # Get span representations (g_i), prepare input, compute mention scores (sm_i)
        mention_reprs = make_mention_reprs(lstm_out, embedded, idx_spans, model)
        mention_scores = model.mention_scorer(mention_reprs)

        # Prune the mention scores down to the top λT after removing overlapping spans
        pruned_idx = prune_mentions(mention_scores, idx_spans, LAMBDA)

        # Get mention pairs, prepare input, compute pairwise scores (sa_ij)
        pairwise_reprs = get_coref_pairs(mention_reprs, pruned_idx, idx_spans, K)
        pairwise_scores = model.pairwise_scorer(pairwise_reprs)

        # Get coreference scores
        ij = 0
        for idx, i in enumerate(pruned_idx):

            antecedents = [] 

            for j in pruned_idx[max(0, idx - K):idx]: # collect all possible (up to K previous) antecedents for span i
                antecedents.append((i, j, ij))
                ij += 1

            coref_repr = [(mention_scores[i] + mention_scores[j] + pairwise_scores[ij]) # compute score_ij
                        for _, j, ij in antecedents] 

            score_ij = torch.stack(coref_repr + [model.dummy]) # throw in dummy variable
            
            y_i = [(idx_spans[j], idx_spans[i]) for i, j, _ in antecedents] # coreferent spans that correspond to each score_ij
    
            yield y_i, score_ij


class Trainer:
    """ Class to train model """
    def __init__(self, train_corpus, model):
        
        self.train_corpus = list(train_corpus)
        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model
        self.optimizer = torch.optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad])
        
    def train_epoch(self, epoch, epoch_steps = 10, CLIP = 5):
        
        self.model.train()
        epoch_loss = []
        documents = sample(self.train_corpus, epoch_steps)

        for doc in tqdm_notebook(documents):
            start = time.time()
            
            # Zero out gradients
            self.optimizer.zero_grad()
            
            # Initialize loss
            losses = []
            
            # Skim the document down to reduce computational overhead
            skimmed_doc = self.token_skim_doc(doc)
            
            # Extract gold coreferences, total number of golds (pre-skim) from the document
            gold_corefs, total_golds = extract_gold_corefs(skimmed_doc)
                        
            for score_ij, pairwise_spans in self.model(skimmed_doc):
                
                # Get the gold coreference span ids 
                gold_pred_idx = [pairwise_spans[gold] for gold in gold_corefs if gold in pairwise_spans]

                # If it's empty, all have been pruned or there are none. Set the gold to the dummy variable
                if not gold_pred_idx:
                    gold_pred_idx = [len(score_ij)-1]
                
                # Conditional probability distribution over all possible previous spans
                probs = F.softmax(score_ij, dim = 0)
                
                # Marginal log-likelihood of correct antecedents implied by gold clustering
                mass = torch.log(sum([probs[idx] for idx in gold_pred_idx]))
                
                # Save loss for this score_ij
                losses.append(mass)
                
            # Aggregate loss
            loss = torch.mean(torch.stack(losses)) * -1
                      
            # Debugging
            end = time.time()
            print('Forward: {0} | Length {1}'.format(end-start, len(skimmed_doc)))
        
            start = time.time()
            loss.backward()
            end = time.time()
            print('Backward: {0}'.format(end-start))
            
            # Save the loss
            epoch_loss.append(loss.data[0])

            # Clip weights
            nn.utils.clip_grad_norm(model.parameters(), CLIP)

            # Step optimizer
            self.optimizer.step()

        print('Epoch: %d | Loss: %.4f | Recall: N/A' % (epoch, np.mean(epoch_loss)))
        
    def train(self, num_epochs = 10, *args, **kwargs):
        """ Train the model for num_epochs """
        for epoch in range(num_epochs):
            self.train_epoch(epoch, *args, **kwargs)
                
    def sent_skim_doc(self, doc, threshold = 50):
        """ Skim off the first 'threshold' sentences from the document """
        sent_count = 0
        for idx, token in enumerate(doc.tokens):
            if token in ['.', '?', '!']:
                sent_count += 1
                if sent_count > threshold:
                    doc.tokens = doc.tokens[:idx]
                    break
        
        return doc
    
    def token_skim_doc(self, doc, threshold = 500):
        """ Skim off the first 'threshold' tokens from the document """
        if len(doc) > threshold:
            doc.tokens = doc.tokens[:threshold]
        return doc

model = CorefScore(input_dim = 300, hidden_dim = 150)
trainer = Trainer(train_corpus, model)
trainer.train(num_epochs = 100, epoch_steps = 1)

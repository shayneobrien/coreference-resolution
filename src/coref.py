# TODO:
# Early stopping

print('Initializing...')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.vocab import Vectors

import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from random import sample
from datetime import datetime
from subprocess import Popen, PIPE
from boltons.iterutils import pairwise
from loader import *
from utils import *


class Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim, hidden_dim=150):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)


class Distance(nn.Module):
    """ Learned, continuous representations for: span widths, distance
    between spans
    """

    bins = [1,2,3,4,8,16,32,64]

    def __init__(self, distance_dim=20):
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.20)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths):
        """ Find which bin a number falls into """
        return to_cuda(torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False
        ))


class Genre(nn.Module):
    """ Learned continuous representations for genre. Zeros if genre unknown.
    """

    genres = ['bc', 'bn', 'mz', 'nw', 'pt', 'tc', 'wb']
    _stoi = {genre: idx+1 for idx, genre in enumerate(genres)}

    def __init__(self, genre_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(len(self.genres)+1, genre_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, labels):
        """ Embedding table lookup """
        return self.embeds(self.stoi(labels))

    def stoi(self, labels):
        """ Locate embedding id for genre """
        indexes = [self._stoi.get(gen) for gen in labels]
        return to_cuda(torch.tensor([i if i is not None else 0 for i in indexes]))


class Speaker(nn.Module):
    """ Learned continuous representations for binary speaker. Zeros if speaker unknown.
    """

    def __init__(self, speaker_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(3, speaker_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, speaker_labels):
        """ Embedding table lookup (see src.utils.speaker_label fnc) """
        return self.embeds(to_cuda(torch.tensor(speaker_labels)))


class CharCNN(nn.Module):
    """ Character-level CNN. Contains character embeddings.
    """

    unk_idx = 1
    vocab = train_corpus.char_vocab
    _stoi = {char: idx+2 for idx, char in enumerate(vocab)}
    pad_size = 15

    def __init__(self, filters, char_dim=8):
        super().__init__()

        self.embeddings = nn.Embedding(len(self.vocab)+2, char_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.pad_size,
                                              out_channels=filters,
                                              kernel_size=n) for n in (3,4,5)])

    def forward(self, sent):
        """ Compute filter-dimensional character-level features for each doc token """
        embedded = self.embeddings(self.sent_to_tensor(sent))
        convolved = torch.cat([F.relu(conv(embedded)) for conv in self.convs], dim=2)
        pooled = F.max_pool1d(convolved, convolved.shape[2]).squeeze(2)
        return pooled

    def sent_to_tensor(self, sent):
        """ Batch-ify a document class instance for CharCNN embeddings """
        tokens = [self.token_to_idx(t) for t in sent]
        batch = self.char_pad_and_stack(tokens)
        return batch

    def token_to_idx(self, token):
        """ Convert a token to its character lookup ids """
        return to_cuda(torch.tensor([self.stoi(c) for c in token]))

    def char_pad_and_stack(self, tokens):
        """ Pad and stack an uneven tensor of token lookup ids """
        skimmed = [t[:self.pad_size] for t in tokens]

        lens = [len(t) for t in skimmed]

        padded = [F.pad(t, (0, self.pad_size-length))
                  for t, length in zip(skimmed, lens)]

        return torch.stack(padded)

    def stoi(self, char):
        """ Lookup char id. <PAD> is 0, <UNK> is 1. """
        idx = self._stoi.get(char)
        return idx if idx else self.unk_idx


class DocumentEncoder(nn.Module):
    """ Document encoder for tokens
    """
    def __init__(self, hidden_dim, char_filters, n_layers=2):
        super().__init__()

        # Unit vector embeddings as per Section 7.1 of paper
        glove_weights = F.normalize(GLOVE.weights())
        turian_weights = F.normalize(TURIAN.weights())

        # GLoVE
        self.glove = nn.Embedding(glove_weights.shape[0], glove_weights.shape[1])
        self.glove.weight.data.copy_(glove_weights)
        self.glove.weight.requires_grad = False

        # Turian
        self.turian = nn.Embedding(turian_weights.shape[0], turian_weights.shape[1])
        self.turian.weight.data.copy_(turian_weights)
        self.turian.weight.requires_grad = False

        # Character
        self.char_embeddings = CharCNN(char_filters)

        # Sentence-LSTM
        self.lstm = nn.LSTM(glove_weights.shape[1]+turian_weights.shape[1]+char_filters,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout
        self.emb_dropout = nn.Dropout(0.50, inplace=True)
        self.lstm_dropout = nn.Dropout(0.20, inplace=True)

    def forward(self, doc):
        """ Convert document words to ids, embed them, pass through LSTM. """

        # Embed document
        embeds = [self.embed(s) for s in doc.sents]

        # Batch for LSTM
        packed, reorder = pack(embeds)

        # Apply embedding dropout
        self.emb_dropout(packed[0])

        # Pass an LSTM over the embeds
        output, _ = self.lstm(packed)

        # Apply dropout
        self.lstm_dropout(output[0])

        # Undo the packing/padding required for batching
        states = unpack_and_unpad(output, reorder)

        return torch.cat(states, dim=0), torch.cat(embeds, dim=0)

    def embed(self, sent):
        """ Embed a sentence using GLoVE, Turian, and character embeddings """

        # Embed the tokens with Glove
        glove_embeds = self.glove(lookup_tensor(sent, GLOVE))

        # Embed again using Turian this time
        tur_embeds = self.turian(lookup_tensor(sent, TURIAN))

        # Character embeddings
        char_embeds = self.char_embeddings(sent)

        # Concatenate them all together
        embeds = torch.cat((glove_embeds, tur_embeds, char_embeds), dim=1)

        return embeds


class MentionScore(nn.Module):
    """ Mention scoring module
    """
    def __init__(self, gi_dim, attn_dim, distance_dim):
        super().__init__()

        self.attention = Score(attn_dim)
        self.width = Distance(distance_dim)
        self.score = Score(gi_dim)

    def forward(self, states, embeds, doc, K=250):
        """ Compute unary mention score for each span
        """

        # Initialize Span objects containing start index, end index, genre, speaker
        spans = [Span(i1=i[0], i2=i[-1], id=idx,
                      speaker=doc.speaker(i), genre=doc.genre)
                 for idx, i in enumerate(compute_idx_spans(doc.sents))]

        # Compute first part of attention over span states (alpha_t)
        attns = self.attention(states)

        # Regroup attn values, embeds into span representations
        # TODO: figure out a way to batch
        span_attns, span_embeds = zip(*[(attns[s.i1:s.i2+1], embeds[s.i1:s.i2+1])
                                        for s in spans])

        # Pad and stack span attention values, span embeddings for batching
        padded_attns, _ = pad_and_stack(span_attns, value=-1e10)
        padded_embeds, _ = pad_and_stack(span_embeds)

        # Weight attention values using softmax
        attn_weights = F.softmax(padded_attns, dim=1)

        # Compute self-attention over embeddings (x_hat)
        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        # Compute span widths (i.e. lengths), embed them
        widths = self.width([len(s) for s in spans])

        # Get LSTM state for start, end indexes
        # TODO: figure out a way to batch
        start_end = torch.stack([torch.cat((states[s.i1], states[s.i2]))
                                 for s in spans])

        # Cat it all together to get g_i, our span representation
        g_i = torch.cat((start_end, attn_embeds, widths), dim=1)

        # Compute each span's unary mention score
        mention_scores = self.score(g_i)

        # Update span object attributes
        # (use detach so we don't get crazy gradients by splitting the tensors)
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores.detach())
        ]

        # Prune down to LAMBDA*len(doc) spans
        spans = prune(spans, len(doc))

        # Update antencedent set (yi) for each mention up to K previous antecedents
        spans = [
            attr.evolve(span, yi=spans[max(0, idx-K):idx])
            for idx, span in enumerate(spans)
        ]

        return spans, g_i, mention_scores


class PairwiseScore(nn.Module):
    """ Coreference pair scoring module
    """
    def __init__(self, gij_dim, distance_dim, genre_dim, speaker_dim):
        super().__init__()

        self.distance = Distance(distance_dim)
        self.genre = Genre(genre_dim)
        self.speaker = Speaker(speaker_dim)

        self.score = Score(gij_dim)

    def forward(self, spans, g_i, mention_scores):
        """ Compute pairwise score for spans and their up to K antecedents
        """

        # Extract raw features
        mention_ids, antecedent_ids, \
            distances, genres, speakers = zip(*[(i.id, j.id,
                                                i.i2-j.i1, i.genre,
                                                speaker_label(i, j))
                                             for i in spans
                                             for j in i.yi])

        # For indexing a tensor efficiently
        mention_ids = to_cuda(torch.tensor(mention_ids))
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids))

        # Embed them
        phi = torch.cat((self.distance(distances),
                         self.genre(genres),
                         self.speaker(speakers)), dim=1)

        # Extract their span representations from the g_i matrix
        i_g = torch.index_select(g_i, 0, mention_ids)
        j_g = torch.index_select(g_i, 0, antecedent_ids)

        # Create s_ij representations
        pairs = torch.cat((i_g, j_g, i_g*j_g, phi), dim=1)

        # Extract mention score for each mention and its antecedents
        s_i = torch.index_select(mention_scores, 0, mention_ids)
        s_j = torch.index_select(mention_scores, 0, antecedent_ids)

        # Score pairs of spans for coreference link
        s_ij = self.score(pairs)

        # Compute pairwise scores for coreference links between each mention and
        # its antecedents
        coref_scores = torch.sum(torch.cat((s_i, s_j, s_ij), dim=1), dim=1, keepdim=True)

        # Update spans with set of possible antecedents' indices, scores
        spans = [
            attr.evolve(span,
                        yi_idx=[((y.i1, y.i2), (span.i1, span.i2)) for y in span.yi]
                        )
            for span, score, (i1, i2) in zip(spans, coref_scores, pairwise_indexes(spans))
        ]

        # Get antecedent indexes for each span
        antecedent_idx = [len(s.yi) for s in spans if len(s.yi)]

        # Split coref scores so each list entry are scores for its antecedents, only.
        # (NOTE that first index is a special case for torch.split, so we handle it here)
        split_scores = [to_cuda(torch.tensor([]))] \
                         + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_var(torch.tensor([[0.]]))
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        # Batch and softmax
        # get the softmax of the scores for each span in the document given
        probs = [F.softmax(tensr) for tensr in with_epsilon]
        
        # pad the scores for each one with a dummy value, 1000 so that the tensors can 
        # be of the same dimension for calculation loss and what not. 
        probs, _ = pad_and_stack(probs, value=1000)
        probs = probs.squeeze()
       
        return spans, probs


class CorefScore(nn.Module):
    """ Super class to compute coreference links between spans
    """
    def __init__(self, embeds_dim,
                       hidden_dim,
                       char_filters=50,
                       distance_dim=20,
                       genre_dim=20,
                       speaker_dim=20):

        super().__init__()

        # Forward and backward pass over the document
        attn_dim = hidden_dim*2

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = attn_dim*2 + embeds_dim + distance_dim

        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim*3 + distance_dim + genre_dim + speaker_dim

        # Initialize modules
        self.encoder = DocumentEncoder(hidden_dim, char_filters)
        self.score_spans = MentionScore(gi_dim, attn_dim, distance_dim)
        self.score_pairs = PairwiseScore(gij_dim, distance_dim, genre_dim, speaker_dim)

    def forward(self, doc):
        """ Enocde document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        states, embeds = self.encoder(doc)

        # Get mention scores for each span, prune
        spans, g_i, mention_scores = self.score_spans(states, embeds, doc)

        # Get pairwise scores for each span combo
        spans, coref_scores = self.score_pairs(spans, g_i, mention_scores)

        return spans, coref_scores


class Trainer:
    """ Class dedicated to training and evaluating the model
    """
    def __init__(self, model, train_corpus, val_corpus, test_corpus,
                    lr=1e-3, steps=100):

        self.__dict__.update(locals())

        self.train_corpus = list(self.train_corpus)
        self.val_corpus = self.val_corpus
        
        self.model = to_cuda(model)

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                            if p.requires_grad],
                                    lr=lr)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=100,
                                                   gamma=0.001)

    def train(self, num_epochs, eval_interval=10, *args, **kwargs):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch, *args, **kwargs)

            # Save often
            self.save_model(str(datetime.now()))

            # Evaluate every eval_interval epochs
            if epoch % eval_interval == 0:
                print('\n\nEVALUATION\n\n')
                self.model.eval()
                results = self.evaluate(self.val_corpus)
                print(results)

    def train_epoch(self, epoch):
        """ Run a training epoch over 'steps' documents """

        # Set model to train (enables dropout)
        self.model.train()

        # Randomly sample documents from the train corpus
        batch = random.sample(self.train_corpus, self.steps)

        epoch_loss, epoch_mentions, epoch_corefs, epoch_identified = [], [], [], []

        for document in tqdm(batch):

            # Randomly truncate document to up to 50 sentences
            doc = document.truncate()

            # Compute loss, number gold links found, total gold links
            loss, mentions_found, total_mentions, \
                corefs_found, total_corefs, corefs_chosen = self.train_doc(doc)

            # Track stats by document for debugging
            print(document, '| Loss: %f | Mentions: %d/%d | Coref recall: %d/%d | Corefs precision: %d/%d' \
                % (loss, mentions_found, total_mentions,
                    corefs_found, total_corefs, corefs_chosen, total_corefs))

            epoch_loss.append(loss)
            epoch_mentions.append(safe_divide(mentions_found, total_mentions))
            epoch_corefs.append(safe_divide(corefs_found, total_corefs))
            epoch_identified.append(safe_divide(corefs_chosen, total_corefs))

        # Step the learning rate decrease scheduler
        self.scheduler.step()

        print('Epoch: %d | Loss: %f | Mention recall: %f | Coref recall: %f | Coref precision: %f' \
                % (epoch, np.mean(epoch_loss), np.mean(epoch_mentions),
                    np.mean(epoch_corefs), np.mean(epoch_identified)))

    def train_doc(self, document):
        """ Compute loss for a forward pass over a document """

        # Extract gold coreference links
        gold_corefs, total_corefs, \
            gold_mentions, total_mentions = extract_gold_corefs(document)

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        mentions_found, corefs_found, corefs_chosen = 0, 0, 0

        # Predict coref probabilites for each span in a document
        spans, probs = self.model(document)

        # Get log-likelihood of correct antecedents implied by gold clustering
        gold_indexes = to_cuda(torch.zeros_like(probs))
        for idx, span in enumerate(spans):

            # Log number of mentions found
            if (span.i1, span.i2) in gold_mentions:
                mentions_found += 1

                # Check which of these tuples are in the gold set, if any
                golds = [
                    i for i, link in enumerate(span.yi_idx)
                    if link in gold_corefs
                ]

                # If gold_pred_idx is not empty, consider the probabilities of the found antecedents
                if golds:
                    gold_indexes[idx, golds] = 1

                    # Progress logging for recall
                    corefs_found += len(golds)
                    found_corefs = sum((probs[idx, golds] > probs[idx, len(span.yi_idx)])).detach()
                    corefs_chosen += found_corefs.item()
                else:
                    # Otherwise, set gold to dummy
                    gold_indexes[idx, len(span.yi_idx)] = 1

        # Negative marginal log-likelihood
        eps = 1e-8
        loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1-eps), dim=0) * -1)

        # Backpropagate
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

        return (loss.item(), mentions_found, total_mentions,
                corefs_found, total_corefs, corefs_chosen)

    def evaluate(self, val_corpus, eval_script='../src/eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """

        # Predict files
        print('Evaluating on validation corpus...')
        predicted_docs = [self.predict(doc) for doc in tqdm(val_corpus)]
        val_corpus.docs = predicted_docs

        # Output results
        golds_file, preds_file = self.to_conll(val_corpus, eval_script)

        # Run perl script
        print('Running Perl evaluation script...')
        p = Popen([eval_script, 'all', golds_file, preds_file], stdout=PIPE)
        stdout, stderr = p.communicate()
        results = str(stdout).split('TOTALS')[-1]

        # Write the results out for later viewing
        with open('../preds/results.txt', 'w+') as f:
            f.write(results)
            f.write('\n\n\n')

        return results

    def predict(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()

        # Pass the document through the model
        spans, probs = self.model(doc)

        # Cluster found coreference links
        for i, span in enumerate(spans):

            # Loss implicitly pushes coref links above 0, rest below 0
            found_corefs = [idx
                            for idx, _ in enumerate(span.yi_idx)
                            if probs[i, idx] > probs[i, len(span.yi_idx)]]

            # If we have any
            if any(found_corefs):

                # Add edges between all spans in the cluster
                for coref_idx in found_corefs:
                    link = spans[coref_idx]
                    graph.add_edge((span.i1, span.i2), (link.i1, link.i2))

        # Extract clusters as nodes that share an edge
        clusters = list(nx.connected_components(graph))

        # Initialize token tags
        token_tags = [[] for _ in range(len(doc))]

        # Add in cluster ids for each cluster of corefs in place of token tag
        for idx, cluster in enumerate(clusters):
            for i1, i2 in cluster:

                if i1 == i2:
                    token_tags[i1].append(f'({idx})')

                else:
                    token_tags[i1].append(f'({idx}')
                    token_tags[i2].append(f'{idx})')

        doc.tags = ['|'.join(t) if t else '-' for t in token_tags]

        return doc

    def to_conll(self, val_corpus, eval_script):
        """ Write to out_file the predictions, return CoNLL metrics results """

        # Make predictions directory if there isn't one already
        golds_file, preds_file = '../preds/golds.txt', '../preds/predictions.txt'
        if not os.path.exists('../preds/'):
            os.makedirs('../preds/')

        # Combine all gold files into a single file (Perl script requires this)
        golds_file_content = flatten([doc.raw_text for doc in val_corpus])
        with io.open(golds_file, 'w', encoding='utf-8', errors='strict') as f:
            for line in golds_file_content:
                f.write(line)

        # Dump predictions
        with io.open(preds_file, 'w', encoding='utf-8', errors='strict') as f:

            for doc in val_corpus:

                current_idx = 0

                for line in doc.raw_text:

                    # Indicates start / end of document or line break
                    if line.startswith('#begin') or line.startswith('#end') or line == '\n':
                        f.write(line)
                        continue
                    else:
                        # Replace the coref column entry with the predicted tag
                        tokens = line.split()
                        tokens[-1] = doc.tags[current_idx]

                        # Increment by 1 so tags are still aligned
                        current_idx += 1

                        # Rewrite it back out
                        f.write('\t'.join(tokens))
                    f.write('\n')

        return golds_file, preds_file

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)


# Initialize model, train
model = CorefScore(embeds_dim=400, hidden_dim=200)
# ?? train for 150 epochs, each each train 100 documents
trainer = Trainer(model, train_corpus, val_corpus, test_corpus, steps=100)
trainer.train(150)

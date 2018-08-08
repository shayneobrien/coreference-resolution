# TODO:
# Early stopping
# No more slicing (is this possible to do..?)

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
    """ Learned, continuous representations for: span size, width between spans
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
        return self.embeds(self.stoi(*args)).squeeze()

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
        self.cnn_dropout = nn.Dropout(0.20)

    def forward(self, *args):
        """ Compute filter-dimensional character-level features for each doc token """
        embedded = self.embeddings(self.doc_to_batch(*args))
        convolved = torch.cat([F.relu(conv(embedded)) for conv in self.convs], dim=2)
        pooled = F.max_pool1d(convolved, convolved.shape[2])
        output = self.cnn_dropout(pooled).squeeze()
        return output

    def doc_to_batch(self, doc):
        """ Batch-ify a document class instance for CharCNN embeddings """
        tokens = [self.token_to_idx(token) for token in doc.tokens]
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

        weights = VECTORS.weights()
        turian_weights = TURIAN.weights()

        # GLoVE
        self.embeddings = nn.Embedding(weights.shape[0], weights.shape[1])
        self.embeddings.weight.data.copy_(weights)

        # Turian
        self.turian = nn.Embedding(turian_weights.shape[0], turian_weights.shape[1])
        self.turian.weight.data.copy_(turian_weights)

        # Character
        self.char_embeddings = CharCNN(char_filters)

        # Graf
        self.lstm = nn.LSTM(weights.shape[1]+turian_weights.shape[1]+char_filters,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout
        self.emb_dropout, self.lstm_dropout = nn.Dropout(0.50), nn.Dropout(0.20)

    def forward(self, documents):
        """ Convert document words to ids, embed them, pass through LSTM. """

        # Embed document
        embedded_docs = [self.embed_doc(doc) for doc in documents]

        # Batch for LSTM
        packed, reorder = pad_and_pack(embedded_docs)

        # Pass an LSTM over the embeds
        states, _ = self.lstm(packed)

        # Undo the packing/padding required for batching
        unpacked = unpack_and_unpad(states, reorder)

        # Apply dropout
        unpacked = [self.lstm_dropout(tensor) for tensor in unpacked]

        return unpacked, embedded_docs

    def embed_doc(self, document):
        """ Embed a document using GLoVE, Turian, and character embeddings """
        # Convert document tokens to look up ids
        tensor = doc_to_tensor(document, VECTORS)

        # Embed the tokens with Glove, apply dropout
        embeds = self.embeddings(tensor)
        embeds = self.emb_dropout(embeds)

        # Convert document tokens to Turian look up IDs
        tur_tensor = doc_to_tensor(document, TURIAN)

        # Embed again using Turian this time, dropout
        tur_embeds = self.turian(tur_tensor)
        tur_embeds = self.emb_dropout(tur_embeds)

        # Character embeddings
        char_embeds = self.char_embeddings(document)

        # Concatenate them all together
        full_embeds = torch.cat((embeds, tur_embeds, char_embeds), dim=1)

        return full_embeds


class MentionScore(nn.Module):
    """ Mention scoring module
    """
    def __init__(self, gi_dim, attn_dim, distance_dim):
        super().__init__()

        self.attention = Score(attn_dim)
        self.width = Distance(distance_dim)
        self.score = Score(gi_dim)

    def forward(self, unpacked, embedded_docs, documents):
        """ Compute unary mention score for each span
        """
        # Cat together for faster computation
        sizes = [i.shape[0] for i in unpacked]
        states = torch.cat(unpacked, dim=0)
        embeds = torch.cat(embedded_docs)

        # Compute widths
        span_lens = [len(s) for doc in documents for s in doc.spans]
        widths = self.width(span_lens)

        # Compute first part of attention over span states
        attns = self.attention(states)
        spans, shift = [], 0

        for idx, doc in tqdm_notebook(enumerate(documents)):

            for s_idx, span in enumerate(doc.spans):

                # Start index, end index of the span
                i1, i2 = span[0]+shift, span[-1]+shift

                # Speaker
                speaker = s_to_speaker(span, doc.speakers)

                # Embeddings, hidden states, raw attn scores for tokens
                # Slicing slows performance. Unsure if this is batch-able.
                span_embeds = embeds[i1:i2+1]
                span_attn = attns[i1:i2+1]

                # Compute the rest of the attention
                attn = F.softmax(span_attn, dim=0)
                attn = sum(attn * span_embeds)

                # Final span representation g_i
                g_i = torch.cat([states[i1], states[i2], attn, widths[shift+s_idx]])
                spans.append(Span(i1, i2, g_i, speaker, doc.genre))

            # Increment shift
            shift += sizes[idx]

        # Compute each span's unary mention score
        mention_reprs = torch.stack([s.g for s in spans])
        mention_scores = self.score(mention_reprs).squeeze()

        # Update the span object
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores)
        ]

        # Regroup spans into their respective documents
        lens = [0] + [len(doc.spans) for doc in documents]
        indexes = [sum(lens[:i+1]) for i, _ in enumerate(lens)]
        spans = [spans[i1:i2] for i1, i2 in pairwise(indexes)]

        # PRUNE
        spans = [prune(doc_spans, len(doc))
                 for doc_spans, doc
                 in zip(spans, documents)]

        return spans


class PairwiseScore(nn.Module):
    """ Coreference pair scoring module
    """
    def __init__(self, gij_dim, distance_dim, genre_dim, speaker_dim):
        super().__init__()

        self.distance = Distance(distance_dim)
        self.genre = Genre(genre_dim)
        self.speaker = Speaker(speaker_dim)

        self.score = Score(gij_dim)

    def forward(self, doc_spans, K=250):
        """ Compute pairwise score for spans and their up to K antecedents
        """

        all_spans = []

        for span_group in tqdm_notebook(doc_spans):

            # Consider only top K antecedents
            spans = [
                attr.evolve(span, yi=span_group[max(0, idx-K):idx])
                for idx, span in enumerate(span_group)
            ]

            # Batch lookup feature dims
            distances, genres, speakers = zip(*[(i.i2-j.i1, i.genre, speaker_label(i, j))
                                                for i in spans
                                                for j in i.yi])

            distances_embs = self.distance(distances)
            genres_embs = self.genre(genres)
            speakers_embs = self.speaker(speakers)

            # Get s_ij representations
            pairs = torch.stack([
                torch.cat([i.g, j.g, i.g*j.g,
                          distances_embs[idx],
                          genres_embs[idx],
                          speakers_embs[idx]
                          ])
                for idx, i in enumerate(spans) for j in i.yi
            ])

            # Score pairs of spans for coreference link
            pairwise_scores = self.score(pairs).squeeze()

            # Indices for pairs indexing back into pairwise_scores
            sa_idx = pairwise_indexes(spans)

            spans_ij = []
            for span, (i1, i2) in zip(spans, pairwise(sa_idx)):

                # sij = score between span i, span j
                sij = [
                    (span.si + yi.si + pair)
                    for yi, pair in zip(span.yi, pairwise_scores[i1:i2])
                ]

                # Dummy variable for if the span is not a mention
                epsilon = to_var(torch.tensor(0.))
                sij = torch.stack(sij + [epsilon])

                # Update span object
                spans_ij.append(attr.evolve(span, sij=sij))

            # Update spans with set of possible antecedents' indices
            spans = [
                attr.evolve(span, yi_idx=[((y.i1, y.i2), (span.i1, span.i2))
                                            for y in span.yi])
                for span in spans_ij
            ]

            all_spans.append(spans)

        return all_spans


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

    def forward(self, documents):
        """ Enocde document
            Predict unary mention scores, prune them
            Predict pairwise coreference scores
        """
        # Encode the document, keep the LSTM hidden states and embedded tokens
        unpacked, embedded_docs = self.encoder(documents)

        # Get mention scores for each span, prune
        spans = self.score_spans(unpacked, embedded_docs, documents)

        # Get pairwise scores for each span combo
        pairs = self.score_pairs(spans)

        return pairs


# Initialize model, train
model = CorefScore(embeds_dim=400, hidden_dim=200)
trainer = Trainer(model, train_corpus, val_corpus, test_corpus)
trainer.train(100)

# TODO:
# Early stopping
# No more slicing (is this possible to do..?)
# Batching documents / convert to sentence LSTM

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

    def stoi(self, num):
        """ Find which bin a number falls into """
        return to_cuda(torch.tensor(
            sum([True for i in self.bins if num >= i]), requires_grad=False
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

    def forward(self, genre):
        """ Embedding table lookup """
        return self.embeds(self.stoi(genre))

    def stoi(self, genre):
        """ Locate embedding id for genre """
        idx = self._stoi.get(genre)
        idx = idx if idx else 0
        return to_cuda(torch.tensor(idx))


class Speaker(nn.Module):
    """ Learned continuous representations for binary speaker. Zeros if speaker unknown.
    """

    def __init__(self, speaker_dim=20):
        super().__init__()

        self.embeds = nn.Sequential(
            nn.Embedding(3, speaker_dim, padding_idx=0),
            nn.Dropout(0.20)
        )

    def forward(self, s1, s2):
        """ Embedding table lookup """
        # Same speaker
        if s1.speaker == s2.speaker:
            idx = torch.tensor(1)

        # Different speakers
        elif s1.speaker != s2.speaker:
            idx = torch.tensor(2)

        # No speaker
        else:
            idx = torch.tensor(0)

        return self.embeds(to_cuda(idx))


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
        batch = self.pad_and_stack(tokens)
        return batch

    def token_to_idx(self, token):
        """ Convert a token to its character lookup ids """
        return to_cuda(torch.tensor([self.stoi(c) for c in token]))

    def pad_and_stack(self, tokens):
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
    def __init__(self, hidden_dim, char_filters):
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

        self.lstm = nn.LSTM(weights.shape[1], hidden_dim, bidirectional=True, batch_first=True)
        self.emb_dropout, self.lstm_dropout = nn.Dropout(0.50), nn.Dropout(0.20)

    def forward(self, document):
        """ Convert document words to ids, embed them, pass through LSTM. """
        # Convert document tokens to look up ids
        tensor = doc_to_tensor(document, VECTORS)
        tensor = tensor.unsqueeze(0)

        # Embed the tokens, regularize
        embeds = self.embeddings(tensor)
        embeds = self.emb_dropout(embeds)

        # Convert document tokens to Turian look up IDs #TODO: align with GLoVE
        tur_tensor = doc_to_tensor(document, TURIAN)
        tur_tensor = tur_tensor.unsqueeze(0)

        # Embed again
        tur_embeds = self.turian(tur_tensor)
        tur_embeds = self.emb_dropout(tur_embeds)

        char_embeds = self.char_embeddings(document).unsqueeze(0)
        full_embeds = torch.cat((embeds, tur_embeds, char_embeds), dim=2)

        # Pass an LSTM over the embeds, regularize
        states, _ = self.lstm(embeds)
        states = self.lstm_dropout(states)

        return states.squeeze(), full_embeds.squeeze()


class MentionScore(nn.Module):
    """ Mention scoring module
    """
    def __init__(self, gi_dim, attn_dim, distance_dim):
        super().__init__()

        self.attention = Score(attn_dim)
        self.width = Distance(distance_dim)
        self.score = Score(gi_dim)

    def forward(self, states, embeds, document, LAMBDA=0.40):
        """ Compute unary mention score for each span

        """
        # Compute first part of attention over span states
        attns = self.attention(states)

        spans = []

        for span in document.spans: # could probably deprecate this
            # Start index, end index of the span
            i1, i2 = span[0], span[-1]

            # Speaker
            speaker = s_to_speaker(span, document.speakers)

            # Embeddings, hidden states, raw attn scores for tokens
            # Slicing slows performance. Unsure if this is batch-able.
            span_embeds = embeds[i1:i2+1]
            span_attn = attns[i1:i2+1]

            # Compute the rest of the attention
            attn = F.softmax(span_attn, dim=0)
            attn = sum(attn * span_embeds)

            # Lookup embedding for width of spans
            size = self.width(len(span))

            # Final span representation g_i
            g_i = torch.cat([states[i1], states[i2], attn, size])
            spans.append(Span(i1, i2, g_i, speaker))

        # Compute each span's unary mention score
        mention_reprs = torch.stack([s.g for s in spans])
        mention_scores = self.score(mention_reprs).squeeze()

        # Update the span object
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores)
        ]

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

    def forward(self, spans, genre, K=250):
        """ Compute pairwise score for spans and their up to K antecedents
        """
        # Consider only top K antecedents
        spans = [
            attr.evolve(span, yi=spans[max(0, idx-K):idx])
            for idx, span in enumerate(spans)
        ]

        # Get s_ij representations
        pairs = torch.stack([
            torch.cat([i.g, j.g, i.g*j.g,
                       self.distance(i.i2-j.i1),
                       self.genre(genre),
                       self.speaker(i, j)])
            for i in spans for j in i.yi
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

        return spans


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
        self.encode_doc = DocumentEncoder(hidden_dim, char_filters)
        self.score_spans = MentionScore(gi_dim, attn_dim, distance_dim)
        self.score_pairs = PairwiseScore(gij_dim, distance_dim, genre_dim, speaker_dim)

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
        spans = prune(spans, len(document))

        # Get pairwise scores for each span combo
        spans = self.score_pairs(spans, document.genre)

        return spans


class Trainer:
    """ Class dedicated to training and evaluating the model
    """
    def __init__(self, model, train_corpus, val_corpus, test_corpus,
                    lr=1e-3, steps=100):
        self.__dict__.update(locals())
        self.train_corpus = list(self.train_corpus)
        self.model = to_cuda(model)

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                            if p.requires_grad],
                                    lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=100,
                                                    gamma=0.001)

    def train(self, num_epochs, *args, **kwargs):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch, *args, **kwargs)
            # Evaluate every now and then
            if epoch % 10 == 0:
                print('\n\nEVALUATION\n\n')
                self.model.eval()
                self.save_model(str(datetime.now()))
                results = self.evaluate(self.val_corpus)
                print(results)

    def train_epoch(self, epoch):
        """ Run a training epoch over 'steps' documents """
        # Set model to train (enables dropout)
        self.model.train()

        # Randomly sample documents from the train corpus
        docs = random.sample(self.train_corpus, self.steps)

        epoch_loss, epoch_mentions, epoch_corefs, epoch_identified = [], [], [], []
        for doc in tqdm(docs):

            # Randomly truncate document to up to 50 sentences
            document = doc.truncate()

            # Compute loss, number gold links found, total gold links
            loss, mentions_found, total_mentions, corefs_found, total_corefs, corefs_chosen = self.train_doc(document)

            # Track stats by document for debugging
            print(document, '| Loss: %f | Mentions found: %d/%d | Coreferences found: %d/%d | Corefs chosen: %d/%d' % (loss,
                                                                                                                        mentions_found, total_mentions,
                                                                                                                        corefs_found, total_corefs,
                                                                                                                        corefs_chosen, total_corefs))

            epoch_loss.append(loss)
            epoch_mentions.append(safe_divide(mentions_found, total_mentions))
            epoch_corefs.append(safe_divide(corefs_found, total_corefs))
            epoch_identified.append(safe_divide(corefs_chosen, total_corefs))

            # Step the learning rate decrease scheduler
            self.scheduler.step()

        print('Epoch: %d | Loss: %f | Mention recall: %f | Coref recall: %f | Coref identifications: %f' % (epoch,
                                                                                                            np.mean(epoch_loss),
                                                                                                            np.mean(epoch_mentions),
                                                                                                            np.mean(epoch_corefs),
                                                                                                            np.mean(epoch_identified)))

    def train_doc(self, document, CLIP=5):
        """ Compute loss for a forward pass over a document """

        # Extract gold coreference links
        gold_corefs, total_corefs, gold_mentions, total_mentions = extract_gold_corefs(document)

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        losses, mentions_found, corefs_found, corefs_chosen = [], [], [], []
        for span in self.model(document):

            # Log number of mentions found
            if (span.i1, span.i2) in gold_mentions:
                mentions_found.append(1)

            # Check which of these tuples are in the gold set, if any
            gold_idx = [
                idx for idx, link in enumerate(span.yi_idx)
                if link in gold_corefs
            ]

            # If gold_pred_idx is empty, set gold to dummy
            if not gold_idx:
                gold_idx = [len(span.sij)-1]
            else:
                corefs_found.append(len(gold_idx))
                found_corefs = [1 for score in span.sij if score > 0.]
                corefs_chosen.append(len(found_corefs))

            # Conditional probability distribution over all possible previous spans
            probs = F.softmax(span.sij, dim=0)

            # Marginal log-likelihood of correct antecedents implied by gold clustering
            mass = torch.log(sum([probs[i] for i in gold_idx]))

            # Save the loss for this span
            losses.append(mass)

        # Negative marginal log-likelihood for minimizing, backpropagate
        loss = sum(losses) * -1
        loss.backward()

        # Clip parameters
#         nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)

        # Step the optimizer
        self.optimizer.step()

        # Compute recall
        mentions_found = sum(mentions_found)
        corefs_found = sum(corefs_found)
        corefs_chosen = sum(corefs_chosen)

        return loss.item(), mentions_found, total_mentions, corefs_found, total_corefs, corefs_chosen

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
        with open('../preds/results.txt', 'w') as f:
            f.write(results)

        return results

    def predict(self, document):
        """ Predict coreference clusters in a document """

        graph = nx.Graph()
        spans = self.model(document)
        for i, span in enumerate(spans):

            found_corefs = [idx
                            for idx, score in enumerate(span.sij)
                            if score > 0.]

            if any(found_corefs):

                for coref_idx in found_corefs:
                    link = spans[coref_idx]
                    graph.add_edge((span.i1, span.i2), (link.i1, link.i2))

        clusters = list(nx.connected_components(graph))

        # Cluster found coreferences
        doc_tags = [[] for _ in range(len(document))]

        for idx, cluster in enumerate(clusters):
            for i1, i2 in cluster:

                if i1 == i2:
                    doc_tags[i1].append(f'({idx})')

                else:
                    doc_tags[i1].append(f'({idx}')
                    doc_tags[i2].append(f'{idx})')

        document.tags = ['|'.join(t) if t else '-' for t in doc_tags]

        return document

    def to_conll(self, val_corpus, eval_script):
        """ Write to out_file the predictions, return CoNLL metrics results """

        # Make predictions directory if there isn't one already
        golds_file, preds_file = '../preds/golds.txt', '../preds/predictions.txt'
        if not os.path.exists('../preds/'):
            os.makedirs('../preds/')

        # Combine all gold files into a single file (Perl script requires this)
        golds_file_content = flatten([doc.raw_text for doc in val_corpus])
        with io.open(preds_file, 'w', encoding='utf-8', errors='strict') as f:
            for line in golds_file_content:
                f.write(line)

        # Dump predictions
        with io.open(filename, 'w', encoding='utf-8', errors='strict') as f:

            current_idx = 0
            for doc in val_corpus:

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
trainer = Trainer(model, train_corpus, val_corpus, test_corpus)
trainer.train(100)

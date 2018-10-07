# Coreference Resolution
PyTorch 0.4.1 | Python 3.6.5

This repository consists of an efficient, annotated PyTorch reimplementation of the EMNLP paper ["End-to-end Neural Coreference Resolution"](https://arxiv.org/pdf/1707.07045.pdf) by Lee et al., 2017.

# Data
The source code assumes access to the English train, test, and development data of OntoNotes Release 5.0. This data should be located in a folder called 'data' inside the main directory. The data consists of 2,802 training documents, 343 development documents, and 348 testing documents. The average length of all documents is 454 words with a maximum length of 4,009 words. The number of mentions and coreferences in each document varies drastically, but is generally correlated with document length.

Since the data require a license from the Linguistic Data Consortium to use, they are thus not supplied here. Information on how to download and preprocess them can be found [here](http://conll.cemantix.org/2012/data.html) and [here](https://catalog.ldc.upenn.edu/ldc2013t19), respectively.

Beyond the data, the source files also assume access to both [Turian embeddings](http://metaoptimize.s3.amazonaws.com/hlbl-embeddings-ACL2010/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt.gz) and [GloVe embeddings](http://nlp.stanford.edu/data/glove.6B.zip).

# Problem Definition
Coreference is defined as occurring when one or more expressions in a document refer back to the an entity that came before it/them. Coreference resolution, then, is the task of finding all expressions that are coreferent with any of the entities found in a given text. This idea is summarized in the below image, courtesy of [Stanford NLP](https://nlp.stanford.edu/projects/coref.shtml).

![](/imgs/problem_intro.png)

# Useful Nomenclature

Oftentimes the nomenclature of coreference resolution can be confusing. Visualizing them makes things a bit easier to understand:

![](/imgs/nomenclature.png)

Words are colored according to whether they are entities or not. Different colored groups of words are members of the same coreference cluster. Entities that are the only member of their cluster are known as 'singleton' entities.

# Why Corefence Resolution is Hard

Coref is hard because entities can be very long and coreferent entities can occur extremely far away from one another. A greedy system would compute every possible span (sequence) of tokens and then compare it to every possible span that came before it. This makes the complexity of the problem O(T<sup>4</sup>), where T is the document length. For a 100 word document this would be 100 million possible options and for the longest document in our dataset, this equates to almost one quadrillion possible combinations.

If this does not make it concrete, imagine that we had the sentence ```Arya Stark walks her direwolf, Nymeria.``` Here we have three entities: ```Arya Stark```, ```her```, and ```Nymeria```. As a native speaker of English it should be trivial to tell that ```her``` refers to ```Arya Stark```. But to a machine with no knowledge, how should it know that ```Arya``` and ```Stark``` should be a single entity rather than two separate ones, that ```Nymeria``` does not refer back to ```her``` even though they are arguably related, or even that that ```Arya Stark walks her direwolf, Nymeria``` is not just one big entity in and of itself?

For another example, consider the sentence ```Napoleon and all of his marvelously dressed, incredibly well trained, loyal troops marched all the way across the Europe to enter into Russia in an, ultimately unsuccessful, effort to conquer it for their country.``` The word ```their``` is referent to ```Napoleon and all of his marvelously dressed, incredibly well trained, loyal troops```; entities can span many, many tokens. Coreferent entities can also occur far away from one another.

# Model Architecture

As a forewarning, this paper presents a beast of a model. The authors present the following series of images to provide clarity as to what the model is doing.

![](/imgs/architecture.png)

#### Token Representation ####
Tokens are represented using 300-dimension static GloVe embeddings, 50-dimensional static Turian embeddings, and 8-dimensional character embeddings from a CNN with 50-dimensional filter sizes 3, 4, and 5. Dropout with p=0.50 is applied to these embeddings. The token representations are passed into a 2-layer bidirectional LSTM with hidden state sizes of 200. Dropout with p=0.20 is applied to the output of the LSTM.

#### Span Representation ####
Using the regularized output, span representations are computed by extracting the LSTM hidden states between the index of the first word and the last word. These are used to compute a weighted sum of the hidden states. Then, we concatenate the first and last index with the weighted attention sum and a 20-dimensional feature representation for the total width (length) of the span under consideration. This is done for all spans up to length 10 in the document.

#### Pruning ####
The span representations are passed into a 3-layer, 150-dimensional feedforward network with ReLU activations and p=0.20 dropout applied between each layer. The output of this feedfoward network is 1-dimensional and represents the 'mention score' of each span in the document. Spans are then pruned in decreasing order of mention score unless, when considering a span i, there exists a previously accepted span j such that START(i) < START(j) <= END(i) < END(j) or START(j) < START(i) <= END(j) < END(j). Only LAMBDA * T spans are kept at the end, where LAMBDA is set to 0.40 and T is the document length.

#### Pairwise Representation ####
For these spans, pairwise representations are computed for a given span i and its antecedent j by concatenating the span representation for span i, the span representation for span j, the dot product between these representations, and 20-dimensional feature embeddings for genre, distance between the spans, and whether or not the two spans have the same speaker.

#### Final Score and Loss ####
These representations are passed into a feedforward network similar to that of scoring the spans. Clusters are then formed for these coreferences by identifying chains of coreference links (e.g. span j and span k both refer to span i). The learning objective is to maximize the log-likelihood of all correct antecedents that were not pruned.

# Results
Originally from the paper,

![](/imgs/results.png)

# Recent Work

The authors have since published [another paper](https://arxiv.org/abs/1804.05392), which achieves an F1 score of 73.0.

# SentimentAnalysis_CNN
Sentiment Analysis for reviews using IMDB Dataset with CNN and LSTM
___

## Introduction

Sentiment Analysis is the process of determining whether a piece of writing is positive, negative or neutral. It’s also known as opinion mining, deriving the opinion or attitude of a speaker. A common use case for this technology is to discover how people feel about a particular topic. In this case, the goal is to determine if a movie review is either positive or negative using various deep learning techniques.

## Data

I chose the IMDB dataset (Maas et al., 2011) which contains 50,000 sentences split equally into training and testing sets. Each training instance contains an entire review written by one individual. 

### Embedding

I also loaded pre-trained word embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/) composed of 400K vocab using 300D vectors. 

Word embeddings are just vectors that represent multiple features of a word. In GloVe, vectors represent relative position between words. One simple way to understand this is to look at the following image:

![](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/linear-relationships.png)

When we inspect these visualizations it becomes apparent that the vectors capture some general, and in fact quite useful, semantic information about words and their relationships to one another. 

## Implementation

This project was implemented using Keras framework with Tensorflow backend.

After loading text data, and embedding file, I create an embedding_matrix with as many entries as unique words in training data, where each row is the equivalent embedding rappresentation. If the word is not present in the embedding file, it's rappresentation would be simply a vector of zeros.

Moreover, I needed to PAD each review to a fixed length. I decided `MAX_SEQUENCE_LENGTH` to be 500 based on following plots:

![](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/box.png)
![](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/hist.png)





```Python
LR = 0.0005
drop_out = 0.3
batch_dim = 64

loss = 'categorical_crossentropy'

# We fix the window size to 11 because the average length of an alpha helix is around eleven residues
# and that of a beta strand is around six.
# See references [6].
m = Sequential()
m.add(Conv1D(128, 11, padding='same', activation='relu', input_shape=(dataset.sequence_len, dataset.amino_acid_residues)))
m.add(Dropout(drop_out))
m.add(Conv1D(64, 11, padding='same', activation='relu'))
m.add(Dropout(drop_out))
m.add(Conv1D(dataset.num_classes, 11, padding='same', activation='softmax'))
opt = optimizers.Adam(lr=LR)
m.compile(optimizer=opt,
          loss=loss,
          metrics=['accuracy', 'mae'])
```

Essentially three different different architectures were used:

- Only CNN
- Only LSTM
- Both CNN and LSTM









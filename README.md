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



## Implementation

This project was implemented using the Keras framework with the Tensorflow backend.

Essentially three different different architectures were used:

- Only CNN
- Only LSTM
- Both CNN and LSTM









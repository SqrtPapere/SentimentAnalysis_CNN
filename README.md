# SentimentAnalysis_CNN
Sentiment Analysis for reviews using IMDB Dataset with CNN and LSTM
___

## Introduction

Sentiment Analysis is the process of determining whether a piece of writing is positive, negative or neutral. It’s also known as opinion mining, deriving the opinion or attitude of a speaker. A common use case for this technology is to discover how people feel about a particular topic. In this case, the goal is to determine if a movie review is either positive or negative using various deep learning techniques.

## Data

I chose the IMDB dataset (Maas et al., 2011) which contains 50,000 sentences split equally into training and testing sets. Each training instance contains an entire review written by one individual. 

### Embedding

I also loaded pre-trained word embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/) composed of 400K vocab using 300D vectors. 

Word embedding is a technique where words are encoded as real-valued vectors in a high-dimensional space, where the similarity between words in terms of meaning translates to closeness in the vector space. One simple way to understand this is to look at the following image:

![](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/linear-relationships.png)

When we inspect these visualizations it becomes apparent that the vectors capture some general, and in fact quite useful, semantic information about words and their relationships to one another. 

## Implementation

This project was implemented using Keras framework with Tensorflow backend.

After loading text data, and embedding file, I create an embedding_matrix with as many entries as unique words in training data (111525 unique tokens), where each row is the equivalent embedding rappresentation. If the word is not present in the embedding file, it's rappresentation would be simply a vector of zeros.

Moreover, I needed to PAD each review to a fixed length. I decided `MAX_SEQUENCE_LENGTH` to be 500 based on following plots:

 [![Box](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/box.png)]() | [![hist](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/hist.png)]() 
|:---:|:---:|
| Box and Whisker | Histogram |

The mean number of word per review is 230 with a variance of 171. Using `MAX_SEQUENCE_LENGTH = 500` you can cover the majority of reviews and remove the outliers with too many words.

Essentially three different different architectures were used:

- Only CNN
- Only LSTM
- Both CNN and LSTM

### Only CNN

```Python
work in progresss...
```

![](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/doblegraph.png)

### Only LSTM

```Python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix],
                                               input_length=MAX_SEQUENCE_LENGTH, trainable=False)

x = embedding_layer(sequence_input)
x = Dropout(0.3)(x)
x = LSTM(100)(x)
prob = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, prob)
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
```

![](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/lstmgraph.png)

### Both CNN and LSTM

```Python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix],
                                               input_length=MAX_SEQUENCE_LENGTH, trainable=False)

x = embedding_layer(sequence_input)
x = Dropout(0.3)(x)
x = Conv1D(200, 5, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = LSTM(100)(x)
x = Dropout(0.3)(x)
prob = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, prob)
optimizer = optimizers.Adam(lr=0.0004)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
```

![](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/doblegraph.png)

## Results
Using 10% train set e evaluation su tutto il testset ....
### Only CNN 
 [![dobleacc](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/dobleacc.png)]() | [![dobleloss](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/dobleloss.png)]() 
|:---:|:---:|
| Accuracy | Loss |

On entire Test Set: `Accuracy = 89.14%`

### Only LSTM 
 [![lstmacc](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/lstmacc.png)]() | [![lstmloss](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/lstmloss.png)]() 
|:---:|:---:|
| Accuracy | Loss |

On entire Test Set: `Accuracy = 88.92%`

### Both CNN and LSTM
 [![dobleacc](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/dobleacc.png)]() | [![dobleloss](https://github.com/SqrtPapere/SentimentAnalysis_CNN/blob/master/Images/dobleloss.png)]() 
|:---:|:---:|
| Accuracy | Loss |

On entire Test Set: `Accuracy = 89.14%`

## References
\[1\]: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

\[2\]: https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

\[3\]: https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

\[4\]: Takeru Miyato, Andrew M. Dai and Ian Goodfellow (2016) -"Virtual Adversarial Training for Semi-Supervised Text Classification"- https://pdfs.semanticscholar.org/a098/6e09559fa6cc173d5c5740aa17030087f0c3.pdf

\[5\] Isaac Caswell, Onkur Sen and Allen Nie - "Exploring Adversarial Learning on Neural Network Models for Text Classification" - https://nlp.stanford.edu/courses/cs224n/2015/reports/20.pdf



# https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/
import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Input, GlobalMaxPooling1D, Dropout, Merge, BatchNormalization
from keras.models import Model
from keras import optimizers

import matplotlib.pyplot as plt

import pickle


def plotting(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


do_early_stopping = True
# top words to be considered in Tokenizer
NUM_WORDS = 20000

# Length of phrases for padding if shorter or cropping if longer
MAX_SEQUENCE_LENGTH = 450

EMBEDDING_DIM = 300

# preparing train-set from text data
train_text = np.load('Res/train_text.npy')
train_label = np.load('Res/train_label.npy')

print('TrainSet is composed of %s texts.' % len(train_text))

# preparing test-set from text data
test_text = np.load('Res/test_text.npy')
test_label = np.load('Res/test_label.npy')

print('TestSet is composed of %s texts.' % len(test_text))

# Formatting text samples and labels in tensors.
with open('Res/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

train_sequences = tokenizer.texts_to_sequences(train_text) # Splits words by space (split=” “), Filters out punctuation, Converts text to lowercase. For each text returns a list of integers (same words a codified by same integer)

test_sequences = tokenizer.texts_to_sequences(test_text)
word_index = tokenizer.word_index # dictionary mapping words (str) to their index starting from 0 (int)
print('Found %s unique tokens.' % len(word_index))

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH) # each element of sequences is cropped or padded to reach maxlen 
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_label = np.asarray(train_label)
test_label = np.asarray(test_label)
print('Shape of data tensor:', train_data.shape)

#shuffle dataset
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
train_data = train_data[indices]
train_label = train_label[indices]

# split the data into a training set and a validation set

num_validation_samples = int(0.1 * train_data.shape[0])

x_train = train_data[:-num_validation_samples]
y_train = train_label[:-num_validation_samples]

x_val = train_data[-num_validation_samples:]
y_val = train_label[-num_validation_samples:]

x_test = test_data
y_test = test_label


embedding_matrix = np.load('Res/embedding_matrix.npy')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(word_index)+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

x = embedding_layer(sequence_input)
x = Dropout(0.3)(x)
x = Conv1D(25, 5, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)
x = Conv1D(20, 5, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(120,activation='sigmoid')(x)
x = Dropout(0.3)(x)
prob = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, prob)

optimizer = optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy', 'mae'])

tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True)

print('model compiled')

print(model.summary()) 

early_stopping = EarlyStopping(monitor='val_loss', patience = 2, mode = 'min')
cp = ModelCheckpoint('bestModel.h5', monitor='val_acc', save_best_only=True, mode='max')


if do_early_stopping:
    print('using early stopping strategy')
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=12, batch_size=128, callbacks = [tensorboard])
else:
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)


loss, acc, mae = model.evaluate(x_test, y_test)

print("loss: "+str(loss))
print("accuracy: "+str(acc)) 
print("mae: "+str(mae)) 

model.save('my_model3.h5')

plotting(history)



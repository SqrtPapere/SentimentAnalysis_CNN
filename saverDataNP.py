import numpy as np

import matplotlib.pyplot as plt

import re
import os

from keras.preprocessing.text import Tokenizer

import pickle



EMBEDDING_DIM = 300
NUM_WORDS = 20000

########## Change these!

dataset_dir = '/Users/francescopegoraro/Desktop/datasets/aclImdb'
glove_dir = '/Users/francescopegoraro/Dropbox/progettoML/glove.6B'

##########

positiveFiles = dataset_dir+'/train/pos'
negativeFiles = dataset_dir+'/train/neg'

positive_testset_dir = dataset_dir+'/test/pos'
negative_testset_dir = dataset_dir+'/test/neg'

if EMBEDDING_DIM == 50:
    glove_embedding_file = glove_dir+'/glove.6B.50d.txt'
elif EMBEDDING_DIM == 100:
    glove_embedding_file = glove_dir+'/glove.6B.100d.txt'
elif EMBEDDING_DIM == 200:
    glove_embedding_file = glove_dir+'/glove.6B.200d.txt'
elif EMBEDDING_DIM == 300:
    glove_embedding_file = glove_dir+'/glove.6B.300d.txt'
else: 
    print("Wrong embedding_dim submitted!")
    exit()



strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def load_text_and_label(positive_set_dir, negative_set_dir):
    texts = []
    labels = []
    for fname in sorted(os.listdir(positive_set_dir)):
        fpath = os.path.join(positive_set_dir, fname)
        f = open(fpath, encoding='latin-1')
        t = f.read()
        i = t.find('\n\n')  # skip header
        if 0 < i:
            t = t[i:]
        t = cleanSentences(t)
        texts.append(t)
        labels.append(1)
        f.close()
    for fname in sorted(os.listdir(negative_set_dir)):
        fpath = os.path.join(negative_set_dir, fname)
        f = open(fpath, encoding='latin-1')
        t = f.read()
        i = t.find('\n\n')  # skip header
        if 0 < i:
            t = t[i:]
        t = cleanSentences(t)
        texts.append(t)
        labels.append(0)
        f.close()
    return texts, labels

print("Loading Train and Test...")

texts, labels = load_text_and_label(positiveFiles, negativeFiles)  # list of text samples and list of label matching texts, 0 is negative, 1 is positive
np.save('Res/train_text.npy', texts)
np.save('Res/train_label.npy', labels)

print("Review length: ")
result = [len(x.split()) for x in texts]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))
# plot review length
binwidth = 5
plt.hist(result, bins=range(1, 1500), rwidth=0.8)
plt.show()

test_texts, test_labels = load_text_and_label(positive_testset_dir, negative_testset_dir)  # list of text samples
np.save('Res/test_text.npy', test_texts)
np.save('Res/test_label.npy', test_labels)

print("Done!")

print("Preparing Tokenizer...")
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(texts) 
test_sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index # dictionary mapping words (str) to their index starting from 0 (int)
print('Found %s unique tokens.' % len(word_index))



# save
with open('Res/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done!")

def load_embedding_from_file(embedding_file_path):
    embeddings_index = {}

    f = open(embedding_file_path)
    for element in f:
        values = element.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

print("Preparing embedding matrix...")
embeddings_index = load_embedding_from_file(glove_embedding_file)

def build_embedding_matrix(word_index_el, embeddings_index_el, EMBEDDING_DIM_val):
    embedding_m = np.zeros((len(word_index_el) + 1, EMBEDDING_DIM_val)) # c'è il +1 perche si vuole partire ad inserire da riga 1 e non da 0
    for word, i in word_index_el.items():
        embedding_vector = embeddings_index_el.get(word)
        if embedding_vector is not None:
            embedding_m[i] = embedding_vector
    return embedding_m

embedding_matrix = build_embedding_matrix(word_index, embeddings_index, EMBEDDING_DIM)

np.save('Res/embedding_matrix.npy', embedding_matrix)

print("Done!")




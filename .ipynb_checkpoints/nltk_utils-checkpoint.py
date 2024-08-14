import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer


def tokenizer(sentence):
    return nltk.word_tokenize(sentence)


def stemmer(word):
    stemmer = PorterStemmer()
    stemmed = stemmer.stem(word.lower())
    return stemmed

def bag_of_words(sentence_tokenized, all_words):
    """
    sentence = ["who", "is", "that"]
    all_words = ["hi", "good", "l", "who", "is", "open", "cool", "that"]
    bag       = [  0,    0,     0,    1,    1,     0,      0,      1]
    """
    sentence_tokenized = [stemmer(w) for w in sentence_tokenized]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
        if w in sentence_tokenized:
            bag[index] = 1.0

    return bag

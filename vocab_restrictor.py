import re
import numpy as np
import pandas as pd
from scipy import spatial
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from grammarize import grammarize
 
module_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3")

with open('bad_words.txt') as f:
    bad_words = set(f.read().split('\n'))
    
def get_vocab(bookpath):
    with open(bookpath) as f:
        text = f.read().lower().replace('\n',' ')
    text = re.sub('\W',' ',text)
    vocab = list(set(text.split())-bad_words)
    return vocab

def encode(wordlist):
    # makes word vectors from list
    return module_use.signatures["question_encoder"](
    tf.constant(wordlist))["outputs"]

def make_kdtree(vocab):
    # creates kdtree for fast nn search
    vectors = encode(vocab)
    df = pd.DataFrame(np.array(vectors),index=vocab)
    kdtree = spatial.cKDTree(df)
    return kdtree

def preserve_pos(nearest_word,word):
    return nearest_word

class vocab_restrictor(object):
    '''
    restricts text to vocabulary in a given book
    '''
    def __init__(self,bookpath):
        self.vocab = get_vocab(bookpath)
        self.kdtree = make_kdtree(self.vocab)
    
    def normalize(self,word):
        # normalizes single word
        gord = ''.join(re.findall('[a-z]',word.lower()))
        if gord in self.vocab or "'" in word:
            return word
        else:
            v = encode(word)
            nn = self.kdtree.query(v,1)[-1][0]
            nearest_word = self.vocab[nn]
            target_word = preserve_pos(nearest_word,word)
            return target_word
        
    def restrict(self,text,mode='test'):
        # restricts text
        words = text.split()
        target_words = [self.normalize(word) for word in words]
                    
        if mode=='test':
            return ' '.join([t if t==w else t.upper()+'('+w+')' 
                             for t,w in zip(target_words,words)])
        elif mode=='lucky':
            return ' '.join(target_words)
        elif mode=='sus':
            return ' '.join([t if t==w else t.upper() 
                             for t,w in zip(target_words,words)])
        elif mode=='grammatical':
            return ' '.join([t if t==w else grammarize(w, t)
                             for t,w in zip(target_words,words)])
import os
import sys
import numpy as np
import pandas as pd

import gensim
from gensim.models import Word2Vec

class W2V():
    """
    This class provides an interface between the normalized and tokenized texts
    from the SciTextProcessor object and a Word2Vec model trained from the
    extracted entities, phrases and tokens.
    """
    def __init__(self, params={}):
        """
        Set parameters for Word2Vec model

        Parameters:
            params (dict, required): Dictionary containing all non-default
                                     parameters
        """
        self.params = params
        if 'min_count' not in self.params.keys():
            self.params['min_count'] = 5
        if 'size' not in self.params.keys():
            self.params['size'] = 200
        if 'window' not in self.params.keys():
            self.params['window'] = 10
        if 'workers' not in self.params.keys():
            self.params['workers'] = 4

    def train(self, tokenized_texts, save=True, save_tag='w2v', save_dir='w2v_files'):
        """
        Trains Word2Vec model and saves to given directory

        Parameters:
            tokenized_texts (dict, required): Dictionary of lists. Each entry is
                                              a nested list of sentences containing
                                              tokens from the text
            save_tag (str): Tag to prepend to saved model
            save_dir (str): Directory to save w2v model
        """
        os.makedirs(save_dir, exit_ok=True)

        # Gather sentences
        sentences = []
        for text in tokenized_texts.values():
            for sentence in text:
                sentences.append(sentence)

        # Train model
        self.model = Word2Vec(sentences, min_count=self.params['min_count'],
                         size=self.params['size'], window=self.params['window'],
                         workers=self.params['workers'], sg=1)
        save_fn = '{}_mc{}_s{}_wdw{}.model'.format(save_tag, self.params['min_count'],
                                                   self.params['size'],
                                                   self.params['window'])

        # Save model
        if save:
            self.model.save(os.path.join(save_dir, save_fn))

    def load(self, path):
        """
        Loads trained Word2Vec model

        Parameters:
            path (str, required): Path to saved model
        """
        self.model = Word2Vec.load(path)

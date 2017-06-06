import tensorflow as tf
import numpy as np
import pandas as pd 
from keras.preprocessing import sequence, text
import string
from gensim.models import KeyedVectors

class Data(object):
    def __init__(self):
        self.train_x1 = None
        self.train_x2 = None
        self.train_labels = None
        self.valid_x1 = None
        self.valid_x2 = None
        self.valid_labels = None
        self.contrastive = False
        self.embedding_matrix = None

    def import_data(self, train_csv):
        print("importing data...")
        df = pd.read_csv(train_csv)
        df = df.dropna(axis=0)
        return df

    def preprocess_char(self, df):
        print("preprocessing data...")
        import string
        vocab_chars = string.ascii_lowercase + '0123456789 '
        vocab2ix_dict = {char:(ix+1) for ix, char in enumerate(vocab_chars)}
        vocab_length = len(vocab_chars) + 1
        def sentence2onehot(sentence, vocab2ix_dict = vocab2ix_dict):
            # translate sentence string into indices
            sentence_ix = [vocab2ix_dict[x] for x in list(sentence) if x in vocab_chars]
            # Pad or crop to embedding dimension
            sentence_ix = (sentence_ix + [0]*self.embedding_dim)[0:self.embedding_dim]
            return(sentence_ix)
        self.train_x1 = np.matrix(df.question1.str.lower().apply(sentence2onehot).tolist())
        self.train_x2 = np.matrix(df.question2.str.lower().apply(sentence2onehot).tolist())
        if self.embedding_matrix is None:
            self.embedding_matrix = tf.diag(tf.ones(shape=[self.embedding_dim]))
        if self.contrastive:
            self.train_labels = np.array(df.is_duplicate)
            return
        else:
            labels_idx = np.array(df.is_duplicate)
            self.train_labels = np.zeros((labels_idx.shape[0], 2))
            for i, x in enumerate(labels_idx):
                self.train_labels[i, int(x)] = 1
            return
    
    def preprocess_word2vec(self, df, save_embedding=False, save_train_data=False):
        print("preprocessing data...")

        

        if self.train_x1 is None or self.train_x2 is None:
            tk = text.Tokenizer(num_words=200000)
            tk.fit_on_texts(list(df.question1.values) + list(df.question2.values.astype(str)))
            word_index = tk.word_index
            train_x1 = tk.texts_to_sequences(df.question1.values)
            self.train_x1 = sequence.pad_sequences(train_x1, maxlen=self.embedding_dim)
            train_x2 = tk.texts_to_sequences(df.question2.values.astype(str))
            self.train_x2 = sequence.pad_sequences(train_x2, maxlen=self.embedding_dim)
        
        if save_train_data:
            print("saving preprocessed training data...")
            np.save("train_x1.npy", self.train_x1)
            np.save("train_x2.npy", self.train_x2)

        if self.embedding_matrix is None:
            if self.train_x1 is not None or self.train_x2 is not None:
                tk = text.Tokenizer(num_words=200000)
                tk.fit_on_texts(list(df.question1.values) + list(df.question2.values.astype(str)))
                word_index = tk.word_index
            print("downloading word2vec...")
            word2vec = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
            num_words = min(200000, len(word_index)+1)
            print("populating embedding matrix...")
            self.embedding_matrix = np.zeros((num_words, self.embedding_dim))
            for word, i in word_index.items():
                if word in word2vec.vocab:
                    self.embedding_matrix[i] = word2vec.word_vec(word)
            if save_embedding:
                print("saving embedding matrix...")
                np.save("embedding.npy", self.embedding_matrix)
        if self.contrastive:
            self.train_labels = np.array(df.is_duplicate)
            return
        else:
            labels_idx = np.array(df.is_duplicate)
            self.train_labels = np.zeros((labels_idx.shape[0], 2))
            for i, x in enumerate(labels_idx):
                self.train_labels[i, int(x)] = 1
            return

    def subsample(self, n_train_samples, n_validation_samples):
        print("subsampling data...")
        train_size = self.train_x1.shape[0]
        global_idx = np.random.choice(train_size, n_train_samples + n_validation_samples, replace=False)
        np.random.shuffle(global_idx)
        train_sample_idx = global_idx[:n_train_samples]
        validation_sample_idx = global_idx[n_train_samples:]
        self.valid_x1 = self.train_x1[validation_sample_idx, :self.embedding_dim]
        self.valid_x2 = self.train_x2[validation_sample_idx, :self.embedding_dim]
        self.train_x1 = self.train_x1[train_sample_idx, :self.embedding_dim]
        self.train_x2 = self.train_x2[train_sample_idx, :self.embedding_dim]
        if self.contrastive:
            self.valid_labels = self.train_labels[validation_sample_idx]
            self.train_labels = self.train_labels[train_sample_idx]
        else:
            self.valid_labels = self.train_labels[validation_sample_idx,:]
            self.train_labels = self.train_labels[train_sample_idx, :]

        
        
    def batch_generator(self, batch_size):
            l = self.train_x1.shape[0]
            if self.contrastive:
                for ndx in range(0, l, batch_size):
                    yield (self.train_x1[ndx:min(ndx + batch_size, l), :],
                        self.train_x2[ndx:min(ndx + batch_size, l), :],
                        self.train_labels[ndx:min(ndx + batch_size, l)],
                        self.valid_x1[ndx:min(ndx + batch_size, l), :],
                        self.valid_x2[ndx:min(ndx + batch_size, l), :],
                        self.valid_labels[ndx:min(ndx + batch_size, l)],
                        )
            else:
                for ndx in range(0, l, batch_size):
                    yield (self.train_x1[ndx:min(ndx + batch_size, l), :],
                        self.train_x2[ndx:min(ndx + batch_size, l), :],
                        self.train_labels[ndx:min(ndx + batch_size, l),:],
                        self.valid_x1[ndx:min(ndx + batch_size, l), :],
                        self.valid_x2[ndx:min(ndx + batch_size, l), :],
                        self.valid_labels[ndx:min(ndx + batch_size, l),:],
                        )
                        
    def run(self, train_csv, n_train_samples=400000, n_validation_samples=10000, embedding_matrix=None, embedding_dim=80, train_x1=None, train_x2=None, save_embedding=False, save_train_data=False, contrastive=False):
        df = self.import_data(train_csv)
        self.contrastive = contrastive
        if embedding_matrix is not None:
            print("loading embedding matrix from %s" % embedding_matrix)
            self.embedding_matrix = np.load(embedding_matrix)
        self.embedding_dim = embedding_dim
        if train_x1 is not None:
            print("loading train_x1 from %s" % train_x1)
            self.train_x1 = np.load(train_x1)
        if train_x2 is not None:
            print("loading train_x2 from %s" % train_x2)
            self.train_x2 = np.load(train_x2)
        self.preprocess_word2vec(df, save_embedding, save_train_data)
        self.subsample(n_train_samples, n_validation_samples)
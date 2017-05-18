import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm

class Data(object):
    def __init__(self):
        self.train_x1 = None
        self.train_x2 = None
        self.train_labels = None
        self.valid_x1 = None
        self.valid_x2 = None
        self.valid_labels = None

    def import_data(self, train_csv):
        print("importing data...")
        df = pd.read_csv(train_csv)
        df = df.dropna(axis=0)
        return df

    def preprocess(self, df):
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
        self.train_labels = np.array(df.is_duplicate)

    def subsample(self, n_train_samples, n_validation_samples):
        print("subsampling data...")
        train_size = self.train_x1.shape[0]
        global_idx = np.random.choice(train_size, n_train_samples + n_validation_samples, replace=False)
        np.random.shuffle(global_idx)
        train_sample_idx = global_idx[:n_train_samples]
        validation_sample_idx = global_idx[n_train_samples:]
        self.valid_x1 = self.train_x1[validation_sample_idx, :self.embedding_dim]
        self.valid_x2 = self.train_x2[validation_sample_idx, :self.embedding_dim]
        self.valid_labels = self.train_labels[validation_sample_idx]
        self.train_x1 = self.train_x1[train_sample_idx, :self.embedding_dim]
        self.train_x2 = self.train_x2[train_sample_idx, :self.embedding_dim]
        self.train_labels = self.train_labels[train_sample_idx]
        
        
    def batch_generator(self, batch_size):
            l = self.train_x1.shape[0]
            for ndx in range(0, l, batch_size):
                yield (self.train_x1[ndx:min(ndx + batch_size, l), :],
                       self.train_x2[ndx:min(ndx + batch_size, l), :],
                       self.train_labels[ndx:min(ndx + batch_size, l)],
                       self.valid_x1[ndx:min(ndx + batch_size, l), :],
                       self.valid_x2[ndx:min(ndx + batch_size, l), :],
                       self.valid_labels[ndx:min(ndx + batch_size, l)],
                       )
    def run(self, train_csv, n_train_samples=400000, n_validation_samples=10000, embedding_dim=80, save = False):
        df = self.import_data(train_csv)
        self.embedding_dim = embedding_dim
        self.preprocess(df)
        self.subsample(n_train_samples, n_validation_samples)
        

class Architecture(object):
    def __init__(self, graph, embedding_dim):
        self.graph = graph
        self.x1 = tf.placeholder(dtype=tf.int32, shape=(None, embedding_dim), name='x1')
        self.x2 = tf.placeholder(dtype=tf.int32, shape=(None, embedding_dim), name='x2')
        self.q1_repr = tf.placeholder(dtype=tf.float32, shape=(embedding_dim, 300), name='x2')
        self.q2_repr = tf.placeholder(dtype=tf.float32, shape=(embedding_dim, 300), name='x2')

    def embed(self, x, embedding_dim):
        return tf.nn.embedding_lookup(tf.diag(tf.ones(shape=[embedding_dim])), x)

    def rnn_temporal_split(self, x, num_steps):
        embed_split = tf.split(axis=1, num_or_size_splits=num_steps, value=x)
        embed_split = [tf.squeeze(x, axis=[1]) for x in embed_split]
        return embed_split

    def stacked_biRNN(self, x, num_steps, cell_type, n_layers, lstm_size):
        xs = self.rnn_temporal_split(x, num_steps)
        fw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)], 
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        bw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)], 
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        fw_cells = [fw_cell(lstm_size) for fw_cell in fw_cells]
        bw_cells = [bw_cell(lstm_size) for bw_cell in bw_cells]
        fw_stack = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_stack = tf.contrib.rnn.MultiRNNCell(bw_cells)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_stack,
                                                                bw_stack,
                                                                xs,
                                                                dtype=tf.float32)
        final_state = outputs[-1]
        return final_state

    def biRNN(self, x, num_steps, cell_type, lstm_size):
        xs = self.rnn_temporal_split(x, num_steps)
        fw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse = None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse = None)}[cell_type]
        bw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse = None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse = None)}[cell_type]
        fw = fw_cell_unit(lstm_size)
        bw = bw_cell_unit(lstm_size)
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw,
                                                                bw,
                                                                xs,
                                                                dtype=tf.float32)
        final_state = outputs[-1]
        return final_state

    def rnn(self, x, num_steps, cell_type, lstm_size):
        xs = self.rnn_temporal_split(x, num_steps)
        fw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse=None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse=None)}[cell_type]
        fw = fw_cell_unit(lstm_size, reuse=None)
        outputs, _, _ = tf.contrib.rnn.static_rnn(fw, xs, dtype=tf.float32)
        final_state = outputs[-1]
        return final_state
    
    def dense_unit(self, x, name, input_size, output_size):
        bn = tf.nn.batch_normalization(x, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        W1 = tf.get_variable(name="W"+name, shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
        h1 = tf.nn.relu(tf.matmul(bn, W1))
        return h1

    def siamese_fc_network(self):
        x1_embed = self.embed(self.x1, 40)
        x2_embed = self.embed(self.x2, 40)
        with tf.variable_scope("x1", reuse=None) as scope:  
            self.q1_repr = self.biRNN(x1_embed, 80, "LSTM", 300)
        with tf.variable_scope("x2", reuse=None) as scope:  
            self.q2_repr = self.biRNN(x2_embed, 80, "LSTM", 300)
        with tf.variable_scope("output", reuse=None) as scope:
            h4 = self.dense_unit(x=tf.concat([self.q1_repr, self.q2_repr], axis=1), name="h4", input_size=300*4, output_size=50)
            h5 = self.dense_unit(x=h4, name="h5", input_size=50, output_size=50)
            h6 = self.dense_unit(x=h5, name="h6", input_size=50, output_size=50)
            h7 = self.dense_unit(x=h6, name="h7", input_size=50, output_size=50)
            h8 = self.dense_unit(x=h7, name="h8", input_size=50, output_size=50)
            output = self.dense_unit(x=h8, name="output", input_size=50, output_size=2)
        return output
    
    def siamese_network(self):
        x1_embed = self.embed(self.x1, 40)
        x2_embed = self.embed(self.x2, 40)
        with tf.variable_scope("x1", reuse=None) as scope:  
            q1_repr = self.biRNN(x1_embed, 80, "LSTM", 300)
        with tf.variable_scope("x2", reuse=None) as scope:  
            q2_repr = self.biRNN(x2_embed, 80, "LSTM", 300)
        return q1_repr, q2_repr



class ContrastiveLoss(object):

    def __init__(self):
        self.labels =  tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')

    def manhattan(self, q1_repr, q2_repr, margin):
        labels_t = tf.to_float(self.labels)
        labels_f = tf.subtract(1.0, tf.to_float(self.labels), name="1-yi")          
        manhattan = tf.squeeze(tf.norm(q1_repr - q2_repr, ord=1, axis=1, keep_dims=True))
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, manhattan, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, manhattan)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = 0.5*tf.reduce_mean(losses, name="loss")
        tf.summary.scalar('contrastive_loss', loss)
        return loss
    
    def euclidean(self, q1_repr, q2_repr, margin):
        labels_t = tf.to_float(self.labels)
        labels_f = tf.subtract(1.0, tf.to_float(self.labels), name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(q1_repr, q2_repr), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = 0.5*tf.reduce_mean(losses, name="loss")
        tf.summary.scalar('contrastive_loss', loss)
        return loss

class Optimization(object):

    def adam(self, loss, lr):
        return tf.train.AdamOptimizer(lr).minimize(loss)

class Build(object):
    def __init__(self, graph):
        self.architecture = Architecture(graph, embedding_dim=80)
        self.loss = ContrastiveLoss()
        self.opt = Optimization()

    def build_siamese(self, graph):
        q1_repr, q2_repr = self.architecture.siamese_network()
        loss = self.loss.manhattan(q1_repr, q2_repr, 1.0)
        opt = self.opt.adam(loss, 0.001)
        return q1_repr, q2_repr, loss, opt

class Display(object):
    def log_train_loss(self, epoch, batch_idx, batch_train_loss):
        tqdm.write("EPOCH: %s, BATCH: %s, LOSS: %s" % (epoch, batch_idx, batch_train_loss))
    def done(self):
        print("Done!")

class Run(object):
        
    def run_siamese(self, train_csv):
        data = Data()
        display = Display()
        data.run(train_csv, n_train_samples=50, n_validation_samples=10, embedding_dim=80, save=False)
        with tf.Graph().as_default() as graph:
           model = Build(data)
           q1_repr, q2_repr, loss, opt = model.build_siamese(graph)
           init = tf.global_variables_initializer()
           with tf.Session(graph=graph) as sess:
               sess.run(init)
               for epoch in range(10):
                 train_iter_ = data.batch_generator(100)  
                 for batch_idx, batch in enumerate(tqdm(train_iter_)):
                    train_x1_batch, train_x2_batch, train_labels_batch, valid_x1_batch, valid_x2_batch, valid_labels_batch = batch
                    _, _, batch_train_loss, _= sess.run([q1_repr, q2_repr, loss, opt], feed_dict={
                                                                                                  model.architecture.x1 : train_x1_batch,
                                                                                                  model.architecture.x2 : train_x2_batch,
                                                                                                  model.loss.labels : train_labels_batch
                                                                                                 })
                    display.log_train_loss(epoch, batch_idx, batch_train_loss)
        display.done()


if __name__ == '__main__':
    Run().run_siamese('train.csv')
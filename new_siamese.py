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
        labels_idx = np.array(df.is_duplicate)
        self.train_labels = np.zeros((labels_idx.shape[0], 2))
        for i, x in enumerate(labels_idx):
            self.train_labels[i, int(x)] = 1

    def subsample(self, n_train_samples, n_validation_samples):
        print("subsampling data...")
        train_size = self.train_x1.shape[0]
        global_idx = np.random.choice(train_size, n_train_samples + n_validation_samples, replace=False)
        np.random.shuffle(global_idx)
        train_sample_idx = global_idx[:n_train_samples]
        validation_sample_idx = global_idx[n_train_samples:]
        self.valid_x1 = self.train_x1[validation_sample_idx, :self.embedding_dim]
        self.valid_x2 = self.train_x2[validation_sample_idx, :self.embedding_dim]
        self.valid_labels = self.train_labels[validation_sample_idx,:]
        self.train_x1 = self.train_x1[train_sample_idx, :self.embedding_dim]
        self.train_x2 = self.train_x2[train_sample_idx, :self.embedding_dim]
        self.train_labels = self.train_labels[train_sample_idx, :]
        
        
    def batch_generator(self, batch_size):
            l = self.train_x1.shape[0]
            for ndx in range(0, l, batch_size):
                yield (self.train_x1[ndx:min(ndx + batch_size, l), :],
                       self.train_x2[ndx:min(ndx + batch_size, l), :],
                       self.train_labels[ndx:min(ndx + batch_size, l),:],
                       self.valid_x1[ndx:min(ndx + batch_size, l), :],
                       self.valid_x2[ndx:min(ndx + batch_size, l), :],
                       self.valid_labels[ndx:min(ndx + batch_size, l),:],
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
       
    def embed(self, input, embedding_dim):
        return tf.nn.embedding_lookup(tf.diag(tf.ones(shape=[embedding_dim])), input)

    def rnn_temporal_split(self, input, num_steps):
        embed_split = tf.split(axis=1, num_or_size_splits=num_steps, value=input)
        embed_split = [tf.squeeze(x, axis=[1]) for x in embed_split]
        return embed_split

    def stacked_biRNN(self, input, num_steps, cell_type, n_layers, network_dim):
        xs = self.rnn_temporal_split(input, num_steps)
        fw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)], 
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        bw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)], 
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        fw_cells = [fw_cell(network_dim) for fw_cell in fw_cells]
        bw_cells = [bw_cell(network_dim) for bw_cell in bw_cells]
        fw_stack = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_stack = tf.contrib.rnn.MultiRNNCell(bw_cells)
        outputs, fw_output_state, bw_output_state = tf.contrib.rnn.static_bidirectional_rnn(fw_stack,
                                                                bw_stack,
                                                                xs,
                                                                dtype=tf.float32)
        
        return outputs, fw_output_state, bw_output_state  

    def biRNN(self, input, num_steps, cell_type, network_dim):
        xs = self.rnn_temporal_split(input, num_steps)
        fw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse = None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse = None)}[cell_type]
        bw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse = None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse = None)}[cell_type]
        fw = fw_cell_unit(network_dim)
        bw = bw_cell_unit(network_dim)
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw,
                                                                bw,
                                                                xs,
                                                                dtype=tf.float32)
        return outputs, output_state_fw, output_state_bw

    def rnn(self, input, num_steps, cell_type, network_dim):
        xs = self.rnn_temporal_split(input, num_steps)
        fw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse=None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse=None)}[cell_type]
        fw = fw_cell_unit(network_dim, reuse=None)
        outputs, _, _ = tf.contrib.rnn.static_rnn(fw, xs, dtype=tf.float32)
        final_state = outputs[-1]
        return final_state
    
    def matchLayer(self, x1_outputs, x1_fw_final, x1_bw_final, x2_outputs, x2_fw_final, x2_bw_final, weight_dim):
        def cosine_similarity(x1, x2):
             x1_norm = tf.nn.l2_normalize(x1, dim=1)
             x2_norm = tf.nn.l2_normalize(x2, dim=1)
             return tf.matmul(x1_norm, tf.transpose(x2_norm, [1, 0]))
        W1 = tf.get_variable(name="W1_match", shape=[160, 200], initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable(name="W2_match", shape=[160, 200], initializer=tf.contrib.layers.xavier_initializer())
        m1 = [cosine_similarity(tf.multiply(W1,x1_out), tf.multiply(W2 ,x1_outputs[-1])) for x1_out in x1_outputs]
        m2 = [cosine_similarity(tf.multiply(W1,x2_out), tf.multiply(W2,x2_outputs[-1])) for x2_out in x2_outputs]
        return m1, m2

    def dense_unit(self, input, name, input_dim, hidden_dim, output_dim):
        bn = tf.nn.batch_normalization(input, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        W1 = tf.get_variable(name="W1_"+name, shape=[input_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="b1_"+name, shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        h1 = tf.nn.relu(tf.matmul(bn, W1) + b1)
        W2 = tf.get_variable(name="W2_"+name, shape=[hidden_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="b2_"+name, shape=[output_dim], initializer=tf.contrib.layers.xavier_initializer())
        out = tf.matmul(h1, W2) + b2
        bn_out = tf.nn.batch_normalization(out, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        return bn_out
    
    def siamese_stacked_fc_network(self):
        embed = self.embed(tf.concat([self.x1, self.x2], axis=1), embedding_dim=80*2)
        with tf.variable_scope("x1", reuse=None) as scope:  
            repr, _, _ = self.stacked_biRNN(input=embed, num_steps=80*2, cell_type="LSTM", n_layers=3, network_dim=300)
        with tf.variable_scope("output", reuse=None) as scope:
            h4 = self.dense_unit(input=repr[-1], name="h4", input_dim=300*2, hidden_dim=300, output_dim=300)
            h5 = self.dense_unit(input=h4, name="h5", input_dim=300, hidden_dim=300, output_dim=300)
            h6 = self.dense_unit(input=h5, name="h6", input_dim=300, hidden_dim=300, output_dim=300)
            h7 = self.dense_unit(input=h6, name="h7", input_dim=300, hidden_dim=300, output_dim=300)
            h8 = self.dense_unit(input=h7, name="h8", input_dim=300, hidden_dim=300, output_dim=300)
            output = self.dense_unit(input=h8, name="output", input_dim=300, hidden_dim=300,  output_dim=2)
        return output

    def siamese_fc_network(self):
        x1_embed = self.embed(self.x1, embedding_dim=80)
        x2_embed = self.embed(self.x2, embedding_dim=80)
        with tf.variable_scope("x1", reuse=None) as scope:  
            q1_repr, _, _ = self.biRNN(input=x1_embed, num_steps=80, cell_type="LSTM", network_dim=512)
        with tf.variable_scope("x2", reuse=None) as scope:  
            q2_repr, _, _ = self.biRNN(input=x2_embed, num_steps=80, cell_type="LSTM", network_dim=512)
        with tf.variable_scope("output", reuse=None) as scope:
            h4 = self.dense_unit(input=tf.concat([q1_repr[-1], q2_repr[-1]], axis=1), name="h4", input_dim=512*4, hidden_dim=300, output_dim=300)
            h5 = self.dense_unit(input=h4, name="h5", input_dim=300, hidden_dim=300, output_dim=300)
            h6 = self.dense_unit(input=h5, name="h6", input_dim=300, hidden_dim=300, output_dim=300)
            h7 = self.dense_unit(input=h6, name="h7", input_dim=300, hidden_dim=300, output_dim=300)
            h8 = self.dense_unit(input=h7, name="h8", input_dim=300, hidden_dim=300, output_dim=300)
            output = self.dense_unit(input=h8, name="output", input_dim=300, hidden_dim=300, output_dim=2)
        return output
    
    def siamese_network(self):
        x1_embed = self.embed(self.x1, embedding_dim=80)
        x2_embed = self.embed(self.x2, embedding_dim=80)
        with tf.variable_scope("x1", reuse=None) as scope:  
            q1_repr, _, _ = self.biRNN(input=x1_embed, num_steps=80, cell_type="LSTM", network_dim=512)
        with tf.variable_scope("x2", reuse=None) as scope:  
            q2_repr, _, _ = self.biRNN(input=x2_embed, num_steps=80, cell_type="LSTM", network_dim=512)
        with tf.variable_scope("output", reuse=None) as scope:
            output = self.dense_unit(input=tf.concat([q1_repr[-1], q2_repr[-1]], axis=1), name="h4", input_dim=512*4,  hidden_dim=300, output_dim=2)
        return output
    
    def match_network(self):
        x1_embed = self.embed(self.x1, embedding_dim=80)
        x2_embed = self.embed(self.x2, embedding_dim=80)
        with tf.variable_scope("x1", reuse=None) as scope:  
            x1_outputs, x1_fw_final, x1_bw_final = self.biRNN(input=x1_embed, num_steps=80, cell_type="LSTM", network_dim=100)
        with tf.variable_scope("x2", reuse=None) as scope:  
            x2_outputs, x2_fw_final, x2_bw_final = self.biRNN(input=x2_embed, num_steps=80, cell_type="LSTM", network_dim=100)
        with tf.variable_scope("match", reuse=None) as scope:
            m1, m2 = self.matchLayer(x1_outputs=x1_outputs, x1_fw_final=x1_fw_final, x1_bw_final=x1_bw_final, x2_outputs=x2_outputs, x2_fw_final=x2_fw_final, x2_bw_final=x2_bw_final,  weight_dim=512)
        with tf.variable_scope("agg_x1", reuse=None) as scope:  
            m1_agg, _, _ = self.biRNN(input=m1, num_steps=100, cell_type="LSTM", network_dim=100)
        with tf.variable_scope("agg_x2", reuse=None) as scope:  
            m2_agg, _, _ = self.biRNN(input=m2, num_steps=100, cell_type="LSTM", network_dim=100)
        with tf.variable_scope("output", reuse=None) as scope:      
            output = self.dense_unit(input=tf.concat([m1_agg[-1], m2_agg[-1]], axis=1), name="h4", input_dim=100*4,  hidden_dim=128, output_dim=2)
        return output

    def merge_siamese_network(self):
        embed = self.embed(tf.concat([self.x1, self.x2], axis = 1), embedding_dim=160)
        with tf.variable_scope("x1", reuse=None) as scope:  
            repr, _, _ = self.biRNN(input=embed, num_steps=160, cell_type="LSTM", network_dim=512)
        with tf.variable_scope("output", reuse=None) as scope:
            output = self.dense_unit(input=repr[-1], name="h4", input_dim=512*2,  hidden_dim=128, output_dim=2)
        return output



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

class SoftmaxLoss(object):
    
    def __init__(self):
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None, 2), name='labels')
    
    def cross_entropy(self, logits):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = self.labels)
        loss = tf.reduce_mean(losses, name="loss")
        tf.summary.scalar('cross_entropy_loss', loss)
        return loss

class Accuracy(object):
    def softmax_accuracy(self, labels, output):
        pred = tf.nn.softmax(output)
        correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

class Optimization(object):

    def adam(self, loss, lr):
        return tf.train.AdamOptimizer(lr).minimize(loss)

class Build(object):
    def __init__(self, graph):
        self.architecture = Architecture(graph, embedding_dim=80)
        self.loss = SoftmaxLoss()
        self.opt = Optimization()
        self.accuracy = Accuracy()
    
    def build_siamese_stacked_fc(self, graph):
        print("building siamese_stacked_fc network...")
        output = self.architecture.siamese_stacked_fc_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.softmax_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged

    def build_siamese_fc(self, graph):
        print("building siamese_fc network...")
        output = self.architecture.siamese_fc_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.softmax_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged

    def build_siamese(self, graph):
        print("building siamese network...")
        output = self.architecture.siamese_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.softmax_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged
    
    def build_match(self, graph):
        print("building match network...")
        output = self.architecture.match_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.softmax_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged

    def build_merge_siamese(self, graph):
        print("building merge siamese network...")
        output = self.architecture.merge_siamese_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.softmax_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged


class Display(object):
    def log_train_loss(self, epoch, batch_idx, batch_train_loss, batch_train_accuracy):
        tqdm.write("EPOCH: %s, BATCH: %s, TRAIN LOSS: %s, TRAIN ACCURACY: %s" % (epoch, batch_idx, batch_train_loss, batch_train_accuracy))
    def log_validation_loss(self, epoch, batch_idx, batch_valid_loss, batch_valid_accuracy):
        tqdm.write("EPOCH: %s, BATCH: %s, VALIDATION LOSS: %s, VALIDATION ACCURACY: %s" % (epoch, batch_idx, batch_valid_loss, batch_valid_accuracy))
    def done(self):
        print("Done!")

class TensorBoard(object):
    def __init__(self, graph, logdir):
        self.writer = tf.summary.FileWriter(logdir,graph)
        self.logdir = logdir

    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    
class Run(object):
        
    def run_siamese(self, train_csv, config):
        data = Data()
        display = Display()
        data.run(train_csv, n_train_samples=config.n_train_samples, n_validation_samples=config.n_validation_samples, embedding_dim=config.embedding_dim, save=False)
        with tf.Graph().as_default() as graph:
           model = Build(data)
           writer = TensorBoard(graph=graph, logdir=config.logdir).writer
           output, loss, accuracy, opt, merged = model.build_merge_siamese(graph)
           init = tf.global_variables_initializer()
           with tf.Session(graph=graph) as sess:
               sess.run(init)
               for epoch in range(10):
                 train_iter_ = data.batch_generator(100)  
                 for batch_idx, batch in enumerate(tqdm(train_iter_)):
                    train_x1_batch, train_x2_batch, train_labels_batch, valid_x1_batch, valid_x2_batch, valid_labels_batch = batch
                    _, batch_train_loss, batch_train_accuracy, _, summary = sess.run([output, loss, accuracy, opt, merged], 
                                                                                    feed_dict={
                                                                                                model.architecture.x1 : train_x1_batch,
                                                                                                model.architecture.x2 : train_x2_batch,
                                                                                                model.loss.labels : train_labels_batch
                                                                                              })
                    batch_valid_loss, batch_valid_accuracy = sess.run([loss, accuracy], feed_dict={
                                                                                                    model.architecture.x1 : valid_x1_batch,
                                                                                                    model.architecture.x2 : valid_x2_batch,
                                                                                                    model.loss.labels : valid_labels_batch
                                                                                                    })
                    display.log_validation_loss(epoch, batch_idx, batch_valid_loss, batch_valid_accuracy)
                    display.log_train_loss(epoch, batch_idx, batch_train_loss, batch_train_accuracy)
                    writer.add_summary(summary, batch_idx)
        display.done()

class Config(object):
    def __init__(self, n_train_samples, n_validation_samples, embedding_dim, logdir):
        self.n_train_samples = n_train_samples
        self.n_validation_samples = n_validation_samples
        self.embedding_dim =embedding_dim
        self.logdir = logdir

if __name__ == '__main__':
    config = Config(n_train_samples=8500, n_validation_samples=1000, embedding_dim=80, logdir="/tmp/quora_logs/siamese_fc_stacked")
    Run().run_siamese('file.csv', config)



## TODO:
## * add word2vec/glove functionality
## * add tensorboard optional
## * make it easier to change dimensions
## * cycle validation set
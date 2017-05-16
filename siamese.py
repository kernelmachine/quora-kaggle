import tensorflow as tf
import pandas as pd
from tensorflow.contrib import learn
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
from nltk.corpus import stopwords
import re

BATCH_SIZE = 2000
VALIDATION_BATCH_SIZE = 100
EMBEDDING_DIM = 80
VOCAB_SIZE = 200000
N_SAMPLES = 298000
N_SICK_SAMPLES = 9800
LSTM_SIZE = 50
N_CLASSES = 2
N_EPOCHS = 50
N_VALIDATION = 10000
class Siamese(object):

    def __init__(self):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=(None, EMBEDDING_DIM), name='x1')
        self.x2 = tf.placeholder(dtype=tf.int32, shape=(None, EMBEDDING_DIM), name='x2')
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')

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

    def create_feed_dict(self, x1, x2, labels):
        return {self.x1 : x2,
                self.x2 : x2,
                self.labels : labels}

    def build_model(self, graph):
        def biRNN(x):
            n_layers = 3
            embed = tf.nn.embedding_lookup(tf.diag(tf.ones(shape=[EMBEDDING_DIM])), x)
            embed_split = tf.split(axis=1, num_or_size_splits=EMBEDDING_DIM, value=embed)
            embed_split = [tf.squeeze(x, axis=[1]) for x in embed_split]
            fw_cell_unit = tf.contrib.rnn.BasicLSTMCell#tf.nn.rnn_cell.BasicLSTMCell
            bw_cell_unit = tf.contrib.rnn.BasicLSTMCell#tf.nn.rnn_cell.BasicLSTMCell

            fw = fw_cell_unit(LSTM_SIZE, reuse=None)

            bw = bw_cell_unit(LSTM_SIZE, reuse=None)
            fw_stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, reuse=None) for _ in range(3)])
            bw_stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, reuse=None) for _ in range(3)])
            # Forward direction cell
            # Backward direction cell

            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_stack,
                                                                    bw_stack,
                                                                    embed_split,
                                                                    dtype=tf.float32)

            # temporal_mean = tf.add_n(outputs) / EMBEDDING_DIM
            # Ws = tf.get_variable(name='Ws', shape=[2*LSTM_SIZE, 1], initializer=tf.contrib.layers.xavier_initializer())
            # self.variable_summaries(Ws)
            # bs = tf.get_variable(name='bs', shape=[1], initializer=tf.random_normal_initializer(stddev=0.1))
            # self.variable_summaries(bs)
            # final_output = tf.matmul(temporal_mean, Ws) + bs
            # tf.summary.histogram('pre_activations', final_output)
            # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
            final_state = outputs[-1]
            return final_state
        with tf.variable_scope("x1", reuse=None) as scope:
            q1_repr = biRNN(self.x1)
        with tf.variable_scope("x2", reuse=None) as scope:
            q2_repr = biRNN(self.x2)

        # scores = tf.exp(-tf.squeeze(tf.norm(q1_repr - q2_repr, ord=1, axis=1, keep_dims=True)))

        def loss_with_step(q1_repr, q2_repr):
            margin = 5.0
            labels_t = tf.to_float(self.labels)
            labels_f = tf.subtract(1.0, tf.to_float(self.labels), name="1-yi")          # labels_ = !labels;
            manhattan = tf.squeeze(tf.norm(q1_repr - q2_repr, ord=1, axis=1, keep_dims=True))
            C = tf.constant(margin, name="C")
            pos = tf.multiply(labels_t, manhattan, name="y_x_eucd")
            neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, manhattan)), name="Ny_C-eucd")
            losses = tf.add(pos, neg, name="losses")
            loss = 0.5*tf.reduce_mean(losses, name="loss")
            return loss
        loss = loss_with_step(q1_repr, q2_repr)
        # loss = tf.to_float(self.labels) * tf.square(scores) + (1.0 - tf.to_float(self.labels)) * tf.square(tf.maximum((1.0 - scores), 0.0))
        # loss = 0.5*tf.reduce_mean(loss)
        tf.summary.scalar('contrastive_loss', loss)
        train_opt = tf.train.AdamOptimizer().minimize(loss)
        merged = tf.summary.merge_all()
        return loss, train_opt, merged, q1_repr, q2_repr

def preprocess(df):
    import string
    def clean_text(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))

        # # Clean the text
        # text = re.sub(r"[^A-Za-z0-9]", " ", text)
        # text = re.sub(r"what's", "", text)
        # text = re.sub(r"What's", "", text)
        # text = re.sub(r"\'s", " ", text)
        # text = re.sub(r"\'ve", " have ", text)
        # text = re.sub(r"can't", "cannot ", text)
        # text = re.sub(r"n't", " not ", text)
        # text = re.sub(r"I'm", "i am", text)
        # text = re.sub(r" m ", " am ", text)
        # text = re.sub(r"\'re", " are ", text)
        # text = re.sub(r"\'d", " would ", text)
        # text = re.sub(r"\'ll", " will ", text)
        # text = re.sub(r"60k", " 60000 ", text)
        # text = re.sub(r" e g ", " eg ", text)
        # text = re.sub(r" b g ", " bg ", text)
        # text = re.sub(r"\0s", "0", text)
        # text = re.sub(r" 9 11 ", "911", text)
        # text = re.sub(r"e-mail", "email", text)
        # text = re.sub(r"\s{2,}", " ", text)
        # text = re.sub(r"quikly", "quickly", text)
        # text = re.sub(r" usa ", " america ", text)
        # text = re.sub(r" USA ", " america ", text)
        # text = re.sub(r" u s ", " america ", text)
        # text = re.sub(r" uk ", " england ", text)
        # text = re.sub(r" UK ", " england ", text)
        # text = re.sub(r"imrovement", "improvement", text)
        # text = re.sub(r"intially", "initially", text)
        # text = re.sub(r" dms ", "direct messages ", text)
        # text = re.sub(r"demonitization", "demonetization", text)
        # text = re.sub(r"actived", "active", text)
        # text = re.sub(r"kms", " kilometers ", text)
        # text = re.sub(r"KMs", " kilometers ", text)
        # text = re.sub(r" cs ", " computer science ", text)
        # text = re.sub(r" upvotes ", " up votes ", text)
        # text = re.sub(r" iPhone ", " phone ", text)
        # text = re.sub(r"\0rs ", " rs ", text)
        # text = re.sub(r"calender", "calendar", text)
        # text = re.sub(r"ios", "operating system", text)
        # text = re.sub(r"programing", "programming", text)
        # text = re.sub(r"bestfriend", "best friend", text)
        # text = re.sub(r"III", "3", text)
        # text = re.sub(r"the US", "america", text)

        # Optionally, remove stop words
        if remove_stop_words:
            text = text.split()
            text = [w for w in text if not w in stop_words]
            text = " ".join(text)

        # Optionally, shorten words to their stems
        if stem_words:
            from nltk.stem.snowball import SnowballStemmer
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        # Return a list of words
        return(text)

    vocab_chars = string.ascii_lowercase + '0123456789 '
    vocab2ix_dict = {char:(ix+1) for ix, char in enumerate(vocab_chars)}
    vocab_length = len(vocab_chars) + 1
    def sentence2onehot(sentence, vocab2ix_dict = vocab2ix_dict, max_sentence_length = EMBEDDING_DIM):
        # translate sentence string into indices
        sentence = clean_text(sentence)
        sentence_ix = [vocab2ix_dict[x] for x in list(sentence) if x in vocab_chars]
        # Pad or crop to max_sentence_len
        sentence_ix = (sentence_ix + [0]*max_sentence_length)[0:max_sentence_length]
        return(sentence_ix)
    x1 = np.matrix(df.question1.str.lower().apply(sentence2onehot).tolist())
    x2 = np.matrix(df.question2.str.lower().apply(sentence2onehot).tolist())
    labels = np.array(df.is_duplicate)
    return x1, x2, labels

def batch_generator(x1, x2, labels, batch_size):
        l = x1.shape[0]
        for ndx in range(0, l, batch_size):
            yield (x1[ndx:min(ndx + batch_size, l), :], x2[ndx:min(ndx + batch_size, l), :], labels[ndx:min(ndx + batch_size, l)])

if __name__=='__main__':
    print("reading training data...")
    # train = pd.read_csv("train.csv")
    # train = train.dropna(axis = 0)
    # train = train.groupby('is_duplicate').apply(lambda x: x.sample(n=149000))
    # # test = pd.read_csv("test.csv")
    # # test = test.dropna(axis = 0)
    # print("preprocessing training data...")
    # train_x1, train_x2, train_labels = preprocess(train)
    # np.save('train_x1', train_x1)
    # np.save('train_x2', train_x2)
    # np.save('train_labels', train_labels)
    # print("preprocessing sick data...")
    # sick = pd.read_csv("sick.csv")
    # sick_x1, sick_x2, sick_labels = preprocess(sick)
    # np.save('sick_x1', sick_x1)
    # np.save('sick_x2', sick_x2)
    # np.save('sick_labels', sick_labels)

    train_x1 = np.load('train_x1.npy')
    train_x2 = np.load('train_x2.npy')
    train_labels = np.load('train_labels.npy')

    sick_x1 = np.load('sick_x1.npy')
    sick_x2 = np.load('sick_x2.npy')
    sick_labels = np.load('sick_labels.npy')

    print("subsampling...")
    train_x1 = train_x1[np.random.choice(train_x1.shape[0], N_SAMPLES, replace=False), :EMBEDDING_DIM]
    train_x2 = train_x2[np.random.choice(train_x2.shape[0], N_SAMPLES, replace=False), :EMBEDDING_DIM]
    train_labels = train_labels[np.random.choice(train_labels.shape[0], N_SAMPLES, replace=False)]
    valid_x1 = train_x1[np.random.choice(train_x1.shape[0], N_VALIDATION, replace=False), :EMBEDDING_DIM]
    valid_x2 = train_x2[np.random.choice(train_x2.shape[0], N_VALIDATION, replace=False), :EMBEDDING_DIM]
    valid_labels = train_labels[np.random.choice(train_labels.shape[0], N_VALIDATION, replace=False)]

    sick_x1 = sick_x1[np.random.choice(sick_x1.shape[0], N_SICK_SAMPLES, replace=False), :EMBEDDING_DIM]
    sick_x2 = sick_x2[np.random.choice(sick_x2.shape[0], N_SICK_SAMPLES, replace=False), :EMBEDDING_DIM]
    sick_labels = sick_labels[np.random.choice(sick_labels.shape[0], N_SICK_SAMPLES, replace=False)]

    train_x1 = np.concatenate([sick_x1, train_x1], axis = 0)
    train_x2 = np.concatenate([sick_x2, train_x2], axis = 0)
    train_labels = np.concatenate([sick_labels, train_labels], axis = 0)
    print("running model...")
    with tf.Graph().as_default() as graph:
        siamese = Siamese()
        loss, train_opt, merged, q1_repr, q2_repr = siamese.build_model(graph)
        train_writer = tf.summary.FileWriter('/tmp/quora_logs' + '/train_7',
                                            graph)
        init = tf.global_variables_initializer()
        with tf.Session(graph=graph) as sess:
            sess.run(init)
            q1_reprs = []
            q2_reprs = []
            print("pre-training...")
        
            for i in range(N_EPOCHS):
                train_iter_ = batch_generator(train_x1,train_x2, train_labels, BATCH_SIZE)
                for ix, (train_x1_batch, train_x2_batch, train_labels_batch) in enumerate(tqdm(train_iter_, total=(N_SAMPLES + N_SICK_SAMPLES) / BATCH_SIZE)):
                    feed_dict = siamese.create_feed_dict(x1=train_x1_batch, x2=train_x2_batch, labels = train_labels_batch)
                    batch_train_loss, _, summary, q1_repr_val, q2_repr_val = sess.run([loss, train_opt, merged, q1_repr, q2_repr], feed_dict=feed_dict)
                    train_writer.add_summary(summary, ix)
                    q1_reprs.append(q1_repr_val)
                    q2_reprs.append(q2_repr_val)
                    if ix % 100 == 0:
                        tqdm.write("    EPOCH: %s, BATCH: %s, TRAIN CONTRASTIVE LOSS: %s" % (i, ix, batch_train_loss))
                    if ix % 500 == 0:
                        tqdm.write("RUNNING VALIDATION....")
                        valid_iter_ = batch_generator(valid_x1,valid_x2, valid_labels, VALIDATION_BATCH_SIZE)
                        valid_losses = []
                        for valid_x1_batch, valid_x2_batch, valid_labels_batch in valid_iter_:
                            valid_feed_dict = siamese.create_feed_dict(x1=valid_x1_batch, x2=valid_x2_batch, labels = valid_labels_batch)
                            batch_valid_loss = loss.eval(feed_dict=valid_feed_dict)
                            valid_losses.append(batch_valid_loss)
                        tqdm.write("        VALID BATCH: %s,  VALID CONTRASTIVE LOSS: %s" % (ix, sum(valid_losses)/float(len(valid_losses))))
            # test_iter_ = batch_generator(test_x1,test_x2, test_labels, BATCH_SIZE)
            # for ix, (test_x1_batch, test_x2_batch, test_labels_batch) in enumerate(tqdm(train_iter_, total=N_SAMPLES / BATCH_SIZE)):
            #     feed_dict = siamese.create_feed_dict(x1=test_x1_batch, x2=test_x2_batch, labels = train_labels_batch)
            print("Optimization Finished!")
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            import ipdb; ipdb.set_trace()
            reduced_q1 = PCA(n_components=2).fit_transform(np.matrix(q1_reprs))
            reduced_q2 = PCA(n_components=2).fit_transform(np.matrix(q2_reprs))
            plt.scatter(reduced_q1[:,0], reduced_q1[:,1], c = train_labels)
            plt.scatter(reduced_q2[:,0], reduced_q2[:,1], c = train_labels)

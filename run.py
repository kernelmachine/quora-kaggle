import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from lib.data import Data
from lib.display import Display, TensorBoard
from lib.build import BuildSiamese

class Config(object):
    def __init__(self, n_train_samples, n_validation_samples, embedding_dim, logdir, contrastive, save):
        self.n_train_samples = n_train_samples
        self.n_validation_samples = n_validation_samples
        self.embedding_dim =embedding_dim
        self.logdir = logdir
        self.contrastive = contrastive
        self.save = save

class Run(object):
        
    def run_siamese(self, train_csv, config):
        data = Data()
        display = Display()
        data.run(train_csv, 
                 n_train_samples=config.n_train_samples, 
                 n_validation_samples=config.n_validation_samples, 
                 embedding_dim=config.embedding_dim, 
                 contrastive=config.contrastive, 
                 save=config.save)
        with tf.Graph().as_default() as graph:
           model = BuildSiamese(data)
           writer = TensorBoard(graph=graph, logdir=config.logdir).writer
           output, loss, accuracy, opt, merged = model.build_siamese_stacked_fc(graph)
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
                                                                                                model.loss.labels : train_labels_batch,
                                                                                                model.architecture.embedding_matrix : data.embedding_matrix
                                                                                              })
                    batch_valid_loss, batch_valid_accuracy = sess.run([loss, accuracy], feed_dict={
                                                                                                    model.architecture.x1 : valid_x1_batch,
                                                                                                    model.architecture.x2 : valid_x2_batch,
                                                                                                    model.loss.labels : valid_labels_batch,
                                                                                                    model.architecture.embedding_matrix : data.embedding_matrix
                                                                                                    })
                    display.log_validation_loss(epoch, batch_idx, batch_valid_loss, batch_valid_accuracy)
                    display.log_train_loss(epoch, batch_idx, batch_train_loss, batch_train_accuracy)
                    writer.add_summary(summary, batch_idx)
        display.done()

if __name__ == '__main__':
    config = Config(n_train_samples=298000, 
                    n_validation_samples=10000,
                    embedding_dim=300, 
                    logdir="/tmp/quora_logs/siamese_fc_stacked", 
                    contrastive=False, 
                    save=False)
    Run().run_siamese('train.csv', config)



## TODO:
## * add word2vec/glove functionality
## * add tensorboard optional
## * make it easier to change dimensions
## * cycle validation set
import tensorflow as tf
from lib.architecture import Network
from lib.optimization import *

class BuildContrastiveSiamese(object):
    def __init__(self, graph):
        self.network = Network(graph, embedding_dim=300)
        self.loss = ContrastiveLoss()
        self.opt = Optimization()
        self.accuracy = Accuracy()
    
    def build_contrastive_siamese(self, graph):
        print("building constrastive siamese network...")
        output = self.network.contrastive_siamese_network()
        loss = self.loss.contrastive_loss(output, 5.0)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.distance_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged   


class BuildSiamese(object):
    def __init__(self, graph):
        self.network = Network(graph, embedding_dim=300)
        self.loss = SigmoidLoss()
        self.opt = Optimization()
        self.accuracy = Accuracy()
    
    def build_fc(self, graph):
        print("building siamese_stacked_fc network...")
        output = self.network.fc_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.sigmoid_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged

    def build_siamese_stacked_fc(self, graph):
        print("building siamese_stacked_fc network...")
        output = self.network.siamese_stacked_fc_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.sigmoid_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged

    def build_siamese_fc(self, graph):
        print("building siamese_fc network...")
        output = self.network.siamese_fc_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.sigmoid_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged

    def build_siamese(self, graph):
        print("building siamese network...")
        output = self.network.siamese_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.sigmoid_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged
    
    def build_match(self, graph):
        print("building match network...")
        output = self.network.match_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.sigmoid_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged

    def build_merge_siamese(self, graph):
        print("building merge siamese network...")
        output = self.network.merge_siamese_network()
        loss = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.001)
        acc = self.accuracy.sigmoid_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, acc, opt, merged
    
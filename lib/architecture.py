import tensorflow as tf

class Layer(object):
    def __init__(self, graph, max_len, embedding_dim):
        self.graph = graph
        self.x1 = tf.placeholder(dtype=tf.int32, shape=(None, max_len), name='x1')
        self.x2 = tf.placeholder(dtype=tf.int32, shape=(None, max_len), name='x2')
        self.embedding_matrix = tf.placeholder(dtype=tf.float32, shape=(None, embedding_dim), name='x2')

    def embed(self, input):
        return tf.nn.embedding_lookup(self.embedding_matrix, input)

    def rnn_temporal_split(self, input):
        num_steps = input.get_shape().as_list()[1]
        embed_split = tf.split(axis=1, num_or_size_splits=num_steps, value=input)
        embed_split = [tf.squeeze(x, axis=[1]) for x in embed_split]
        return embed_split

    def stacked_biRNN(self, input, cell_type, n_layers, network_dim):
        xs = self.rnn_temporal_split(input)
        dropout = lambda y : tf.contrib.rnn.DropoutWrapper(y, output_keep_prob=0.5, seed=42)

        fw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)], 
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        bw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)], 
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        fw_cells = [dropout(fw_cell(network_dim)) for fw_cell in fw_cells]
        bw_cells = [dropout(bw_cell(network_dim)) for bw_cell in bw_cells]
        fw_stack = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_stack = tf.contrib.rnn.MultiRNNCell(bw_cells)
        outputs, fw_output_state, bw_output_state = tf.contrib.rnn.static_bidirectional_rnn(fw_stack,
                                                                bw_stack,
                                                                xs,
                                                                dtype=tf.float32)
        
        return outputs, fw_output_state, bw_output_state  

    def biRNN(self, input, cell_type, network_dim):
        xs = self.rnn_temporal_split(input)
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

    def rnn(self, input, cell_type, network_dim):
        xs = self.rnn_temporal_split(input)
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
        d = tf.nn.dropout(h1, keep_prob = 0.5, seed = 42)
        W2 = tf.get_variable(name="W2_"+name, shape=[hidden_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="b2_"+name, shape=[output_dim], initializer=tf.contrib.layers.xavier_initializer())
        out = tf.matmul(d, W2) + b2
        bn_out = tf.nn.batch_normalization(out, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        return bn_out

class Network(Layer):
    def __init__(self, graph, max_len, embedding_dim):
        super(Network, self).__init__(graph, max_len, embedding_dim)
    
    def fc_network(self):
        embed = self.embed(tf.concat([self.x1, self.x2], axis=1))
        with tf.variable_scope("output", reuse=None) as scope:
            output = self.dense_unit(embed, "feedforward", input_dim=300*2, hidden_dim=100, output_dim=2)
        return output

    def siamese_stacked_fc_network(self):
        embed = self.embed(tf.concat([self.x1, self.x2], axis=1))
        with tf.variable_scope("x1", reuse=None) as scope:  
            repr, _, _ = self.stacked_biRNN(input=embed, cell_type="LSTM", n_layers=3, network_dim=300)
        with tf.variable_scope("output", reuse=None) as scope:
            h4 = self.dense_unit(input=repr[-1], name="h4", input_dim=300*2, hidden_dim=300, output_dim=300)
            h5 = self.dense_unit(input=h4, name="h5", input_dim=300, hidden_dim=300, output_dim=300)
            h6 = self.dense_unit(input=h5, name="h6", input_dim=300, hidden_dim=300, output_dim=300)
            h7 = self.dense_unit(input=h6, name="h7", input_dim=300, hidden_dim=300, output_dim=300)
            h8 = self.dense_unit(input=h7, name="h8", input_dim=300, hidden_dim=300, output_dim=300)
            output = self.dense_unit(input=h8, name="output", input_dim=300, hidden_dim=300,  output_dim=2)
        return output

    def siamese_fc_network(self):
        x1_embed = self.embed(self.x1)
        x2_embed = self.embed(self.x2)
        with tf.variable_scope("x1", reuse=None) as scope:  
            q1_repr, _, _ = self.biRNN(input=x1_embed, cell_type="LSTM", network_dim=512)
        with tf.variable_scope("x2", reuse=None) as scope:  
            q2_repr, _, _ = self.biRNN(input=x2_embed, cell_type="LSTM", network_dim=512)
        with tf.variable_scope("output", reuse=None) as scope:
            h4 = self.dense_unit(input=tf.concat([q1_repr[-1], q2_repr[-1]], axis=1), name="h4", 
                                 input_dim=512*4, hidden_dim=256, output_dim=128)
            h5 = self.dense_unit(input=h4, name="h5", input_dim=128, hidden_dim=128, output_dim=128)
            h6 = self.dense_unit(input=h5, name="h6", input_dim=128, hidden_dim=128, output_dim=128)
            h7 = self.dense_unit(input=h6, name="h7", input_dim=128, hidden_dim=128, output_dim=128)
            h8 = self.dense_unit(input=h7, name="h8", input_dim=128, hidden_dim=128, output_dim=128)
            output = self.dense_unit(input=h8, name="output", input_dim=128, hidden_dim=64, output_dim=2)
        return output
    
    def siamese_network(self):
        x1_embed = self.embed(self.x1)
        x2_embed = self.embed(self.x2)
        with tf.variable_scope("x1", reuse=None) as scope:  
            q1_repr, _, _ = self.stacked_biRNN(input=x1_embed, cell_type="LSTM", n_layers=3, network_dim=512)
        with tf.variable_scope("x2", reuse=None) as scope:  
            q2_repr, _, _ = self.stacked_biRNN(input=x2_embed, cell_type="LSTM", n_layers=3, network_dim=512)
        with tf.variable_scope("output", reuse=None) as scope:
            output = self.dense_unit(input=tf.concat([q1_repr[-1], q2_repr[-1]], axis=1), name="h4", 
                                     input_dim=512*4,  hidden_dim=300, output_dim=2)
        return output
    
    def match_network(self):
        x1_embed = self.embed(self.x1)
        x2_embed = self.embed(self.x2)
        with tf.variable_scope("x1", reuse=None) as scope:  
            x1_outputs, x1_fw_final, x1_bw_final = self.biRNN(input=x1_embed, cell_type="LSTM", network_dim=100)
        with tf.variable_scope("x2", reuse=None) as scope:  
            x2_outputs, x2_fw_final, x2_bw_final = self.biRNN(input=x2_embed, cell_type="LSTM", network_dim=100)
        with tf.variable_scope("match", reuse=None) as scope:
            m1, m2 = self.matchLayer(x1_outputs=x1_outputs, 
                                     x1_fw_final=x1_fw_final, 
                                     x1_bw_final=x1_bw_final, 
                                     x2_outputs=x2_outputs, 
                                     x2_fw_final=x2_fw_final, 
                                     x2_bw_final=x2_bw_final, 
                                     weight_dim=512)
        with tf.variable_scope("agg_x1", reuse=None) as scope:  
            m1_agg, _, _ = self.biRNN(input=m1, cell_type="LSTM", network_dim=100)
        with tf.variable_scope("agg_x2", reuse=None) as scope:  
            m2_agg, _, _ = self.biRNN(input=m2, cell_type="LSTM", network_dim=100)
        with tf.variable_scope("output", reuse=None) as scope:      
            output = self.dense_unit(input=tf.concat([m1_agg[-1], m2_agg[-1]], axis=1), name="h4", 
                                     input_dim=100*4,  hidden_dim=128, output_dim=2)
        return output

    def merge_siamese_network(self):
        embed = self.embed(tf.concat([self.x1, self.x2], axis = 1))
        with tf.variable_scope("x1", reuse=None) as scope:  
            repr, _, _ = self.biRNN(input=embed, cell_type="LSTM", network_dim=512)
        with tf.variable_scope("output", reuse=None) as scope:
            output = self.dense_unit(input=repr[-1], name="h4", input_dim=512*2,  hidden_dim=128, output_dim=2)
        return output
    
    def contrastive_siamese_network(self):
        x1_embed = self.embed(self.x1)
        x2_embed = self.embed(self.x2)
        with tf.variable_scope("x1", reuse=None) as scope:  
            q1_repr, _, _ = self.biRNN(input=x1_embed, cell_type="LSTM", network_dim=512)
        with tf.variable_scope("x2", reuse=None) as scope:  
            q2_repr, _, _ = self.biRNN(input=x2_embed, cell_type="LSTM", network_dim=512)
        output = tf.squeeze(tf.norm(q1_repr[-1] - q2_repr[-1], ord=1, axis=1, keep_dims=True))
        return output



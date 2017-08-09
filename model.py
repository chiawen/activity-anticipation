import tensorflow as tf

class ActionPredictor(object):
    def __init__(self, n_classes, feature_length, learning_rate, max_length, hidden_size):
        self.n_classes = n_classes
        self.lr = learning_rate
        self.feature_length = feature_length
        self.max_length = max_length
        self.hidden_size = hidden_size
    
    def build_model(self):
        self.seq_in = tf.placeholder(tf.float32, [None, self.max_length, self.feature_length])
        self.seq_length = tf.placeholder(tf.int32, [None])
        self.label = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        seq_flat = tf.reshape(self.seq_in, [-1, self.feature_length])
        # [batch_size * max_length, hidden_size]
        seq_embed = tf.contrib.layers.fully_connected(seq_flat, self.hidden_size)
        seq_embed = tf.reshape(seq_embed, [-1, self.max_length, self.hidden_size])

        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        init_state = cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32)

        outputs, states = tf.nn.dynamic_rnn(cell, self.seq_in, sequence_length=self.seq_length,
                                            initial_state=init_state, dtype=tf.float32)

        last = self.last_relevant(outputs, self.seq_length)
        
        # Fully connected layer
        #fc1 = tf.contrib.layers.fully_connected(last, 20)
        #fc1 = tf.contrib.layers.dropout(fc1, keep_prob=self.keep_prob)
        
        logits = tf.contrib.layers.fully_connected(last, self.n_classes, activation_fn=None)

        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.label, logits=logits))

        self.prediction = tf.argmax(logits, 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.label, 1)), tf.float32))

        # Optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars), 5)
        self.updates = self.optimizer.apply_gradients(zip(self.grads, self.tvars), global_step=self.global_step)
        
    # Refernece: https://stackoverflow.com/questions/41273361/get-the-last-output-of-a-dynamic-rnn-in-tensorflow
    @staticmethod
    def last_relevant(output, length):
        batch_range = tf.range(tf.shape(output)[0])
        last_index = length - 1
        indices = tf.stack([batch_range, last_index], axis=1)
        relevant = tf.gather_nd(output, indices)

        return relevant


from __future__ import print_function

import tensorflow as tf
import numpy as np
from inception_v3 import inception_v3, inception_v3_arg_scope
import inception_preprocessing

slim = tf.contrib.slim

class Extractor():
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint
        self.sess = tf.Session()
        self.width = inception_v3.default_image_size
        self.height = inception_v3.default_image_size
    
        self.build_model()

        self.init_fn = slim.assign_from_checkpoint_fn(
                self.checkpoint, slim.get_model_variables('InceptionV3'))
        self.init_fn(self.sess)

    def build_model(self):
        self.input = tf.placeholder(tf.uint8, [None, None, 3])
        self.processed_image = inception_preprocessing.preprocess_image(self.input, self.height, self.width, is_training=False)
        self.processed_images = tf.expand_dims(self.processed_image, 0)

        with slim.arg_scope(inception_v3_arg_scope()):
            self.logits, self.end_points = inception_v3(self.processed_images, num_classes=1001, is_training=False)
        self.probabilities = tf.nn.softmax(self.logits)

    def extract(self, image):
        # Get the endpoints from each layer
        feed_dict = {self.input: image}
        features = self.sess.run(self.end_points, feed_dict=feed_dict)

        # Get 2048-d features
        prelogits = features['PreLogits']
        dense_features = np.squeeze(prelogits)

        return dense_features

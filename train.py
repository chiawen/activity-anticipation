from __future__ import division

import os
import sys
import json
import logging

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from model import ActionPredictor
import data_utils

logFormatter = logging.Formatter('[%(asctime)s] %(message)s')
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

fileHandler = logging.FileHandler('./activity-anticipation.log')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

tf.flags.DEFINE_integer('feature_size', 2048, 'Length of feature vectors')
tf.flags.DEFINE_integer('hidden_size', 16, 'Size of RNN hidden state')

tf.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.flags.DEFINE_integer('epochs', 15, 'Number of training epochs')
tf.flags.DEFINE_float('val_ratio', 0.2, 'Ratio of validation data')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate')
tf.flags.DEFINE_float('keep_prob', 0.2, 'Dropout keep probability')

tf.flags.DEFINE_string('train_dir', './', 
                        'The directory to save the model files in')
tf.flags.DEFINE_string('train_data_feat', 'features/',
                        'The directory of training data features')
tf.flags.DEFINE_string('train_file', 'train_file.csv',
                        'Training data ids and labels')
tf.flags.DEFINE_string('log_dir', 'log/',
                        'The directory to save checkpoints and log events')
tf.flags.DEFINE_integer('save_every', 30, 'Save frequency')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

#rootLogger.info(FLAGS.__flags)
param_string = '\nParameters:\n'
for attr, value in sorted(FLAGS.__flags.items()):
    param_string = param_string + '{}: {}\n'.format(attr, value)
rootLogger.info(param_string)

def count_trainable_parameters():
    total_parameters = 0
    # Iterating over all variables
    for variable in tf.trainable_variables():
        variable_parameters = 1
        shape = variable.get_shape()
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    
    return total_parameters

def plot_graph(train_curve, val_curve, y_label, dir_path):
    plt.clf()

    curve1, = plt.plot(train_curve, 'r', label='training')
    curve2, = plt.plot(val_curve, 'b', label='validation')
    plt.legend(handles=[curve1, curve2])
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    
    save_path = os.path.join(dir_path, y_label+'.jpg')
    plt.savefig(save_path)


def train_model(model, train, validation, FLAGS):
    with tf.Graph().as_default() as graph:
        
        model.build_model()

        # Summaries
        train_loss_sum = tf.summary.scalar('training/loss', model.loss)
        val_loss_sum = tf.summary.scalar('validation/loss', model.loss)
        train_acc_sum = tf.summary.scalar('training/accuracy', model.accuracy)
        val_acc_sum = tf.summary.scalar('validation/accuracy', model.accuracy)
        # Keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in zip(model.grads, model.tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name),
                                    tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        train_summary = tf.summary.merge([train_loss_sum, train_acc_sum] + grad_summaries)
        validation_summary = tf.summary.merge([val_loss_sum, val_acc_sum])
    
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    
        # Number of model parameters
        model_parameters = count_trainable_parameters()
        rootLogger.info('Number of total trainable parameters: {}'.format(model_parameters))
    
    
    log_dir = os.path.join(FLAGS.train_dir, FLAGS.log_dir)
    sv = tf.train.Supervisor(graph=graph, logdir=log_dir, saver=saver, 
                            save_model_secs=0, summary_op=None)

    train_loss_plt = []
    val_loss_plt = []
    train_acc_plt = []
    val_acc_plt = []
    
    # Start training
    rootLogger.info('Start training')
    with sv.managed_session() as sess:
        # Initialization is taken care of by the managed_session
        
        num_batches = int(np.ceil(train.num_examples() / FLAGS.batch_size))

        if validation.num_examples() > 0:
            val = True
            val_num_batches = int(np.ceil(validation.num_examples() / FLAGS.batch_size))
        else:
            val = False

        for epoch in range(FLAGS.epochs):
            #if sv.should_stop():
            #   break
           
            # Training
            train_loss = []
            train_accuracies = []
            train.reset_index_in_epoch()
            for i in range(num_batches):
                batch_seq_in, batch_label, batch_seq_length = train.next_batch(FLAGS.batch_size)

                feed_dict = {model.seq_in: batch_seq_in,
                            model.label: batch_label,
                            model.seq_length: batch_seq_length,
                            model.keep_prob: FLAGS.keep_prob}               
                
                
                _, loss, acc, curr_step, summary = sess.run([model.updates, 
                                                    model.loss, 
                                                    model.accuracy, 
                                                    model.global_step,
                                                    train_summary], 
                                                 feed_dict=feed_dict)
                
                train_loss.append(loss)
                train_accuracies.append(acc)
                
                sv.summary_computed(sess, summary)
                
            
            train_loss_mean = np.mean(train_loss)
            train_accuracies_mean = np.mean(train_accuracies)

            rootLogger.info('Training step {}, epoch {}'.format(curr_step, epoch))
            rootLogger.info('Training loss: {:.4f}, training accuracy: {:.4f}'
                                .format(train_loss_mean, train_accuracies_mean))
            
            train_loss_plt.append(train_loss_mean)
            train_acc_plt.append(train_accuracies_mean)
            
            # Compute loss over validation data
            if val:
                val_loss = []
                val_accuracies = []
                validation.reset_index_in_epoch()
                for i in range(val_num_batches):
                    batch_seq_in, batch_label, batch_seq_length = validation.next_batch(FLAGS.batch_size)

                    feed_dict = {model.seq_in: batch_seq_in,
                                model.label: batch_label,
                                model.seq_length: batch_seq_length,
                                model.keep_prob: 1.0}
                    
                    loss, acc, summary = sess.run([model.loss, 
                                            model.accuracy,
                                            validation_summary], 
                                            feed_dict=feed_dict)
                    val_loss.append(loss)
                    val_accuracies.append(acc)
                    sv.summary_computed(sess, summary)
                val_loss_mean = np.mean(val_loss)                
                val_accuracies_mean = np.mean(val_accuracies)
                
            rootLogger.info('Validation loss: {:.4f}, validation accuracy: {:.4f}'
                                .format(val_loss_mean, val_accuracies_mean)) 
        
            val_loss_plt.append(val_loss_mean)
            val_acc_plt.append(val_accuracies_mean)


            # Plot accuracy and loss graph
            plot_graph(train_loss_plt, val_loss_plt, 'loss', FLAGS.train_dir)
            plot_graph(train_acc_plt, val_acc_plt, 'accuracy', FLAGS.train_dir)

            # Save checkpoints
            if curr_step % FLAGS.save_every == 0:
                sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
                rootLogger.info('Saved model checkpoint to {}'.format(sv.save_path))

        # Finish training
        sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
        rootLogger.info('Finished training! Saved model checkpoint to {}'.format(sv.save_path))

def main(_):

    # Get data
    train, validation, data_msg = data_utils.get_data_sets(FLAGS.train_file, FLAGS.train_data_feat, one_hot=True, val_ratio=FLAGS.val_ratio, max_seq_length=30, data_augmentation=True, extra_samples=4)
    rootLogger.info(data_msg)
    rootLogger.info('training data size: {}'.format(train.num_examples()))
    rootLogger.info('validation data size {}'.format(validation.num_examples()))

    rootLogger.info('training data.shape: {}'.format(train.data().shape))
    rootLogger.info('training labels.shape: {}'.format(train.labels().shape))
    n_classes = train.labels().shape[1]

    # Get the model
    model = ActionPredictor(n_classes, FLAGS.feature_size, FLAGS.learning_rate, 30,
                            FLAGS.hidden_size)

    # Start training
    train_model(model, train, validation, FLAGS)



if __name__ == '__main__':
    tf.app.run()
